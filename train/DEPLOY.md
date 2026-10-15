# Развёртывание frame-модели в проде: гайд backend-разработчика

Основная NLU-модель телефонного регистратурного контура (решение в
корневом README): decoder-only Transformer 6.84M (256/8/8), вход — текст
реплики (после ASR), выход — семантический фрейм из замкнутого набора
токенов. Никаких GPU, внешних сервисов и секретов не нужно: инференс на
CPU ~10 мс/запрос (замер: Python 3.12, torch 2.4.1+cpu, один поток).

## Контракт модели

Вход: строка реплики пациента (русский, нижний/любой регистр, знаки
препинания допустимы — обучение видело три STT-стииля).

Выход: строка токенов фрейма, всегда синтаксически валидная — генерация
идёт под конечным автоматом (`constrained.py`), невалидную
последовательность модель выдать не может:

```
<INTENT> <SPECIALTY> <TIME> T_SCOPE T_DAY T_TIME </TIME> <SYMPTOM_*>... [<NEEDS_CLARIFICATION>] <END_FRAME>
```

- 13 интентов + `<YES>`/`<NO>` (голые подтверждения) + `<UNSUPPORTED>`;
- 9 специальностей + `<UNKNOWN_SPECIALTY>`;
- время — фактор-токены из замкнутой сетки (`T_TOMORROW`, `T_FRI`,
  `T_H17_30`, `T_AMBIG`, ...) либо `<T_UNKNOWN>`;
- порядок симптомов не значим: сравнивайте как множество.

## Артефакты и совместимость (главный источник поломок)

| Файл | Роль |
| :-- | :-- |
| `models/model.pt` | веса + конфиг (`{"d_model": 256, "n_layers": 8, "n_heads": 8}`) |
| `models/tokenizer_v2.json` | токенизатор ОСНОВНОЙ модели (ByteLevel BPE + спецтокены, HF `tokenizers`) |
| `train/model.py` | `TinyGPT` — архитектура |
| `train/constrained.py` | `generate_frame()` + автомат; **импортирует `synth/vocab.py`** (списки допустимых токенов автомата) |
| `synth/vocab.py` | единый источник истины словаря |

Три правила:

1. **Токенизатор — только `tokenizer_v2.json`.** `models/tokenizer.json`
   принадлежит теггеру (старые id токенов): модель молча выдаст мусор.
   Проверка при старте: у токенизатора должны существовать id у
   `<BOS>/<USER>/<ASSISTANT>/<EOS>/<PAD>` (assert уже внутри
   `FrameModelNlu`).
2. `model.pt`, `tokenizer_v2.json` и `synth/vocab.py` обязаны быть из
   одного коммита: словарь задаёт списки токенов автомата. Обновлять —
   только комплектом.
3. `models/model.onnx`, `*.tflite` в репозитории устарели относительно
   текущих весов — для ONNX-пути перегенерируйте (ниже).

## Быстрый старт: FastAPI-сервис

Минимальный рабочий сервис (файл `api.py` в корне репозитория,
`.venv-tr` или любое окружение с `torch`+`tokenizers`+`fastapi`):

```python
# uvicorn api:app --host 0.0.0.0 --port 8000
import sys
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "train"))

from model import TinyGPT                      # noqa: E402
from constrained import generate_frame         # noqa: E402

app = FastAPI()

tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer_v2.json"))
ckpt = torch.load(ROOT / "models" / "model.pt", map_location="cpu",
                  weights_only=False)
model = TinyGPT(tok.get_vocab_size(), **ckpt["config"])
model.load_state_dict(ckpt["model"])
model.eval()
IDS = {t: tok.token_to_id(t) for t in
       ("<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>")}
assert all(v is not None for v in IDS.values()), "не тот токенизатор?"


class ParseReq(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse")
def parse(req: ParseReq):
    frame, tokens = generate_frame(
        model, tok, req.text, torch.device("cpu"),
        bos=IDS["<BOS>"], user=IDS["<USER>"],
        assistant=IDS["<ASSISTANT>"], eos=IDS["<EOS>"],
        pad=IDS["<PAD>"])
    return {"frame": frame, "tokens": tokens}
```

Модель грузится один раз на процесс; эндпоинт синхронный (uvicorn
исполняет его в threadpool) — `generate_frame` обёрнут в
``@torch.no_grad()`` и потокобезопасен на общих весах в режиме eval.
Батчинг не нужен: один запрос = ≤24 авторегресс-шагов по ~0.5–1 мс.

Ответ: `{"frame": "<BOOK> <GP> <TIME> T_TOMORROW T_MORNING </TIME> <END_FRAME>", ...}`.

## Разбор фрейма в структуру

Вариант A — готовый адаптер телефонии (только stdlib, тянет `telephone/`):

```python
sys.path.insert(0, str(ROOT))
from datetime import date
from telephone.nlu import frame_from_tokens
frame = frame_from_tokens(tokens, text, today=date.today())
# frame.intent / frame.specialty / frame.symptoms / frame.time_prefs ...
```

Он же дозаполняет словарной заглушкой детерминированные слоты (ФИО нет —
для этого есть Natasha-контур, дата рождения, полис, да/нет-оттенки) и
разрешает `T_MON..T_SUN`/`T_TOMORROW` в конкретную дату относительно
`today`. Передавайте `today=date.today()` на каждый запрос — модель
отдаёт только относительные факторы.

Вариант B — тонкий самописный парсер (сервис без `telephone/`):
образец `parse_target()` в `train/train.py` + таблица
`TEMPORAL_ENTRIES` в `synth/vocab.py` для времени.

## Если разворачивается стек телефонии (DialogManager)

`telephone/` — менеджер диалога поверх модели; NLU-адаптер `FrameModelNlu`
уже всё делает (ленивая загрузка, кэш на класс). Обязательные env:

```bash
export TEL_FMODEL_RESOLVER=0    # ВАЖНО: e5-резолвер входа включён кодом
                                # по умолчанию, на текущих весах он портит
                                # фразы («меня записывали» -> «меня
                                # высыпало») — выключать
export TEL_NAMES=natasha        # контур ФИО (off — выключить)
```

Зависимости: `.venv-tr` (python 3.12, torch 2.4.1+cpu, tokenizers,
natasha). Смартфон-пробы — `.llm-local/run_battery.py` (19 сценариев на
StubMis).

## Лёгкий образ: ONNX Runtime вместо torch

Чтобы убрать torch (~200 МБ) из Docker-образа (~50 МБ onnxruntime):

1. Перегенерировать экспорт под текущие веса:
   `.venv-tr/Scripts/python train/export_onnx.py` (скрипт сам гоняет
   проверку парности логитов).
2. Обёртка `OnnxModel` в `train/export_onnx.py` реализует тот же
   `__call__(idx) -> logits`, что и `TinyGPT` — `generate_frame`
   работает без изменений (автомат и токенизатор всё равно нужны:
   `tokenizers` + `synth/vocab.py`).
3. Контракт: вход `int64[1,128]` (хвост — `<PAD>`), выход
   `float32[1,128,V]`, логиты шага — `logits[0, L-1]`.

Docker-скелет (torch-вариант):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir tokenizers fastapi uvicorn
COPY api.py train/model.py train/constrained.py ./
COPY synth/vocab.py synth/
COPY models/model.pt models/tokenizer_v2.json models/
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Обновление модели

1. Пересборка датасета и обучение (см. корневой README):
   `synth/build_dataset.py` → `train/train.py --device dml --epochs 40
   --d-model 256 --n-layers 8 --n-heads 8 --save models/model.pt
   --tokenizer models/tokenizer_v2.json` (DirectML — машина с Radeon;
   чекпойнт проходит самопроверку при сохранении).
2. Приёмка ОБЯЗАТЕЛЬНА перед деплоем:
   - `train/evaluate_all.py model.pt tokenizer_v2.json` — таблица
     метрик против предыдущей версии (ориентир текущей: val/iid exact
     0.992, OOD intent 0.945);
   - `train/test_new_intents.py` — батарейка живых формулировок;
   - `.llm-local/run_battery.py` — если в проде DialogManager.
3. Комплект `model.pt + tokenizer_v2.json + synth/vocab.py` из одного
   коммита; для ONNX-пути — перегенерация экспорта.
4. Канарейка: доля трафика на новую версию + сравнение
   intent-распределения и доли `<UNSUPPORTED>` со старой.

## Мониторинг и известные отказы

- **Главный отказ-режим: незнакомая формулировка → `<UNSUPPORTED>`.**
  На свежих живых парафразах ~50% (батарейка `test_new_intents.py`);
  главный сигнал для мониторинга — рост доли `UNSUPPORTED` в трафике.
  Это мягкая деградация: реплика уходит оператору/дефолт-сценарию.
- Косвенные отмены («не приду, снимите запись», «приём не актуален») и
  отыменные формы («отмена визита») — вне покрытия (после словарной
  страховки `RuleNlu` частично закрываются: у неё есть «не приду» и
  «снять запись»).
- `models/failures_report.md` — полный список провалов на внешних
  наборах (HASD/CLINC): читать перед демо заказчику.
- Симптомы сравнивать множеством (порядок в target не фиксирован).
- Специальность из реплики: врач по симптомам модель НЕ выбирает — это
  правило регламента (таблица 2.3), не ограничение модели.

## Чеклист деплоя

- [ ] `model.pt` + `tokenizer_v2.json` + `synth/vocab.py` из одного коммита
- [ ] стартовый assert на спецтокены токенизатора прошёл
- [ ] `TEL_FMODEL_RESOLVER=0` (если используется `FrameModelNlu`)
- [ ] `/health` отвечает, smoke-прогон 5–10 фраз по телефону прошёл
- [ ] метрики приёмки (`evaluate_all`, батарейка) зафиксированы рядом с версией
- [ ] алерт на долю `<UNSUPPORTED>` подключён
