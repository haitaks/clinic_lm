# Интеграция обученной модели: бэкенд и Android

> Актуальный гайд по прод-развёртыванию бэкенда — **`train/DEPLOY.md`**
> (основная модель = `train/` + `tokenizer_v2.json`). Ниже — краткая
> сводка; примеры приведены к текущим артефактам.
>
> ВАЖНО: `model.onnx`/`*.tflite` в этом каталоге устарели относительно
> текущего `model.pt` — перед использованием перегенерируйте
> (`train/export_onnx.py`, затем `train/export_tflite.py`).

Всё для интеграции уже подготовлено: модель экспортирована в ONNX и TFLite
с проверкой парности (`train/export_onnx.py`, `train/export_tflite.py`),
для Android написана инструкция (`models/ANDROID_INTEGRATION.md`). Ниже —
простейшие пути для обоих случаев.

| Файл | Размер | Назначение |
| :-- | :-- | :-- |
| `model.pt` | 24 МБ | чекпойнт torch — основной вариант для бэкенда |
| `model.onnx` | 24 МБ | ONNX-экспорт, вход `int64[1,128]`, выход `float32[1,128,V]` (перегенерировать) |
| `model.tflite` | 24 МБ | TFLite fp32 — основной вариант для Android (офлайн) |
| `model_int8.tflite` | 7 МБ | TFLite weight-int8 — то же, компактнее для APK |
| `tokenizer_v2.json` | 0.3 МБ | токенизатор ОСНОВНОЙ модели (HF tokenizers); `tokenizer.json` — теггера, для неё старые id! |
| `failures_report.md` | — | кейсы, которые модель НЕ обрабатывает (читать до демо) |

## Бэкенд — самый простой вариант: FastAPI + готовый `generate_frame`

Готовая функция инференса «текст → фрейм» — `generate_frame()` в
`train/constrained.py`: загружает модель, жадно генерирует с ограничением
автоматом и возвращает строку токенов. Весь сервис — один файл ~25 строк:

```python
# api.py  →  uvicorn api:app
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer
from model import TinyGPT                      # train/
from constrained import generate_frame         # train/

app = FastAPI()
tok = Tokenizer.from_file("models/tokenizer_v2.json")   # НЕ tokenizer.json (теггер)!
ckpt = torch.load("models/model.pt", map_location="cpu")
model = TinyGPT(tok.get_vocab_size(), **ckpt["config"]); model.load_state_dict(ckpt["model"]); model.eval()
ids = {t: tok.token_to_id(t) for t in ("<BOS>","<EOS>","<USER>","<ASSISTANT>","<PAD>")}

class Req(BaseModel):
    text: str

@app.post("/parse")
def parse(req: Req):
    frame, tokens = generate_frame(model, tok, req.text, torch.device("cpu"), **ids)
    return {"frame": frame, "tokens": tokens}
```

Запуск: `pip install fastapi uvicorn torch tokenizers`, затем
`uvicorn api:app` (из корня проекта, с `train/` в `sys.path`).

Этого достаточно: модель 24 МБ, CPU-инференс — миллисекунды, батчинг не нужен.
ONNX на бэкенде не обязателен — он нужен был для Android. Если захочется
облегчить Docker-образ (~200 МБ torch → ~50 МБ onnxruntime), обёртка
`OnnxModel` уже лежит в `train/export_onnx.py` — можно скопировать как есть.

Канонические токены времени (`<DATE_TOMORROW>` и т.п.) бэкенд считает
хардкод-парсером из `TEMPORAL_ENTRIES` (см. README), а не моделью: `/parse`
отдаёт фрейм + сырой спан, а маппинг в слоты расписания делается рядом
отдельной чистой функцией.

## Android — два пути по возрастанию сложности

### Вариант 1 (проще некуда): HTTP-вызов бэкенда

Если офлайн-режим не критичен — просто POST на `/parse`. Пять строк
Retrofit/Ktor, ноль портирования. Минусы: нужна сеть/сервер, STT-текст
гоняется по сети.

### Вариант 2 (штатный, офлайн): TFLite / LiteRT

Для него всё готово, шаги:

1. Зависимость `org.tensorflow:tensorflow-lite` (или
   `com.google.ai.edge.litert`), файлы `model_int8.tflite` (или
   `model.tflite`) и `tokenizer.json` — в `assets/`.
2. Интерпретатор — стандартные 3 строки TFLite API. Контракт тот же,
   что у ONNX: вход `int64[1,128]` (хвост паддится `<PAD>`), выход
   `float32[1,128,V]`, очередной токен — `logits[0, L-1]`.
3. Токенизатор: либо JNI-обёртка над Rust-`tokenizers`, либо прямая
   реализация BPE на Kotlin по `vocab`+`merges` из json (~200 строк) —
   оба варианта описаны в `ANDROID_INTEGRATION.md`.
4. Порт `FrameAutomaton` из `train/constrained.py` на Kotlin (~150 строк) —
   это обязательно: именно автомат гарантирует валидный фрейм всегда.

Важно: если в приложении живёт sherpa-onnx (ASR), берите именно TFLite —
Java-обёртка ONNX Runtime конфликтует с нативной сборкой sherpa по ABI
(две `libonnxruntime.so` в одном процессе; детали — `../Conclusion.md`).
Чистый ONNX-путь (`model.onnx` + `com.microsoft.onnxruntime:onnxruntime-android`)
годится только для приложений без sherpa.

Перегенерация артефактов после переобучения: `train/export_onnx.py`,
затем `train/export_tflite.py` (второй сам гоняет проверки парности
логитов и сквозной генерации — упало, значит в APK ничего менять нельзя).

## На что обратить внимание до демо

- `failures_report.md` — список кейсов, которые модель проваливает
  (невидимые синонимы специальностей, академический стиль HASD, часть живой
  речи). Метрики на тестах высокие (val/iid exact 0.992, актуальные цифры —
  в `metrics.json` и корневом README), но тест ≠ реальная речь.
- Списки токенов автомата (специальности/симптомы/время) имеют единственный
  источник истины — `synth/vocab.py`. При порте на Kotlin и при любых
  правках словаря синхронизируйтесь с ним, иначе автомат молча зарежет
  валидные токены.

## Итог

Для бэкенда — FastAPI поверх готового `generate_frame` (меньше часа работы).
Для Android — либо HTTP на этот же сервис, либо ONNX Runtime по
`ANDROID_INTEGRATION.md`, если нужен офлайн.
