# Интеграция модели на Android

## Файлы

| Файл | Размер | Что это |
| :-- | :-- | :-- |
| `model.tflite` | 24 МБ (fp32) | decoder-only Transformer, веса из `model.pt` |
| `model_int8.tflite` | 7 МБ (weight int8) | то же, dynamic-range квантизация; фреймы совпадают с fp32 |
| `model.onnx` | 24 МБ (fp32) | ONNX-экспорт — для бэкенда или приложений БЕЗ sherpa-onnx |
| `tokenizer.json` | ~0.3 МБ | ByteLevel BPE + спецтокены (формат HF tokenizers) |
| `failures_report.md` | — | кейсы, которые модель НЕ обрабатывает (читать до демо) |

Runtime: **TensorFlow Lite / LiteRT**. Если в приложении уже есть sherpa-onnx
(ASR), это единственный мирный вариант: Java-обёртка ONNX Runtime тянет
вторую `libonnxruntime.so` с несовместимым ABI (версионированные символы
QNN-сборки sherpa против неверсионированных Microsoft), один процесс —
одна библиотека с таким именем. TFLite — независимый рантайм, конфликт
не возникает. ONNX-путь (`model.onnx`) годится только когда sherpa
в процессе не живёт.

Зависимость Gradle: `org.tensorflow:tensorflow-lite` (новое имя —
`com.google.ai.edge.litert`), артефакты — в `assets/`. Оба tflite
проверены парностью с torch (логиты fp32 до 1e-5, сквозная генерация
совпадает) — см. `train/export_tflite.py`.

## Контракт ввода/вывода (одинаков для tflite и onnx)

```
input_ids : int64[1, 128]          // всегда 128, хвост паддится <PAD>
logits    : float32[1, 128, 3000]
```

Последовательность: `<BOS> <USER> {токены текста} <ASSISTANT> {целевые токены}`,
далее паддинг `<PAD>` до 128. Каузальная маска внутри модели гарантирует,
 что паддинг-хвост не влияет на предыдущие позиции.
Для очередного токена брать `logits[0, L-1, :]`, где L — текущая длина
(без паддинга), и применять маску автомата.

## Цикл генерации (жадный, с автоматом слотов)

```
seq = [BOS, USER] + tokenize(текст) + [ASSISTANT]
state = INTENT
повторять до 24 шагов:
    logits = model(seq)[0, -1]
    allowed = допустимые токены по state      // см. таблицу ниже
    next = argmax(logits[allowed])
    если next == EOS: прекратить
    seq += next; state = переход(state, next)
```

Состояния автомата (референс — `train/constrained.py`):

```
INTENT      -> {BOOK, RESCHEDULE, CANCEL, UNSUPPORTED}
UNSUPPORTED -> END_FRAME;  END_FRAME -> EOS
SPECIALTY   -> 10 токенов специальностей (включая UNKNOWN_SPECIALTY)
TIME        -> {T_UNKNOWN} | <TIME> ...
внутри <TIME>: факторы категорий ОБЛАСТЬ -> ДЕНЬ -> ВРЕМЯ, по одному из
  категории, минимум один; закрывать слот можно только после фактора
SYMPTOMS    -> любое число разных <SYMPTOM_*> | NEEDS_CLARIFICATION | END_FRAME
NEEDS_CLARIFICATION -> END_FRAME
```

Автомат гарантирует синтаксически валидный фрейм всегда — невалидный
вывод невозможен физически, от модели требуется только выбор значений.

Вызов модели через TFLite (Kotlin), один шаг:

```kotlin
val opts = Interpreter.Options().apply { numThreads = 2 }
val tflite = Interpreter(loadModelFile("model_int8.tflite"), opts)
// input: Array<LongArray> формы [1][128] (хвост — padId), output:
// Array<Array<FloatArray>> формы [1][128][3000]
val inBuf = arrayOf(LongArray(128).also { seqCopyInto(it, padId) })
val outBuf = arrayOf(Array(128) { FloatArray(3000) })
tflite.run(inBuf, outBuf)
val logits = outBuf[0][seqLen - 1]        // для очередного токена
val next = allowedIds.maxBy { logits[it] } // argmax по маске автомата
```

## Токенизация на устройстве

`tokenizer.json` — стандартный формат HF `tokenizers`: сначала матчинг
спецтокенов (все `<...>` и `T_*` — единые id), затем ByteLevel-претокенайзер
и BPE-слияния. Варианты: JNI-обёртка над `tokenizers` (Rust) или прямая
реализация BPE на Kotlin по `vocab` + `merges` из json (~200 строк).

## Ориентиры производительности

- ~6M параметров, seq = 128 на каждый шаг (KV-кэша нет, контекст
  пересчитывается целиком — как в ONNX-варианте): полный фрейм
  (7-15 шагов) — десятки-сотни мс на средних смартфонах; fp32 tflite
  ~24 МБ, int8 ~7 МБ, RAM < 50 МБ.
- Текст после STT: можно подавать как есть (модель обучалась на трёх
  стилях регистра/пунктуации); спецтокены не токенизировать текстом —
  только через id.

## Что модель умеет и не умеет (кратко)

Умеет: BOOK/RESCHEDULE/CANCEL/UNSUPPORTED, 9 специальностей (включая хирурга
и психотерапевта) + UNKNOWN,
13 симптомов, время фактор-токенами (`T_NEXT_WEEK T_FRI T_H17_30`).
Провалы — см. `failures_report.md`: невидимые синонимы специальностей,
академический стиль HASD, часть живой разговорной речи.
