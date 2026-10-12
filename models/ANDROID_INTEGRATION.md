# Интеграция модели на Android (Kotlin, TFLite / LiteRT)

> **Основной подход теперь — теггер** (один forward, сборка фрейма на
> Kotlin): гайд в `tagger/ANDROID_INTEGRATION.md`, артефакты
> `tagger/models/`. Этот документ — развёртывание GPT-варианта
> (авторегрессия + constrained decoding), он остаётся резервным контуром;
> токенайзер `tokenizer.json` общий для обеих архитектур.

Полный гайд: от файлов до работающего `parse(text): String` на телефоне.
Референсная реализация всей логики — `train/constrained.py` +
`train/export_tflite.py` (Python); здесь её точный порт на Kotlin.

## 1. Файлы

| Файл | Размер | Что это |
| :-- | :-- | :-- |
| `model_int8.tflite` | 7 МБ (weight int8) | **основной вариант**: dynamic-range квантизация, фреймы совпадают с fp32 |
| `model.tflite` | 24 МБ (fp32) | то же без квантизации — если важна максимальная точность логитов |
| `tokenizer.json` | ~0.3 МБ | ByteLevel BPE + спецтокены (формат HF `tokenizers`) |
| `kotlin_testvectors.json` | ~20 КБ | тест-векторы: текст → BPE-ids и ожидаемый фрейм (для юнит-теста) |
| `model.onnx` | 27 МБ | для бэкенда или приложений БЕЗ sherpa-onnx (см. ниже) |
| `failures_report.md` | — | кейсы, которые модель НЕ обрабатывает (читать до демо) |

Runtime: **TensorFlow Lite / LiteRT**. Если в приложении уже есть sherpa-onnx
(ASR), это единственный мирный вариант: Java-обёртка ONNX Runtime тянет
вторую `libonnxruntime.so` с несовместимым ABI (версионированные символы
QNN-сборки sherpa против неверсионированных Microsoft), один процесс —
одна библиотека с таким именем. TFLite — независимый рантайм, конфликт
не возникает. ONNX-путь годится только когда sherpa в процессе не живёт.

## 2. Зависимости и ассеты

```kotlin
// build.gradle(.kts) модуля app
dependencies {
    implementation("com.google.ai.edge.litert:litert:1.3.0")   // новое имя
    // или старое: implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.json:json:20240303")   // парс tokenizer.json/тестов
}
androidResources { noCompress += "tflite" }    // ассеты не сжимать
```

В `app/src/main/assets/` положить: `model_int8.tflite`,
`tokenizer.json`, `kotlin_testvectors.json` (последний — только для
инструментальных тестов, в release можно не включать).

## 3. Контракт модели

```
вход  : input_ids int64[1, 128]     // всегда 128, хвост паддится <PAD>
выход : logits    float32[1, 128, 3000]
```

Последовательность: `<BOS> <USER> {токены текста} <ASSISTANT> {целевые
токены}` + паддинг `<PAD>` до 128. Каузальная маска внутри модели
гарантирует, что хвост не влияет на предыдущие позиции. Для очередного
токена брать `logits[0, L-1, :]` (L — текущая длина без паддинга) и
делать argmax по маске автомата. KV-кэша нет: каждый шаг пересчитывает
весь контекст (7–15 шагов на фрейм, десятки–сотни мс на средних
смартфонах; numThreads = 2–4).

Идентификаторы спецтокенов (файксированы в `tokenizer.json`):

```
<PAD>=0 <UNK>=1 <BOS>=2 <EOS>=3 <USER>=4 <ASSISTANT>=5
```

## 4. Токенизатор на Kotlin (ByteLevel BPE)

Прямая реализация по `vocab` + `merges` из `tokenizer.json` (~100 строк,
без зависимостей). Алгоритм идентичен HF `tokenizers`: byte→unicode
таблица GPT-2 → regex-претокенайзер → BPE-слияния по рангу.

```kotlin
class BpeTokenizer(tokenizerJson: String) {
    private val vocab: Map<String, Int>
    private val ranks: Map<Pair<String, String>, Int>
    private val added: List<Pair<String, Int>>   // спецтокены, матчятся первыми
    private val byteToUni: Map<Int, String>

    init {
        val root = JSONObject(tokenizerJson)
        vocab = root.getJSONObject("model").getJSONObject("vocab")
            .let { v -> buildMap { v.keys().forEach { put(it, v.getInt(it)) } } }
        ranks = root.getJSONObject("model").getJSONArray("merges")
            .let { m -> buildMap {
                for (i in 0 until m.length()) {
                    val p = m.getString(i).split(" ")
                    put(p[0] to p[1], i)
                } } }
        added = root.getJSONArray("added_tokens")
            .let { a -> buildList {
                for (i in 0 until a.length()) {
                    val t = a.getJSONObject(i)
                    add(t.getString("content") to t.getInt("id"))
                } } }
        // GPT-2 byte-level: печатаемые латинские байты остаются собой,
        // остальные получают коды 256+n
        val bs = buildList {
            addAll(33..126); addAll(161..172); addAll(174..255)
        }
        var n = 0
        byteToUni = buildMap {
            for (b in 0..255) put(b, if (b in bs) b.toChar().toString()
                                   else (256 + n++).toChar().toString())
        }
    }

    private val splitRegex = Regex(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")

    fun encode(text: String): List<Int> {
        val out = ArrayList<Int>()
        var rest = text
        // 1) спецтокены вырезаются до BPE (в пользовательском тексте их
        //    обычно нет, но контракт должен соблюдаться)
        while (rest.isNotEmpty()) {
            val hit = added.filter { it.first.length > 1 && rest.startsWith(it.first) }
                .maxByOrNull { it.first.length }
            if (hit == null) break
            out.add(hit.second)
            rest = rest.substring(hit.first.length)
        }
        // 2) regex-сплит + byte-level + BPE-слияния
        for (piece in splitRegex.findAll(rest).map { it.value }) {
            var syms = piece.map { byteToUni[it.code]!! }
            while (syms.size > 1) {
                val best = syms.zipWithNext().filter { it in ranks }
                    .minByOrNull { ranks[it]!! } ?: break
                syms = buildList {
                    var i = 0
                    while (i < syms.size) {
                        if (i < syms.size - 1 && syms[i] to syms[i + 1] == best) {
                            add(syms[i] + syms[i + 1]); i += 2
                        } else { add(syms[i]); i += 1 }
                    }
                }
            }
            for (s in syms) out.add(vocab[s] ?: vocab["<UNK>"]!!)
        }
        return out
    }
}
```

Сверка с Python: для каждой фразы из `kotlin_testvectors.json` поле
`bpe_ids` — эталон (см. тест в §7).

## 5. Автомат слотов на Kotlin

Точный порт `FrameAutomaton` из `train/constrained.py`. Автомат
гарантирует синтаксически валидный фрейм всегда — невалидный вывод
невозможен физически, от модели требуется только выбор значений.

```kotlin
object FrameVocab {
    val INTENTS = listOf("<BOOK>", "<RESCHEDULE>", "<CANCEL>", "<UNSUPPORTED>")
    val SPECIALTY = listOf(
        "<GP>", "<DENTIST>", "<DERMATOLOGIST>", "<GYNECOLOGIST>",
        "<OPHTHALMOLOGIST>", "<NEUROLOGIST>", "<PSYCHOLOGIST>",
        "<SURGEON>", "<ENT>", "<UNKNOWN_SPECIALTY>")
    val SYMPTOMS = listOf(
        "<SYMPTOM_TOOTHACHE>", "<SYMPTOM_SORE_THROAT>", "<SYMPTOM_FEVER>",
        "<SYMPTOM_RASH>", "<SYMPTOM_BACK_PAIN>", "<SYMPTOM_HEADACHE>",
        "<SYMPTOM_EYE_PAIN>", "<SYMPTOM_ANXIETY>", "<SYMPTOM_COUGH>",
        "<SYMPTOM_STOMACH_PAIN>", "<SYMPTOM_DIZZINESS>",
        "<SYMPTOM_INSOMNIA>", "<SYMPTOM_ALLERGY>")
    // категории времени: ранг растёт ОБЛАСТЬ(0) -> ДЕНЬ(1) -> ВРЕМЯ(2),
    // внутри <TIME> факторы идут строго по возрастанию ранга
    val TIME_SCOPE = setOf(
        "T_NEXT_MONTH", "T_NEXT_WEEK", "T_THIS_MONTH", "T_THIS_WEEK", "T_WEEKEND")
    val TIME_DAY = setOf(
        "T_DAY_AFTER_TOMORROW", "T_FRI", "T_IN_2_DAYS", "T_IN_2_WEEKS",
        "T_IN_3_DAYS", "T_IN_MONTH", "T_IN_WEEK", "T_MON", "T_SAT", "T_SUN",
        "T_THU", "T_TODAY", "T_TOMORROW", "T_TUE", "T_WED")
    // относительное время («утром», «после работы»)
    val TIME_REL = setOf(
        "T_AFTERNOON", "T_AFTER_18", "T_AFTER_NOON", "T_AFTER_SHIFT",
        "T_AFTER_WORK", "T_BEFORE_NOON", "T_EARLY", "T_EVENING",
        "T_MORNING", "T_SOON")
    // сетка слотов приёма: 8:00-20:45 с шагом 15 минут (52 слота)
    val TIME_SLOTS = setOf(
        "T_H08_00", "T_H08_15", "T_H08_30", "T_H08_45",
        "T_H09_00", "T_H09_15", "T_H09_30", "T_H09_45",
        "T_H10_00", "T_H10_15", "T_H10_30", "T_H10_45",
        "T_H11_00", "T_H11_15", "T_H11_30", "T_H11_45",
        "T_H12_00", "T_H12_15", "T_H12_30", "T_H12_45",
        "T_H13_00", "T_H13_15", "T_H13_30", "T_H13_45",
        "T_H14_00", "T_H14_15", "T_H14_30", "T_H14_45",
        "T_H15_00", "T_H15_15", "T_H15_30", "T_H15_45",
        "T_H16_00", "T_H16_15", "T_H16_30", "T_H16_45",
        "T_H17_00", "T_H17_15", "T_H17_30", "T_H17_45",
        "T_H18_00", "T_H18_15", "T_H18_30", "T_H18_45",
        "T_H19_00", "T_H19_15", "T_H19_30", "T_H19_45",
        "T_H20_00", "T_H20_15", "T_H20_30", "T_H20_45")
}

class FrameAutomaton(private val idOf: (String) -> Int?) {
    private enum class S { INTENT, SPECIALTY, TIME, FACTORS, SYMPTOMS,
                           CLARIF, END_FRAME, DONE }
    private var state = S.INTENT
    private var lastRank = -1
    private var emitted = 0
    private val symptoms = mutableSetOf<String>()

    private fun rank(t: String) = when (t) {
        in FrameVocab.TIME_SCOPE -> 0
        in FrameVocab.TIME_DAY -> 1
        else -> 2
    }

    private fun ids(ts: Collection<String>) =
        ts.mapNotNull { idOf(it) }.toSet()

    fun allowed(): Set<Int> = when (state) {
        S.INTENT -> ids(FrameVocab.INTENTS)
        S.END_FRAME -> setOf(idOf("<END_FRAME>")!!)
        S.SPECIALTY -> ids(FrameVocab.SPECIALTY)
        S.TIME -> ids(setOf("<T_UNKNOWN>", "<TIME>"))
        S.FACTORS -> {
            val ok = (FrameVocab.TIME_SCOPE + FrameVocab.TIME_DAY +
                      FrameVocab.TIME_REL + FrameVocab.TIME_SLOTS +
                      setOf("T_AMBIG"))      // «в 15:00 или в 16:00»
                .filter { rank(it) > lastRank }
            (if (emitted > 0) ids(ok) + idOf("</TIME>")!! else ids(ok))
        }
        S.SYMPTOMS -> ids(FrameVocab.SYMPTOMS.filter { it !in symptoms }) +
                setOf(idOf("<NEEDS_CLARIFICATION>")!!, idOf("<END_FRAME>")!!)
        S.CLARIF -> setOf(idOf("<END_FRAME>")!!)
        S.DONE -> setOf(idOf("<EOS>")!!)
    }

    private val tokenById: Map<Int, String> = buildMap {
        (FrameVocab.INTENTS + FrameVocab.SPECIALTY + FrameVocab.SYMPTOMS +
         listOf("<T_UNKNOWN>", "<TIME>", "</TIME>", "<NEEDS_CLARIFICATION>",
                "<END_FRAME>", "<EOS>") +
         FrameVocab.TIME_REL + FrameVocab.TIME_SLOTS + listOf("T_AMBIG"))
            .forEach { idOf(it)?.let { id -> put(id, it) } }
    }

    fun step(id: Int) {
        val tok = tokenById[id] ?: run { state = S.DONE; return }
        state = when (state) {
            S.INTENT -> if (tok == "<UNSUPPORTED>") S.END_FRAME else S.SPECIALTY
            S.SPECIALTY -> S.TIME
            S.TIME -> if (tok == "<T_UNKNOWN>") S.SYMPTOMS else S.FACTORS
            S.FACTORS -> when (tok) {
                "</TIME>" -> S.SYMPTOMS
                else -> { lastRank = rank(tok); emitted++; S.FACTORS }
            }
            S.SYMPTOMS -> when {
                tok in symptoms -> S.SYMPTOMS          // дубль игнорируется
                tok == "<NEEDS_CLARIFICATION>" -> S.CLARIF
                else -> { if (tok.startsWith("<SYMPTOM_")) symptoms += tok
                          if (tok == "<END_FRAME>") S.DONE else S.SYMPTOMS }
            }
            S.CLARIF -> if (tok == "<END_FRAME>") S.DONE else S.CLARIF
            S.END_FRAME -> if (tok == "<END_FRAME>") S.DONE else S.END_FRAME
            S.DONE -> S.DONE
        }
    }

    val isDone: Boolean get() = state == S.DONE
}
```

## 6. Инференс: ClinicNLU

```kotlin
class ClinicNLU(context: Context) {
    private val vocabSize = 3000
    private val ctxLen = 128
    private val padId = 0; private val bosId = 2; private val eosId = 3
    private val userId = 4; private val assistantId = 5

    private val tflite: Interpreter
    private val tok: BpeTokenizer
    private val vocab: JSONObject
    private val idToToken: Map<Int, String>
    private val autom: FrameAutomaton

    init {
        val mmBuf = context.assets.openFd("model_int8.tflite").use { fd ->
            fd.createInputStream().channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
        tflite = Interpreter(mmBuf, Interpreter.Options().apply { numThreads = 2 })
        val tokJson = context.assets.open("tokenizer.json")
            .bufferedReader().readText()
        tok = BpeTokenizer(tokJson)
        vocab = JSONObject(tokJson).getJSONObject("model").getJSONObject("vocab")
        idToToken = buildMap { vocab.keys().forEach { put(vocab.getInt(it), it) } }
        autom = FrameAutomaton { t -> vocab.optInt(t, -1).takeIf { it >= 0 } }
    }

    /** text -> semantic frame, напр. "<BOOK> <DENTIST> <TIME> ... <END_FRAME>" */
    fun parse(text: String, maxSteps: Int = 24): String {
        val seq = ArrayList<Int>(listOf(bosId, userId) + tok.encode(text))
        seq.add(assistantId)
        val inBuf = arrayOf(LongArray(ctxLen) { padId.toLong() })
        val outBuf = arrayOf(Array(ctxLen) { FloatArray(vocabSize) })
        val frame = StringBuilder()
        for (step in 0 until maxSteps) {
            // при переполнении контекста берём ПОСЛЕДНИЕ 128 токенов
            // (как seq[-context:] в train/constrained.py)
            val len = seq.size.coerceAtMost(ctxLen)
            val start = seq.size - len
            java.util.Arrays.fill(inBuf[0], padId.toLong())
            for (i in 0 until len) inBuf[0][i] = seq[start + i].toLong()
            tflite.run(inBuf, outBuf)
            val logits = outBuf[0][len - 1]
            val allowed = autom.allowed()
            if (allowed.isEmpty()) break
            var bestId = -1; var bestVal = Float.NEGATIVE_INFINITY
            for (id in allowed)
                if (logits[id] > bestVal) { bestVal = logits[id]; bestId = id }
            if (bestId == eosId) break
            frame.append(idToToken[bestId] ?: "<UNK>").append(' ')
            seq.add(bestId)
            autom.step(bestId)
            if (autom.isDone) break
        }
        return frame.toString().trim()
    }
}
```

Использование:

```kotlin
val nlu = ClinicNLU(context)
val frame = nlu.parse("запишите меня к зубному в следующую пятницу вечером, зуб ноет")
// <BOOK> <DENTIST> <TIME> T_NEXT_WEEK T_FRI T_EVENING </TIME> <SYMPTOM_TOOTHACHE> <END_FRAME>
```

Разбор фрейма в поля (intent / specialty / время / симптомы) — по пробелам;
формат описан в `../README.md`, секция «Схема целевой последовательности».

## 7. Юнит-тест на тест-векторах

`kotlin_testvectors.json` сгенерирован из той же модели (`train/
gen_kotlin_testvectors.py`): для 12 фраз — эталонные `bpe_ids`
(токенизатор) и `expected_frame` (сквозной инференс int8-модели +
автомат). Тест проверяет обе части конвейера:

```kotlin
@Test fun tokenizerMatchesPython() {
    val cases = loadCases()          // парс kotlin_testvectors.json
    for (c in cases)
        assertEquals(c.bpeIds, nluDebugTokenize(c.text))
}

@Test fun framesMatchInt8Model() {
    val cases = loadCases()
    for (c in cases)
        assertEquals(c.expectedFrame, nlu.parse(c.text))
}
```

Расхождение на `bpe_ids` — баг Kotlin-токенизатора; расхождение на
`expected_frame` при верных ids — баг инференса/автомата (или другая
модель в assets). Для тестов берите `numThreads = 1` (как в проверках
`export_tflite.py`) — на результат это не влияет, но сравнение честнее.

## 8. Что важно знать

- **STT-текст подаётся как есть**: модель обучалась на трёх стилях
  регистра/пунктуации. Спецтокены не токенизировать текстом — только id.
- **Незнакомые слова** (hold-out синонимы: «кожник», «дантист»...) модель
  честно отдаёт как `<UNKNOWN_SPECIALTY>` часто с
  `<NEEDS_CLARIFICATION>` — это контракт, а не ошибка; UI должен уметь
  показывать уточняющий вопрос.
- **Провалы** собраны в `failures_report.md` — прочитать до демо.
- Модель ~6.8M параметров (d_model=256, 8 слоёв, 8 голов), контекст 128
  токенов; длина пользовательского текста усекается до ~110 токенов
  (бюджет в `train/train.py: encode_example`).
- Перегенерация артефактов после переобучения: `train/export_onnx.py`,
  затем `train/export_tflite.py` (второй сам гоняет проверки парности —
  упало, значит в APK ничего менять нельзя), затем
  `train/gen_kotlin_testvectors.py` для обновления тест-векторов.
