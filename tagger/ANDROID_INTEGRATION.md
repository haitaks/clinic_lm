# Развертывание теггера на Android (Kotlin, LiteRT) — ОСНОВНОЙ подход

Теггер — joint-модель: интент классифицируется по всей фразе, слова
размечаются тегами, фрейм собирается детерминированным Kotlin-кодом.
Это **другая архитектура**, чем GPT-вариант из `models/` (decoder-only
с авторегрессией): здесь **один прогон модели** вместо 7–15 шагов
генерации, нет автомата слотов — синтаксическую правильность фрейма
гарантирует сборка. Референс сборки — `tagger/assemble.py` (Python),
этот гайд — его точный порт.

## 1. Файлы в `app/src/main/assets/`

| Файл | Размер | Что это |
| :-- | :-- | :-- |
| `tagger_int8.tflite` | 7.5 МБ | энкодер 6.8M + 2 головы, int8 (weight) |
| `tokenizer.json` | ~0.3 МБ | ByteLevel BPE — ОБЩИЙ с GPT-контуром (из `models/`) |
| `vocab_kotlin.json` | ~50 КБ | словари сборки: иголки спец/симптомов, таблица времени, категории (генератор — `tagger/gen_kotlin_vocab.py`) |
| `kotlin_vectors.json` | ~25 КБ | тест-векторы: 19 кейсов (текст → BPE-ids, теги, интент, фрейм от int8-модели) |

Gradle: `implementation("com.google.ai.edge.litert:litert:1.3.0")`,
`androidResources { noCompress += "tflite" }`.

## 2. Контракт модели

```
вход  : input_ids int64[1, 128]     // <BOS>=2 + bpe-токены текста, хвост <PAD>=0
выход : intent      float32[1, 4]   // book | reschedule | cancel | unsupported
       tags        float32[1, 128, 4]  // O=0 SPEC=1 SYM=2 TIME=3
тег i-го токена текста = argmax(tags[0, i+1])   // позиция 0 — BOS
```

Токенизатор — тот же `BpeTokenizer` (см. `models/ANDROID_INTEGRATION.md`
§4: ByteLevel BPE по vocab+merges, ~100 строк). Спецтокены в тексте
матчатся до BPE — в пользовательской речи их нет.

## 3. Пайплайн (Kotlin)

```
текст → bpe.ids → [BOS]+ids → tflite(1 прогон) → intent + теги
     → спаны (подряд идущие токены одного тега)
     → + словарные спаны SPEC/SYM (иголки vocab_kotlin.json; TIME — только от модели)
     → канонизация спанов словарём
     → правила сборки → строка фрейма
```

### 3.1. Матчинг иголок (границы слов + склонения)

Повторяет `synth/validate.py:_contains`: многословная иголка ищется
целиком, однословная ловит хвост-склонение до 3 букв («стоматолог»
находит «с стоматологом», но не «последнем»).

```kotlin
object Needles {
    private fun fold(s: String) = s.lowercase().replace('ё', 'е')

    fun regex(needle: String): Regex {
        val esc = Regex.escape(needle)
        return if (' ' in needle)
            Regex("(?<![а-яёa-z])$esc(?![а-яёa-z])")
        else Regex("(?<![а-яёa-z])$esc[а-яё]{0,3}(?![а-яёa-z])")
    }

    /** Все непересекающиеся матчи поверхностей класса, слева направо,
     *  в каждой позиции выигрывает самый длинный. -> (start, end, key) */
    fun scan(text: String, surfaces: Map<String, List<String>>):
            List<Triple<Int, Int, String>> {
        val f = fold(text)
        val raw = ArrayList<Triple<Int, Int, String>>()
        for ((key, list) in surfaces)
            for (n in list)
                for (m in regex(fold(n)).findAll(f))
                    raw.add(Triple(m.range.first, m.range.last + 1, key))
        raw.sortWith(compareBy({ -it.second + it.first }, { it.first }))
        val out = ArrayList<Triple<Int, Int, String>>()
        var pos = 0
        val byStart = raw.groupBy { it.first }
        while (pos < f.length) {
            val cands = byStart[pos]
            if (cands != null) {
                out.add(cands.first())         // самый длинный в позиции
                pos = cands.first().second
            } else pos++
        }
        return out.sortedBy { it.first }
    }
}
```

### 3.2. Инференс и сборка

```kotlin
class TaggerNLU(context: Context) {
    private val ctxLen = 128
    private val bosId = 2L; private val padId = 0L
    private val tflite: Interpreter
    private val intentOutIx: Int; private val tagsOutIx: Int
    private val bpe: BpeTokenizer
    private val vocab: JSONObject
    private val specSurfaces: Map<String, List<String>>
    private val symSurfaces: Map<String, List<String>>
    private val temporal: Map<String, List<String>>   // фраза -> факторы
    private val catOf: Map<String, Int>               // токен -> ранг 0/1/2

    init {
        val mm = context.assets.openFd("tagger_int8.tflite").use { fd ->
            fd.createInputStream().channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
        tflite = Interpreter(mm, Interpreter.Options().apply { numThreads = 2 })
        // выходы различаем по форме: intent [1,4], tags [1,128,4] —
        // порядок в графе не гарантирован
        intentOutIx = if (tflite.getOutputTensor(0).shape().size == 2) 0 else 1
        tagsOutIx = 1 - intentOutIx
        bpe = BpeTokenizer(context.assets.open("tokenizer.json")
            .bufferedReader().readText())
        vocab = JSONObject(context.assets.open("vocab_kotlin.json")
            .bufferedReader().readText())
        fun surfaces(sec: String) =
            vocab.getJSONObject(sec).keys().associateWith { k ->
                vocab.getJSONObject(sec).getJSONObject(k)
                    .getJSONArray("surfaces").let { a ->
                        List(a.length()) { a.getString(it) } } }
        specSurfaces = surfaces("spec"); symSurfaces = surfaces("sym")
        temporal = vocab.getJSONObject("temporal").keys().associateWith {
            val a = vocab.getJSONObject("temporal").getJSONArray(it)
            List(a.length()) { a.getString(it) } }
        val cats = vocab.getJSONObject("time_categories")
        catOf = buildMap {
            listOf("scope" to cats.getJSONArray("scope"),
                   "day" to cats.getJSONArray("day"),
                   "time" to cats.getJSONArray("time")).forEachIndexed {
                    rank, (_, arr) ->
                for (i in 0 until arr.length()) put(arr.getString(i), rank) }
        }
    }

    fun parse(text: String): String {
        val enc = bpe.encode(text)                       // List<Int>
        val ids = LongArray(ctxLen) { padId }
        ids[0] = bosId
        enc.forEachIndexed { i, t -> if (i + 1 < ctxLen) ids[i + 1] = t }
        val len = minOf(enc.size + 1, ctxLen)
        val outInt = arrayOf(FloatArray(4))
        val outTags = arrayOf(Array(ctxLen) { FloatArray(4) })
        tflite.run(arrayOf(ids), mapOf(intentOutIx to outInt, tagsOutIx to outTags))
        val intentIx = outInt[0].indices.maxBy { outInt[0][it] }
        val intents = listOf("book", "reschedule", "cancel", "unsupported")
        val tags = (1 until len).map {                  // без BOS-позиции
            outTags[0][it].indices.maxBy { i2 -> outTags[0][it][i2] } }

        // --- спаны из тегов (по offsets BPE; Ġ-токены начинают слово) ---
        val offsets = bpe.offsets(text)
        data class Span(val s: Int, val e: Int, val tag: Int)
        val modelSpans = ArrayList<Span>()
        var cur: Span? = null
        for (i in 0 until tags.size) {
            val t = tags[i]
            if (t == 0) { cur?.let(modelSpans::add); cur = null; continue }
            val (s, e) = offsets[i]
            if (cur != null && cur.tag == t && s <= cur.e)
                cur = Span(cur.s, maxOf(cur.e, e), t)
            else { cur?.let(modelSpans::add); cur = Span(s, e, t) }
        }
        cur?.let(modelSpans::add)

        // --- словарный мерж: SPEC/SYM из словаря, TIME только модель ---
        val dictSpec = Needles.scan(text, specSurfaces).map { (s, e, _) -> Span(s, e, 1) }
        val dictSym = Needles.scan(text, symSurfaces).map { (s, e, _) -> Span(s, e, 2) }
        val all = (modelSpans + dictSpec + dictSym)
            .distinctBy { it.s to it.e to it.tag }.sortedBy { it.s }

        fun canon(span: Span, surfaces: Map<String, List<String>>) =
            Needles.scan(text.substring(span.s, span.e), surfaces)
                .map { it.third }

        var specs = all.filter { it.tag == 1 }.flatMap { canon(it, specSurfaces) }.distinct()
        var syms = all.filter { it.tag == 2 }.flatMap { canon(it, symSurfaces) }.distinct()
        var times = all.filter { it.tag == 3 }.map { text.substring(it.s, it.e) }

        // --- ревизия интента: unsupported + найденная спец/симптом ---
        var intent = intents[intentIx]
        if (intent == "unsupported" && (specs.isNotEmpty() || syms.isNotEmpty()))
            intent = "book"
        if (intent == "unsupported") return "<UNSUPPORTED> <END_FRAME>"

        // --- словарное время для валидных интентов (модель не дала) ---
        if (times.isEmpty()) {
            temporal.keys.sortedByDescending { it.length }
                .firstNotNullOfOrNull { p ->
                    Needles.regex(p).find(text.lowercase().replace('ё', 'е'))
                        ?.let { times = listOf(text.substring(it.range.first, it.range.last + 1)) } }
        }

        // --- сборка (правила = tagger/assemble.py) ---
        val p = ArrayList<String>()
        p.add("<${intent.uppercase()}>")
        var clarif = false
        when {
            specs.size >= 2 -> { p.add("<UNKNOWN_SPECIALTY>"); clarif = true }
            specs.size == 1 -> p.add(specToken(specs[0]))
            else -> { p.add("<UNKNOWN_SPECIALTY>")
                      if (intent == "book") clarif = true }
        }
        val factors = if (times.isNotEmpty()) timeFactors(times[0]) else null
        if (factors != null) p.add("<TIME> ${factors.joinToString(" ")} </TIME>")
        else p.add("<T_UNKNOWN>")
        syms.forEach { p.add(symToken(it)) }
        if (clarif) p.add("<NEEDS_CLARIFICATION>")
        p.add("<END_FRAME>")
        return p.joinToString(" ")
    }

    private fun specToken(key: String) =
        vocab.getJSONObject("spec").getJSONObject(key).getString("token")
    private fun symToken(key: String) =
        vocab.getJSONObject("sym").getJSONObject(key).getString("token")

    /** Точная таблица -> «или»-альтернатива (день + T_AMBIG) ->
     *  поиск известных фраз внутри спана (категории без дублей). */
    private fun timeFactors(span: String): List<String>? {
        val key = span.lowercase().replace('ё', 'е').trim()
        temporal[key]?.let { return it }
        if (" или " in key) {
            val slotRe = Regex("""\b([0-9]|1[0-9]|2[0-3])[:. ]([0-5]\d)\b""")
            val left = key.substringBefore(" или ")
            val right = key.substringAfter(" или ")
            val m = slotRe.find(left)
            if (m != null && slotRe.find(right) != null) {
                var dayPart = left.substring(0, m.range.first).trim().trimEnd(',')
                while (dayPart.split(" ").lastOrNull() in
                       setOf("в", "к", "на", "около", "после", "до"))
                    dayPart = dayPart.substringBeforeLast(" ").trim()
                val day = if (dayPart.isEmpty()) emptyList()
                          else temporal[dayPart] ?: emptyList()
                return day + "T_AMBIG"
            }
        }
        val f = key
        val found = ArrayList<Pair<IntRange, List<String>>>()
        for (phrase in temporal.keys.sortedByDescending { it.length }) {
            for (m in Needles.regex(phrase).findAll(f)) {
                if (found.none { it.first.intersects(m.range) })
                    found.add(m.range to (temporal[phrase] ?: continue))
            }
        }
        val facs = ArrayList<String>(); val seenCat = HashSet<Int>()
        for ((_, fs) in found.sortedBy { it.first.first }) {
            for (t in fs) {
                val c = catOf[t] ?: 2
                if (seenCat.add(c)) facs.add(t)
            }
        }
        return facs.ifEmpty { null }
    }
}

private fun IntRange.intersects(o: IntRange) = first <= o.last && o.first <= last
```

`bpe.offsets(text)` — список `(start, end)` для каждого токена (HF
`tokenizers` даёт offsets; в `BpeTokenizer` из основного гайда добавьте
сохранение `enc.offsets`).

### 3.3. Routing «симптомы → врач» (рекомендация при UNKNOWN_SPECIALTY)

Если фрейм вернул `<UNKNOWN_SPECIALTY>` и есть симптомы — UI получает
рекомендацию: первичный врач + альтернативы. Логика (референс
`tagger/routing.py`, данные в `vocab_kotlin.json` → `routing`):
пересечение списков симптомов («боль в глазу + голова» → офтальмолог),
при пустом пересечении — голосование по объединению, порядок — таблица
приоритетов `sym_priority` (точка настройки под клинику), gp — дефолт.
Для cancel рекомендации не показываются.

```kotlin
object SymptomRouter {
    private lateinit var symToSpecs: Map<String, List<String>>
    private lateinit var prio: Map<String, List<String>>
    private lateinit var tokens: Map<String, String>
    private lateinit var symKeyByToken: Map<String, String>
    private lateinit var profileSize: Map<String, Int>   // спец -> n симптомов

    fun init(vocab: JSONObject) {
        val r = vocab.getJSONObject("routing")
        symToSpecs = r.getJSONObject("sym_to_specs").keys().associateWith {
            val a = r.getJSONObject("sym_to_specs").getJSONArray(it)
            List(a.length()) { i -> a.getString(i) } }
        prio = r.getJSONObject("sym_priority").keys().associateWith {
            val a = r.getJSONObject("sym_priority").getJSONArray(it)
            List(a.length()) { i -> a.getString(i) } }
        tokens = r.getJSONObject("spec_tokens").keys().associateWith {
            it to r.getJSONObject("spec_tokens").getString(it) }
        symKeyByToken = buildMap {
            val sym = vocab.getJSONObject("sym")
            sym.keys().forEach { k ->
                put(sym.getJSONObject(k).getString("token"), k) }
        }
        profileSize = symToSpecs.values.flatten().groupingBy { it }.eachCount()
    }

    /** symKeys — канон-ключи симптомов (не токены). */
    fun route(symKeys: List<String>): Pair<String, List<String>>? {
        if (symKeys.isEmpty()) return null
        val sets = symKeys.map { symToSpecs[it]?.toSet() ?: emptySet() }
        val inter = sets.reduce { a, b -> a intersect b }
        val pool = if (inter.isNotEmpty()) inter
                   else sets.reduce { a, b -> a union b }
        val p = prio[symKeys.first()] ?: emptyList()
        val ordered = pool.sortedWith(compareBy(
            { s -> if (s in p) p.indexOf(s)
                   else p.size + if (s == "gp") 1 else 0 },
            { profileSize[it] ?: 0 }))
        return ordered.first() to ordered.drop(1)
    }

    /** Фрейм-строка -> (primaryToken, altTokens); null — рекомендация
     *  неуместна (специальность названа / cancel / нет симптомов). */
    fun fromFrame(frame: String): Pair<String, List<String>>? {
        val t = frame.split(" ")
        if (t.size < 2 || t[0] !in setOf("<BOOK>", "<RESCHEDULE>")) return null
        if (t[1] != "<UNKNOWN_SPECIALTY>") return null
        val keys = t.mapNotNull { symKeyByToken[it] }
        if (keys.isEmpty()) return null
        val (p, alts) = route(keys) ?: return null
        return (tokens[p] ?: p) to alts.map { tokens[it] ?: it }
    }
}
```

## 4. Юнит-тест

`kotlin_vectors.json`: для каждого кейса `bpe_ids` (токенизатор),
`expected_tags` + `expected_intent` (инференс) и `expected_frame`
(сквозной путь). Расхождение ids — баг токенизатора; тегов — инференса;
фрейма при верных тегах — баг Kotlin-сборки (сверять с Python-эталоном
`tagger/assemble.py`). Для тестов `numThreads = 1`.

## 5. Производительность и границы

- Один forward [1,128]: десятки мс на средних смартфонах (GPT-контур
  делал 7–15 таких прогонов); RAM < 50 МБ; APK +7.5 МБ (int8).
- Модель отвечает за интент/теги/контекст (отрицания «не отменяй»),
  словарь — за точные синонимы и время: несмежные куски времени
  («перенеси на послезавтра к 8:30») берут первый спан.
- Числовые слоты «15:45» и «или»-альтернативы парсятся таблицей+regex —
  Duckling или иные сервисы не нужны, всё офлайн.
- Регенерация после переобучения: `tagger/train_tagger.py` →
  `tagger/export_tflite.py` → `tagger/gen_kotlin_vocab.py` →
  `tagger/gen_kotlin_vectors.py`.
