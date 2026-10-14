# -*- coding: utf-8 -*-
"""NLU-слой менеджера диалога.

Контракт: NluClient.parse(текст, сегодня, options) → Frame. Дальше
менеджер диалога работает только с фреймом («модель разбирает строку на
токены, дальше работаем по скрипту»). Реализации:

- RuleNlu — детерминированный словарный разбор (Приложения А.1–А.3,
  триггеры раздела 5). Используется в демо, пока не подключена модель.
- NatashaNames — модель имён (NamesExtractor) для дозаполнения Frame.fio.
- FixtureNlu — RuleNlu + подстановка ФИО из списка пациентов Заглушки
  МИС (демо/ручные прогоны).
- FrameModelNlu — адаптер к самописной frame-модели
  (train/constrained.generate_frame, models/model.pt). Загружается лениво.
- TaggerNlu — адаптер к теггеру из tagger/ (один forward: интент + теги
  SPEC/SYM/TIME, tagger/tagger.pt). <UNSUPPORTED> — авторитетное решение
  (other): причину не разбираем, запрос уходит оператору с текстом реплики.
- LlmNlu — готовая LLM через OpenAI-совместимый HTTP-эндпоинт
  (TEL_LLM_*; вывод ограничен json-схемой фрейма — constrained decoding).

Слоты, которые модель не заполнила (ФИО, дата рождения, полис, да/нет),
дозаполняются детерминированными парсерами slots.py — контур одинаков
для всех реализаций. Врача по симптомам модель НЕ выбирает: специальность
берётся из реплики, порядок записи — из таблицы 2.3 (clinic_data).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import date

from . import slots, textutils as tu
from .clinic_data import DOCTOR_KEYWORDS, SPECIALTIES, SPECIALTY_KEYWORDS
from .schema import Frame, Fio, Intent, TimePrefs

# ---------------------------------------------------------------------------
# Детерминированные таблицы интентов (триггеры раздела 5 + Приложение А.2).
# Порядок важен: сначала специфичные действия, потом запись, потом справки.
# Жалоба/медвопрос/красные флаги ловит сквозной слой safety.py.

_INTENT_PATTERNS: list[tuple[Intent, tuple[str, ...]]] = [
    (Intent.RESCHEDULE, ("перенесите", "перенести", "перезапишите",
                         "перезаписаться", "поменять время", "сдвинуть",
                         "на другой день", "перепишите", "не могу прийти",
                         "не смогу прийти", "не смогу прийти в",
                         "перенесите на", "не получится",
                         "не получится прийти", "не смогу прийти завтра")),
    (Intent.CANCEL, ("отмените", "отменить", "не приду", "не приедем",
                     "снять запись", "снимите запись", "вычеркните",
                     "уже не надо", "уже не нужно", "отмени запись",
                     "не нужна запись", "выздоровел", "отмените мою")),
    (Intent.CHECK_BOOKING, ("на когда я записан", "на когда записана",
                            "во сколько мне", "какой кабинет",
                            "когда мой прием", "когда моя запись",
                            "записан ли", "записана ли", "на какое время",
                            "мой талон", "у меня запись есть",
                            "есть ли запись", "куда мне прийти")),
    (Intent.HOME_VISIT, ("вызвать врача", "вызовите врача", "врача на дом",
                         "врача домой", "на дом", "домой врача",
                         "чтобы пришли домой", "нужен врач на дом")),
    (Intent.WAITLIST, ("лист ожидания", "в очередь", "в лист ожидания",
                       "запишите в очередь", "позвоните когда появится",
                       "позвоните когда будет")),
    (Intent.LAB_STATUS, ("анализы", "результаты", "готовы ли",
                         "кровь пришла", "пришли результаты", "гормоны готовы",
                         "анализы пришли")),
    (Intent.PAID, ("сколько стоит", "стоимость", "цена", "платно",
                   "за деньги", "по прайсу")),
    (Intent.DOCS, ("прикрепиться", "прикрепление", "прикреплен",
                   "прикреплена", "как прикрепиться", "справка", "справку",
                   "больничный", "выписку", "выписка", "рецепт",
                   "что взять", "что взять с собой", "документы",
                   "медкомиссия", "медкомиссию")),
    (Intent.SCHEDULE, ("когда принимает", "график", "расписание",
                       "до скольки", "работаете ли", "во сколько работает",
                       "когда работает", "часы работы", "когда прием",
                       "когда ведет прием", "когда дежурный",
                       "когда дежурирует")),
    (Intent.BOOK, ("запишите", "записаться", "записать", "талон", "номерок",
                   "попасть на прием", "на прием к", "хочу к", "нужно к",
                   "мне бы к", "мне к", "надо к", "нужна запись",
                   "запишите меня", "на прием", "запишитесь", "хочу попасть",
                   "записаться к", "попасть к")),
]

# Разделители нескольких интентов в одной реплике (раздел 4):
# «отмените четверг и запишите к лору».
_MULTI_SEPARATORS = (" и ", ", потом ", ", затем ", "; ", ", после этого ",
                     ", еще ", ", ещё ")


def _match_intent(n: str) -> Intent | None:
    for intent, patterns in _INTENT_PATTERNS:
        for p in patterns:
            if p in n:
                return intent
    return None


# Ключевые слова для boolean-полей фрейма: это внутренности словарной
# заглушки, НЕ часть менеджера диалога.
_MORE_WORDS = ("еще", "ещё", "другое", "другой", "другая", "посмотреть",
               "что еще", "что ещё", "варианты")
_BEST_DOCTOR_WORDS = ("хорош", "получше", "быстрее", "скорее", "попобыстрее")
_PAPER_REFERRAL_WORDS = ("бумажн",)
_EMERGENCY_CLEARED_WORDS = ("год назад", "было раньше", "уже все хорошо",
                            "уже всё хорошо", "просто спросить",
                            "спрашиваю для", "в прошлом", "все нормально",
                            "всё нормально", "уже прошло", "обошлось",
                            "все в порядке", "всё в порядке",
                            "сейчас все в порядке", "сейчас всё в порядке",
                            "продолжим")
_AMBULANCE_CALLED_WORDS = ("уже вызвали", "скорая едет", "уже позвонили",
                           "скорую вызвали", "скорая уже", "дожидаюсь скорую")
_OPT_OUT_WORDS = ("не звоните", "хватит звонить", "не звоните больше")
_WRONG_PERSON_WORDS = ("это не она", "это не я", "нет это не", "не ее",
                       "не её", "другой человек", "это сын", "это дочь",
                       "муж взял")
_HOSPITAL_WORDS = ("умер", "умерла", "в стационаре", "в больнице",
                   "реанимации", "скончался")


class RuleNlu:
    """Словарная заглушка модели: заполняет фрейм ключевыми словами
    и детерминированными парсерами slots.py."""

    def parse(self, text: str, today: date,
              options: list | None = None) -> Frame:
        n = tu.normalize(text)
        frame = Frame(raw=text)
        frame.answer, frame.maybe = slots.parse_yesno(text)
        if options:
            frame.choice = slots.match_option(options, text, today)
        frame.more = any(w in n for w in _MORE_WORDS)
        frame.best_doctor = any(w in n for w in _BEST_DOCTOR_WORDS)
        frame.paper_referral = any(w in n for w in _PAPER_REFERRAL_WORDS)
        frame.emergency_cleared = any(w in n
                                      for w in _EMERGENCY_CLEARED_WORDS)
        frame.ambulance_called = any(w in n for w in _AMBULANCE_CALLED_WORDS)
        frame.opt_out = any(w in n for w in _OPT_OUT_WORDS)
        frame.wrong_person = any(w in n for w in _WRONG_PERSON_WORDS)
        frame.hospital = any(w in n for w in _HOSPITAL_WORDS)
        frame.relation = slots.parse_relation(text)
        frame.faq_id = _match_faq_id(text)

        # специальность (длинные ключевые слова приоритетнее)
        best: tuple[int, str] | None = None
        for key, kws in SPECIALTY_KEYWORDS.items():
            for kw in kws:
                kw = tu.normalize(kw)
                if kw in n and (best is None or len(kw) > best[0]):
                    best = (len(kw), key)
        if best:
            spec_key = best[1]
            if spec_key.endswith("_child"):
                frame.for_child = True
                spec_key = spec_key[: -len("_child")]
            frame.specialty = spec_key

        # врач: фамилия рядом с предлогом «к»/«у» или «принимает»,
        # чтобы ФИО пациента («орлова мария ивановна») не считалось врачом
        if any(p in n for p in ("к ", "у ", "принимает", "запишите")):
            for surname, full in DOCTOR_KEYWORDS.items():
                if surname in n:
                    frame.doctor = full
                    break

        frame.time_prefs = slots.parse_time_prefs(text, today)

        # «записать ребёнка/дочку» / «записать маму»
        for w in ("ребенка", "ребенку", "дочку", "дочери", "сына", "сыну",
                  "внука", "внучку"):
            if w in n:
                frame.for_child = True
        for w in ("маму", "маме", "папу", "папе", "мужу", "мужа",
                  "бабушку", "бабушке", "дедушку", "дедушке"):
            if w in n:
                frame.for_other_person = True

        # дата рождения — детерминированный парсер (А.3)
        frame.dob = slots.extract_dob(text, today)
        frame.policy_no = slots.parse_policy_no(text)

        # интент (+ несколько интентов в одной реплике: под-фреймы частей)
        frame.intent = _match_intent(n)
        for sep in _MULTI_SEPARATORS:
            if sep in n and frame.intent is not None:
                _, right = n.split(sep, 1)
                sub = self.parse(right, today)
                if sub.intent is not None and sub.intent is not frame.intent:
                    frame.extra_intents = (sub.intent,)
                    frame.extra_frames = (sub,)
                break

        if (frame.intent is None and frame.answer is None
                and frame.policy_no is None and not frame.maybe
                and frame.fio is None and frame.choice is None
                and frame.faq_id is None and not frame.more):
            frame.unknown = True
        return frame


def _match_faq_id(text: str) -> str | None:
    from .phrases import FAQ
    n = tu.normalize(text)
    best: tuple[int, str] | None = None
    for entry in FAQ:
        for kw in entry["keywords"]:
            kwn = tu.normalize(kw)
            if kwn in n and (best is None or len(kwn) > best[0]):
                best = (len(kwn), entry["id"])
    return best[1] if best else None


class NatashaNames:
    """Модель имён на Natasha (NamesExtractor): ФИО пациента, если основной
    NLU не заполнил. Загрузка natasha ленивая — пакет тяжёлый.

    Фильтры от шума экстрактора: месяцы («марта» как имя) и одиночные
    фамилии без имени/отчества (чаще всего это врач, не пациент).
    """

    _MONTHS = {"января", "февраля", "марта", "апреля", "мая", "июня",
               "июля", "августа", "сентября", "октября", "ноября",
               "декабря"}

    def __init__(self):
        self._extractor = None

    def extract(self, text: str) -> Fio | None:
        if self._extractor is None:
            from natasha import MorphVocab, NamesExtractor
            self._extractor = NamesExtractor(MorphVocab())
        best: tuple[int, Fio] | None = None
        for m in self._extractor(text):
            f = m.fact
            last = (f.last or "").strip()
            first = (f.first or "").strip()
            middle = (f.middle or "").strip()
            if len(last) < 3 or not (first or middle):
                continue
            if (first.lower() in self._MONTHS
                    or last.lower() in self._MONTHS
                    or middle.lower() in self._MONTHS):
                continue
            score = bool(first) + bool(middle)
            if best is None or score > best[0]:
                best = (score, Fio(last, first, middle))
        return best[1] if best else None


_NAMES_SINGLETON: "NatashaNames | None" = None


def names_backstop(text: str) -> Fio | None:
    """Контур модели имён (README «Подключение реальной модели», п. 4):
    Natasha дозаполняет Frame.fio, когда основная NLU оставила его пустым.
    Ленивая загрузка, кэш на уровне модуля; TEL_NAMES=off — выключить."""
    global _NAMES_SINGLETON
    if os.environ.get("TEL_NAMES", "natasha") == "off":
        return None
    if _NAMES_SINGLETON is None:
        _NAMES_SINGLETON = NatashaNames()
    return _NAMES_SINGLETON.extract(text)


def _apply_names_backstop(frame: Frame, text: str) -> Frame:
    """frame.fio пуст → достать ФИО контуром имён (на месте)."""
    if frame.fio is None:
        fio = names_backstop(text)
        if fio is not None:
            frame.fio = fio
            frame.unknown = False
    return frame


class FixtureNlu(RuleNlu):
    """Заглушка «другой модели» извлечения ФИО для демо: ФИО
    подставляется сопоставлением с известным списком пациентов Заглушки
    МИС. Остальные слоты — как у RuleNlu. В прод-контуре ФИО извлекает
    Natasha (names_backstop) или основная модель."""

    def __init__(self, known: list[Fio]):
        self.known = [
            (tu.normalize(f.full()).split(), f) for f in known if f.full()
        ]

    def parse(self, text: str, today: date,
              options: list | None = None) -> Frame:
        frame = super().parse(text, today, options)
        n = tu.normalize(text)
        for parts, fio in self.known:
            if all(p in n for p in parts):
                frame.fio = fio
                frame.unknown = False
                return frame
        # неизвестного пациента «модель» извлекает по заглавным буквам —
        # эвристика заглушки, не менеджера диалога
        caps = [w.strip(",.") for w in text.split()
                if len(w) > 2 and w[0].isupper() and w[1:].isalpha()]
        if len(caps) >= 2:
            frame.fio = Fio(*caps[:3])
            frame.unknown = False
        return frame


# ---------------------------------------------------------------------------
# Адаптер самописной frame-модели (train/constrained.generate_frame).
# Модель обучена на полный каталог интентов телефонного регистратурного
# контура (INTENT_TOKENS в synth/vocab.py) + тривиальные да/нет.

_TOKEN2INTENT = {
    "<BOOK>": Intent.BOOK,
    "<RESCHEDULE>": Intent.RESCHEDULE,
    "<CANCEL>": Intent.CANCEL,
    "<CHECK_BOOKING>": Intent.CHECK_BOOKING,
    "<HOME_VISIT>": Intent.HOME_VISIT,
    "<SCHEDULE>": Intent.SCHEDULE,
    "<LAB_STATUS>": Intent.LAB_STATUS,
    "<PAID>": Intent.PAID,
    "<WAITLIST>": Intent.WAITLIST,
    "<DOCS>": Intent.DOCS,
    "<COMPLAINT>": Intent.COMPLAINT,
    "<MEDICAL>": Intent.MEDICAL,
    "<UNSUPPORTED>": Intent.OTHER,
}
_TOKEN2ANSWER = {"<YES>": True, "<NO>": False}
_TOKEN2SPECIALTY = {"<GP>": "gp", "<DENTIST>": "dentist",
                    "<DERMATOLOGIST>": "dermatologist",
                    "<GYNECOLOGIST>": "gynecologist",
                    "<OPHTHALMOLOGIST>": "ophthalmologist",
                    "<NEUROLOGIST>": "neurologist",
                    "<PSYCHOLOGIST>": "psychologist",
                    "<SURGEON>": "surgeon", "<ENT>": "ent"}
_TIME_FACTORS_DAY = {"T_MON": 0, "T_TUE": 1, "T_WED": 2, "T_THU": 3,
                     "T_FRI": 4, "T_SAT": 5, "T_SUN": 6}
_TIME_FACTORS_PART = {"T_MORNING": "morning", "T_AFTERNOON": "afternoon",
                      "T_EVENING": "evening", "T_EARLY": "early",
                      "T_AFTER_NOON": "afternoon", "T_SOON": None}


class FrameModelNlu:
    """Модель из train/constrained.py: текст → строка фрейма → Frame.

    Загрузка тяжёлых зависимостей (torch) — ленивая, при первом parse().
    Незнакомые слова входа заменяются ближайшим знакомым синонимом по e5
    (train/oov_input.py, TEL_FMODEL_RESOLVER=0 — выключить).
    """

    _shared: dict | None = None      # кэш: model, tokenizer, generate_frame

    def __init__(self, model_path: str = "models/model.pt",
                 tokenizer_path: str = "models/tokenizer_v2.json"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._resolver = None

    def _load(self):
        import sys
        from pathlib import Path

        import torch
        from tokenizers import Tokenizer

        if FrameModelNlu._shared is not None:
            return

        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root / "train"))
        from constrained import generate_frame  # noqa: E402
        from model import TinyGPT               # noqa: E402

        tok = Tokenizer.from_file(str(root / self.tokenizer_path))
        state = torch.load(root / self.model_path, map_location="cpu",
                           weights_only=False)
        cfg = state.get("config", {})
        model = TinyGPT(
            vocab_size=state["model"]["tok.weight"].shape[0],
            d_model=cfg.get("d_model", 256),
            n_layers=cfg.get("n_layers", 8),
            n_heads=cfg.get("n_heads", 8))
        model.load_state_dict(state["model"])
        model.eval()
        FrameModelNlu._shared = {"model": model, "tok": tok,
                                 "generate_frame": generate_frame}

    def parse(self, text: str, today: date,
              options: list | None = None) -> Frame:
        self._load()
        shared = FrameModelNlu._shared
        if self._resolver is None and os.environ.get(
                "TEL_FMODEL_RESOLVER", "1") != "0":
            try:
                from train.oov_input import E5SynonymResolver
                self._resolver = E5SynonymResolver()
            except Exception:
                self._resolver = False     # e5 недоступен — без резолвера
        if self._resolver:
            text = self._resolver(text)
        tok = shared["tok"]
        ids = {t: tok.token_to_id(t) for t in
               ("<BOS>", "<USER>", "<ASSISTANT>", "<EOS>", "<PAD>")}
        assert all(v is not None for v in ids.values()), \
            f"токенизатор {self.tokenizer_path} без спецтокенов диалога"
        _, semantic = shared["generate_frame"](
            shared["model"], tok, text, device="cpu",
            bos=ids["<BOS>"], user=ids["<USER>"],
            assistant=ids["<ASSISTANT>"], eos=ids["<EOS>"],
            pad=ids["<PAD>"])
        frame = frame_from_tokens(semantic, text, today, options)
        return _apply_names_backstop(frame, text)


def frame_from_tokens(semantic: list[str], text: str, today: date,
                      options: list | None = None) -> Frame:
    """Токены фрейма → Frame + дозаполнение словарной заглушкой."""
    frame = Frame(raw=text)
    prefs = TimePrefs()
    in_time = False
    for tok in semantic:
        if tok in _TOKEN2INTENT:
            frame.intent = _TOKEN2INTENT[tok]
        elif tok in _TOKEN2ANSWER:
            frame.answer = _TOKEN2ANSWER[tok]
        elif tok in _TOKEN2SPECIALTY:
            spec = _TOKEN2SPECIALTY[tok]
            # таблица 2.3 телефонии уже словаря модели: чужие ключи
            # (dermatologist, psychologist) отбрасываем в страховку
            if spec in SPECIALTIES:
                frame.specialty = spec
        elif tok == "<TIME>":
            in_time = True
        elif tok == "</TIME>":
            in_time = False
        elif in_time and tok in _TIME_FACTORS_DAY:
            prefs.weekday = _TIME_FACTORS_DAY[tok]
            prefs.date = tu.resolve_weekday(today, prefs.weekday)
        elif in_time and tok == "T_TOMORROW":
            prefs.date = today + __import__("datetime").timedelta(days=1)
        elif in_time and tok in _TIME_FACTORS_PART:
            prefs.daypart = _TIME_FACTORS_PART[tok]
        elif tok.startswith("<SYMPTOM_"):
            frame.symptoms = frame.symptoms + (tok,)

    # дозаполнение словарной заглушкой (ФИО — только модель имён)
    det = RuleNlu().parse(text, today, options)
    if frame.answer is None:
        frame.answer, frame.maybe = det.answer, det.maybe
    frame.dob = det.dob
    frame.policy_no = det.policy_no
    frame.choice = det.choice
    frame.relation = det.relation
    frame.more = det.more
    frame.best_doctor = det.best_doctor
    frame.paper_referral = det.paper_referral
    frame.emergency_cleared = det.emergency_cleared
    frame.ambulance_called = det.ambulance_called
    frame.opt_out = det.opt_out
    frame.wrong_person = det.wrong_person
    frame.hospital = det.hospital
    frame.for_child = det.for_child
    frame.for_other_person = det.for_other_person
    frame.faq_id = det.faq_id
    if frame.specialty is None:
        frame.specialty = det.specialty
    if frame.doctor is None:
        frame.doctor = det.doctor
    if prefs.date or prefs.daypart or prefs.weekday:
        frame.time_prefs = prefs
    elif det.time_prefs:
        frame.time_prefs = det.time_prefs
    if frame.intent is None:
        frame.intent, frame.extra_frames = det.intent, det.extra_frames
        frame.unknown = det.unknown
    return frame


class TaggerNlu:
    """Теггер из tagger/ (один forward: интент + BIO-теги SPEC/SYM/TIME)
    → токены фрейма → Frame. Загрузка тяжёлых зависимостей ленивая,
    веса кэшируются на уровне класса (каждый звонок создаёт свой NLU).

    Теггер знает 4 интента (BOOK/RESCHEDULE/CANCEL/UNSUPPORTED) и 9
    специальностей; <UNSUPPORTED> — авторитетное решение (other): причину
    не разбираем, запрос уходит оператору с текстом реплики. Словарная
    страховка дозаполняет только незаполненные слоты, интент не
    перекрывает. После переобучения на полный каталог расширить
    _TOKEN2INTENT/_TOKEN2SPECIALTY — адаптер подхватит сам.
    """

    _shared: dict | None = None      # кэш: predict_frame, model, tokenizer

    def __init__(self, model_path: str = "tagger/tagger.pt",
                 tokenizer_path: str = "models/tokenizer.json"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def _load(self):
        import sys
        from pathlib import Path

        import torch
        from tokenizers import Tokenizer

        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root / "tagger"))
        from evaluate import predict_frame  # noqa: E402
        from model import Tagger             # noqa: E402

        tok = Tokenizer.from_file(str(root / self.tokenizer_path))
        ckpt = torch.load(root / self.model_path, map_location="cpu",
                          weights_only=False)
        model = Tagger(tok.get_vocab_size())
        model.load_state_dict(ckpt["model"])
        model.eval()
        TaggerNlu._shared = {"fn": predict_frame, "model": model, "tok": tok}

    def parse(self, text: str, today: date,
              options: list | None = None) -> Frame:
        if TaggerNlu._shared is None:
            self._load()
        shared = TaggerNlu._shared
        frame_str = shared["fn"](shared["model"], shared["tok"],
                                 text, device="cpu")
        frame = frame_from_tokens(frame_str.split(), text, today, options)
        return _apply_names_backstop(frame, text)


# ---------------------------------------------------------------------------
# Готовая LLM через OpenAI-совместимый эндпоинт (самостоятельный контур —
# раздел 8 регламента: только аттестованный/локальный эндпоинт).

_LLM_SYSTEM_PROMPT = """Ты — семантический парсер реплик пациента поликлиники (русский язык, текст после ASR).
Верни ТОЛЬКО JSON без пояснений со схемой:
{"intent": "book|reschedule|cancel|check_booking|home_visit|schedule|lab_status|paid|waitlist|docs|complaint|medical|other|null",
 "specialty": "gp|pediatrician|surgeon|surgeon_child|ophthalmologist|ophthalmologist_child|ent|ent_child|gynecologist|dentist|cardiologist|neurologist|endocrinologist|gastroenterologist|urologist|pulmonologist|prophylaxis|fluorography|ultrasound|null",
 "doctor": "фамилия|null", "answer": true|false|null, "maybe": true|false,
 "fio": {"last":"","first":"","patronymic":""}|null, "dob": "YYYY-MM-DD|null",
 "policy_no": "16 цифр|null", "time": {"date":"YYYY-MM-DD|null","weekday":0-6|null,"daypart":"morning|afternoon|evening|early|late|null","exclude_weekday":0-6|null},
 "choice": 0|1|2|null, "relation": "mother|father|guardian|relative|self|null",
 "more": true|false, "best_doctor": true|false, "paper_referral": true|false,
 "emergency_cleared": true|false, "ambulance_called": true|false,
 "opt_out": true|false, "wrong_person": true|false, "hospital": true|false,
 "for_child": true|false, "for_other_person": true|false, "faq_id": "строка|null",
 "unknown": true|false}
Интенты: book запись на приём, reschedule перенос, cancel отмена, check_booking проверка своей записи, home_visit врач на дом, schedule график/режим работы, lab_status готовность анализов, paid платные услуги, waitlist лист ожидания, docs документы/прикрепление/справки, complaint жалоба, medical медицинский вопрос (лекарства/лечение/оценка результатов), other прочее.
Специальность указывай только если пациент назвал её явно; врача по симптомам НЕ выбирай.
choice — индекс варианта пациента из пронумерованного списка в запросе (0-базированный), если ассистент предлагал варианты.
Отвечай строго по смыслу реплики; если поле неизвестно — null/false."""


def _extract_json(content: str) -> dict:
    """Достать JSON-объект из ответа LLM: слабые локальные модели любят
    обернуть его в ```json-забор или сопроводить текстом."""
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s).strip()
    start = s.find("{")
    if start < 0:
        raise ValueError(f"в ответе LLM нет JSON: {content[:200]!r}")
    obj, _ = json.JSONDecoder().raw_decode(s[start:])
    return obj


def _nullable_str(**constraints: int) -> dict:
    """Схема nullable-строки: {"type": ["string","null"]} + ограничения
    длины (llama.cpp переводит их в грамматику)."""
    schema: dict = {"type": ["string", "null"]}
    schema.update(constraints)
    return schema


def _build_frame_schema() -> dict:
    """JSON Schema фрейма для response_format=json_schema: сервер строит
    из неё грамматику constrained decoding — модель физически не может
    выдать JSON мимо схемы. Специальности и интенты — из кода
    (SPECIALTIES / Intent), не руками."""
    properties = {
        "intent": {"enum": [i.value for i in Intent] + [None]},
        "specialty": {"enum": [*SPECIALTIES, None]},
        "doctor": _nullable_str(),
        "answer": {"type": ["boolean", "null"]},
        "maybe": {"type": "boolean"},
        "fio": {
            "type": ["object", "null"],
            "properties": {
                "last": {"type": "string"},
                "first": {"type": "string"},
                "patronymic": {"type": "string"},
            },
            "required": ["last", "first", "patronymic"],
            "additionalProperties": False,
        },
        "dob": _nullable_str(minLength=10, maxLength=10),
        "policy_no": _nullable_str(minLength=16, maxLength=16),
        "time": {
            "type": ["object", "null"],
            "properties": {
                "date": _nullable_str(minLength=10, maxLength=10),
                "weekday": {"type": ["integer", "null"],
                            "minimum": 0, "maximum": 6},
                "daypart": {"enum": ["morning", "afternoon", "evening",
                                     "early", "late", None]},
                "exclude_weekday": {"type": ["integer", "null"],
                                    "minimum": 0, "maximum": 6},
            },
            "required": ["date", "weekday", "daypart", "exclude_weekday"],
            "additionalProperties": False,
        },
        "choice": {"type": ["integer", "null"], "minimum": 0},
        "relation": {"enum": ["mother", "father", "guardian", "relative",
                              "self", None]},
        "more": {"type": "boolean"},
        "best_doctor": {"type": "boolean"},
        "paper_referral": {"type": "boolean"},
        "emergency_cleared": {"type": "boolean"},
        "ambulance_called": {"type": "boolean"},
        "opt_out": {"type": "boolean"},
        "wrong_person": {"type": "boolean"},
        "hospital": {"type": "boolean"},
        "for_child": {"type": "boolean"},
        "for_other_person": {"type": "boolean"},
        "faq_id": _nullable_str(),
        "unknown": {"type": "boolean"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_FRAME_SCHEMA = _build_frame_schema()


class LlmNlu:
    """OpenAI-совместимый /chat/completions → Frame. Конфигурация из env
    (TEL_LLM_BASE_URL / TEL_LLM_MODEL / TEL_LLM_API_KEY / TEL_LLM_TIMEOUT).
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None,
                 timeout: int | None = None):
        self.base_url = (base_url or os.environ.get("TEL_LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("TEL_LLM_MODEL", "")
        self.api_key = api_key or os.environ.get("TEL_LLM_API_KEY", "")
        self.timeout = (timeout if timeout is not None else
                        int(os.environ.get("TEL_LLM_TIMEOUT", "10")))

    def parse(self, text: str, today: date,
              options: list | None = None) -> Frame:
        if not self.base_url:
            raise RuntimeError("TEL_LLM_BASE_URL не задан")
        user_msg = text
        if options:
            listed = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
            user_msg = f"Варианты, предложенные ассистентом:\n{listed}\n\nРеплика пациента: {text}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0,
        }
        # constrained decoding: сервер строит из схемы грамматику и не даёт
        # модели сгенерировать JSON мимо неё (TEL_LLM_SCHEMA=0 — выключить).
        # Формат — OpenAI-стандарт (json_schema.json_schema.strict); llama.cpp
        # его поддерживает, плоская вложенность schema: рядом с type — нет
        # (молча игнорируется).
        use_schema = os.environ.get("TEL_LLM_SCHEMA", "1") != "0"
        if use_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "frame", "strict": True,
                                "schema": _FRAME_SCHEMA},
            }
        data = self._post(payload)
        if data is None and use_schema:
            # эндпоинт не понял json_schema — раз без ограничения
            payload.pop("response_format")
            data = self._post(payload)
        content = data["choices"][0]["message"]["content"]
        frame = frame_from_json(_extract_json(content), text, today, options)
        return _apply_names_backstop(frame, text)

    def _post(self, payload: dict) -> dict | None:
        """POST /chat/completions; None — эндпоинт отверг запрос с
        response_format (HTTP 400), иначе исключение уходит наверх."""
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 400 and "response_format" in payload:
                return None
            raise


def frame_from_json(data: dict, text: str, today: date,
                    options: list | None = None) -> Frame:
    frame = Frame(raw=text)
    if data.get("intent"):
        try:
            frame.intent = Intent(data["intent"])
        except ValueError:
            pass
    spec = data.get("specialty")
    frame.specialty = spec if spec in SPECIALTIES else None
    frame.doctor = data.get("doctor")
    frame.answer = data.get("answer")
    frame.maybe = bool(data.get("maybe"))
    choice = data.get("choice")
    frame.choice = choice if isinstance(choice, int) and choice >= 0 else None
    frame.relation = data.get("relation")
    for key in ("more", "best_doctor", "paper_referral", "emergency_cleared",
                "ambulance_called", "opt_out", "wrong_person", "hospital"):
        setattr(frame, key, bool(data.get(key)))
    fio = data.get("fio")
    if isinstance(fio, dict) and fio.get("last"):
        frame.fio = Fio(fio.get("last", ""), fio.get("first", ""),
                        fio.get("patronymic", ""))
    dob = data.get("dob")
    if isinstance(dob, str) and len(dob) == 10:
        try:
            frame.dob = date.fromisoformat(dob)
        except ValueError:
            frame.dob = None
    pol = data.get("policy_no")
    frame.policy_no = pol if isinstance(pol, str) and len(pol) == 16 else None
    t = data.get("time") or {}
    if t.get("date") or t.get("weekday") or t.get("daypart") or t.get("exclude_weekday"):
        t_date = None
        if isinstance(t.get("date"), str) and len(t["date"]) == 10:
            try:
                t_date = date.fromisoformat(t["date"])
            except ValueError:
                t_date = None
        prefs = TimePrefs(
            date=t_date,
            weekday=t.get("weekday"), daypart=t.get("daypart"),
            exclude_weekday=t.get("exclude_weekday"))
        frame.time_prefs = prefs
    frame.for_child = bool(data.get("for_child"))
    frame.for_other_person = bool(data.get("for_other_person"))
    frame.faq_id = data.get("faq_id") or None
    frame.unknown = bool(data.get("unknown"))

    # страховка словарной заглушкой — только там, где LLM не ответила
    det = RuleNlu().parse(text, today, options)
    if frame.dob is None:
        frame.dob = det.dob
    if frame.policy_no is None:
        frame.policy_no = det.policy_no
    if frame.answer is None and not frame.unknown:
        frame.answer, frame.maybe = det.answer, det.maybe
    if frame.choice is None:
        frame.choice = det.choice
    if frame.specialty is None:
        frame.specialty = det.specialty
    if frame.doctor is None:
        frame.doctor = det.doctor
    if frame.time_prefs is None:
        frame.time_prefs = det.time_prefs
    if frame.faq_id is None:
        frame.faq_id = det.faq_id
    if frame.intent is None:
        frame.intent = det.intent
    return frame


NluClient = RuleNlu | FrameModelNlu | LlmNlu | TaggerNlu
