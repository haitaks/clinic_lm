# -*- coding: utf-8 -*-
"""NLU-слой менеджера диалога.

Контракт: NluClient.parse(текст, сегодня) → Frame. Дальше менеджер диалога
работает только с фреймом («модель разбирает строку на токены, дальше
работаем по скрипту»). Реализации:

- RuleNlu — детерминированный словарный разбор (Приложения А.1–А.3,
  триггеры сценариев раздела 5). Используется в тестах и демо, пока
  самописная модель не переобучена на полный каталог интентов.
- FrameModelNlu — адаптер к самописной frame-модели
  (train/constrained.generate_frame, models/model.pt). Загружается лениво.
- LlmNlu — готовая LLM через OpenAI-совместимый HTTP-эндпоинт
  (TEL_LLM_BASE_URL / TEL_LLM_MODEL / TEL_LLM_API_KEY).

Слоты, которые модель не заполнила (ФИО, дата рождения, полис, да/нет),
дозаполняются детерминированными парсерами slots.py — контур одинаков
для всех реализаций.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import date

from . import slots, textutils as tu
from .clinic_data import DOCTOR_KEYWORDS, SPECIALTIES, SPECIALTY_KEYWORDS
from .schema import Frame, Fio, Intent, TimePrefs

# ---------------------------------------------------------------------------
# Детерминированные таблицы интентов (триггеры раздела 5 + Приложение А.2).
# Порядок важен: сначала специфичные действия, потом запись, потом справки.

_INTENT_PATTERNS: list[tuple[Intent, tuple[str, ...]]] = [
    (Intent.RESCHEDULE, ("перенесите", "перенести", "перезапишите", "перезаписаться",
                         "поменять время", "сдвинуть", "на другой день",
                         "перепишите", "не могу прийти", "не смогу прийти",
                         "не смогу прийти в", "перенесите на", "не получится",
                         "не получится прийти", "не смогу прийти завтра")),
    (Intent.CANCEL, ("отмените", "отменить", "не приду", "не приедем",
                     "снять запись", "снимите запись", "вычеркните",
                     "уже не надо", "уже не нужно", "отмени запись",
                     "не нужна запись", "выздоровел", "отмените мою")),
    (Intent.CHECK_BOOKING, ("на когда я записан", "на когда записана",
                            "во сколько мне", "какой кабинет", "когда мой прием",
                            "когда моя запись", "записан ли", "записана ли",
                            "на какое время", "мой талон", "у меня запись есть",
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
    (Intent.DOCS, ("прикрепиться", "прикрепление", "прикреплен", "прикреплена",
                   "как прикрепиться", "справка", "справку", "больничный",
                   "выписку", "выписка", "рецепт", "что взять", "что взять с собой",
                   "документы", "медкомиссия", "медкомиссию")),
    (Intent.SCHEDULE, ("когда принимает", "график", "расписание", "до скольки",
                       "работаете ли", "во сколько работает", "когда работает",
                       "часы работы", "когда прием", "когда ведет прием",
                       "когда дежурный", "когда дежурирует")),
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
_WRONG_PERSON_WORDS = ("это не она", "это не я", "нет это не", "не ее", "не её",
                       "другой человек", "это сын", "это дочь", "муж взял")
_HOSPITAL_WORDS = ("умер", "умерла", "в стационаре", "в больнице",
                   "реанимации", "скончался")


class RuleNlu:
    """Словарная заглушка модели: заполняет фрейм ключевыми словами
    и детерминированными парсерами slots.py. Используется в тестах
    и демо, пока не подключена реальная модель."""

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


class FixtureNlu(RuleNlu):
    """Заглушка «другой модели» извлечения ФИО для тестов и демо.

    Реальная модель имён подключается отдельным контуром (как FrameModelNlu
    или LlmNlu); здесь ФИО подставляется сопоставлением с известным списком
    пациентов Заглушки МИС: если в реплике прозвучали все части имени —
    Frame.fio заполнен. Остальные слоты — как у RuleNlu.
    """

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
        from .schema import Fio as _Fio
        caps = [w.strip(",.") for w in text.split()
                if len(w) > 2 and w[0].isupper() and w[1:].isalpha()]
        if len(caps) >= 2:
            frame.fio = _Fio(*caps[:3])
            frame.unknown = False
        return frame


# ---------------------------------------------------------------------------
# Адаптер самописной frame-модели (train/constrained.generate_frame).
# Модель переобучается на полный каталог интентов отдельно (другая машина).

_TOKEN2INTENT = {"<BOOK>": Intent.BOOK, "<RESCHEDULE>": Intent.RESCHEDULE,
                 "<CANCEL>": Intent.CANCEL, "<UNSUPPORTED>": Intent.OTHER}
_TOKEN2SPECIALTY = {"<GP>": "gp", "<DENTIST>": "dentist",
                    "<GYNECOLOGIST>": "gynecologist",
                    "<OPHTHALMOLOGIST>": "ophthalmologist",
                    "<NEUROLOGIST>": "neurologist",
                    "<SURGEON>": "surgeon", "<ENT>": "ent"}
_TIME_FACTORS_DAY = {"T_MON": 0, "T_TUE": 1, "T_WED": 2, "T_THU": 3,
                     "T_FRI": 4, "T_SAT": 5, "T_SUN": 6}
_TIME_FACTORS_PART = {"T_MORNING": "morning", "T_AFTERNOON": "afternoon",
                      "T_EVENING": "evening", "T_EARLY": "early",
                      "T_AFTER_NOON": "afternoon", "T_SOON": None}


class FrameModelNlu:
    """Модель из train/constrained.py: текст → строка фрейма → Frame.

    Загрузка тяжёлых зависимостей (torch) — ленивая, при первом parse().
    Пока модель знает только BOOK/RESCHEDULE/CANCEL/UNSUPPORTED —
    остальные поля берутся из детерминированных парсеров.
    """

    def __init__(self, model_path: str = "models/model.pt",
                 tokenizer_path: str = "models/tokenizer.json"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._ready = False

    def _load(self):
        import sys
        from pathlib import Path

        import torch
        from tokenizers import Tokenizer

        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root / "train"))
        from constrained import generate_frame  # noqa: E402
        from model import TinyGPT               # noqa: E402

        self._generate_frame = generate_frame
        tok = Tokenizer.from_file(str(root / self.tokenizer_path))
        state = torch.load(root / self.model_path, map_location="cpu")
        self._model = TinyGPT(vocab_size=state["embed.weight"].shape[0])
        self._model.load_state_dict(state)
        self._model.eval()
        self._tokenizer = tok
        self._torch = torch
        self._ready = True

    def parse(self, text: str, today: date,
              options: list | None = None) -> Frame:
        if not self._ready:
            self._load()
        _, semantic = self._generate_frame(
            self._model, self._tokenizer, text, device="cpu")
        return frame_from_tokens(semantic, text, today, options)


def frame_from_tokens(semantic: list[str], text: str, today: date,
                      options: list | None = None) -> Frame:
    """Токены фрейма → Frame + дозаполнение словарной заглушкой."""
    frame = Frame(raw=text)
    prefs = TimePrefs()
    in_time = False
    for tok in semantic:
        if tok in _TOKEN2INTENT:
            frame.intent = _TOKEN2INTENT[tok]
        elif tok in _TOKEN2SPECIALTY:
            frame.specialty = _TOKEN2SPECIALTY[tok]
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


# ---------------------------------------------------------------------------
# Готовая LLM через OpenAI-совместимый эндпоинт (самостоятельный контур —
# раздел 8 регламента: только аттестованный/локальный эндпоинт).

_LLM_SYSTEM_PROMPT = """Ты — семантический парсер реплик пациента поликлиники (русский язык, текст после ASR).
Верни ТОЛЬКО JSON без пояснений со схемой:
{"intent": "I-01..I-14|null", "specialty": "gp|pediatrician|surgeon|ophthalmologist|ent|gynecologist|dentist|cardiologist|neurologist|endocrinologist|gastroenterologist|urologist|pulmonologist|prophylaxis|fluorography|ultrasound|null",
 "doctor": "фамилия|null", "answer": true|false|null, "maybe": true|false,
 "fio": {"last":"","first":"","patronymic":""}|null, "dob": "YYYY-MM-DD|null",
 "policy_no": "16 цифр|null", "time": {"date":"YYYY-MM-DD|null","weekday":0-6|null,"daypart":"morning|afternoon|evening|early|late|null","exclude_weekday":0-6|null},
 "choice": 0|1|2|null, "relation": "mother|father|guardian|relative|self|null",
 "more": true|false, "best_doctor": true|false, "paper_referral": true|false,
 "emergency_cleared": true|false, "ambulance_called": true|false,
 "opt_out": true|false, "wrong_person": true|false, "hospital": true|false,
 "for_child": true|false, "for_other_person": true|false, "faq_id": "строка|null",
 "unknown": true|false}
Интенты: I-01 запись, I-02 перенос, I-03 отмена, I-04 проверка записи, I-05 врач на дом, I-06 график, I-07 анализы, I-08 платное, I-09 лист ожидания, I-10 документы/прикрепление, I-12 жалоба, I-13 медицинский вопрос, I-14 прочее.
choice — индекс варианта пациента из пронумерованного списка в запросе (0-базированный), если ассистент предлагал варианты.
Отвечай строго по смыслу реплики; если поле неизвестно — null/false."""


class LlmNlu:
    """OpenAI-совместимый /chat/completions → Frame. Конфигурация из env."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: int = 10):
        self.base_url = (base_url or os.environ.get("TEL_LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("TEL_LLM_MODEL", "")
        self.api_key = api_key or os.environ.get("TEL_LLM_API_KEY", "")
        self.timeout = timeout

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
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"]
        return frame_from_json(json.loads(content), text, today, options)


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
        frame.dob = date.fromisoformat(dob)
    pol = data.get("policy_no")
    frame.policy_no = pol if isinstance(pol, str) and len(pol) == 16 else None
    t = data.get("time") or {}
    if t.get("date") or t.get("weekday") or t.get("daypart") or t.get("exclude_weekday"):
        prefs = TimePrefs(
            date=date.fromisoformat(t["date"]) if t.get("date") else None,
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


NluClient = RuleNlu | FrameModelNlu | LlmNlu
