# -*- coding: utf-8 -*-
"""Сквозные правила, проверяемые на КАЖДОЙ реплике пациента (7.3).

Порядок проверки — «безопасность выше всего» (7.1):
красный флаг А (R-01) → флаг Б (R-11) → оператор (R-02) → жалоба (R-03) →
«вы робот» → возражение против записи разговора (3.4) →
острая зубная боль (R-09) → медицинский вопрос (R-04) → организация (R-14).

Красные флаги распознаются по словарю 7.4: ложная тревога допустима,
пропуск — нет. При подключении семантической NLU-модели этот слой
остаётся обязательной страховкой.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import textutils as tu
from .schema import Priority, RouteRule, Target

# Категория А — возможная угроза жизни → R-01 (103/112 + регистратор, P1)
RED_FLAGS_A: list[list[str]] = [
    # Сердце, грудь
    ["боль в груди", "болит грудь", "давит за грудиной", "давит в груди",
     "давит грудь", "жжет в груди", "отдает в левую руку",
     "сердечный приступ", "инфаркт", "болит за грудиной"],
    # Дыхание
    ["не может дышать", "не могу дышать", "задыхается", "задыхаюсь",
     "синеет", "хрипит", "подавился", "подавилась",
     "не хватает воздуха"],
    # Сознание
    ["потерял сознание", "потеряла сознание", "без сознания",
     "не приходит в себя", "не могу разбудить", "упал в обморок",
     "в обмороке"],
    # Инсульт
    ["перекосило лицо", "отнялась рука", "отнялась нога", "не может говорить",
     "не могу говорить", "речь невнятная", "онемела половина тела",
     "парализовало"],
    # Судороги
    ["судороги", "трясет не останавливается", "припадок"],
    # Кровотечение
    ["сильное кровотечение", "кровь не останавливается", "рвота кровью",
     "кровь изо рта", "кровотечение"],
    # Травмы, отравления
    ["дтп", "сильно ударился головой", "ударилась головой",
     "упал с высоты", "упала с высоты", "ожог большой", "отравился",
     "отравилась", "проглотил", "проглотила", "укусила змея",
     "выпил таблетки"],
    # Аллергия
    ["отекает горло", "отекает лицо", "не может глотать", "не могу глотать",
     "анафилактический шок", "отек квинке"],
    # Дети
    ["младенец вялый", "не просыпается", "у грудного ребенка температура",
     "у грудничка температура", "ребенок посинел", "ребенок проглотил"],
    # Беременность
    ["беременная кровотечение", "отошли воды",
     "сильная боль в животе при беременности", "начались схватки"],
    # Угроза себе или другим
    ["не хочу жить", "хочу покончить с собой", "наглотался таблеток",
     "угрожает убить", "хочу умереть", "убью"],
]

# Категория Б — острое состояние без явной угрозы жизни → R-11 (P1)
RED_FLAGS_B: list[list[str]] = [
    ["очень сильная боль", "невыносимо болит", "боль усиливается",
     "невыносимая боль"],
    ["температура сорок", "температура не сбивается",
     "температура третий день"],
    ["стало намного хуже", "резко поднялось давление", "поднялось давление",
     "кружится голова не могу встать", "резко хуже"],
    ["подвернул ногу распухла", "порезался глубоко", "глубокий порез"],
]

# Просьба об операторе (Приложение А.2): основы слов-обращений + просьбы.
# Токенный матчинг, чтобы «в интересах другого человека» не считалось
# просьбой об операторе. «Регистратура» намеренно не входит: слишком
# часто означает саму поликлинику, а не просьбу о сотруднике.
_PERSON_BASES = ("оператор", "человек", "сотрудник", "специалист",
                 "регистратор", "девушк", "жив")
_REQUEST_WORDS = {"соедините", "позовите", "дайте", "хочу", "можно",
                  "переведите", "передайте", "надо", "нужно", "давайте",
                  "пригласите"}
_PERSON_SUFFIXES = ("", "а", "у", "е", "и", "ой", "ом", "ем", "ого",
                    "ую", "ым", "ей")

COMPLAINT_WORDS = ["пожаловаться", "жалоба", "жалобу", "безобразие",
                   "нагрубил", "нагрубила", "напишу в министерство",
                   "в министерство", "позор", "хочу пожаловаться",
                   "обращусь"]

ROBOT_WORDS = ["вы робот", "ты робот", "вы человек", "ты человек",
               "вы живой", "вы автомат", "это робот", "вы не человек",
               "с вами робот"]

RECORDING_OBJECTION_WORDS = ["не записывайте", "против записи",
                             "отключите запись", "я против записи",
                             "не хочу чтобы записывалось", "уберите запись",
                             "запись отключите"]

MEDICAL_WORDS = ["что выпить", "мне выпить", "выпить от", "что принять",
                 "мне принять", "принять от", "какое лекарство",
                 "какие таблетки", "как лечить", "чем лечить",
                 "нормальный ли", "нормальные ли", "что делать при",
                 "можно ли принимать", "какой у меня холестерин",
                 "что значит анализ", "дозировка"]

# Вопрос о ЗНАЧЕНИЯХ результатов — отдельная ветка (3.8.4): статусы можно,
# значения и их оценку — никогда.
MEDICAL_VALUES_WORDS = ["холестерин", "показатели", "результаты хорошие",
                        "результаты плохие", "анализы нормальные",
                        "анализы в норме ли", "что показали анализы",
                        "расшифровать анализ", "расшифровку"]

GRATITUDE_WORDS = ["благодар", "спасибо врачу", "хочу поблагодарить",
                   "хвалить врача"]

COMPLAINT_WRITTEN_WORDS = ["письменн", "письменную жалобу", "напишу жалобу",
                           "заявление в прокуратуру", "в минздрав напишу"]

TOOTH_PAIN_WORDS = ["острая зубная боль", "травма зуба", "отек десны",
                    "флюс", "зуб невыносимо", "разболелся зуб",
                    "сильно болит зуб"]

ORGANIZATION_WORDS = {"полиции", "полиция", "телевидение",
                      "юрист", "юриста", "юридическое", "юридический",
                      "редакция", "редакции", "газета", "прессслужба"}


@dataclass
class PolicyHit:
    """Сработавшее сквозное правило."""

    kind: str                  # red_a / red_b / operator / complaint / robot /
                               # recording_objection / medical / tooth /
                               # organization
    rule: RouteRule | None = None
    target: Target | None = None
    priority: Priority | None = None
    phrase_group: str = ""     # сработавшая фраза/группа (для журнала)


def _match_phrases(text: str, groups: list[list[str]]) -> str | None:
    n = tu.normalize(text)
    for group in groups:
        for phrase in group:
            if phrase in n:
                return phrase
    return None


def _is_person_word(tok: str) -> bool:
    return any(tok.startswith(base) and tok[len(base):] in _PERSON_SUFFIXES
               for base in _PERSON_BASES)


def _operator_request(n: str) -> bool:
    toks = n.split()
    if not any(_is_person_word(t) for t in toks):
        return False
    if set(toks) & _REQUEST_WORDS:
        return True
    # короткая реплика из одних обращений: «оператора», «человека пожалуйста»
    return len(toks) <= 3


def check_utterance(text: str) -> PolicyHit | None:
    """Проверка реплики по сквозным правилам, приоритет — сверху вниз."""
    n = tu.normalize(text)
    phrase = _match_phrases(n, RED_FLAGS_A)
    if phrase:
        return PolicyHit("red_a", RouteRule.R01, None, Priority.P1, phrase)
    phrase = _match_phrases(n, RED_FLAGS_B)
    if phrase:
        return PolicyHit("red_b", RouteRule.R11, Target.REGISTRAR,
                         Priority.P1, phrase)
    if _operator_request(n):
        return PolicyHit("operator", RouteRule.R02, Target.REGISTRAR,
                         Priority.P2, "operator")
    if any(w in n for w in COMPLAINT_WORDS):
        if any(w in n for w in GRATITUDE_WORDS):
            return PolicyHit("gratitude", None, None, None, "gratitude")
        if any(w in n for w in COMPLAINT_WRITTEN_WORDS):
            return PolicyHit("complaint_written", RouteRule.R03,
                             Target.SENIOR, Priority.P3, "written")
        return PolicyHit("complaint", RouteRule.R03, Target.SENIOR,
                         Priority.P2, "complaint")
    if any(w in n for w in ROBOT_WORDS):
        return PolicyHit("robot", None, None, None, "robot")
    if any(w in n for w in RECORDING_OBJECTION_WORDS):
        return PolicyHit("recording_objection", None, None, None, "recording")
    if any(w in n for w in TOOTH_PAIN_WORDS):
        return PolicyHit("tooth", RouteRule.R09, Target.DENTISTRY,
                         Priority.P2, "tooth")
    if any(w in n for w in MEDICAL_VALUES_WORDS):
        return PolicyHit("medical_values", RouteRule.R04, Target.REGISTRAR,
                         Priority.P3, "values")
    if any(w in n for w in MEDICAL_WORDS):
        return PolicyHit("medical", RouteRule.R04, Target.REGISTRAR,
                         Priority.P3, "medical")
    if set(n.split()) & ORGANIZATION_WORDS:
        return PolicyHit("organization", RouteRule.R14, Target.SENIOR,
                         Priority.P3, "organization")
    return None
