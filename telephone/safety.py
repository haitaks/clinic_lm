# -*- coding: utf-8 -*-
"""Сквозной слой безопасности: проверяется на КАЖДОЙ реплике пациента
(раздел 7.3 регламента: «безопасность выше всего»).

Порядок: красный флаг А → флаг Б → просьба об операторе → жалоба →
«вы робот» → возражение против записи разговора → острая зубная боль →
медицинский вопрос. Словарь красных флагов — из раздела 7.4: ложная
тревога допустима, пропуск — нет. При семантической NLU-модели этот слой
остаётся обязательной страховкой.

Здесь же два мини-потока: EmergencyFlow (С-14) и MedicalFlow (граничная
фраза 3.8) — они прерывают/обслуживают любые сценарии.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import phrases as ph
from . import textutils as tu
from .scenarios.common import Scenario, StepResult
from .schema import EndCall, NotifySenior, Target, Transfer

# Категория А — возможная угроза жизни: 103/112 + перевод, приоритет 1
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

# Категория Б — острое состояние без явной угрозы жизни → оператор, приоритет 1
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


@dataclass
class SafetyHit:
    """Сработавшее сквозное правило; решение применяет менеджер."""

    kind: str                   # red_a / red_b / operator / complaint /
                                # gratitude / complaint_written / robot /
                                # recording_objection / tooth /
                                # medical_values / medical
    phrase: str = ""            # сработавшая фраза/группа (для журнала)


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


def check_utterance(text: str) -> SafetyHit | None:
    """Проверка реплики, приоритет — сверху вниз (7.3)."""
    n = tu.normalize(text)
    phrase = _match_phrases(n, RED_FLAGS_A)
    if phrase:
        return SafetyHit("red_a", phrase)
    phrase = _match_phrases(n, RED_FLAGS_B)
    if phrase:
        return SafetyHit("red_b", phrase)
    if _operator_request(n):
        return SafetyHit("operator", "operator")
    if any(w in n for w in COMPLAINT_WORDS):
        if any(w in n for w in GRATITUDE_WORDS):
            return SafetyHit("gratitude", "gratitude")
        if any(w in n for w in COMPLAINT_WRITTEN_WORDS):
            return SafetyHit("complaint_written", "written")
        return SafetyHit("complaint", "complaint")
    if any(w in n for w in ROBOT_WORDS):
        return SafetyHit("robot", "robot")
    if any(w in n for w in RECORDING_OBJECTION_WORDS):
        return SafetyHit("recording_objection", "recording")
    if any(w in n for w in TOOTH_PAIN_WORDS):
        return SafetyHit("tooth", "tooth")
    if any(w in n for w in MEDICAL_VALUES_WORDS):
        return SafetyHit("medical_values", "values")
    if any(w in n for w in MEDICAL_WORDS):
        return SafetyHit("medical", "medical")
    return None


# ---------------------------------------------------------------------------
# Экстренная ситуация (С-14): прерывает любой сценарий; никаких
# уточняющих вопросов, идентификации и предложений записи.

class EmergencyFlow(Scenario):
    """Двухшаговый поток: 103/112 + перевод; «скорая уже вызвана» —
    завершение; «всё в порядке» — возврат к прерванному сценарию."""

    label = "emergency"

    def start(self, ctx) -> StepResult:
        ctx.collect("красный флаг", ctx.collected.get("красный флаг",
                                                      "категория А"))
        return StepResult(
            label=self.label,
            reply=ph.EMERGENCY_MAIN,
            actions=[
                Transfer(target=Target.OPERATOR, priority=1, card=ctx.card(
                    "Экстренный: " + ctx.last_user[:120], "экстренная ситуация")),
                NotifySenior("Экстренный звонок: " + ctx.last_user[:120]),
            ],
            status="emergency")

    def handle_utterance(self, ctx, frame, text):
        if frame.ambulance_called:
            return StepResult(label=self.label, reply=ph.EMERGENCY_ALREADY,
                              actions=[EndCall(reason="скорая вызвана")],
                              done=True, status="emergency")
        if frame.emergency_cleared:
            # ложная тревога: фраза о 103 уже прозвучала (безопасность
            # важнее) — возвращаемся к исходному сценарию
            return StepResult(label=self.label,
                              reply=ph.EMERGENCY_FALSE_ALARM +
                              " Возвращаюсь к вашему вопросу.",
                              resume_previous=True)
        # пациент продолжает говорить об остром состоянии
        return StepResult(
            label=self.label,
            reply=ph.EMERGENCY_TRANSFER,
            actions=[Transfer(target=Target.OPERATOR, priority=1, card=ctx.card(
                "Продолжает описывать острое состояние: "
                + ctx.last_user[:100], "экстренная ситуация"))],
            done=True, status="emergency")


class MedicalFlow(Scenario):
    """Граничная фраза 3.8: без советов, предложение записи к терапевту;
    при настойчивости — перевод на оператора."""

    label = "medical"

    def __init__(self, reply: str = ""):
        self.custom_reply = reply or ph.MEDICAL_BOUNDARY

    def start(self, ctx):
        return StepResult(label=self.label, reply=self.custom_reply,
                          expecting="да/нет")

    def handle_utterance(self, ctx, frame, text):
        if frame.answer is True:
            from .scenarios.booking import BookingFlow
            return StepResult(label=self.label,
                              handoff=BookingFlow(specialty="gp"))
        if frame.answer is False:
            return StepResult(label=self.label, reply=ph.ANYTHING_ELSE,
                              done=True, status="info")
        if frame.intent is not None or frame.faq_id is not None:
            return StepResult(label=self.label, progress=False)
        # пациент настаивает после граничной фразы → оператор
        return StepResult(
            label=self.label,
            reply=ph.MEDICAL_PERSIST,
            actions=[Transfer(target=Target.OPERATOR, priority=3, card=ctx.card(
                "Медицинский вопрос, пациент настаивает", "медвопрос"))],
            done=True, status="transferred")
