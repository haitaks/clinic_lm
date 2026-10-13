# -*- coding: utf-8 -*-
"""Схема данных менеджера диалога телефонии.

Соответствует регламенту РГ-ТЕЛ-01 (скрипт.docx):
- каталог интентов — раздел 4 (I-01…I-14);
- сценарии — раздел 5 (С-01…С-15);
- матрица маршрутизации — раздел 7.3 (R-01…R-17), адресаты — 7.2 (М0…М5);
- коды результата звонка — Приложение Г;
- карточка перевода — Приложение В.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, time


class Intent(str, enum.Enum):
    """Каталог обращений (интентов), раздел 4."""

    BOOK = "I-01"            # Запись на приём — С-01/С-02
    RESCHEDULE = "I-02"      # Перенос записи — С-03
    CANCEL = "I-03"          # Отмена записи — С-04
    CHECK_BOOKING = "I-04"   # Проверка своей записи — С-05
    HOME_VISIT = "I-05"      # Вызов врача на дом — С-06 (этап 2)
    SCHEDULE = "I-06"        # График врача / поликлиники — С-07
    LAB_STATUS = "I-07"      # Готовность анализов — С-08 (этап 2)
    PAID = "I-08"            # Платные услуги — С-09 (этап 2)
    WAITLIST = "I-09"        # Лист ожидания — С-11 (этап 2)
    DOCS = "I-10"            # Прикрепление, полис, документы — С-13/FAQ (этап 2)
    EMERGENCY = "I-11"       # Экстренная ситуация — С-14
    COMPLAINT = "I-12"       # Жалоба, конфликт — С-15
    MEDICAL = "I-13"         # Медицинский вопрос — граничная фраза 3.8
    OTHER = "I-14"           # Прочее / нестандартное


class ScenarioId(str, enum.Enum):
    C01 = "С-01"
    C02 = "С-02"
    C03 = "С-03"
    C04 = "С-04"
    C05 = "С-05"
    C06 = "С-06"
    C07 = "С-07"
    C08 = "С-08"
    C09 = "С-09"
    C10 = "С-10"
    C11 = "С-11"
    C12 = "С-12"
    C13 = "С-13"
    C14 = "С-14"
    C15 = "С-15"


class ResultCode(str, enum.Enum):
    """Коды результата звонка, Приложение Г."""

    BOOKED = "BOOKED"
    BOOKED_GP = "BOOKED_GP"
    BOOKED_PAID = "BOOKED_PAID"
    RESCHEDULED = "RESCHEDULED"
    CANCELLED = "CANCELLED"
    CONFIRMED = "CONFIRMED"
    INFO_GIVEN = "INFO_GIVEN"
    WAITLISTED = "WAITLISTED"
    HOME_VISIT_REQUESTED = "HOME_VISIT_REQUESTED"
    ESCALATED = "ESCALATED"
    CALLBACK = "CALLBACK"
    EMERGENCY = "EMERGENCY"
    COMPLAINT = "COMPLAINT"
    GRATITUDE = "GRATITUDE"
    WRONG_PERSON = "WRONG_PERSON"
    VOICEMAIL = "VOICEMAIL"
    NO_ANSWER = "NO_ANSWER"
    OPT_OUT = "OPT_OUT"
    DROPPED = "DROPPED"


class RouteRule(str, enum.Enum):
    """Матрица маршрутизации 7.3. Проверка сверху вниз, срабатывает первое."""

    R01 = "R-01"   # красный флаг категории А → 103/112 + регистратор, P1
    R02 = "R-02"   # просьба об операторе → М2
    R03 = "R-03"   # жалоба → М3
    R04 = "R-04"   # медицинский вопрос, пациент настаивает → М2
    R05 = "R-05"   # не понял 2 раза подряд → М2
    R06 = "R-06"   # сбой/таймаут МИС → М2 или обратный звонок
    R07 = "R-07"   # не удалось идентифицировать → М2
    R08 = "R-08"   # запрос вне каталога и особые случаи → М2
    R09 = "R-09"   # острая зубная боль → М4-С
    R10 = "R-10"   # платные вне справочника → М4-П
    R11 = "R-11"   # красный флаг категории Б → М2 → М4-Н, P1
    R12 = "R-12"   # вопросы других организаций/лекобеспечение → М5 (122)
    R13 = "R-13"   # оформление полиса → М5 (СМО)
    R14 = "R-14"   # звонок от организации → М3
    R15 = "R-15"   # другой язык, плохо слышит → М2
    R16 = "R-16"   # отзыв ПДн, копии записей → М3
    R17 = "R-17"   # тишина/помехи → завершение


class Target(str, enum.Enum):
    """Адресаты маршрутизации 7.2."""

    AMBULANCE = "М0"        # скорая 103/112
    GA = "М1"               # сам ассистент
    REGISTRAR = "М2"        # регистратор, общая очередь
    SENIOR = "М3"           # старший администратор
    DENTISTRY = "М4-С"      # стоматологическое отделение
    PAID = "М4-П"           # платное отделение
    URGENT = "М4-Н"         # кабинет неотложной помощи
    EXTERNAL = "М5"         # внешние (122, СМО, ЕПГУ)
    CALLBACK = "CB"         # обратный звонок


class Priority(str, enum.Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


# ---------------------------------------------------------------------------
# Значения из МИС (контракты передаются между сценарием и клиентом МИС)

@dataclass
class Fio:
    last: str = ""
    first: str = ""
    patronymic: str = ""

    def full(self) -> str:
        return " ".join(p for p in (self.last, self.first, self.patronymic) if p)

    def name_patronymic(self) -> str:
        return " ".join(p for p in (self.first, self.patronymic) if p)

    def initials(self) -> str:
        """«Петров И. Н.» — для карточки перевода."""
        parts = [self.last]
        for p in (self.first, self.patronymic):
            if p:
                parts.append(p[0].upper() + ".")
        return " ".join(parts)


@dataclass
class MisPatient:
    id: str
    fio: Fio
    dob: date
    phone: str
    attached: bool = True
    district: int | None = None
    gp_doctor: str = ""            # участковый врач (ФИО)
    pediatric: bool = False        # пациент педиатрического отделения (0–17)
    trusted_caller: bool = True    # ANI совпадает с телефоном из МИС / сам пациент


@dataclass
class Slot:
    doctor: str
    specialty: str                 # ключ clinic_data.SPECIALTIES
    date: date
    time: time
    cabinet: int

    def key(self) -> tuple:
        return (self.doctor, self.date, self.time)


@dataclass
class Appointment:
    id: str
    patient_id: str
    specialty: str
    doctor: str
    date: date
    time: time
    cabinet: int
    status: str = "Записан"        # «Записан» / «Отменена»
    source: str = "ГА"
    referral_id: str | None = None
    paid: bool = False
    late_cancel: bool = False

    def describe(self) -> str:
        """«запись к терапевту Кузнецовой в четверг в восемь сорок пять»."""
        from . import textutils as tu
        return (f"запись к {tu.specialty_dat(self.specialty)} {tu.doctor_gen(self.doctor)} "
                f"{tu.date_words(self.date)} в {tu.time_words(self.time)}")


@dataclass
class Referral:
    id: str
    specialty: str
    issued: date
    valid_days: int = 30
    cito: bool = False             # «подозрение на онкологию» — не озвучивается


@dataclass
class DoctorScheduleInfo:
    doctor: str
    specialty: str
    days: str                      # «каждый будний день»
    hours: str                     # «с восьми до четырнадцати»
    cabinet: int
    on_leave_until: date | None = None
    substitute: str = ""


@dataclass
class LabOrder:
    name: str
    taken: date
    status: str                    # «готов» / «в работе»
    expected: date | None = None


# ---------------------------------------------------------------------------
# Результат NLU: фрейм, с которым работает менеджер диалога

@dataclass
class TimePrefs:
    """Желаемые дата/время, Приложение А.3."""

    date: date | None = None       # конкретная дата, если однозначно разрешилась
    weekday: int | None = None     # 0=пн … 6=вс, «в пятницу»
    daypart: str | None = None     # morning / afternoon / evening / early / late
    next_week: bool = False        # «на следующей неделе»
    in_days: int | None = None     # «через неделю» = 7 дней
    exclude_weekday: int | None = None   # «не в пятницу»


@dataclass
class Frame:
    """Семантический фрейм реплики — ВСЁ, что модель извлекает из текста.

    Менеджер диалога — маршрутизатор по скрипту: он не разбирает реплики,
    а принимает решения только по полям фрейма. Модель отвечает в том
    числе за выбор варианта из предложенных (choice — 0-базированный
    индекс в списке options последнего вопроса ГА).
    """

    raw: str = ""
    intent: Intent | None = None
    extra_intents: tuple[Intent, ...] = ()   # несколько интентов в одной реплике
    extra_frames: tuple["Frame", ...] = ()  # под-фреймы остальных интентов
    specialty: str | None = None             # ключ clinic_data.SPECIALTIES
    doctor: str | None = None                # фамилия врача, как назвали
    time_prefs: TimePrefs | None = None
    fio: Fio | None = None                   # ФИО пациента — извлекает модель имён
    dob: date | None = None
    policy_no: str | None = None
    answer: bool | None = None               # да / нет
    maybe: bool = False                      # «не знаю, может быть»
    choice: int | None = None                # выбранный вариант из options
    relation: str | None = None              # кем приходится пациенту (С-12)
    more: bool = False                       # «ещё варианты»
    best_doctor: bool = False                # «к хорошему врачу» (без оценок)
    paper_referral: bool = False             # «бумажное направление»
    emergency_cleared: bool = False          # «уже всё в порядке» (С-14)
    ambulance_called: bool = False           # «скорую уже вызвали» (С-14)
    opt_out: bool = False                    # «не звоните больше» (С-10)
    wrong_person: bool = False               # трубку взял другой (С-10)
    hospital: bool = False                   # пациент в стационаре / умер (С-10)
    symptoms: tuple[str, ...] = ()
    for_child: bool = False                  # «записать ребёнка/дочку»
    for_other_person: bool = False           # «записать маму», «в интересах…»
    faq_id: str | None = None                # вопрос из справочника FAQ (раздел 6)
    unknown: bool = False                    # NLU не понял реплику


# ---------------------------------------------------------------------------
# Действия и ответ хода

@dataclass
class DmAction:
    pass


@dataclass
class Transfer(DmAction):
    target: Target
    rule: RouteRule
    priority: Priority
    card: "TransferCard"


@dataclass
class SmsRequest(DmAction):
    phone: str
    text: str
    kind: str = ""


@dataclass
class EndCall(DmAction):
    reason: str = ""


@dataclass
class CallbackTask(DmAction):
    card: "TransferCard"
    within_minutes: int = 30


@dataclass
class NotifySenior(DmAction):
    text: str


@dataclass
class TransferCard:
    """Карточка перевода, Приложение В."""

    call_id: str
    time_start: str
    time_transfer: str
    phone: str                      # маскированный ANI
    patient: str                    # «№ 204518, Петров И. Н., 05.05.1962» или «не идентифицирован»
    intent: str                     # «I-01 Запись на приём»
    collected: str                  # всё, что пациент уже сообщил
    rule: str                       # «R-08»
    priority: str                   # «P3»
    summary: str                    # 1–2 фразы, до 200 символов
    dissatisfaction: bool = False
    recording_consent: bool = True

    def render(self) -> str:
        rows = [
            ("ID звонка", self.call_id),
            ("Время", f"{self.time_start} / {self.time_transfer}"),
            ("Номер", self.phone),
            ("Пациент", self.patient),
            ("Интент", self.intent),
            ("Собрано", self.collected or "—"),
            ("Причина перевода", self.rule),
            ("Приоритет", self.priority),
            ("Резюме", self.summary),
            ("Недовольство", "да" if self.dissatisfaction else "нет"),
            ("Согласие на запись", "да" if self.recording_consent else "нет"),
        ]
        return " | ".join(f"{k}: {v}" for k, v in rows)


@dataclass
class DmTurn:
    """Ответ менеджера на реплику/событие: что сказать и что сделать."""

    reply: str
    actions: list[DmAction] = field(default_factory=list)
    scenario: ScenarioId | None = None
    result_code: ResultCode | None = None
    rule: RouteRule | None = None
    expecting: str = ""             # подсказка для отладки: чего ждём дальше


# ---------------------------------------------------------------------------
# События телефонии

@dataclass
class SilenceEvent:
    seconds: int = 5


@dataclass
class DtmfEvent:
    digits: str


@dataclass
class VoicemailEvent:
    """Телефония определила автоответчик (исходящий звонок)."""

    pass


ClockEvent = SilenceEvent | DtmfEvent | VoicemailEvent
