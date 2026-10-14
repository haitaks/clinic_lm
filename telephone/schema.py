# -*- coding: utf-8 -*-
"""Схема данных менеджера диалога телефонии.

Интенты и цели перевода — читаемые имена (book, reschedule, operator…),
без кодов регламента (I-xx, R-xx, М-xx): маршрутизатор работает понятными
сущностями, соответствие регламенту РГ-ТЕЛ-01 фиксируется в комментариях
и справочниках clinic_data.py.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, time


class Intent(str, enum.Enum):
    """Каталог обращений (раздел 4 регламента, имена читаемые)."""

    BOOK = "book"               # запись на приём (С-01/С-02)
    RESCHEDULE = "reschedule"   # перенос записи (С-03)
    CANCEL = "cancel"           # отмена записи (С-04)
    CHECK_BOOKING = "check_booking"  # проверка своей записи (С-05)
    HOME_VISIT = "home_visit"   # вызов врача на дом (С-06) → оператор
    SCHEDULE = "schedule"       # график врача/поликлиники (С-07) → ответ
    LAB_STATUS = "lab_status"   # готовность анализов (С-08) → оператор
    PAID = "paid"               # платные услуги (С-09) → платное отделение
    WAITLIST = "waitlist"       # лист ожидания (С-11) → оператор
    DOCS = "docs"               # документы/прикрепление (С-13) → мини-ответ
    COMPLAINT = "complaint"     # жалоба (С-15) → старший администратор
    MEDICAL = "medical"         # медицинский вопрос (3.8): граничная фраза
    OTHER = "other"             # нераспознанное → оператор с текстом реплики


class Target(str, enum.Enum):
    """Куда переводим звонок (7.2). Экстренные номера ГА проговаривает,
    но переводит на оператора — скорая набирается пациентом."""

    OPERATOR = "operator"       # регистратор
    SENIOR = "senior"           # старший администратор
    PAID = "paid"               # платное отделение
    DENTISTRY = "dentistry"     # стоматология (острая зубная боль, доб. 2)


# Итог звонка — читаемые строки (бывшие коды Приложения Г).
BOOKED = "booked"
BOOKED_GP = "booked_gp"         # записан к терапевту за направлением
RESCHEDULED = "rescheduled"
CANCELLED = "cancelled"
CONFIRMED = "confirmed"         # исходящее напоминание: придёт
INFO = "info"
TRANSFERRED = "transferred"
COMPLAINT = "complaint"
EMERGENCY = "emergency"
VOICEMAIL = "voicemail"
WRONG_PERSON = "wrong_person"
OPT_OUT = "opt_out"
DROPPED = "dropped"


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
    referral_id: str | None = None
    late_cancel: bool = False

    def describe(self) -> str:
        """«запись к терапевту Кузнецовой в четверг в восемь сорок пять»."""
        from . import textutils as tu
        return (f"запись к {tu.specialty_dat(self.specialty)} "
                f"{tu.doctor_gen(self.doctor)} {tu.date_words(self.date)} "
                f"в {tu.time_words(self.time)}")


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

    Менеджер диалога — маршрутизатор по скрипту: реплики он не разбирает,
    решения принимает только по полям фрейма. Модель отвечает в том числе
    за выбор варианта из предложенных (choice — 0-базированный индекс
    в списке options последнего вопроса ГА). Симптомы — сырьё для карточки
    перевода, по ним врач НЕ выбирается (выбор — по таблице 2.3 регламента).
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
    relation: str | None = None              # кем приходится пациенту (звонок за другого)
    more: bool = False                       # «ещё варианты»
    best_doctor: bool = False                # «к хорошему врачу» (без оценок врачей)
    paper_referral: bool = False             # «бумажное направление»
    emergency_cleared: bool = False          # «уже всё в порядке» (ложная тревога)
    ambulance_called: bool = False           # «скорую уже вызвали»
    opt_out: bool = False                    # «не звоните больше» (исходящие)
    wrong_person: bool = False               # трубку взял другой (исходящие)
    hospital: bool = False                   # пациент в стационаре / умер
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
    """Перевод звонка. Карточка — простой dict (см. DialogContext.card)."""

    target: Target
    card: dict
    priority: int = 3            # 1 — экстренно, 2 — старший администратор, 3 — обычное


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
    """Адресат недоступен (вне часов работы) — перезвонить."""

    card: dict
    within_minutes: int = 30


@dataclass
class NotifySenior(DmAction):
    text: str


@dataclass
class DmTurn:
    """Ответ менеджера на реплику/событие: что сказать и что сделать."""

    reply: str
    actions: list[DmAction] = field(default_factory=list)
    scenario: str = ""               # метка сценария для отладки
    status: str | None = None        # итог звонка (booked/transferred/…), если завершён
    expecting: str = ""              # подсказка для отладки: чего ждём дальше


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
