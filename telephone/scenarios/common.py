# -*- coding: utf-8 -*-
"""Общие компоненты сценариев: контекст диалога, результат шага, каскад
идентификации пациента (3.5) и предложение слотов (≤3 вариантов, 3.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from .. import phrases as ph
from .. import textutils as tu
from ..schema import (DmAction, EndCall, MisPatient, Slot, SmsRequest,
                      Target, Transfer)
from ..mis import SlotTaken


# ---------------------------------------------------------------------------
# Контекст диалога, разделяемый менеджером и сценариями

@dataclass
class DialogContext:
    mis: object
    now: object
    ani: str | None = None
    call_id: str = ""
    recording_consent: bool = True
    patient: MisPatient | None = None
    caller_relation: str | None = None     # self / mother / father / relative…
    mis_errors: int = 0                    # сбои МИС подряд
    collected: dict[str, str] = field(default_factory=dict)
    journal: list[dict] = field(default_factory=list)
    last_user: str = ""
    last_assistant: str = ""

    @property
    def today(self) -> date:
        return self.now.date()

    def collect(self, key: str, value) -> None:
        v = str(value)
        if self.collected.get(key) != v:
            self.collected[key] = v

    def log(self, role: str, text: str, **extra) -> None:
        self.journal.append({"role": role, "text": text, "ts": self.now,
                             **extra})

    def patient_label(self) -> str:
        if self.patient:
            return (f"№ {self.patient.id}, {self.patient.fio.initials()}, "
                    f"{self.patient.dob.isoformat()}")
        named = []
        if self.collected.get("ФИО"):
            named.append(self.collected["ФИО"])
        if self.collected.get("ДР"):
            named.append(self.collected["ДР"])
        return "не идентифицирован" + ("; названо: " + ", ".join(named)
                                       if named else "")

    def card(self, summary: str, topic: str = "") -> dict:
        """Карточка перевода — простой словарь для оператора."""
        phone = "скрыт"
        if self.ani:
            phone = "+7 ***-***-" + self.ani[-4:]
        return {
            "call_id": self.call_id,
            "time": (self.journal[0]["ts"].strftime("%H:%M:%S")
                     if self.journal else self.now.strftime("%H:%M:%S")),
            "phone": phone,
            "patient": self.patient_label(),
            "topic": topic or self.collected.get("тема", ""),
            "collected": "; ".join(f"{k}: {v}"
                                   for k, v in self.collected.items()
                                   if k != "тема"),
            "summary": summary[:200],
            "dissatisfaction": self.collected.get("недовольство") == "да",
            "recording_consent": self.recording_consent,
        }

    def sms_doctor(self, doctor: str) -> str:
        """«Кузнецова А. С.» — для SMS (фамилия + инициалы)."""
        parts = doctor.split()
        if len(parts) >= 3:
            return f"{parts[0]} {parts[1][0]}. {parts[2][0]}."
        if len(parts) == 2:
            return f"{parts[0]} {parts[1][0]}."
        return doctor


def render_card(card: dict) -> str:
    rows = [("ID", card["call_id"]), ("время", card["time"]),
            ("номер", card["phone"]), ("пациент", card["patient"]),
            ("тема", card.get("topic") or "—"),
            ("собрано", card.get("collected") or "—"),
            ("резюме", card["summary"]),
            ("недовольство", "да" if card.get("dissatisfaction") else "нет")]
    return " | ".join(f"{k}: {v}" for k, v in rows)


# ---------------------------------------------------------------------------
# Результат шага сценария

@dataclass
class StepResult:
    reply: str = ""
    actions: list[DmAction] = field(default_factory=list)
    label: str = ""                  # метка сценария для отладки
    done: bool = False
    status: str | None = None        # итог звонка (booked/…), если завершён
    handoff: "Scenario | None" = None  # продолжить другим сценарием
    progress: bool = True            # False = не распознано (лестница)
    resume_previous: bool = False    # экстренный: ложное срабатывание
    mis_retry: bool = False          # сбой МИС: не считаем непониманием
    expecting: str = ""              # подсказка: чего ждём дальше


class Scenario:
    """FSM сценария: шаги обсуждаются с менеджером через StepResult."""

    label: str = ""

    def start(self, ctx: DialogContext) -> StepResult:
        raise NotImplementedError

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        raise NotImplementedError

    def handle_event(self, ctx: DialogContext, ev) -> StepResult | None:
        return None

    def current_options(self) -> list:
        """Варианты последнего вопроса ГА — передаются NLU, чтобы модель
        смогла вернуть choice (индекс выбранного варианта)."""
        chooser = getattr(self, "chooser", None)
        return list(chooser.options) if chooser else []


# ---------------------------------------------------------------------------
# Вспомогательные конструкции реплик

def offer_phrase(slot: Slot, today: date) -> str:
    """Слот → «завтра, в четверг в восемь сорок пять» (3.2: дата с днём
    недели, время словами)."""
    wd = tu.WEEKDAYS_ACC[slot.date.weekday()]
    if (slot.date - today).days == 1:
        return f"завтра, {wd} в {tu.time_words(slot.time)}"
    if (slot.date - today).days <= 3:
        return f"{wd} в {tu.time_words(slot.time)}"
    return f"{tu.date_words(slot.date)}, в {tu.time_words(slot.time)}"


def options_phrase(options: list[Slot], today: date) -> str:
    parts = [offer_phrase(s, today) for s in options]
    if len(parts) > 1:
        return ", или ".join([", ".join(parts[:-1]), parts[-1]])
    return parts[0]


def booking_sms(ctx: DialogContext, appt, kind: str = "booked") -> SmsRequest:
    doctor = ctx.sms_doctor(appt.doctor)
    if kind == "booked":
        text = ph.SMS_BOOKED.format(date=appt.date.strftime("%d.%m"),
                                    time=appt.time.strftime("%H:%M"),
                                    cab=appt.cabinet, doctor=doctor)
    elif kind == "rescheduled":
        text = ph.SMS_RESCHEDULED.format(date=appt.date.strftime("%d.%m"),
                                         time=appt.time.strftime("%H:%M"),
                                         cab=appt.cabinet, doctor=doctor)
    else:
        text = ph.SMS_CANCELLED.format(date=appt.date.strftime("%d.%m"),
                                       time=appt.time.strftime("%H:%M"))
    return SmsRequest(phone=ctx.patient.phone, text=text, kind=kind)


def waitlist_transfer(ctx: DialogContext, summary: str) -> list[DmAction]:
    """Нет слотов → лист ожидания + уведомление старшему администратору
    о нарушении сроков ПГГ."""
    from ..schema import NotifySenior
    return [
        Transfer(target=Target.OPERATOR, priority=3, card=ctx.card(
            summary, "лист ожидания")),
        NotifySenior(f"ПГГ под угрозой: {summary}"),
    ]


# ---------------------------------------------------------------------------
# Каскад идентификации пациента (3.5):
# ФИО + дата рождения → повтор (по буквам) → полис ОМС → оператор

class IdentificationFlow:
    def __init__(self, ask_phrase: str = ph.ASK_FIO_DOB):
        self.state = "ask"          # ask → spell → policy
        self.ask_phrase = ask_phrase
        self._policy_retried = False

    def start(self, ctx: DialogContext) -> StepResult:
        self.state = "ask"
        return StepResult(reply=self.ask_phrase,
                          expecting="ФИО и дата рождения")

    def handle_event(self, ctx: DialogContext, ev) -> StepResult | None:
        if self.state == "policy" and hasattr(ev, "digits"):
            digits = ev.digits
            if len(digits) == 16:
                found = ctx.mis.search_patient(policy=digits)
                return self._found(ctx, found)
        return None

    def handle(self, ctx: DialogContext, frame, text: str) -> StepResult | None:
        """None — пациент идентифицирован (ctx.patient заполнен),
        сценарий продолжает со следующего шага."""
        if self.state in ("ask", "spell"):
            fio, dob = frame.fio, frame.dob
            if fio is None or dob is None:
                if self.state == "spell":
                    self.state = "policy"
                    return StepResult(reply=ph.ASK_POLICY,
                                      expecting="номер полиса")
                return StepResult(reply=self.ask_phrase, progress=False,
                                  expecting="ФИО и дата рождения")
            ctx.collect("ФИО", fio.full())
            ctx.collect("ДР", dob.isoformat())
            found = ctx.mis.search_patient(fio=fio, dob=dob, ani=ctx.ani)
            if len(found) == 1:
                return self._ok(ctx, found[0])
            if self.state == "ask":
                self.state = "spell"
                return StepResult(reply=ph.ASK_FIO_SPELL,
                                  expecting="ФИО по буквам и дата рождения")
            self.state = "policy"
            return StepResult(reply=ph.ASK_POLICY, expecting="номер полиса")
        # state == policy
        pol = frame.policy_no
        if pol is None:
            if frame.unknown:
                # пациент не может назвать полис → оператор (3.5.6)
                return self._fail(ctx)
            if self._policy_retried:
                return self._fail(ctx)
            self._policy_retried = True
            return StepResult(reply=ph.ASK_POLICY, progress=False,
                              expecting="номер полиса")
        ctx.collect("полис", pol)
        found = ctx.mis.search_patient(policy=pol)
        return self._found(ctx, found)

    def _found(self, ctx, found) -> StepResult | None:
        if len(found) == 1:
            return self._ok(ctx, found[0])
        return self._fail(ctx)

    def _ok(self, ctx, patient: MisPatient) -> None:
        ctx.patient = patient
        ctx.collect("пациент МИС", patient.id)
        self.state = "done"
        return None

    def _fail(self, ctx) -> StepResult:
        return StepResult(
            label="identify",
            reply=ph.IDENT_FAIL,
            actions=[Transfer(target=Target.OPERATOR, priority=3, card=ctx.card(
                "Пациент не найден по ФИО, дате рождения и полису",
                "идентификация")),
                EndCall(reason="не идентифицирован")],
            done=True, status="transferred")


# ---------------------------------------------------------------------------
# Предложение и выбор слотов: не более 3 вариантов за раз (3.2),
# перебор «ещё», уточнение части суток, «к хорошему врачу» без оценок.

class SlotChooser:
    def __init__(self, specialty: str, doctor: str | None = None,
                 daypart: str | None = None, fixed_date: date | None = None,
                 date_to: date | None = None,
                 exclude_weekday: int | None = None):
        self.specialty = specialty
        self.doctor = doctor
        self.daypart = daypart
        self.fixed_date = fixed_date
        self.date_to = date_to
        self.exclude_weekday = exclude_weekday
        self.all_slots: list[Slot] = []
        self.options: list[Slot] = []
        self.offset = 0
        self.chosen: Slot | None = None

    def _load(self, ctx: DialogContext) -> list[Slot]:
        date_from = ctx.today
        date_to = self.date_to or ctx.today + timedelta(days=14)
        if self.fixed_date:
            date_from = date_to = self.fixed_date
        slots = ctx.mis.free_slots(self.specialty, doctor=self.doctor,
                                   date_from=date_from, date_to=date_to,
                                   daypart=self.daypart, today=ctx.today)
        # запись на сегодня — не позднее чем за 30 минут до слота
        cutoff = ctx.now + timedelta(minutes=30)
        slots = [s for s in slots
                 if s.date > ctx.today or
                 (s.time.hour, s.time.minute) >= (cutoff.hour, cutoff.minute)]
        if self.exclude_weekday is not None:
            slots = [s for s in slots
                     if s.date.weekday() != self.exclude_weekday]
        return slots

    def offer(self, ctx: DialogContext, prefix: str = "",
              relaxed_note: str = "") -> StepResult:
        """Показать ближайшие ≤3 слота; если ничего — лист ожидания."""
        slots = self._load(ctx)
        relaxed = False
        if not slots and (self.daypart or self.fixed_date):
            # фильтр слишком строгий — предлагаем ближайшее без него
            self.daypart = self.fixed_date = None
            slots = self._load(ctx)
            relaxed = True
        if not slots:
            summary = (f"Нет слотов: {self.specialty}"
                       + (f", врач {self.doctor}" if self.doctor else ""))
            return StepResult(reply=ph.NO_SLOTS,
                              actions=waitlist_transfer(ctx, summary),
                              done=True, status="transferred")
        self.all_slots = slots
        self.options = slots[:3]
        self.offset = 3
        note = relaxed_note if relaxed else ""
        reply = (prefix or "") + ph.OFFER_SLOTS.format(
            options=options_phrase(self.options, ctx.today))
        if note:
            reply = note + " " + reply
        return StepResult(reply=reply, expecting="выбор времени")

    def choose(self, ctx: DialogContext, frame, text: str):
        """Выбор слота по фрейму (решение принимает модель, не парсинг).
        → ("picked", slot) | ("again", StepResult) | ("none", StepResult)
        | ("miss", None)."""
        if frame.choice is not None and 0 <= frame.choice < len(self.options):
            self.chosen = self.options[frame.choice]
            return "picked", self.chosen
        # единственный вариант и согласие — «да, давайте»
        if len(self.options) == 1 and frame.answer is True:
            self.chosen = self.options[0]
            return "picked", self.chosen
        prefs = frame.time_prefs
        if prefs is not None and prefs.daypart and frame.answer is None:
            self.daypart = prefs.daypart
            return "again", self.offer(
                ctx, relaxed_note="В это время свободно следующее.")
        if frame.more:
            nxt = self.all_slots[self.offset:self.offset + 3]
            self.offset += 3
            if nxt:
                self.options = nxt
                return "again", StepResult(reply=ph.OFFER_SLOTS_MORE.format(
                    options=options_phrase(nxt, ctx.today)))
            return "none", StepResult(
                reply="Другого времени в ближайшие две недели нет. "
                      + ph.NO_SLOTS,
                actions=waitlist_transfer(
                    ctx, "Пациент не выбрал слоты: " + self.specialty),
                done=True, status="transferred")
        if frame.best_doctor:
            self.options = self.all_slots[:3]
            return "again", StepResult(reply=ph.GOOD_DOCTOR.format(
                options=options_phrase(self.options, ctx.today)))
        if frame.answer is False:
            return "none", StepResult(
                reply=ph.NO_SLOTS,
                actions=waitlist_transfer(
                    ctx, "Пациент отказался от слотов: " + self.specialty),
                done=True, status="transferred")
        return "miss", None
