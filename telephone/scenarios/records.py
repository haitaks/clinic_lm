# -*- coding: utf-8 -*-
"""С-03 Перенос записи, С-04 Отмена записи, С-05 Проверка своей записи."""
from __future__ import annotations

from .. import phrases as ph
from .. import textutils as tu
from ..mis import SlotTaken
from ..schema import (Appointment, Intent, Priority, ResultCode, RouteRule,
                      ScenarioId, SmsRequest, Target, Transfer)
from .base import (DialogContext, IdentificationFlow, Scenario, SlotChooser,
                   StepResult, booking_sms, options_phrase)


def _future_bookings(ctx: DialogContext) -> list[Appointment]:
    """Активные записи на будущее (прошедшие не показываем)."""
    books = ctx.mis.active_bookings(ctx.patient.id)
    return [b for b in books if b.date > ctx.today or
            (b.date == ctx.today and (b.time.hour, b.time.minute) >
             (ctx.now.hour, ctx.now.minute))]


class RescheduleScenario(Scenario):
    """С-03: список записей → подтверждение исходной → новые слоты →
    атомарный перенос (старая отменяется только после создания новой)."""

    id = ScenarioId.C03
    intent_label = "I-02 Перенос записи"

    def __init__(self, frame=None, target_booking: Appointment | None = None):
        self.ident = IdentificationFlow()
        self.state = "start"        # identify/which/confirm_src/offer/confirm
        self.bookings: list[Appointment] = []
        self.source: Appointment | None = target_booking
        self.chooser: SlotChooser | None = None

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", "перенос записи")
        if ctx.patient is not None and self.source is not None:
            return self._have_source(ctx)
        if ctx.patient is not None:
            return self._after_ident(ctx)
        self.state = "identify"
        return self.ident.start(ctx)

    def current_options(self) -> list:
        if self.state == "which":
            return self.bookings[:3]
        chooser = getattr(self, "chooser", None)
        return list(chooser.options) if chooser else []

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if self.state == "identify":
            res = self.ident.handle(ctx, frame, text)
            if res is not None:
                return res
            return self._after_ident(ctx)

        if self.state == "which":
            if frame.choice is not None and frame.choice < len(self.bookings[:3]):
                self.source = self.bookings[:3][frame.choice]
                return self._have_source(ctx)
            return StepResult(reply=ph.ASK_WHICH_BOOKING.format(
                list=self._list_phrase(ctx)), progress=False)

        if self.state == "confirm_src":
            if frame.answer is True:
                return self._offer(ctx)
            if frame.answer is False:
                # возможно, пациент имел в виду другую запись
                if len(self.bookings) > 1:
                    self.state = "which"
                    return StepResult(reply=ph.ASK_WHICH_BOOKING.format(
                        list=self._list_phrase(ctx)))
                return StepResult(reply="Хотите вместо этого записаться "
                                        "заново?", progress=False)
            return StepResult(progress=False)

        if self.state == "offer":
            status, payload = self.chooser.choose(ctx, frame, text)
            if status == "picked":
                return self._confirm(ctx)
            if status in ("again", "none"):
                return payload
            return StepResult(reply=ph.OFFER_SLOTS.format(
                options=options_phrase(self.chooser.options, ctx.today)),
                progress=False)

        if self.state == "confirm":
            if frame.answer is True:
                return self._move(ctx)
            if frame.answer is False:
                return self._offer(ctx)
            return StepResult(progress=False)
        return StepResult(progress=False)

    def _after_ident(self, ctx: DialogContext) -> StepResult:
        thanks = ph.THANKS_NAME.format(
            name=ctx.patient.fio.name_patronymic()) + " "
        self.bookings = _future_bookings(ctx)
        if not self.bookings:
            return StepResult(reply=thanks + ph.NO_BOOKINGS, done=True)
        if len(self.bookings) == 1:
            self.source = self.bookings[0]
            return self._have_source(ctx, prefix=thanks)
        self.state = "which"
        return StepResult(reply=thanks + ph.ASK_WHICH_BOOKING.format(
            list=self._list_phrase(ctx)))

    def _list_phrase(self, ctx: DialogContext) -> str:
        parts = [f"{tu.specialty_nom(b.specialty)} — "
                 f"{tu.date_words(b.date, with_weekday=False)} "
                 f"в {tu.time_words(b.time)}"
                 for b in self.bookings[:3]]
        return "; ".join(parts)

    def _have_source(self, ctx: DialogContext, prefix: str = "") -> StepResult:
        self.state = "confirm_src"
        src = self.source
        ctx.collect("переносимая запись",
                    f"{tu.specialty_nom(src.specialty)} {src.date.isoformat()} "
                    f"{src.time.strftime('%H:%M')}")
        return StepResult(reply=prefix + ph.CONFIRM_MOVE_SOURCE.format(
            specialty=tu.specialty_nom(src.specialty),
            doctor=tu.doctor_gen(src.doctor),
            date=tu.date_words(src.date), time=tu.time_words(src.time)),
            expecting="да/нет")

    def _offer(self, ctx: DialogContext) -> StepResult:
        prefs_daypart = None
        self.chooser = SlotChooser(self.source.specialty,
                                   daypart=prefs_daypart)
        self.state = "offer"
        return self.chooser.offer(ctx, relaxed_note="У этого врача другое "
                                                    "время. Ближайшее:")

    def _confirm(self, ctx: DialogContext) -> StepResult:
        slot = self.chooser.chosen
        self.state = "confirm"
        return StepResult(reply=ph.CONFIRM_MOVE.format(
            date=tu.date_words_move(slot.date),
            time=tu.time_words(slot.time),
            cabinet=tu.number_words(slot.cabinet),
            old_date=tu.date_words(self.source.date, with_weekday=False)))

    def _move(self, ctx: DialogContext) -> StepResult:
        try:
            appt = ctx.mis.reschedule_booking(self.source.id,
                                              self.chooser.chosen)
        except SlotTaken:
            return StepResult(
                reply="Извините, не получилось перенести — время только что "
                      "заняли. Ваша запись в силе. " + ph.OFFER_SLOTS.format(
                          options=options_phrase(self.chooser.options[:2],
                                                 ctx.today)),
                actions=[Transfer(target=Target.REGISTRAR, rule=RouteRule.R06,
                                  priority=Priority.P3,
                                  card=ctx.build_card(
                                      RouteRule.R06, Priority.P3,
                                      "Слот занят при переносе записи",
                                      self.intent_label))],
                done=True, result_code=ResultCode.ESCALATED)
        return StepResult(reply=ph.RESCHEDULE_DONE,
                          actions=[booking_sms(ctx, appt, "rescheduled")],
                          done=True, result_code=ResultCode.RESCHEDULED)


class CancelScenario(Scenario):
    """С-04: перед отменей один раз предлагаем перенос (правило С-04)."""

    id = ScenarioId.C04
    intent_label = "I-03 Отмена записи"

    def __init__(self, frame=None):
        self.ident = IdentificationFlow()
        self.state = "start"        # identify/which/cancel_or_move/confirm
        self.bookings: list[Appointment] = []
        self.source: Appointment | None = None
        self.move_offered = False

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", "отмена записи")
        if ctx.patient is not None:
            return self._after_ident(ctx)
        self.state = "identify"
        return self.ident.start(ctx)

    def current_options(self) -> list:
        if self.state == "which":
            return self.bookings[:3]
        return []

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if self.state == "identify":
            res = self.ident.handle(ctx, frame, text)
            if res is not None:
                return res
            return self._after_ident(ctx)

        if self.state == "which":
            top = self.bookings[:3]
            if frame.choice is not None and frame.choice < len(top):
                self.source = top[frame.choice]
                return self._cancel_or_move(ctx)
            return StepResult(reply=ph.ASK_WHICH_BOOKING.format(
                list="; ".join(
                    f"{tu.specialty_nom(b.specialty)} — "
                    f"{tu.date_words(b.date, with_weekday=False)}"
                    for b in top)),
                progress=False)

        if self.state == "cancel_or_move":
            if frame.intent == Intent.RESCHEDULE:
                return StepResult(handoff=RescheduleScenario(
                    target_booking=self.source))
            if frame.intent == Intent.CANCEL:
                return self._confirm(ctx)
            if frame.answer is True:
                # первый ответ «да» на «Перенести или отменить?» — перенос
                return StepResult(handoff=RescheduleScenario(
                    target_booking=self.source))
            if frame.answer is False:
                return StepResult(reply=ph.ANYTHING_ELSE, done=True)
            return StepResult(reply="Перенести или отменить?", progress=False)

        if self.state == "confirm":
            if frame.answer is True:
                return self._cancel(ctx)
            if frame.answer is False:
                self.state = "cancel_or_move"
                return StepResult(reply="Перенести или отменить?",
                                  expecting="перенести/отменить")
            return StepResult(progress=False)
        return StepResult(progress=False)

    def _after_ident(self, ctx: DialogContext) -> StepResult:
        thanks = ph.THANKS_NAME.format(
            name=ctx.patient.fio.name_patronymic()) + " "
        self.bookings = _future_bookings(ctx)
        if not self.bookings:
            return StepResult(reply=thanks + ph.NO_BOOKINGS, done=True)
        if len(self.bookings) == 1:
            self.source = self.bookings[0]
            return self._cancel_or_move(ctx, prefix=thanks)
        self.state = "which"
        return StepResult(reply=thanks + "У вас несколько записей: " +
                          "; ".join(
                              f"{tu.specialty_nom(b.specialty)} — "
                              f"{tu.date_words(b.date, with_weekday=False)} "
                              f"в {tu.time_words(b.time)}"
                              for b in self.bookings[:3]) +
                          ". Какую отменить?")

    def _by_date(self, ctx: DialogContext, text: str) -> Appointment | None:
        """Не используется: выбор записи возвращает модель (Frame.choice)."""
        return None

    def _cancel_or_move(self, ctx: DialogContext, prefix: str = "") -> StepResult:
        self.state = "cancel_or_move"
        self.move_offered = True
        src = self.source
        # поздняя отмена (менее 2 часов) — отметка для статистики (С-04)
        late = (src.date == ctx.today and (
            src.time.hour * 60 + src.time.minute -
            (ctx.now.hour * 60 + ctx.now.minute)) < 120)
        if late:
            ctx.collect("поздняя отмена", "да")
        return StepResult(reply=prefix + ph.CANCEL_OR_MOVE.format(
            specialty=tu.specialty_nom(src.specialty),
            date=tu.date_words(src.date), time=tu.time_words(src.time)),
            expecting="перенести/отменить")

    def _confirm(self, ctx: DialogContext) -> StepResult:
        self.state = "confirm"
        src = self.source
        return StepResult(reply=ph.CONFIRM_CANCEL.format(
            specialty=tu.specialty_nom(src.specialty),
            date=tu.date_words(src.date, with_weekday=False)),
            expecting="да/нет")

    def _cancel(self, ctx: DialogContext) -> StepResult:
        src = self.source
        reason = "не требуется"
        if ctx.collected.get("поздняя отмена") == "да":
            reason = "поздняя отмена"
        appt = ctx.mis.cancel_booking(src.id, reason)
        sms = booking_sms(ctx, appt, "cancelled")
        return StepResult(reply=ph.CANCEL_DONE, actions=[sms],
                          done=True, result_code=ResultCode.CANCELLED)


class CheckBookingScenario(Scenario):
    """С-05: сообщения о записях — только самому пациенту (3.5)."""

    id = ScenarioId.C05
    intent_label = "I-04 Проверка записи"

    def __init__(self, frame=None):
        self.ident = IdentificationFlow()
        self.state = "start"        # identify/third_party/announce

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", "проверка записи")
        if ctx.patient is not None:
            return self._after_ident(ctx)
        self.state = "identify"
        return self.ident.start(ctx)

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if self.state == "identify":
            res = self.ident.handle(ctx, frame, text)
            if res is not None:
                return res
            return self._after_ident(ctx)

        if self.state == "third_party":
            rel = frame.relation
            if rel == "self":
                return self._announce(ctx)
            if rel is not None or frame.answer is not None:
                return StepResult(
                    reply=ph.THIRD_PARTY_REFUSAL,
                    actions=[SmsRequest(phone=ctx.patient.phone,
                                        text=ph.SMS_TO_PATIENT,
                                        kind="to_patient")],
                    done=True, result_code=ResultCode.INFO_GIVEN)
            return StepResult(reply=ph.THIRD_PARTY_ASK, progress=False)

        if self.state == "announce":
            if frame.answer is True:
                b = _future_bookings(ctx)[0] if _future_bookings(ctx) else None
                if b:
                    sms = SmsRequest(
                        phone=ctx.patient.phone,
                        text=ph.SMS_REMINDER.format(
                            date=b.date.strftime("%d.%m"),
                            time=b.time.strftime("%H:%M"), cab=b.cabinet),
                        kind="reminder")
                    return StepResult(reply=ph.SMS_SENT, actions=[sms],
                                      done=True,
                                      result_code=ResultCode.INFO_GIVEN)
                return StepResult(reply=ph.SMS_SENT, done=True,
                                  result_code=ResultCode.INFO_GIVEN)
            if frame.answer is False:
                return StepResult(reply=ph.ANYTHING_ELSE_SHORT, done=True,
                                  result_code=ResultCode.INFO_GIVEN)
            return StepResult(progress=False)
        return StepResult(progress=False)

    def _after_ident(self, ctx: DialogContext) -> StepResult:
        # АНИ не совпадает с телефоном из МИС → проверяем, кто звонит (3.5)
        if ctx.ani and ctx.patient.phone:
            ani_digits = "".join(ch for ch in ctx.ani if ch.isdigit())[-10:]
            pat_digits = "".join(ch for ch in ctx.patient.phone
                                 if ch.isdigit())[-10:]
            if ani_digits != pat_digits and ctx.caller_relation is None:
                self.state = "third_party"
                return StepResult(
                    reply=ph.THANKS_NAME.format(
                        name=ctx.patient.fio.name_patronymic()) + " " +
                    ph.THIRD_PARTY_ASK,
                    expecting="сам / в интересах другого")
        return self._announce(ctx)

    def _announce(self, ctx: DialogContext) -> StepResult:
        books = _future_bookings(ctx)
        if not books:
            return StepResult(reply=ph.NO_BOOKINGS_CHECK, done=True)
        self.state = "announce"
        b = books[0]
        reply = ph.ANNOUNCE_BOOKING.format(
            name=ctx.patient.fio.name_patronymic(),
            specialty=tu.specialty_nom(b.specialty),
            doctor=tu.doctor_gen(b.doctor),
            date=tu.date_words(b.date), time=tu.time_words(b.time),
            cabinet=tu.number_words(b.cabinet))
        if len(books) > 1:
            others = "; ".join(
                f"{tu.specialty_nom(x.specialty)} "
                f"{tu.date_words(x.date, with_weekday=False)}"
                for x in books[1:3])
            reply = (f"У вас {len(books)} записи. {reply} "
                     f"Также: {others}.")
        return StepResult(reply=reply, expecting="да/нет")
