# -*- coding: utf-8 -*-
"""Действия с существующими записями: перенос, отмена, проверка.
Один поток: идентификация → выбор записи (если несколько) → действие.
Сведения о записях — только самому пациенту (3.5)."""
from __future__ import annotations

from .. import phrases as ph
from .. import textutils as tu
from ..mis import SlotTaken
from ..schema import (CANCELLED, CONFIRMED, INFO, RESCHEDULED, Intent,
                      SmsRequest, Target, Transfer)
from .common import (DialogContext, IdentificationFlow, Scenario, SlotChooser,
                     StepResult, booking_sms, options_phrase)


def _future_bookings(ctx: DialogContext) -> list:
    """Активные записи на будущее (прошедшие не показываем)."""
    books = ctx.mis.active_bookings(ctx.patient.id)
    return [b for b in books if b.date > ctx.today or
            (b.date == ctx.today and (b.time.hour, b.time.minute) >
             (ctx.now.hour, ctx.now.minute))]


class AppointmentsFlow(Scenario):
    """action: reschedule | cancel | check_booking."""

    label = "appointments"

    def __init__(self, frame=None, action: str = "reschedule",
                 target_booking=None):
        self.action = action or (frame.intent.value
                                 if frame and frame.intent else "reschedule")
        self.ident = IdentificationFlow()
        self.state = "identify"    # identify/which/third_party/…по действию
        self.bookings: list = []
        self.source = target_booking
        self.chooser: SlotChooser | None = None

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("тема", {"reschedule": "перенос записи",
                             "cancel": "отмена записи",
                             "check_booking": "проверка записи"}[self.action])
        if ctx.patient is not None:
            return self._after_ident(ctx)
        return self.ident.start(ctx)

    def current_options(self) -> list:
        if self.state == "which":
            return self.bookings[:3]
        chooser = getattr(self, "chooser", None)
        return list(chooser.options) if chooser else []

    # -- реплики --

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
                return self._have_source(ctx)
            return StepResult(label=self.label,
                              reply=self._which_question(ctx),
                              progress=False)

        if self.state == "third_party":
            rel = frame.relation
            if rel == "self":
                return self._announce(ctx)
            if rel is not None or frame.answer is not None:
                return StepResult(
                    label=self.label,
                    reply=ph.THIRD_PARTY_REFUSAL,
                    actions=[SmsRequest(phone=ctx.patient.phone,
                                        text=ph.SMS_TO_PATIENT,
                                        kind="to_patient")],
                    done=True, status=INFO)
            return StepResult(label=self.label, reply=ph.THIRD_PARTY_ASK,
                              progress=False)

        if self.state == "announce":
            # из проверки записи пациент может сразу просить перенос/отмену
            if frame.intent == Intent.CANCEL:
                self.action = "cancel"
                self.source = _future_bookings(ctx)[0] \
                    if _future_bookings(ctx) else None
                if self.source is not None:
                    return self._have_source(ctx)
            if frame.intent == Intent.RESCHEDULE:
                self.action = "reschedule"
                self.source = _future_bookings(ctx)[0] \
                    if _future_bookings(ctx) else None
                if self.source is not None:
                    return self._have_source(ctx)
            if frame.answer is True:
                books = _future_bookings(ctx)
                if books:
                    b = books[0]
                    sms = SmsRequest(
                        phone=ctx.patient.phone,
                        text=ph.SMS_REMINDER.format(
                            date=b.date.strftime("%d.%m"),
                            time=b.time.strftime("%H:%M"), cab=b.cabinet),
                        kind="reminder")
                    return StepResult(label=self.label, reply=ph.SMS_SENT,
                                      actions=[sms], done=True, status=INFO)
                return StepResult(label=self.label, reply=ph.SMS_SENT,
                                  done=True, status=INFO)
            if frame.answer is False:
                return StepResult(label=self.label,
                                  reply=ph.ANYTHING_ELSE_SHORT, done=True,
                                  status=INFO)
            return StepResult(label=self.label, progress=False)

        if self.action == "reschedule":
            return self._reschedule_turn(ctx, frame, text)
        if self.action == "cancel":
            return self._cancel_turn(ctx, frame)
        return StepResult(label=self.label, progress=False)

    # -- общие шаги --

    def _after_ident(self, ctx: DialogContext) -> StepResult:
        thanks = ph.THANKS_NAME.format(
            name=ctx.patient.fio.name_patronymic()) + " "
        if self.source is not None:      # запись уже выбрана (дубль/очередь)
            return self._have_source(ctx, prefix=thanks)
        self.bookings = _future_bookings(ctx)
        if not self.bookings:
            return StepResult(label=self.label,
                              reply=thanks + ph.NO_BOOKINGS, done=True,
                              status=INFO)
        # проверка записи — только самому пациенту (3.5)
        if self.action == "check_booking" and ctx.ani and ctx.patient.phone:
            ani_digits = "".join(ch for ch in ctx.ani if ch.isdigit())[-10:]
            pat_digits = "".join(ch for ch in ctx.patient.phone
                                 if ch.isdigit())[-10:]
            if ani_digits != pat_digits and ctx.caller_relation is None:
                self.state = "third_party"
                return StepResult(label=self.label,
                                  reply=thanks + ph.THIRD_PARTY_ASK,
                                  expecting="сам / в интересах другого")
        if len(self.bookings) == 1:
            self.source = self.bookings[0]
            return self._have_source(ctx, prefix=thanks)
        self.state = "which"
        return StepResult(label=self.label,
                          reply=thanks + self._which_question(ctx))

    def _which_question(self, ctx) -> str:
        verb = {"reschedule": "Какую переносить?",
                "cancel": "Какую отменить?",
                "check_booking": "Какая интересует?"}[self.action]
        return "У вас несколько записей: " + self._list_phrase(ctx) + ". " + verb

    def _list_phrase(self, ctx) -> str:
        parts = [f"{tu.specialty_nom(b.specialty)} — "
                 f"{tu.date_words(b.date, with_weekday=False)} "
                 f"в {tu.time_words(b.time)}"
                 for b in self.bookings[:3]]
        return "; ".join(parts)

    def _have_source(self, ctx: DialogContext, prefix: str = "") -> StepResult:
        src = self.source
        ctx.collect("запись", f"{tu.specialty_nom(src.specialty)} "
                    f"{src.date.isoformat()} {src.time.strftime('%H:%M')}")
        if self.action == "reschedule":
            self.state = "confirm_src"
            return StepResult(
                label=self.label,
                reply=prefix + ph.CONFIRM_MOVE_SOURCE.format(
                    specialty=tu.specialty_nom(src.specialty),
                    doctor=tu.doctor_gen(src.doctor),
                    date=tu.date_words(src.date), time=tu.time_words(src.time)),
                expecting="да/нет")
        if self.action == "cancel":
            self.state = "cancel_or_move"
            # поздняя отмена (менее 2 часов) — отметка для статистики
            late = (src.date == ctx.today and (
                src.time.hour * 60 + src.time.minute -
                (ctx.now.hour * 60 + ctx.now.minute)) < 120)
            if late:
                ctx.collect("поздняя отмена", "да")
            return StepResult(
                label=self.label,
                reply=prefix + ph.CANCEL_OR_MOVE.format(
                    specialty=tu.specialty_nom(src.specialty),
                    date=tu.date_words(src.date), time=tu.time_words(src.time)),
                expecting="перенести/отменить")
        return self._announce(ctx)

    def _announce(self, ctx: DialogContext) -> StepResult:
        books = _future_bookings(ctx)
        if not books:
            return StepResult(label=self.label, reply=ph.NO_BOOKINGS_CHECK,
                              done=True, status=INFO)
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
        return StepResult(label=self.label, reply=reply, expecting="да/нет")

    # -- перенос --

    def _reschedule_turn(self, ctx, frame, text) -> StepResult:
        if self.state == "confirm_src":
            if frame.answer is True:
                return self._offer(ctx)
            if frame.answer is False:
                if len(self.bookings) > 1:
                    self.state = "which"
                    return StepResult(label=self.label,
                                      reply=self._which_question(ctx))
                return StepResult(label=self.label,
                                  reply="Хотите вместо этого записаться "
                                        "заново?", progress=False)
            return StepResult(label=self.label, progress=False)

        if self.state == "offer":
            status, payload = self.chooser.choose(ctx, frame, text)
            if status == "picked":
                return self._confirm_move(ctx)
            if status in ("again", "none"):
                return payload
            return StepResult(label=self.label, reply=ph.OFFER_SLOTS.format(
                options=options_phrase(self.chooser.options, ctx.today)),
                progress=False)

        if self.state == "confirm":
            if frame.answer is True:
                return self._move(ctx)
            if frame.answer is False:
                return self._offer(ctx)
            return StepResult(label=self.label, progress=False)
        return StepResult(label=self.label, progress=False)

    def _offer(self, ctx: DialogContext) -> StepResult:
        self.chooser = SlotChooser(self.source.specialty)
        self.state = "offer"
        return self.chooser.offer(ctx, relaxed_note="У этого врача другое "
                                                    "время. Ближайшее:")

    def _confirm_move(self, ctx) -> StepResult:
        slot = self.chooser.chosen
        self.state = "confirm"
        return StepResult(
            label=self.label,
            reply=ph.CONFIRM_MOVE.format(
                date=tu.date_words_move(slot.date),
                time=tu.time_words(slot.time),
                cabinet=tu.number_words(slot.cabinet),
                old_date=tu.date_words(self.source.date, with_weekday=False)))

    def _move(self, ctx) -> StepResult:
        try:
            appt = ctx.mis.reschedule_booking(self.source.id,
                                              self.chooser.chosen)
        except SlotTaken:
            return StepResult(
                label=self.label,
                reply="Извините, не получилось перенести — время только что "
                      "заняли. Ваша запись в силе. " +
                      ph.OFFER_SLOTS.format(options=options_phrase(
                          self.chooser.options[:2], ctx.today)),
                actions=[Transfer(target=Target.OPERATOR, priority=3,
                                  card=ctx.card("Слот занят при переносе "
                                                "записи", "перенос записи"))],
                done=True, status="transferred")
        return StepResult(label=self.label, reply=ph.RESCHEDULE_DONE,
                          actions=[booking_sms(ctx, appt, "rescheduled")],
                          done=True, status=RESCHEDULED)

    # -- отмена --

    def _cancel_turn(self, ctx, frame) -> StepResult:
        if self.state == "cancel_or_move":
            if frame.intent == Intent.RESCHEDULE:
                return self._switch_to_reschedule(ctx)
            if frame.intent == Intent.CANCEL:
                return self._confirm_cancel(ctx)
            if frame.answer is True:
                # первый ответ «да» на «Перенести или отменить?» — перенос
                return self._switch_to_reschedule(ctx)
            if frame.answer is False:
                return StepResult(label=self.label, reply=ph.ANYTHING_ELSE,
                                  done=True)
            return StepResult(label=self.label,
                              reply="Перенести или отменить?", progress=False)

        if self.state == "confirm":
            if frame.answer is True:
                return self._cancel(ctx)
            if frame.answer is False:
                self.state = "cancel_or_move"
                return StepResult(label=self.label,
                                  reply="Перенести или отменить?",
                                  expecting="перенести/отменить")
            return StepResult(label=self.label, progress=False)
        return StepResult(label=self.label, progress=False)

    def _switch_to_reschedule(self, ctx) -> StepResult:
        self.action = "reschedule"
        self.state = "confirm_src"
        return self._have_source(ctx)

    def _confirm_cancel(self, ctx) -> StepResult:
        self.state = "confirm"
        src = self.source
        return StepResult(
            label=self.label,
            reply=ph.CONFIRM_CANCEL.format(
                specialty=tu.specialty_nom(src.specialty),
                date=tu.date_words(src.date, with_weekday=False)),
            expecting="да/нет")

    def _cancel(self, ctx) -> StepResult:
        src = self.source
        reason = ("поздняя отмена"
                  if ctx.collected.get("поздняя отмена") == "да"
                  else "не требуется")
        appt = ctx.mis.cancel_booking(src.id, reason)
        return StepResult(label=self.label, reply=ph.CANCEL_DONE,
                          actions=[booking_sms(ctx, appt, "cancelled")],
                          done=True, status=CANCELLED)
