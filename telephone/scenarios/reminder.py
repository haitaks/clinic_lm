# -*- coding: utf-8 -*-
"""Исходящий звонок-напоминание о приёме: проверка адресата →
напоминание → подтверждение / перенос / отмена / чужой человек /
автоответчик / отказ от обзвона."""
from __future__ import annotations

from .. import phrases as ph
from .. import textutils as tu
from ..mis import SlotTaken
from ..schema import (CANCELLED, CONFIRMED, EndCall, INFO, Intent,
                      NotifySenior, RESCHEDULED, SmsRequest, VOICEMAIL,
                      WRONG_PERSON, OPT_OUT)
from .common import (DialogContext, Scenario, SlotChooser, StepResult,
                     booking_sms, options_phrase)


class ReminderFlow(Scenario):

    label = "reminder"

    def __init__(self, patient, appointment):
        self.patient = patient
        self.appointment = appointment
        self.state = "verify"   # verify/attend/offer_move/confirm/cancel_q
        self.chooser: SlotChooser | None = None

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.patient = self.patient
        ctx.collect("тема", "напоминание о приёме")
        return StepResult(label=self.label, reply=ph.OUT_GREETING.format(
            name=self.patient.fio.name_patronymic()),
            expecting="подтверждение адресата")

    def handle_event(self, ctx: DialogContext, ev) -> StepResult | None:
        from ..schema import VoicemailEvent
        if isinstance(ev, VoicemailEvent):
            return StepResult(
                label=self.label, reply=ph.OUT_VOICEMAIL,
                actions=[EndCall(reason="voicemail")],
                done=True, status=VOICEMAIL)
        return None

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if self.state == "verify":
            if frame.answer is True:
                self.state = "attend"
                a = self.appointment
                return StepResult(
                    label=self.label,
                    reply=ph.OUT_REMINDER.format(
                        specialty=tu.specialty_dat(a.specialty),
                        doctor=tu.doctor_gen(a.doctor),
                        date=tu.date_words(a.date), time=tu.time_words(a.time),
                        cabinet=tu.number_words(a.cabinet)),
                    expecting="приду / не приду")
            if frame.wrong_person or frame.answer is False:
                return StepResult(
                    label=self.label,
                    reply=ph.OUT_WRONG_PERSON.format(
                        name=self.patient.fio.name_patronymic()),
                    actions=[SmsRequest(phone=ctx.patient.phone,
                                        text=ph.SMS_TO_PATIENT,
                                        kind="to_patient")],
                    done=True, status=WRONG_PERSON)
            return StepResult(label=self.label, reply=ph.OUT_GREETING.format(
                name=self.patient.fio.name_patronymic()), progress=False)

        if self.state == "attend":
            if frame.opt_out:
                ctx.collect("отказ от обзвона", "да")
                return StepResult(label=self.label, reply=ph.OUT_OPT_OUT,
                                  done=True, status=OPT_OUT)
            if frame.hospital:
                ctx.mis.cancel_booking(self.appointment.id,
                                       "пациент в стационаре")
                return StepResult(
                    label=self.label, reply=ph.OUT_BAD_NEWS,
                    actions=[NotifySenior(
                        "Скорректировать данные пациента: "
                        + self.patient.id)],
                    done=True, status=CANCELLED)
            if frame.intent == Intent.RESCHEDULE:
                return self._offer_move(ctx)
            if frame.intent == Intent.CANCEL:
                return self._cancel_q()
            if frame.answer is True:
                return StepResult(label=self.label, reply=ph.OUT_CONFIRMED_OK,
                                  done=True, status=CONFIRMED)
            if frame.maybe:
                return StepResult(label=self.label, reply=ph.OUT_MAYBE,
                                  done=True, status=INFO)
            if frame.answer is False:
                return self._cancel_q()
            return StepResult(label=self.label, reply="Вы придёте?",
                              progress=False)

        if self.state == "cancel_q":
            if frame.intent == Intent.RESCHEDULE or frame.answer is True:
                return self._offer_move(ctx)
            if frame.intent == Intent.CANCEL or frame.answer is False:
                appt = ctx.mis.cancel_booking(self.appointment.id,
                                              "не требуется")
                return StepResult(label=self.label, reply=ph.CANCEL_DONE,
                                  actions=[booking_sms(ctx, appt, "cancelled")],
                                  done=True, status=CANCELLED)
            return StepResult(label=self.label,
                              reply="Перенести или отменить?", progress=False)

        if self.state == "offer_move":
            status, payload = self.chooser.choose(ctx, frame, text)
            if status == "picked":
                self.state = "confirm"
                slot = self.chooser.chosen
                return StepResult(
                    label=self.label,
                    reply=ph.CONFIRM_MOVE.format(
                        date=tu.date_words_move(slot.date),
                        time=tu.time_words(slot.time),
                        cabinet=tu.number_words(slot.cabinet),
                        old_date=tu.date_words(self.appointment.date,
                                               with_weekday=False)),
                    expecting="да/нет")
            if status in ("again", "none"):
                return payload
            return StepResult(label=self.label,
                              reply="Есть время: " + options_phrase(
                                  self.chooser.options, ctx.today)
                                  + ". Что удобнее?", progress=False)

        if self.state == "confirm":
            if frame.answer is True:
                try:
                    appt = ctx.mis.reschedule_booking(self.appointment.id,
                                                      self.chooser.chosen)
                except SlotTaken:
                    return StepResult(
                        label=self.label,
                        reply="Извините, время только что заняли. Ваша запись "
                              "в силе. " + ph.OUT_MAYBE,
                        done=True, status="transferred")
                return StepResult(
                    label=self.label,
                    reply="Готово, SMS придёт на этот номер. Спасибо, что "
                          "предупредили. Всего доброго!",
                    actions=[booking_sms(ctx, appt, "rescheduled")],
                    done=True, status=RESCHEDULED)
            if frame.answer is False:
                return self._offer_move(ctx)
            return StepResult(label=self.label, progress=False)
        return StepResult(label=self.label, progress=False)

    def _offer_move(self, ctx) -> StepResult:
        self.chooser = SlotChooser(self.appointment.specialty)
        self.state = "offer_move"
        return self.chooser.offer(ctx, prefix="Давайте перенесу. ",
                                  relaxed_note="Ближайшее свободное время:")

    def _cancel_q(self) -> StepResult:
        self.state = "cancel_q"
        return StepResult(label=self.label,
                          reply="Хотите вместо этого перенести запись на "
                                "другое время?",
                          expecting="перенести/отменить")
