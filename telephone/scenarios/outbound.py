# -*- coding: utf-8 -*-
"""С-10 Исходящий звонок-напоминание о приёме."""
from __future__ import annotations

from .. import phrases as ph
from .. import textutils as tu
from ..mis import SlotTaken
from ..schema import (Appointment, EndCall, Intent, MisPatient, NotifySenior,
                      ResultCode, ScenarioId, VoicemailEvent)
from .base import DialogContext, Scenario, SlotChooser, StepResult, \
    booking_sms, options_phrase


class ReminderCallScenario(Scenario):
    """С-10: проверка адресата → напоминание → подтверждение / перенос /
    отмена. Коды: CONFIRMED, RESCHEDULED, CANCELLED, WRONG_PERSON,
    VOICEMAIL, OPT_OUT, ESCALATED."""

    id = ScenarioId.C10
    intent_label = "I-01 Напоминание (исходящий)"

    def __init__(self, patient: MisPatient, appointment: Appointment):
        self.patient = patient
        self.appointment = appointment
        self.state = "verify"       # verify/attend/offer_move/confirm/cancel_q
        self.chooser: SlotChooser | None = None

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.patient = self.patient
        ctx.collect("интент", "напоминание о приёме")
        return StepResult(reply=ph.OUT_GREETING.format(
            name=self.patient.fio.name_patronymic()),
            expecting="подтверждение адресата")

    def handle_event(self, ctx: DialogContext, ev) -> StepResult | None:
        if isinstance(ev, VoicemailEvent):
            return StepResult(
                reply=ph.OUT_VOICEMAIL,
                actions=[EndCall(reason="voicemail")],
                done=True, result_code=ResultCode.VOICEMAIL)
        return None

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if self.state == "verify":
            if frame.answer is True:
                self.state = "attend"
                a = self.appointment
                return StepResult(reply=ph.OUT_REMINDER.format(
                    specialty=tu.specialty_dat(a.specialty),
                    doctor=tu.doctor_gen(a.doctor),
                    date=tu.date_words(a.date), time=tu.time_words(a.time),
                    cabinet=tu.number_words(a.cabinet)),
                    expecting="приду / не приду")
            if frame.wrong_person or frame.answer is False:
                return StepResult(
                    reply=ph.OUT_WRONG_PERSON.format(
                        name=self.patient.fio.name_patronymic()),
                    actions=[_sms_to_patient(ctx)],
                    done=True, result_code=ResultCode.WRONG_PERSON)
            return StepResult(reply=ph.OUT_GREETING.format(
                name=self.patient.fio.name_patronymic()), progress=False)

        if self.state == "attend":
            if frame.opt_out:
                ctx.collect("отказ от обзвона", "да")
                return StepResult(reply=ph.OUT_OPT_OUT, done=True,
                                  result_code=ResultCode.OPT_OUT)
            if frame.hospital:
                ctx.mis.cancel_booking(self.appointment.id,
                                       "пациент в стационаре")
                return StepResult(
                    reply=ph.OUT_BAD_NEWS,
                    actions=[NotifySenior(
                        "Скорректировать данные пациента: "
                        + self.patient.id)],
                    done=True, result_code=ResultCode.CANCELLED)
            if frame.intent == Intent.RESCHEDULE:
                return self._offer_move(ctx)
            if frame.intent == Intent.CANCEL:
                self.state = "cancel_q"
                return StepResult(reply="Хотите вместо этого перенести "
                                        "запись на другое время?",
                                  expecting="перенести/отменить")
            if frame.answer is True:
                return StepResult(reply=ph.OUT_CONFIRMED_OK, done=True,
                                  result_code=ResultCode.CONFIRMED)
            if frame.maybe:
                return StepResult(reply=ph.OUT_MAYBE, done=True,
                                  result_code=ResultCode.INFO_GIVEN)
            if frame.answer is False:
                self.state = "cancel_q"
                return StepResult(reply="Хотите вместо этого перенести "
                                        "запись на другое время?",
                                  expecting="перенести/отменить")
            return StepResult(reply="Вы придёте?", progress=False)

        if self.state == "cancel_q":
            if frame.intent == Intent.RESCHEDULE or frame.answer is True:
                return self._offer_move(ctx)
            if frame.intent == Intent.CANCEL or frame.answer is False:
                appt = ctx.mis.cancel_booking(self.appointment.id,
                                              "не требуется")
                return StepResult(reply=ph.CANCEL_DONE,
                                  actions=[booking_sms(ctx, appt,
                                                       "cancelled")],
                                  done=True, result_code=ResultCode.CANCELLED)
            return StepResult(reply="Перенести или отменить?", progress=False)

        if self.state == "offer_move":
            status, payload = self.chooser.choose(ctx, frame, text)
            if status == "picked":
                self.state = "confirm"
                slot = self.chooser.chosen
                return StepResult(reply=ph.CONFIRM_MOVE.format(
                    date=tu.date_words_move(slot.date),
                    time=tu.time_words(slot.time),
                    cabinet=tu.number_words(slot.cabinet),
                    old_date=tu.date_words(self.appointment.date,
                                           with_weekday=False)),
                    expecting="да/нет")
            if status in ("again", "none"):
                return payload
            return StepResult(reply="Есть время: " + options_phrase(
                self.chooser.options, ctx.today) + ". Что удобнее?",
                progress=False)

        if self.state == "confirm":
            if frame.answer is True:
                try:
                    appt = ctx.mis.reschedule_booking(self.appointment.id,
                                                      self.chooser.chosen)
                except SlotTaken:
                    return StepResult(reply="Извините, время только что "
                                            "заняли. Ваша запись в силе. "
                                            + ph.OUT_MAYBE,
                                      done=True,
                                      result_code=ResultCode.ESCALATED)
                return StepResult(
                    reply="Готово, SMS придёт на этот номер. Спасибо, что "
                          "предупредили. Всего доброго!",
                    actions=[booking_sms(ctx, appt, "rescheduled")],
                    done=True, result_code=ResultCode.RESCHEDULED)
            if frame.answer is False:
                return self._offer_move(ctx)
            return StepResult(progress=False)
        return StepResult(progress=False)

    def _offer_move(self, ctx: DialogContext) -> StepResult:
        self.chooser = SlotChooser(self.appointment.specialty)
        self.state = "offer_move"
        res = self.chooser.offer(
            ctx, prefix="Давайте перенесу. ",
            relaxed_note="Ближайшее свободное время:")
        return res


def _sms_to_patient(ctx: DialogContext):
    from ..schema import SmsRequest
    return SmsRequest(phone=ctx.patient.phone, text=ph.SMS_TO_PATIENT,
                      kind="to_patient")
