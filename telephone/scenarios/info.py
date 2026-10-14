# -*- coding: utf-8 -*-
"""Справки: график работы (врача/отделения/поликлиники) и FAQ раздела 6
регламента. Ответ — готовой фразой из справочника, затем предложение
записи или маршрутизация по followup-таблице FAQ."""
from __future__ import annotations

from .. import phrases as ph
from .. import textutils as tu
from ..schema import INFO, Intent, Target, Transfer
from .common import DialogContext, Scenario, StepResult

# идентификаторы ответов раздела 6 — их возвращает модель (Frame.faq_id)
_FAQ_BY_ID = {entry["id"]: entry for entry in ph.FAQ}


class InfoFlow(Scenario):
    """График (intent=schedule), FAQ-вопрос (frame.faq_id) или документы/
    прикрепление (intent=docs). Идентификация пациента не требуется."""

    label = "info"

    def __init__(self, frame=None):
        self.frame = frame
        self.state = "answer"      # answer → offer_booking
        self._specialty = None

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("тема", "справка")
        if self.frame is not None and self.frame.faq_id:
            return self._answer_faq(ctx, self.frame.faq_id)
        if self.frame is not None and self.frame.intent == Intent.DOCS:
            return self._docs_answer(ctx)
        return self._answer_schedule(ctx)

    def _docs_answer(self, ctx) -> StepResult:
        return StepResult(label=self.label, reply=ph.ATTACH_HOWTO +
                          " " + ph.ANYTHING_ELSE_SHORT, done=True,
                          status=INFO)

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if self.state != "offer_booking":
            return StepResult(label=self.label, progress=False)
        if frame.answer is True:
            from .booking import BookingFlow
            return StepResult(label=self.label,
                              handoff=BookingFlow(specialty=self._specialty))
        if frame.answer is False:
            return StepResult(label=self.label, reply="Хорошо. Всего доброго!",
                              done=True, status=INFO)
        if frame.intent is not None:
            return StepResult(label=self.label, progress=False)
        return StepResult(label=self.label, reply="Записать вас?",
                          progress=False)

    # -- график --

    def _answer_schedule(self, ctx) -> StepResult:
        doctor = self.frame.doctor if self.frame else None
        specialty = self.frame.specialty if self.frame else None
        self.state = "offer_booking"
        if doctor:
            infos = ctx.mis.doctor_schedule(doctor.split()[0])
            if not infos:
                return StepResult(label=self.label, reply=ph.DOCTOR_NOT_FOUND,
                                  done=True, status=INFO)
            info = infos[0]
            self._specialty = info.specialty
            if info.on_leave_until:
                return StepResult(label=self.label, reply=ph.DOCTOR_ON_LEAVE.format(
                    doctor=info.doctor,
                    date=tu.date_words(info.on_leave_until, with_weekday=False),
                    substitute=info.substitute))
            return StepResult(label=self.label, reply=ph.DOCTOR_SCHEDULE.format(
                doctor=info.doctor, days=info.days, hours=info.hours,
                cabinet=tu.number_words(info.cabinet)))
        if specialty:
            infos = [i for i in ctx.mis.doctor_schedule("")
                     if i.specialty == specialty]
            self._specialty = specialty
            if len(infos) == 2:
                return StepResult(
                    label=self.label,
                    reply=ph.TWO_DOCTORS_SCHEDULE.format(
                        first=infos[0].doctor.split()[0], hours1=infos[0].hours,
                        second=infos[1].doctor.split()[0], hours2=infos[1].hours,
                        cabinet=tu.number_words(infos[0].cabinet)))
            if len(infos) == 1:
                return StepResult(
                    label=self.label,
                    reply=ph.DOCTOR_SCHEDULE.format(
                        doctor=infos[0].doctor, days=infos[0].days,
                        hours=infos[0].hours,
                        cabinet=tu.number_words(infos[0].cabinet)))
        return StepResult(label=self.label, reply=ph.CLINIC_SCHEDULE,
                          expecting="да/нет")

    # -- FAQ --

    def _answer_faq(self, ctx, faq_id: str) -> StepResult:
        entry = _FAQ_BY_ID.get(faq_id)
        if entry is None:
            # вопрос из справочника не найден — оператор с текстом реплики
            return StepResult(
                label=self.label,
                reply="Наличие уточнит регистратор. " + ph.TRANSFER_OPERATOR,
                actions=[Transfer(target=Target.OPERATOR, priority=3,
                                  card=ctx.card(ctx.last_user[:150],
                                                "вопрос вне справочника"))],
                done=True, status="transferred")
        ctx.collect("вопрос", entry["id"])
        answer = entry["answer"]
        followup = entry["followup"]
        if followup == "":
            return StepResult(label=self.label, reply=answer, done=True,
                              status=INFO)
        if followup == "offer_booking":
            self.state = "offer_booking"
            return StepResult(label=self.label, reply=answer, status=INFO,
                              expecting="да/нет")
        # ответ звучит, затем действие/сценарий из followup-таблицы
        return self._followup(ctx, answer, followup)

    def _followup(self, ctx, answer: str, followup: str) -> StepResult:
        from .appointments import AppointmentsFlow
        from .booking import BookingFlow
        if followup == "cancel":
            return StepResult(label=self.label, reply=answer,
                              handoff=AppointmentsFlow(action="cancel"))
        if followup == "reschedule":
            return StepResult(label=self.label, reply=answer,
                              handoff=AppointmentsFlow(action="reschedule"))
        if followup == "home_visit":
            return StepResult(
                label=self.label,
                reply=answer + " " + ph.TRANSFER_OPERATOR,
                actions=[Transfer(target=Target.OPERATOR, priority=3,
                                  card=ctx.card(ctx.last_user[:150],
                                                "вызов врача на дом"))],
                done=True, status="transferred")
        if followup == "waitlist":
            return StepResult(
                label=self.label,
                reply=answer + " Соединяю с регистратором для оформления "
                               "листа ожидания.",
                actions=[Transfer(target=Target.OPERATOR, priority=3,
                                  card=ctx.card(ctx.last_user[:150],
                                                "лист ожидания"))],
                done=True, status="transferred")
        if followup == "paid":
            return StepResult(
                label=self.label,
                reply=answer + " " + ph.TRANSFER_PAID,
                actions=[Transfer(target=Target.PAID, priority=3,
                                  card=ctx.card(ctx.last_user[:150],
                                                "платные услуги"))],
                done=True, status="transferred")
        if followup == "attach":
            return self._attach_answer(answer)
        if followup == "escalate":
            return StepResult(
                label=self.label,
                reply=answer + " " + ph.TRANSFER_OPERATOR,
                actions=[Transfer(target=Target.OPERATOR, priority=3,
                                  card=ctx.card(ctx.last_user[:150],
                                                "вопрос вне справочника"))],
                done=True, status="transferred")
        return StepResult(label=self.label, reply=answer, done=True,
                          status=INFO)

    def _attach_answer(self, answer: str) -> StepResult:
        reply = ((answer + " ") if answer else "") + ph.ATTACH_HOWTO
        return StepResult(label=self.label, reply=reply, done=True,
                          status=INFO)
