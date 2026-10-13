# -*- coding: utf-8 -*-
"""С-07 Справка о графике работы + FAQ (раздел 6 регламента)."""
from __future__ import annotations

from .. import phrases as ph
from .. import textutils as tu
from ..schema import Intent, Priority, ResultCode, RouteRule, ScenarioId, \
    Target, Transfer
from .base import DialogContext, Scenario, StepResult

# идентификаторы ответов раздела 6 — их возвращает модель (Frame.faq_id)
_FAQ_BY_ID = {entry["id"]: entry for entry in ph.FAQ}


class ScheduleScenario(Scenario):
    """С-07: график врача/подразделения/поликлиники, затем предложение
    записи. Идентификация не требуется."""

    id = ScenarioId.C07
    intent_label = "I-06 График"

    def __init__(self, frame=None):
        self.frame = frame
        self.state = "start"        # start → offer_booking

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", "график/справка")
        doctor = self.frame.doctor if self.frame else None
        specialty = self.frame.specialty if self.frame else None
        if doctor:
            infos = ctx.mis.doctor_schedule(doctor.split()[0])
            if not infos:
                return StepResult(reply=ph.DOCTOR_NOT_FOUND, done=True,
                                  result_code=ResultCode.INFO_GIVEN)
            info = infos[0]
            if info.on_leave_until:
                self.state = "offer_booking"
                self._specialty = info.specialty
                return StepResult(reply=ph.DOCTOR_ON_LEAVE.format(
                    doctor=info.doctor,
                    date=tu.date_words(info.on_leave_until,
                                       with_weekday=False),
                    substitute=info.substitute))
            self.state = "offer_booking"
            self._specialty = info.specialty
            return StepResult(reply=ph.DOCTOR_SCHEDULE.format(
                doctor=info.doctor, days=info.days, hours=info.hours,
                cabinet=tu.number_words(info.cabinet)))
        if specialty:
            infos = [i for i in ctx.mis.doctor_schedule("")
                     if i.specialty == specialty]
            self.state = "offer_booking"
            self._specialty = specialty
            if len(infos) == 2:
                return StepResult(reply=ph.TWO_DOCTORS_SCHEDULE.format(
                    first=infos[0].doctor.split()[0], hours1=infos[0].hours,
                    second=infos[1].doctor.split()[0], hours2=infos[1].hours,
                    cabinet=tu.number_words(infos[0].cabinet)))
            if len(infos) == 1:
                return StepResult(reply=ph.DOCTOR_SCHEDULE.format(
                    doctor=infos[0].doctor, days=infos[0].days,
                    hours=infos[0].hours,
                    cabinet=tu.number_words(infos[0].cabinet)))
        self.state = "offer_booking"
        return StepResult(reply=ph.CLINIC_SCHEDULE,
                          expecting="да/нет")

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if self.state == "offer_booking":
            if frame.answer is True:
                from .booking import BookingScenario
                return StepResult(handoff=BookingScenario(
                    specialty=getattr(self, "_specialty", None)))
            if frame.answer is False:
                return StepResult(reply="Хорошо. Всего доброго!", done=True,
                                  result_code=ResultCode.INFO_GIVEN)
            if frame.intent is not None:
                return StepResult(progress=False)
            return StepResult(reply="Записать вас?", progress=False)
        return StepResult(progress=False)


class FaqScenario(Scenario):
    """FAQ раздела 6: готовые ответы + маршрут «действие» из таблицы."""

    id = ScenarioId.C07
    intent_label = "I-10 Документы и справки"

    def __init__(self, frame=None):
        self.frame = frame
        self.state = "start"        # start → offer
        self.entry: dict | None = None
        self._specialty = None

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", "справка")
        faq_id = self.frame.faq_id if self.frame else None
        self.entry = _FAQ_BY_ID.get(faq_id)
        if self.entry is None:
            from .transferred import TransferredScenario
            return StepResult(handoff=TransferredScenario(Intent.DOCS))
        ctx.collect("вопрос", self.entry["id"])
        followup = self.entry["followup"]
        answer = self.entry["answer"]
        if followup == "":
            return StepResult(reply=answer, done=True,
                              result_code=ResultCode.INFO_GIVEN)
        if followup == "offer_booking":
            self.state = "offer"
            return StepResult(reply=answer, result_code=ResultCode.INFO_GIVEN,
                              expecting="да/нет")
        # ответ звучит, затем действие/сценарий из таблицы раздела 6
        step = self._followup(ctx, followup)
        if step.handoff is not None:
            return StepResult(reply=answer + " " + step.reply
                              if step.reply else answer,
                              handoff=step.handoff)
        return StepResult(reply=answer + " " + step.reply
                          if step.reply else answer,
                          actions=step.actions, done=step.done,
                          result_code=step.result_code)

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if self.state == "offer":
            if frame.answer is True:
                from .booking import BookingScenario
                return StepResult(handoff=BookingScenario(
                    specialty=self._specialty))
            if frame.answer is False:
                return StepResult(reply=ph.ANYTHING_ELSE_SHORT, done=True,
                                  result_code=ResultCode.INFO_GIVEN)
            if frame.intent is not None:
                return StepResult(progress=False)
            return StepResult(reply="Записать вас?", progress=False)
        return StepResult(progress=False)

    def _followup(self, ctx: DialogContext, followup: str) -> StepResult:
        from .booking import BookingScenario
        from .records import CancelScenario, RescheduleScenario
        from .transferred import TransferredScenario
        routes = {
            "cancel": lambda: StepResult(handoff=CancelScenario()),
            "reschedule": lambda: StepResult(handoff=RescheduleScenario()),
            "home_visit": lambda: StepResult(
                handoff=TransferredScenario(Intent.HOME_VISIT)),
            "waitlist": lambda: StepResult(
                handoff=TransferredScenario(Intent.WAITLIST)),
            "paid": lambda: StepResult(
                handoff=TransferredScenario(Intent.PAID)),
            "attach": lambda: StepResult(
                handoff=TransferredScenario(Intent.DOCS)),
            "escalate": lambda: StepResult(
                reply="Наличие уточнит регистратор. " + ph.R08_TRANSFER,
                actions=[Transfer(
                    target=Target.REGISTRAR, rule=RouteRule.R08,
                    priority=Priority.P3,
                    card=ctx.build_card(
                        RouteRule.R08, Priority.P3,
                        "Вопрос вне справочника ГА", self.intent_label))],
                done=True, result_code=ResultCode.ESCALATED),
        }
        return routes[followup]()
