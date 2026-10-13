# -*- coding: utf-8 -*-
"""С-12 Звонок в интересах ребёнка/другого, С-14 Экстренная ситуация,
С-15 Жалоба, граничный медицинский вопрос (3.8).

Все решения — по полям фрейма NLU (relation, emergency_cleared,
ambulance_called и т.д.); сценарии реплики не разбирают."""
from __future__ import annotations

from .. import phrases as ph
from .. import textutils as tu
from ..schema import (Intent, MisPatient, NotifySenior, Priority, ResultCode,
                      RouteRule, ScenarioId, Target, Transfer)
from .base import DialogContext, IdentificationFlow, Scenario, StepResult


class ThirdPartyScenario(Scenario):
    """С-12: идентификация ПАЦИЕНТА (ребёнка/другого человека), вопрос
    «кем приходитесь», затем основной сценарий от его имени."""

    id = ScenarioId.C12
    intent_label = "I-01 Запись в интересах пациента"

    def __init__(self, inner: Scenario, child: bool = True):
        self.inner = inner
        self.child = child
        self.ident = IdentificationFlow(
            ph.ASK_FIO_DOB_CHILD if child else ph.ASK_FIO_DOB_PATIENT)
        self.state = "start"     # identify → relation

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", "запись в интересах другого")
        if ctx.patient is not None:
            return self._ask_relation(ctx)
        self.state = "identify"
        return self.ident.start(ctx)

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if self.state == "identify":
            res = self.ident.handle(ctx, frame, text)
            if res is not None:
                return res
            return self._ask_relation(ctx)

        if self.state == "relation":
            rel = frame.relation
            if rel is None and frame.answer is True:
                # ответ «да» на вопрос «Вы мама Алисы?» / «кем приходитесь?»
                rel = "mother" if self.child else "representative"
            if rel is None:
                if frame.answer is False:
                    return StepResult(reply=ph.ASK_RELATION_FALLBACK,
                                      expecting="кем приходится")
                return StepResult(reply=self._relation_question(ctx),
                                  progress=False)
            ctx.caller_relation = rel
            ctx.collect("звонящий", rel)
            reply = ""
            if rel == "relative":
                reply = ph.NON_REPRESENTATIVE_NOTE + " "
            return StepResult(reply=reply, handoff=self.inner)
        return StepResult(progress=False)

    def _ask_relation(self, ctx: DialogContext) -> StepResult:
        self.state = "relation"
        return StepResult(reply=self._relation_question(ctx),
                          expecting="кем приходитесь")

    def _relation_question(self, ctx: DialogContext) -> str:
        p: MisPatient = ctx.patient
        if self.child:
            return ph.ASK_RELATION_CHILD.format(name=p.fio.first)
        return ph.ASK_RELATION_FALLBACK


class EmergencyScenario(Scenario):
    """С-14: прерывает любой сценарий; никаких уточняющих вопросов,
    идентификации и предложений записи — только 103/112 и перевод P1."""

    id = ScenarioId.C14
    intent_label = "I-11 Экстренная ситуация"

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", "экстренная ситуация")
        ctx.collect("флаг", ctx.collected.get("красный флаг", "категория А"))
        return StepResult(
            reply=ph.EMERGENCY_MAIN,
            actions=[
                Transfer(target=Target.REGISTRAR, rule=RouteRule.R01,
                         priority=Priority.P1,
                         card=ctx.build_card(
                             RouteRule.R01, Priority.P1,
                             "Экстренный: " + ctx.last_user[:120],
                             self.intent_label)),
                NotifySenior("Экстренный звонок: " + ctx.last_user[:120]),
            ],
            result_code=ResultCode.EMERGENCY)

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if frame.ambulance_called:
            from ..schema import EndCall
            return StepResult(reply=ph.EMERGENCY_ALREADY, actions=[EndCall(
                reason="скорая вызвана")], done=True,
                result_code=ResultCode.EMERGENCY)
        if frame.emergency_cleared:
            # ложное срабатывание: фраза о 103 уже прозвучала (безопасность
            # важнее), возвращаемся к исходному сценарию
            return StepResult(reply=ph.EMERGENCY_FALSE_ALARM +
                              " Возвращаюсь к вашему вопросу.",
                              resume_previous=True)
        # пациент продолжает говорить об остром состоянии
        return StepResult(
            reply=ph.EMERGENCY_TRANSFER,
            actions=[Transfer(target=Target.REGISTRAR,
                              rule=RouteRule.R01, priority=Priority.P1,
                              card=ctx.build_card(
                                  RouteRule.R01, Priority.P1,
                                  "Продолжает описывать острое состояние: "
                                  + ctx.last_user[:100],
                                  self.intent_label))],
            done=True, result_code=ResultCode.EMERGENCY)


class ComplaintScenario(Scenario):
    """С-15: принять жалобу, не обсуждая по существу, → старший
    администратор (М3, P2)."""

    id = ScenarioId.C15
    intent_label = "I-12 Жалоба"

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", "жалоба")
        ctx.collect("недовольство", "да")
        return StepResult(
            reply=ph.R03_TRANSFER,
            actions=[Transfer(target=Target.SENIOR, rule=RouteRule.R03,
                              priority=Priority.P2,
                              card=ctx.build_card(
                                  RouteRule.R03, Priority.P2,
                                  "Жалоба: " + ctx.last_user[:120],
                                  self.intent_label))],
            done=True, result_code=ResultCode.COMPLAINT)


class MedicalBoundaryScenario(Scenario):
    """I-13: граничная фраза 3.8 — без советов, предложение записи;
    при настойчивости — перевод R-04."""

    id = ScenarioId.C01
    intent_label = "I-13 Медицинский вопрос"

    def __init__(self):
        self.state = "offered"      # offered → done
        self.count = 0

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", "медицинский вопрос")
        self.count += 1
        return StepResult(reply=ph.MEDICAL_BOUNDARY, expecting="да/нет")

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        if frame.answer is True:
            from .booking import BookingScenario
            return StepResult(handoff=BookingScenario(specialty="gp"))
        if frame.answer is False:
            return StepResult(reply=ph.ANYTHING_ELSE, done=True,
                              result_code=ResultCode.INFO_GIVEN)
        if frame.intent is not None or frame.faq_id is not None:
            return StepResult(progress=False)
        # пациент настаивает после граничной фразы → R-04
        return StepResult(
            reply=ph.MEDICAL_PERSIST,
            actions=[Transfer(target=Target.REGISTRAR, rule=RouteRule.R04,
                              priority=Priority.P3,
                              card=ctx.build_card(
                                  RouteRule.R04, Priority.P3,
                                  "Медицинский вопрос, пациент настаивает",
                                  self.intent_label))],
            done=True, result_code=ResultCode.ESCALATED)
