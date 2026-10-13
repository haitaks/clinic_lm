# -*- coding: utf-8 -*-
"""Менеджер диалога телефонии (регламент РГ-ТЕЛ-01).

Конвейер хода:
1. приветствие с уведомлением о записи разговора (3.3–3.4);
2. сквозные правила на каждой реплике (policy, матрица 7.3) — красные
   флаги прерывают любой сценарий;
3. активный сценарий получает фрейм NLU и реплику целиком;
4. свободное состояние — классификация интента и запуск сценария;
5. лестница непонимания (3.2): переспрос → предложение оператора →
   автоперевод R-05.

Менеджер не знает, кто разобрал реплику (самописная frame-модель,
готовая LLM или словарный RuleNlu) — только контракт NluClient.parse.
"""
from __future__ import annotations

from datetime import datetime

from . import phrases as ph
from . import policy
from .mis import MisUnavailable
from .schema import (Appointment, CallbackTask, DmAction, DmTurn, EndCall,
                     Frame, Intent, MisPatient, NotifySenior, Priority,
                     ResultCode, RouteRule, SmsRequest, Target, Transfer,
                     VoicemailEvent)
from .scenarios import EmergencyScenario, MedicalBoundaryScenario, \
    ReminderCallScenario, create_scenario
from .scenarios.base import DialogContext, Scenario, StepResult
from .clinic_data import target_available


class DialogManager:
    """Один звонок = один экземпляр. Потокобезопасностью не занимается."""

    def __init__(self, nlu, mis, now: datetime, ani: str | None = None,
                 call_id: str | None = None,
                 outbound: tuple[MisPatient, Appointment] | None = None):
        self.nlu = nlu
        self.ctx = DialogContext(
            mis=mis, now=now, ani=ani,
            call_id=call_id or now.strftime("CALL-%Y%m%d-%H%M%S"))
        self.scenario: Scenario | None = None
        self.suspended: Scenario | None = None      # прерван С-14
        self.pending_frames: list = []              # очередь интентов (раздел 4)
        self.state = "greeted"      # greeted / active / wrap / closed
        self.result_code: ResultCode | None = None
        self.misses = 0
        self.silence_streak = 0
        self.outbound = outbound

    # ------------------------------------------------------------------
    # Жизненный цикл звонка

    def start_call(self) -> DmTurn:
        if self.outbound is not None:
            patient, appt = self.outbound
            self.scenario = ReminderCallScenario(patient, appt)
            self.state = "active"
            return self._finish_turn(self.scenario.start(self.ctx))
        return self._turn(DmTurn(reply=ph.GREETING_IN,
                                 expecting="интент звонка"))

    def handle_utterance(self, text: str) -> DmTurn:
        if self.state == "closed":
            return self._turn(DmTurn(reply=""))
        self.ctx.last_user = text
        self.ctx.log("patient", text)
        self.silence_streak = 0

        hit = policy.check_utterance(text)
        if hit is not None:
            turn = self._apply_policy(hit)
            if turn is not None:
                return turn

        if self.scenario is not None:
            frame = self.nlu.parse(text, self.ctx.today,
                                   options=self.scenario.current_options())
            return self._finish_turn(
                self._run(self.scenario, frame, text))

        frame = self.nlu.parse(text, self.ctx.today)
        if self.state == "wrap":
            if frame.answer is False and frame.intent is None:
                return self._farewell()
            if frame.answer is True and frame.intent is None:
                return self._turn(DmTurn(reply=ph.SURE_WHAT,
                                         expecting="интент"))
        if frame.intent is not None:
            if frame.extra_frames:
                self.pending_frames.extend(frame.extra_frames)
            return self._start_scenario(frame)
        # интент не распознан, но модель узнала вопрос из справочника FAQ
        if frame.faq_id is not None:
            return self._start_scenario(frame, force_faq=True)
        return self._ladder()

    def handle_event(self, ev) -> DmTurn:
        """События телефонии: тишина (R-17), DTMF, автоответчик."""
        if self.state == "closed":
            return self._turn(DmTurn(reply=""))
        if isinstance(ev, VoicemailEvent) and isinstance(
                self.scenario, ReminderCallScenario):
            res = self.scenario.handle_event(self.ctx, ev)
            if res is not None:
                return self._finish_turn(res)

        if hasattr(ev, "digits"):
            if self.scenario is not None:
                res = self.scenario.handle_event(self.ctx, ev)
                if res is not None:
                    return self._finish_turn(res)
            return self._turn(DmTurn(reply=""))

        # тишина
        self.silence_streak += 1
        if self.silence_streak == 1:
            return self._turn(DmTurn(reply=ph.SILENCE_1))
        self.result_code = self.result_code or ResultCode.DROPPED
        self.state = "closed"
        return self._turn(DmTurn(
            reply=ph.SILENCE_2, actions=[EndCall(reason="тишина")],
            result_code=self.result_code))

    def summary(self) -> dict:
        """Итог звонка: код результата, собранные данные, журнал."""
        return {"call_id": self.ctx.call_id,
                "result_code": self.result_code,
                "state": self.state,
                "patient": self.ctx.patient_label(),
                "collected": dict(self.ctx.collected),
                "recording_consent": self.ctx.recording_consent,
                "journal": list(self.ctx.journal)}

    # ------------------------------------------------------------------
    # Сквозные правила

    def _apply_policy(self, hit: policy.PolicyHit) -> DmTurn | None:
        if hit.kind == "red_a":
            self.ctx.collect("красный флаг", hit.phrase_group)
            if isinstance(self.scenario, EmergencyScenario):
                # красный флаг звучит снова — немедленная передача P1
                return self._finish_turn(StepResult(
                    reply=ph.EMERGENCY_TRANSFER,
                    actions=self._transfer_actions(
                        Target.REGISTRAR, RouteRule.R01, Priority.P1,
                        "Повторный красный флаг: " + hit.phrase_group),
                    done=True, result_code=ResultCode.EMERGENCY))
            if self.scenario is not None:
                self.suspended = self.scenario
            self.scenario = EmergencyScenario()
            self.state = "active"
            # инцидент фиксируется сразу, даже если окажется ложной тревогой
            self.result_code = ResultCode.EMERGENCY
            return self._finish_turn(self.scenario.start(self.ctx))

        if hit.kind == "red_b":
            self.ctx.collect("красный флаг Б", hit.phrase_group)
            return self._finish_turn(StepResult(
                reply=ph.R11_TRANSFER + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.REGISTRAR, RouteRule.R11, Priority.P1,
                    "Острое состояние: " + hit.phrase_group),
                done=True, result_code=ResultCode.ESCALATED))

        if hit.kind == "operator":
            return self._finish_turn(StepResult(
                reply=ph.R02_TRANSFER + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.REGISTRAR, RouteRule.R02, Priority.P2,
                    "Пациент попросил сотрудника"),
                done=True, result_code=ResultCode.ESCALATED))

        if hit.kind == "complaint":
            self.scenario = create_scenario(
                _frame_with_intent(None, Intent.COMPLAINT))
            self.state = "active"
            return self._finish_turn(self.scenario.start(self.ctx))

        if hit.kind == "gratitude":
            return self._finish_turn(StepResult(
                reply=ph.GRATITUDE, done=True,
                result_code=ResultCode.GRATITUDE))

        if hit.kind == "complaint_written":
            return self._finish_turn(StepResult(
                reply=ph.COMPLAINT_WRITTEN, done=True,
                result_code=ResultCode.COMPLAINT))

        if hit.kind == "robot":
            return self._turn(DmTurn(reply=ph.ROBOT_ANSWER,
                                     expecting="ответ на вопрос ГА"))

        if hit.kind == "recording_objection":
            self.ctx.recording_consent = False
            tail = (" " + self.ctx.last_assistant) if self.scenario \
                else " " + ph.ANYTHING_ELSE
            return self._turn(DmTurn(
                reply=ph.RECORDING_OFF + tail))

        if hit.kind == "tooth":
            return self._finish_turn(StepResult(
                reply=ph.R09_TRANSFER + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.DENTISTRY, RouteRule.R09, Priority.P2,
                    "Острая зубная боль: " + hit.phrase_group),
                done=True, result_code=ResultCode.ESCALATED))

        if hit.kind == "medical_values":
            reply = ph.LAB_VALUES_REFUSAL
            if self.scenario is not None:
                return self._turn(DmTurn(reply=reply,
                                         expecting="продолжение сценария"))
            self.scenario = MedicalBoundaryScenario()
            self.state = "active"
            return self._finish_turn(self.scenario.start(self.ctx))

        if hit.kind == "medical":
            if self.scenario is not None:
                # граничная фраза внутри сценария: отвечаем и продолжаем
                return self._turn(DmTurn(
                    reply=ph.MEDICAL_BOUNDARY,
                    expecting="продолжение сценария"))
            self.scenario = MedicalBoundaryScenario()
            self.state = "active"
            return self._finish_turn(self.scenario.start(self.ctx))

        if hit.kind == "organization":
            return self._finish_turn(StepResult(
                reply=ph.R14_TRANSFER + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.SENIOR, RouteRule.R14, Priority.P3,
                    "Звонок от организации"),
                done=True, result_code=ResultCode.ESCALATED))
        return None

    def _transfer_actions(self, target: Target, rule: RouteRule,
                          priority: Priority, summary: str) -> list[DmAction]:
        card = self.ctx.build_card(rule, priority, summary)
        if target_available(target.value, self.ctx.now):
            return [Transfer(target=target, rule=rule, priority=priority,
                             card=card)]
        # вне часов работы адресата — обратный звонок (7.6)
        within = 15 if priority == Priority.P2 else 30
        return [CallbackTask(card=card, within_minutes=within)]

    # ------------------------------------------------------------------
    # Сценарии

    def _start_scenario(self, frame, force_faq: bool = False) -> DmTurn:
        if force_faq:
            from .scenarios.info import FaqScenario
            scenario = FaqScenario(frame=frame)
        else:
            scenario = create_scenario(frame)
        if scenario is None:
            return self._ladder()
        self.scenario = scenario
        self.state = "active"
        return self._finish_turn(scenario.start(self.ctx))

    def _run(self, scenario: Scenario, frame, text: str) -> StepResult:
        try:
            res = scenario.handle_utterance(self.ctx, frame, text)
            self.ctx.mis_errors = 0
            return res
        except MisUnavailable:
            self.ctx.mis_errors += 1
            if self.ctx.mis_errors >= 2:
                return StepResult(
                    reply=ph.MIS_DOWN_TRANSFER + " " + ph.HANDOVER,
                    actions=self._transfer_actions(
                        Target.REGISTRAR, RouteRule.R06, Priority.P3,
                        "МИС не отвечает дважды"),
                    done=True, result_code=ResultCode.ESCALATED)
            return StepResult(
                reply=ph.MIS_WAIT + " " + ph.MIS_DOWN_RETRY + " " +
                self.ctx.last_assistant, mis_retry=True)

    def _finish_turn(self, res: StepResult) -> DmTurn:
        # цепочка передач другому сценарию (С-01→С-02, С-01→С-03, С-12→С-01…)
        hops = 0
        while res.handoff is not None and hops < 4:
            nxt = res.handoff
            self.scenario = nxt
            start_res = self._run_start(nxt)
            res = StepResult(
                reply=" ".join(x for x in (res.reply, start_res.reply) if x),
                actions=res.actions + start_res.actions,
                done=start_res.done,
                result_code=start_res.result_code or res.result_code,
                progress=start_res.progress)
            hops += 1

        if res.resume_previous and self.suspended is not None:
            self.scenario = self.suspended
            self.suspended = None
            res.done = False
            res.result_code = None
            self.state = "active"
        elif res.done:
            if res.result_code is not None:
                self.result_code = res.result_code
            # перевод звонка или завершение — ГА больше не ведёт диалог
            ended = any(isinstance(a, (EndCall, Transfer, CallbackTask))
                        for a in res.actions)
            self.scenario = None
            self.state = "closed" if ended else "wrap"
            # несколько интентов в одном звонке — по очереди (раздел 4)
            if self.state == "wrap" and self.pending_frames:
                nxt_frame = self.pending_frames.pop(0)
                res = self._chain_next(res, nxt_frame)

        if res.mis_retry:
            return self._turn(DmTurn(
                reply=res.reply,
                scenario=self.scenario.id if self.scenario else None,
                expecting="повтор реплики после сбоя МИС"))

        if not res.progress:
            return self._ladder()

        self.misses = 0
        return self._turn(DmTurn(
            reply=res.reply,
            actions=res.actions,
            scenario=self.scenario.id if self.scenario else None,
            result_code=self.result_code,
            expecting=res.expecting))

    def _chain_next(self, res: StepResult, frame) -> StepResult:
        scenario = create_scenario(frame)
        if scenario is None:
            return res
        self.scenario = scenario
        self.state = "active"
        start_res = self._run_start(scenario)
        return StepResult(
            reply=" ".join(x for x in (res.reply, start_res.reply) if x),
            actions=res.actions + start_res.actions,
            done=start_res.done, result_code=start_res.result_code,
            progress=start_res.progress)

    def _run_start(self, scenario: Scenario) -> StepResult:
        try:
            res = scenario.start(self.ctx)
            self.ctx.mis_errors = 0
            return res
        except MisUnavailable:
            self.ctx.mis_errors += 1
            if self.ctx.mis_errors >= 2:
                return StepResult(
                    reply=ph.MIS_DOWN_TRANSFER + " " + ph.HANDOVER,
                    actions=self._transfer_actions(
                        Target.REGISTRAR, RouteRule.R06, Priority.P3,
                        "МИС не отвечает дважды"),
                    done=True, result_code=ResultCode.ESCALATED)
            return StepResult(reply=ph.MIS_DOWN_RETRY, mis_retry=True)

    # ------------------------------------------------------------------
    # Общие ходы

    def _ladder(self) -> DmTurn:
        """Лестница непонимания: переспрос → оператор → автоперевод (R-05)."""
        self.misses += 1
        if self.misses >= 3:
            self.state = "closed"
            self.result_code = ResultCode.ESCALATED
            return self._turn(DmTurn(
                reply=ph.LADDER_TRANSFER + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.REGISTRAR, RouteRule.R05, Priority.P3,
                    "ГА не понимает пациента (3 раза подряд)"),
                result_code=ResultCode.ESCALATED))
        reply = ph.LADDER_OFFER if self.misses == 2 else ph.LADDER_REPHRASE
        return self._turn(DmTurn(reply=reply,
                                 scenario=self.scenario.id
                                 if self.scenario else None,
                                 expecting="повтор реплики"))

    def _farewell(self) -> DmTurn:
        self.state = "closed"
        return self._turn(DmTurn(
            reply=ph.FAREWELL, actions=[EndCall(reason="завершение")],
            result_code=self.result_code))

    def _turn(self, turn: DmTurn) -> DmTurn:
        # побочные эффекты на Заглушке МИС (в проде — исполнитель действий)
        for a in turn.actions:
            if isinstance(a, NotifySenior) and hasattr(self.ctx.mis,
                                                      "notify_senior"):
                self.ctx.mis.notify_senior(a.text)
            elif isinstance(a, SmsRequest) and hasattr(self.ctx.mis,
                                                       "send_sms"):
                self.ctx.mis.send_sms(a.phone, a.text)
        if turn.reply:
            self.ctx.last_assistant = turn.reply
            self.ctx.log("ga", turn.reply)
        return turn


def _frame_with_intent(frame, intent: Intent):
    """Копия фрейма с другим интентом (для очереди нескольких интентов)."""
    from dataclasses import replace
    if frame is None:
        return Frame(intent=intent)
    return replace(frame, intent=intent, extra_intents=())
