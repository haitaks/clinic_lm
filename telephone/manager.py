# -*- coding: utf-8 -*-
"""Менеджер диалога телефонии: маршрутизатор реплик по интентам.

Поток хода:
1. приветствие с уведомлением о записи разговора (3.3–3.4);
2. сквозной слой безопасности на каждой реплике (safety.py) — красные
   флаги прерывают любой сценарий;
3. активный сценарий получает фрейм NLU и реплику целиком;
4. свободное состояние — диспетчер интентов (таблица ROUTING ниже);
5. лестница непонимания (3.2): переспрос → предложение оператора →
   автоперевод.

Менеджер не знает, кто разобрал реплику (словарная заглушка, теггер или
LLM) — только контракт NluClient.parse. Какого врача предложить — решает
таблица 2.3 (clinic_data), не модель.
"""
from __future__ import annotations

from datetime import datetime

from . import phrases as ph
from . import safety
from .clinic_data import target_available
from .mis import MisUnavailable
from .schema import (COMPLAINT, DROPPED, EMERGENCY, EndCall, Frame, Intent,
                     MisPatient, NotifySenior, SmsRequest, Target, Transfer,
                     VoicemailEvent, Appointment, CallbackTask, DmAction,
                     DmTurn)
from .scenarios.appointments import AppointmentsFlow
from .scenarios.booking import BookingFlow
from .scenarios.common import DialogContext, Scenario, StepResult
from .scenarios.info import InfoFlow
from .scenarios.reminder import ReminderFlow
from .safety import EmergencyFlow, MedicalFlow

# Интенты, закрываемые простым переводом: интент → (адресат, тема карточки)
SIMPLE_TRANSFERS: dict[Intent, tuple[Target, str]] = {
    Intent.HOME_VISIT: (Target.OPERATOR, "вызов врача на дом"),
    Intent.LAB_STATUS: (Target.OPERATOR, "готовность анализов"),
    Intent.WAITLIST: (Target.OPERATOR, "лист ожидания"),
    Intent.PAID: (Target.PAID, "платные услуги"),
    Intent.OTHER: (Target.OPERATOR, "запрос вне каталога"),
}

# Интенты, запускающие сценарий
def _scenario_for(frame: Frame) -> Scenario | None:
    if frame.intent == Intent.BOOK:
        return BookingFlow(frame=frame)
    if frame.intent == Intent.RESCHEDULE:
        return AppointmentsFlow(frame=frame, action="reschedule")
    if frame.intent == Intent.CANCEL:
        return AppointmentsFlow(frame=frame, action="cancel")
    if frame.intent == Intent.CHECK_BOOKING:
        return AppointmentsFlow(frame=frame, action="check_booking")
    if frame.intent in (Intent.SCHEDULE, Intent.DOCS):
        return InfoFlow(frame=frame)
    if frame.intent == Intent.MEDICAL:
        return MedicalFlow()
    return None


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
        self.suspended: Scenario | None = None      # прерван экстренным
        self.pending_frames: list = []              # очередь интентов (раздел 4)
        self.state = "greeted"      # greeted / active / wrap / closed
        self.status: str | None = None              # итог звонка
        self.misses = 0
        self.silence_streak = 0
        self.outbound = outbound

    # ------------------------------------------------------------------
    # Жизненный цикл звонка

    def start_call(self) -> DmTurn:
        if self.outbound is not None:
            patient, appt = self.outbound
            self.scenario = ReminderFlow(patient, appt)
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

        hit = safety.check_utterance(text)
        if hit is not None:
            turn = self._apply_safety(hit)
            if turn is not None:
                return turn

        if self.scenario is not None:
            frame = self.nlu.parse(text, self.ctx.today,
                                   options=self.scenario.current_options())
            return self._finish_turn(
                self._run(self.scenario, frame, text))

        frame = self.nlu.parse(text, self.ctx.today)
        if self.state == "greeted" and frame.answer is False \
                and frame.intent is None:
            return self._farewell()          # «нет, спасибо» на приветствие
        if self.state == "wrap":
            if frame.answer is False and frame.intent is None:
                return self._farewell()
            if frame.answer is True and frame.intent is None:
                return self._turn(DmTurn(reply=ph.SURE_WHAT,
                                         expecting="интент"))
        if frame.intent is not None:
            if frame.extra_frames:
                self.pending_frames.extend(frame.extra_frames)
            return self._dispatch(frame)
        # интент не распознан, но модель узнала вопрос из справочника FAQ
        if frame.faq_id is not None:
            return self._dispatch(Frame(raw=text, faq_id=frame.faq_id,
                                        specialty=frame.specialty,
                                        doctor=frame.doctor))
        return self._ladder()

    def handle_event(self, ev) -> DmTurn:
        """События телефонии: тишина, DTMF, автоответчик."""
        if self.state == "closed":
            return self._turn(DmTurn(reply=""))
        if isinstance(ev, VoicemailEvent) and isinstance(
                self.scenario, ReminderFlow):
            res = self.scenario.handle_event(self.ctx, ev)
            if res is not None:
                return self._finish_turn(res)

        if hasattr(ev, "digits"):
            if self.scenario is not None:
                res = self.scenario.handle_event(self.ctx, ev)
                if res is not None:
                    return self._finish_turn(res)
            return self._turn(DmTurn(reply=""))

        # тишина ×2 — завершение
        self.silence_streak += 1
        if self.silence_streak == 1:
            return self._turn(DmTurn(reply=ph.SILENCE_1))
        self.status = self.status or DROPPED
        self.state = "closed"
        return self._turn(DmTurn(
            reply=ph.SILENCE_2, actions=[EndCall(reason="тишина")],
            status=self.status))

    def summary(self) -> dict:
        """Итог звонка: статус, собранные данные, журнал."""
        return {"call_id": self.ctx.call_id,
                "status": self.status,
                "state": self.state,
                "patient": self.ctx.patient_label(),
                "collected": dict(self.ctx.collected),
                "recording_consent": self.ctx.recording_consent,
                "journal": list(self.ctx.journal)}

    # ------------------------------------------------------------------
    # Диспетчер интентов

    def _dispatch(self, frame: Frame) -> DmTurn:
        intent = frame.intent
        if intent in SIMPLE_TRANSFERS:
            target, topic = SIMPLE_TRANSFERS[intent]
            ack = ph.TRANSFER_ACK.get(intent.value, ph.TRANSFER_OPERATOR)
            return self._finish_turn(StepResult(
                label="transfer",
                reply=ack + " " + ph.HANDOVER,
                actions=self._transfer_actions(target, 3,
                                               topic + "; " + frame.raw[:100],
                                               topic),
                done=True, status="transferred"))
        if intent == Intent.COMPLAINT:
            self.ctx.collect("недовольство", "да")
            return self._finish_turn(StepResult(
                label="complaint",
                reply=ph.COMPLAINT_TRANSFER + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.SENIOR, 2, "Жалоба: " + frame.raw[:120], "жалоба"),
                done=True, status=COMPLAINT))
        scenario = _scenario_for(frame)
        if scenario is None:
            return self._ladder()
        self.scenario = scenario
        self.state = "active"
        return self._finish_turn(scenario.start(self.ctx))

    # ------------------------------------------------------------------
    # Сквозной слой безопасности

    def _apply_safety(self, hit: safety.SafetyHit) -> DmTurn | None:
        if hit.kind == "red_a":
            self.ctx.collect("красный флаг", hit.phrase)
            if isinstance(self.scenario, EmergencyFlow):
                # красный флаг звучит снова — немедленная передача
                return self._finish_turn(StepResult(
                    label="emergency",
                    reply=ph.EMERGENCY_TRANSFER,
                    actions=self._transfer_actions(
                        Target.OPERATOR, 1,
                        "Повторный красный флаг: " + hit.phrase,
                        "экстренная ситуация"),
                    done=True, status=EMERGENCY))
            if self.scenario is not None:
                self.suspended = self.scenario
            self.scenario = EmergencyFlow()
            self.state = "active"
            # инцидент фиксируется сразу, даже если окажется ложной тревогой
            self.status = EMERGENCY
            return self._finish_turn(self.scenario.start(self.ctx))

        if hit.kind == "red_b":
            self.ctx.collect("красный флаг Б", hit.phrase)
            return self._finish_turn(StepResult(
                label="safety",
                reply=ph.URGENT_TRANSFER + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.OPERATOR, 1,
                    "Острое состояние: " + hit.phrase, "острое состояние"),
                done=True, status="transferred"))

        if hit.kind == "operator":
            return self._finish_turn(StepResult(
                label="safety",
                reply=ph.TRANSFER_OPERATOR + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.OPERATOR, 2, "Пациент попросил сотрудника",
                    "перевод на сотрудника"),
                done=True, status="transferred"))

        if hit.kind == "complaint":
            self.ctx.collect("недовольство", "да")
            return self._finish_turn(StepResult(
                label="complaint",
                reply=ph.COMPLAINT_TRANSFER + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.SENIOR, 2, "Жалоба: " + self.ctx.last_user[:120],
                    "жалоба"),
                done=True, status=COMPLAINT))

        if hit.kind == "gratitude":
            return self._finish_turn(StepResult(
                label="safety", reply=ph.GRATITUDE, done=True, status="info"))

        if hit.kind == "complaint_written":
            return self._finish_turn(StepResult(
                label="complaint", reply=ph.COMPLAINT_WRITTEN, done=True,
                status=COMPLAINT))

        if hit.kind == "robot":
            return self._turn(DmTurn(reply=ph.ROBOT_ANSWER,
                                     expecting="ответ на вопрос ГА"))

        if hit.kind == "recording_objection":
            self.ctx.recording_consent = False
            tail = (" " + self.ctx.last_assistant) if self.scenario \
                else " " + ph.ANYTHING_ELSE
            return self._turn(DmTurn(reply=ph.RECORDING_OFF + tail))

        if hit.kind == "tooth":
            return self._finish_turn(StepResult(
                label="safety",
                reply=ph.TRANSFER_DENTISTRY + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.DENTISTRY, 2,
                    "Острая зубная боль: " + hit.phrase, "острая зубная боль"),
                done=True, status="transferred"))

        if hit.kind == "medical_values":
            reply = ph.LAB_VALUES_REFUSAL
            if self.scenario is not None:
                return self._turn(DmTurn(reply=reply,
                                         expecting="продолжение сценария"))
            self.scenario = MedicalFlow(reply)
            self.state = "active"
            return self._finish_turn(self.scenario.start(self.ctx))

        if hit.kind == "medical":
            if self.scenario is not None:
                # граничная фраза внутри сценария: отвечаем и продолжаем
                return self._turn(DmTurn(reply=ph.MEDICAL_BOUNDARY,
                                         expecting="продолжение сценария"))
            self.scenario = MedicalFlow()
            self.state = "active"
            return self._finish_turn(self.scenario.start(self.ctx))
        return None

    def _transfer_actions(self, target: Target, priority: int, summary: str,
                          topic: str) -> list[DmAction]:
        card = self.ctx.card(summary, topic)
        if target_available(target.value, self.ctx.now):
            return [Transfer(target=target, priority=priority, card=card)]
        # вне часов работы адресата — обратный звонок
        within = 15 if priority == 2 else 30
        return [CallbackTask(card=card, within_minutes=within)]

    # ------------------------------------------------------------------
    # Прогон сценария

    def _start_scenario(self, scenario: Scenario) -> DmTurn:
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
                    label=scenario.label,
                    reply=ph.MIS_DOWN_TRANSFER + " " + ph.HANDOVER,
                    actions=self._transfer_actions(
                        Target.OPERATOR, 3, "МИС не отвечает дважды",
                        "сбой МИС"),
                    done=True, status="transferred")
            return StepResult(
                label=scenario.label,
                reply=ph.MIS_WAIT + " " + ph.MIS_DOWN_RETRY + " " +
                self.ctx.last_assistant, mis_retry=True)

    def _finish_turn(self, res: StepResult) -> DmTurn:
        # цепочка передач другому сценарию (справка → запись и т.п.)
        hops = 0
        while res.handoff is not None and hops < 4:
            nxt = res.handoff
            self.scenario = nxt
            start_res = self._run_start(nxt)
            res = StepResult(
                label=nxt.label,
                reply=" ".join(x for x in (res.reply, start_res.reply) if x),
                actions=res.actions + start_res.actions,
                done=start_res.done,
                status=start_res.status or res.status,
                progress=start_res.progress)
            hops += 1

        if res.resume_previous and self.suspended is not None:
            self.scenario = self.suspended
            self.suspended = None
            res.done = False
            res.status = None
            self.state = "active"
        elif res.done:
            if res.status is not None:
                self.status = res.status
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
                scenario=res.label or (self.scenario.label
                                       if self.scenario else ""),
                expecting="повтор реплики после сбоя МИС"))

        if not res.progress:
            return self._ladder()

        self.misses = 0
        return self._turn(DmTurn(
            reply=res.reply,
            actions=res.actions,
            scenario=res.label or (self.scenario.label
                                   if self.scenario else ""),
            status=self.status,
            expecting=res.expecting))

    def _chain_next(self, res: StepResult, frame) -> StepResult:
        scenario = _scenario_for(frame)
        if scenario is None and frame.intent in SIMPLE_TRANSFERS:
            target, topic = SIMPLE_TRANSFERS[frame.intent]
            return StepResult(
                label="transfer",
                reply=" ".join(x for x in (
                    res.reply,
                    ph.TRANSFER_ACK.get(frame.intent.value,
                                        ph.TRANSFER_OPERATOR) + " "
                    + ph.HANDOVER) if x),
                actions=res.actions + self._transfer_actions(
                    target, 3, topic + "; " + frame.raw[:100], topic),
                done=True, status="transferred")
        if scenario is None:
            return res
        self.scenario = scenario
        self.state = "active"
        start_res = self._run_start(scenario)
        return StepResult(
            label=scenario.label,
            reply=" ".join(x for x in (res.reply, start_res.reply) if x),
            actions=res.actions + start_res.actions,
            done=start_res.done, status=start_res.status,
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
                    label=scenario.label,
                    reply=ph.MIS_DOWN_TRANSFER + " " + ph.HANDOVER,
                    actions=self._transfer_actions(
                        Target.OPERATOR, 3, "МИС не отвечает дважды",
                        "сбой МИС"),
                    done=True, status="transferred")
            return StepResult(label=scenario.label,
                              reply=ph.MIS_DOWN_RETRY, mis_retry=True)

    # ------------------------------------------------------------------
    # Общие ходы

    def _ladder(self) -> DmTurn:
        """Лестница непонимания: переспрос → оператор → автоперевод."""
        self.misses += 1
        if self.misses >= 3:
            self.state = "closed"
            self.status = "transferred"
            return self._turn(DmTurn(
                reply=ph.LADDER_TRANSFER + " " + ph.HANDOVER,
                actions=self._transfer_actions(
                    Target.OPERATOR, 3, "ГА не понимает пациента "
                    "(3 раза подряд)", "непонимание"),
                status="transferred"))
        reply = ph.LADDER_OFFER if self.misses == 2 else ph.LADDER_REPHRASE
        return self._turn(DmTurn(reply=reply,
                                 scenario=self.scenario.label
                                 if self.scenario else "",
                                 expecting="повтор реплики"))

    def _farewell(self) -> DmTurn:
        self.state = "closed"
        return self._turn(DmTurn(
            reply=ph.FAREWELL, actions=[EndCall(reason="завершение")],
            status=self.status))

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
