# -*- coding: utf-8 -*-
"""Этап-2 интенты (С-06, С-08, С-09, С-11, С-13): распознаём, собираем
суть в карточку перевода и передаём сотруднику по матрице 7.3.
На этапе 2 пилота эти сценарии получат собственные FSM."""
from __future__ import annotations

from .. import phrases as ph
from ..schema import (Intent, Priority, ResultCode, RouteRule, ScenarioId,
                      Target, Transfer)
from .base import DialogContext, Scenario, StepResult

_DEFAULT_TARGETS: dict[str, tuple[Target, RouteRule, Priority, str]] = {
    # intent → (адресат, правило, приоритет, резюме)
    "I-05": (Target.REGISTRAR, RouteRule.R08, Priority.P3,
             "Заявка на вызов врача на дом (сценарий С-06, этап 2)"),
    "I-07": (Target.REGISTRAR, RouteRule.R08, Priority.P3,
             "Готовность анализов (сценарий С-08, этап 2)"),
    "I-08": (Target.PAID, RouteRule.R10, Priority.P3,
             "Платные услуги (сценарий С-09, этап 2)"),
    "I-09": (Target.REGISTRAR, RouteRule.R08, Priority.P3,
             "Лист ожидания (сценарий С-11, этап 2)"),
    "I-10": (Target.REGISTRAR, RouteRule.R08, Priority.P3,
             "Прикрепление/документы (сценарий С-13, этап 2)"),
    "I-14": (Target.REGISTRAR, RouteRule.R08, Priority.P3,
             "Запрос вне каталога интентов"),
}


class TransferredScenario(Scenario):
    """Подтверждение обращения и перевод с карточкой (7.7)."""

    id = ScenarioId.C07

    def __init__(self, intent: Intent, target: Target | None = None,
                 rule: RouteRule | None = None,
                 priority: Priority | None = None,
                 reply: str | None = None, summary: str | None = None):
        self.intent = intent
        t, r, p, default_summary = _DEFAULT_TARGETS.get(
            intent.value, (Target.REGISTRAR, RouteRule.R08, Priority.P3,
                           "Обращение передано сотруднику"))
        self.target = target or t
        self.rule = rule or r
        self.priority = priority or p
        self.custom_reply = reply
        self.summary = summary or default_summary
        self.intent_label = f"{intent.value} Перевод"

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", self.intent.value)
        reply = self.custom_reply or ph.TRANSFERRED_ACK.get(
            self.intent.value, "Хорошо. " + ph.R08_TRANSFER)
        if not self.custom_reply:
            reply += " " + ph.HANDOVER
        return StepResult(
            reply=reply,
            actions=[Transfer(
                target=self.target, rule=self.rule, priority=self.priority,
                card=ctx.build_card(self.rule, self.priority,
                                    self.summary + "; " + ctx.last_user[:100],
                                    f"{self.intent.value} Перевод"))],
            done=True, result_code=ResultCode.ESCALATED)
