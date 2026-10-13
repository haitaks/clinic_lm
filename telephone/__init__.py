# -*- coding: utf-8 -*-
"""Менеджер диалога телефонии регистратуры (регламент РГ-ТЕЛ-01).

Быстрый старт::

    from datetime import datetime
    from telephone import DialogManager, RuleNlu, StubMis

    dm = DialogManager(nlu=RuleNlu(), mis=StubMis(),
                       now=datetime(2026, 9, 30, 12, 0),
                       ani="+79001234567")
    turn = dm.start_call()
    turn = dm.handle_utterance("Мне бы записаться к терапевту")
    print(turn.reply)          # реплика ГА
    print(turn.actions)        # переводы, SMS, завершение
"""
from .manager import DialogManager
from .mis import MisUnavailable, SlotTaken, StubMis
from .nlu import FixtureNlu, FrameModelNlu, LlmNlu, RuleNlu
from .schema import DmTurn, Frame, Intent, ResultCode

__all__ = ["DialogManager", "RuleNlu", "FixtureNlu", "FrameModelNlu",
           "LlmNlu", "StubMis", "MisUnavailable", "SlotTaken", "DmTurn",
           "Frame", "Intent", "ResultCode"]
