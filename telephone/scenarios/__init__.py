# -*- coding: utf-8 -*-
"""Сценарии менеджера диалога: booking (запись), appointments (перенос/
отмена/проверка), info (график + FAQ), reminder (исходящее напоминание).
Общие компоненты (идентификация, выбор слотов) — в common.py."""
from .appointments import AppointmentsFlow
from .booking import BookingFlow
from .common import (DialogContext, IdentificationFlow, Scenario,
                     SlotChooser, StepResult)
from .info import InfoFlow
from .reminder import ReminderFlow

__all__ = ["AppointmentsFlow", "BookingFlow", "DialogContext",
           "IdentificationFlow", "InfoFlow", "ReminderFlow", "Scenario",
           "SlotChooser", "StepResult"]
