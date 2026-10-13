# -*- coding: utf-8 -*-
"""Реестр сценариев: интент → сценарий (маршрутизация по каталогу 4)."""
from __future__ import annotations

from ..schema import Frame, Intent
from .base import Scenario
from .booking import BookingScenario, ReferralBookingScenario
from .info import FaqScenario, ScheduleScenario
from .outbound import ReminderCallScenario
from .records import CancelScenario, CheckBookingScenario, RescheduleScenario
from .special import (ComplaintScenario, EmergencyScenario,
                      MedicalBoundaryScenario, ThirdPartyScenario)
from .transferred import TransferredScenario

__all__ = [
    "BookingScenario", "ReferralBookingScenario", "CancelScenario",
    "CheckBookingScenario", "RescheduleScenario", "ScheduleScenario",
    "FaqScenario", "ReminderCallScenario", "ThirdPartyScenario",
    "EmergencyScenario", "ComplaintScenario", "MedicalBoundaryScenario",
    "TransferredScenario", "create_scenario",
]


def create_scenario(frame: Frame) -> Scenario | None:
    """Интент фрейма → стартовый сценарий. None — интент не определён."""
    intent = frame.intent
    if intent is None:
        return None
    if intent == Intent.BOOK:
        if frame.for_child or frame.for_other_person:
            return ThirdPartyScenario(BookingScenario(frame=frame))
        return BookingScenario(frame=frame)
    if intent == Intent.RESCHEDULE:
        return RescheduleScenario(frame=frame)
    if intent == Intent.CANCEL:
        return CancelScenario(frame=frame)
    if intent == Intent.CHECK_BOOKING:
        return CheckBookingScenario(frame=frame)
    if intent == Intent.SCHEDULE:
        return ScheduleScenario(frame=frame)
    if intent == Intent.DOCS:
        return FaqScenario(frame=frame)
    if intent == Intent.MEDICAL:
        return MedicalBoundaryScenario()
    if intent == Intent.EMERGENCY:
        return EmergencyScenario()
    if intent == Intent.COMPLAINT:
        return ComplaintScenario()
    # этап-2 интенты и прочее → перевод с карточкой
    return TransferredScenario(intent)
