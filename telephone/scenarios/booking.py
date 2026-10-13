# -*- coding: utf-8 -*-
"""С-01 Запись на приём (прямая запись) и С-02 Запись по направлению."""
from __future__ import annotations

from datetime import date, timedelta

from .. import phrases as ph
from .. import textutils as tu
from ..clinic_data import SPECIALTIES
from ..mis import SlotTaken
from ..schema import (Intent, Priority, ResultCode, RouteRule, ScenarioId,
                      Target, Transfer)
from .base import (DialogContext, IdentificationFlow, Scenario, SlotChooser,
                   StepResult, booking_sms, options_phrase, waitlist_transfer)


def _under_18(dob: date, today: date) -> bool:
    try:
        cutoff = date(today.year - 18, today.month, today.day)
    except ValueError:                       # 29 февраля
        cutoff = date(today.year - 18, today.month, 28)
    return dob > cutoff


class BookingScenario(Scenario):
    """С-01: идентификация → проверки → слоты → подтверждение → запись.

    Ветвления (5.С-01): не найден (в каскаде), не прикреплён, нет слота
    в пределах 24 часов у участкового, не свой участок, младше 18 (С-12),
    дубль записи (С-03), слот заняли, специальность по направлению (С-02).
    """

    id = ScenarioId.C01
    intent_label = "I-01 Запись на приём"

    def __init__(self, frame=None, specialty: str | None = None,
                 doctor: str | None = None):
        self.specialty = specialty or (frame.specialty if frame else None)
        self.doctor = doctor or (frame.doctor if frame else None)
        self.prefs = frame.time_prefs if frame else None
        self.ident = IdentificationFlow()
        self.chooser: SlotChooser | None = None
        self.state = "start"          # ask_specialty/identify/…/offer/confirm
        self.pending_booking = None   # дубль записи (предложение С-03)
        self.gp_for_referral = False
        self.last_frame = None

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("интент", "запись на приём")
        if self.specialty:
            ctx.collect("специальность", tu.specialty_nom(self.specialty))
            if (SPECIALTIES[self.specialty]["referral"]
                    and not isinstance(self, ReferralBookingScenario)):
                return StepResult(handoff=ReferralBookingScenario(
                    specialty=self.specialty, doctor=self.doctor,
                    prefs=self.prefs))
        if ctx.patient is not None:      # пациент уже идентифицирован (С-12)
            return self._post_ident(ctx)
        if not self.specialty:
            self.state = "ask_specialty"
            return StepResult(reply=ph.ASK_SPECIALTY, expecting="специальность")
        self.state = "identify"
        return self.ident.start(ctx)

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        self.last_frame = frame
        if self.state == "ask_specialty":
            if frame.specialty:
                self.specialty = frame.specialty
                ctx.collect("специальность", tu.specialty_nom(self.specialty))
                if (SPECIALTIES[self.specialty]["referral"]
                        and not isinstance(self, ReferralBookingScenario)):
                    return StepResult(handoff=ReferralBookingScenario(
                        specialty=self.specialty, prefs=self.prefs))
                if ctx.patient is None:
                    self.state = "identify"
                    return self.ident.start(ctx)
                return self._post_ident(ctx)
            if frame.doctor:
                self.doctor = frame.doctor
                ctx.collect("врач", frame.doctor)
                if ctx.patient is None:
                    self.state = "identify"
                    return self.ident.start(ctx)
                return self._post_ident(ctx)
            return StepResult(reply=ph.ASK_SPECIALTY, progress=False,
                              expecting="специальность")

        if self.state == "identify":
            res = self.ident.handle(ctx, frame, text)
            if res is not None:
                return res
            return self._post_ident(ctx)

        if self.state == "attach_question":
            if frame.intent == Intent.DOCS:
                from .transferred import TransferredScenario
                return StepResult(handoff=TransferredScenario(
                    Intent.DOCS, reply=(
                        "Прикрепиться можно двумя способами: подать заявление "
                        "на Госуслугах или прийти в регистратуру, окно номер "
                        "один. Нужны паспорт, полис ОМС и СНИЛС. Обычно это "
                        "занимает до трёх рабочих дней.")))
            if frame.intent == Intent.PAID:
                from .transferred import TransferredScenario
                return StepResult(handoff=TransferredScenario(Intent.PAID))
            if frame.answer is False:
                return StepResult(reply=ph.ANYTHING_ELSE, done=True)
            return StepResult(reply="Рассказать, как прикрепиться, или "
                                    "записать на платный приём?",
                              progress=False)

        if self.state == "dup":
            if frame.answer is True:
                from .records import RescheduleScenario
                return StepResult(handoff=RescheduleScenario(
                    target_booking=self.pending_booking))
            if frame.answer is False:
                return self._offer(ctx)
            return StepResult(reply=ph.DUP_BOOKING.format(
                specialty=tu.specialty_nom(self.specialty), date=""),
                progress=False)

        if self.state == "offer":
            status, payload = self.chooser.choose(ctx, frame, text)
            if status == "picked":
                return self._confirm(ctx)
            if status == "again":
                return payload
            if status == "none":
                return payload
            return StepResult(reply=ph.OFFER_SLOTS.format(options=options_phrase(
                self.chooser.options, ctx.today)), progress=False)

        if self.state == "confirm":
            if frame.answer is True:
                return self._book(ctx)
            if frame.answer is False:
                return self._offer(ctx)
            return StepResult(reply="Скажите «да» или «нет» — или назовите "
                                    "другое время.", progress=False)
        return StepResult(progress=False)

    # -- шаги --

    def _post_ident(self, ctx: DialogContext) -> StepResult:
        p = ctx.patient
        thanks = ph.THANKS_NAME.format(name=p.fio.name_patronymic()) + " "
        if not p.attached:
            self.state = "attach_question"
            return StepResult(reply=thanks + ph.NOT_ATTACHED +
                              " Рассказать, как прикрепиться, или записать "
                              "на платный приём?")
        if p.pediatric or _under_18(p.dob, ctx.today):
            if ctx.caller_relation is None:
                # пациент младше 18 → С-12 (звонит представитель)
                from .special import ThirdPartyScenario
                return StepResult(handoff=ThirdPartyScenario(BookingScenario(
                    specialty=self.specialty, doctor=self.doctor)))
        dup = [b for b in ctx.mis.active_bookings(p.id)
               if b.specialty == self.specialty]
        if dup and not self.gp_for_referral:
            self.pending_booking = dup[0]
            self.state = "dup"
            return StepResult(reply=thanks + ph.DUP_BOOKING.format(
                specialty=tu.specialty_nom(self.specialty),
                date=tu.date_words(dup[0].date, with_weekday=False)))
        return self._offer(ctx, prefix=thanks)
    def _offer(self, ctx: DialogContext, prefix: str = "") -> StepResult:
        doctor = self.doctor
        if self.specialty == "gp" and ctx.patient and ctx.patient.gp_doctor:
            doctor = doctor or ctx.patient.gp_doctor
            if ctx.patient.gp_doctor and not self._gp_has_urgent_slot(ctx):
                # нет слота в пределах 24 часов — дежурный/другой терапевт
                duty = ctx.mis.free_slots(
                    "gp", date_from=ctx.today,
                    date_to=ctx.today + timedelta(days=1), today=ctx.today)
                duty = [s for s in duty if s.doctor != ctx.patient.gp_doctor]
                if duty:
                    self.chooser = SlotChooser("gp")
                    self.chooser.all_slots = duty
                    self.chooser.options = duty[:3]
                    self.chooser.offset = 3
                    self.state = "offer"
                    return StepResult(
                        reply=prefix + ph.NO_URGENT_GP_SLOT + " " +
                        ph.OFFER_SLOTS.format(
                            options=options_phrase(duty[:3], ctx.today)))
                summary = "Нет слотов к участковому терапевту в пределах 24 часов"
                return StepResult(reply=prefix + ph.NO_SLOTS,
                                  actions=waitlist_transfer(ctx, summary),
                                  done=True, result_code=ResultCode.ESCALATED)
        self.chooser = SlotChooser(
            self.specialty, doctor=doctor,
            daypart=self.prefs.daypart if self.prefs else None,
            fixed_date=self.prefs.date if self.prefs else None,
            exclude_weekday=self.prefs.exclude_weekday
            if self.prefs else None)
        note = ""
        if self.specialty == "gp" and doctor:
            note = f"Ваш участковый терапевт — {doctor}. "
        self.state = "offer"
        return self.chooser.offer(ctx, prefix=prefix + note,
                                  relaxed_note="Именно в это время свободного "
                                               "нет. Ближайшее:")

    def _gp_has_urgent_slot(self, ctx: DialogContext) -> bool:
        slots = ctx.mis.free_slots(
            "gp", doctor=ctx.patient.gp_doctor, date_from=ctx.today,
            date_to=ctx.today + timedelta(days=1), today=ctx.today)
        return bool(slots)

    def _confirm(self, ctx: DialogContext) -> StepResult:
        slot = self.chooser.chosen
        self.state = "confirm"
        return StepResult(reply=ph.CONFIRM_BOOKING.format(
            specialty=tu.specialty_nom(self.specialty), doctor=slot.doctor,
            date=tu.date_words(slot.date), time=tu.time_words(slot.time),
            cabinet=tu.number_words(slot.cabinet)),
            expecting="подтверждение да/нет")

    def _book(self, ctx: DialogContext) -> StepResult:
        slot = self.chooser.chosen
        try:
            appt = ctx.mis.create_booking(ctx.patient.id, slot)
        except SlotTaken:
            rest = [s for s in self.chooser.all_slots
                    if s.key() != slot.key()][:3]
            if not rest:
                return StepResult(
                    reply="Извините, это время только что заняли. " + ph.NO_SLOTS,
                    actions=waitlist_transfer(
                        ctx, "Слот занят при подтверждении: " + self.specialty),
                    done=True, result_code=ResultCode.ESCALATED)
            self.chooser.options = rest
            self.state = "offer"
            return StepResult(reply=ph.SLOT_TAKEN.format(
                options=options_phrase(rest, ctx.today)))
        code = ResultCode.BOOKED_GP if self.gp_for_referral else ResultCode.BOOKED
        return StepResult(reply=ph.BOOKED_DONE,
                          actions=[booking_sms(ctx, appt, "booked")],
                          done=True, result_code=code)


class ReferralBookingScenario(BookingScenario):
    """С-02: запись к специалисту по направлению (кардиолог, невролог…)."""

    id = ScenarioId.C02
    intent_label = "I-01 Запись по направлению"

    def __init__(self, specialty: str | None = None, doctor: str | None = None,
                 prefs=None, frame=None):
        super().__init__(frame=frame, specialty=specialty, doctor=doctor)
        self.prefs = prefs or self.prefs
        self.referral = None
        self.asked_gp = False

    def _post_ident(self, ctx: DialogContext) -> StepResult:
        p = ctx.patient
        thanks = ph.THANKS_NAME.format(name=p.fio.name_patronymic()) + " "
        paper = self.last_frame is not None and self.last_frame.paper_referral
        if paper:
            return StepResult(
                reply=ph.PAPER_REFERRAL,
                actions=[Transfer(
                    target=Target.REGISTRAR, rule=RouteRule.R08,
                    priority=Priority.P3,
                    card=ctx.build_card(
                        RouteRule.R08, Priority.P3,
                        "Бумажное направление из другой МО, требует ручной "
                        "регистрации", self.intent_label))],
                done=True, result_code=ResultCode.ESCALATED)
        ref = ctx.mis.find_referral(p.id, self.specialty)
        if ref is None:
            self.state = "ask_gp"
            return StepResult(reply=thanks + ph.NO_REFERRAL.format(
                specialty=tu.specialty_dat(self.specialty)))
        self.referral = ref
        thanks = (f"{p.fio.name_patronymic()}, вижу ваше направление к "
                  f"{tu.specialty_dat(self.specialty)}. ")
        date_to = ref.issued + timedelta(days=ref.valid_days)
        if ref.cito:
            # отметку не озвучиваем (С-02): попадаем к врачу в ≤3 рабочих дня
            date_to = ctx.today + timedelta(days=3)
            slots = ctx.mis.free_slots(self.specialty, date_from=ctx.today,
                                       date_to=date_to, today=ctx.today)
            if not slots:
                return StepResult(
                    reply="По вашему направлению нужно попасть к врачу "
                          "в ближайшие дни. Соединяю с администратором — "
                          "он организует приём быстро.",
                    actions=[Transfer(
                        target=Target.SENIOR, rule=RouteRule.R08,
                        priority=Priority.P2,
                        card=ctx.build_card(
                            RouteRule.R08, Priority.P2,
                            "Направление с пометкой срочности, слотов нет",
                            self.intent_label))],
                    done=True, result_code=ResultCode.ESCALATED)
        self.chooser = SlotChooser(
            self.specialty, doctor=self.doctor,
            daypart=self.prefs.daypart if self.prefs else None,
            fixed_date=self.prefs.date if self.prefs else None,
            date_to=date_to)
        self.state = "offer"
        return self.chooser.offer(ctx, prefix=thanks)

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        self.last_frame = frame
        if self.state == "ask_gp":
            if frame.paper_referral:
                return StepResult(
                    reply=ph.PAPER_REFERRAL,
                    actions=[Transfer(
                        target=Target.REGISTRAR, rule=RouteRule.R08,
                        priority=Priority.P3,
                        card=ctx.build_card(
                            RouteRule.R08, Priority.P3,
                            "Бумажное направление из другой МО",
                            self.intent_label))],
                    done=True, result_code=ResultCode.ESCALATED)
            if frame.answer is True:
                self.specialty = "gp"
                self.gp_for_referral = True
                ctx.collect("специальность", "терапевт (за направлением)")
                return self._offer(ctx)
            if frame.answer is False:
                return StepResult(reply=ph.ANYTHING_ELSE, done=True)
            return StepResult(reply=ph.NO_REFERRAL.format(
                specialty=tu.specialty_dat(self.specialty)), progress=False)
        return super().handle_utterance(ctx, frame, text)

    def _book(self, ctx: DialogContext) -> StepResult:
        slot = self.chooser.chosen
        try:
            appt = ctx.mis.create_booking(ctx.patient.id, slot,
                                          referral_id=self.referral.id
                                          if self.referral else None)
        except SlotTaken:
            return super()._book(ctx)
        code = ResultCode.BOOKED_GP if self.gp_for_referral \
            else ResultCode.BOOKED
        return StepResult(
            reply="Готово. Возьмите паспорт, полис и результаты обследований, "
                  "если они есть на руках. " + ph.ANYTHING_ELSE_SHORT,
            actions=[booking_sms(ctx, appt, "booked")],
            done=True, result_code=code)
