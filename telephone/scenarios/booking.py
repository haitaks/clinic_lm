# -*- coding: utf-8 -*-
"""Запись на приём: прямая (терапевт, хирург, ЛОР…) и по направлению
(кардиолог, невролог…). Порядок записи решает таблица 2.3 (clinic_data),
не модель. Ребёнок 0–17 — детский вариант специальности + вопрос «кем
приходитесь» (звонит представитель)."""
from __future__ import annotations

from datetime import date, timedelta

from .. import phrases as ph
from .. import textutils as tu
from ..clinic_data import SPECIALTIES, child_specialty, default_specialty
from ..mis import SlotTaken
from ..schema import BOOKED, BOOKED_GP, Intent, Target, Transfer
from .common import (DialogContext, IdentificationFlow, Scenario,
                     SlotChooser, StepResult, booking_sms, options_phrase,
                     waitlist_transfer)


def _under_18(dob: date, today: date) -> bool:
    try:
        cutoff = date(today.year - 18, today.month, today.day)
    except ValueError:                       # 29 февраля
        cutoff = date(today.year - 18, today.month, 28)
    return dob > cutoff


class BookingFlow(Scenario):
    """Идентификация → проверки (прикреплён, ребёнок, дубль, направление)
    → слоты → подтверждение → запись + SMS.

    Ветвления: не найден (каскад идентификации), не прикреплён, нет слота
    в пределах 24 часов у участкового (дежурный), дубль записи, слот заняли,
    специальность по направлению без направления → терапевт.
    """

    label = "booking"

    def __init__(self, frame=None, specialty: str | None = None,
                 doctor: str | None = None):
        self.specialty = specialty or (frame.specialty if frame else None)
        self.doctor = doctor or (frame.doctor if frame else None)
        self.prefs = frame.time_prefs if frame else None
        self.ident = IdentificationFlow()
        self.chooser: SlotChooser | None = None
        self.state = "start"       # ask_specialty/identify/relation/
                                   # attach_question/dup/ask_gp/offer/confirm
        self.pending_booking = None        # дубль записи
        self.gp_for_referral = False       # записываем к терапевту за направлением
        self.referral = None
        self.last_frame = None

    # -- запуск --

    def start(self, ctx: DialogContext) -> StepResult:
        ctx.collect("тема", "запись на приём")
        if not self.specialty:
            # специальность не прозвучала — дефолт по таблице 2.3:
            # участковый терапевт (педиатр, если пациент уже известен и младше 18)
            known_child = ctx.patient is not None and (
                ctx.patient.pediatric or _under_18(ctx.patient.dob, ctx.today))
            self.specialty = (default_specialty(True) if known_child
                              else None)
            if self.specialty is None:
                self.state = "ask_specialty"
                return StepResult(label=self.label, reply=ph.ASK_SPECIALTY,
                                  expecting="специальность")
        return self._with_specialty(ctx)

    def _with_specialty(self, ctx: DialogContext) -> StepResult:
        ctx.collect("специальность", tu.specialty_nom(self.specialty))
        if SPECIALTIES[self.specialty]["mode"] == "referral":
            return self._after_ident_entry(ctx, referral=True)
        return self._after_ident_entry(ctx, referral=False)

    def _after_ident_entry(self, ctx, referral: bool) -> StepResult:
        if referral and not isinstance(self, _ReferralBranch):
            return StepResult(label=self.label, handoff=_ReferralBranch(self))
        if ctx.patient is not None:
            return self._post_ident(ctx)
        self.state = "identify"
        return self.ident.start(ctx)

    # -- реплики --

    def handle_utterance(self, ctx: DialogContext, frame, text: str) -> StepResult:
        self.last_frame = frame
        if self.state == "ask_specialty":
            if frame.specialty:
                self.specialty = frame.specialty
                return self._with_specialty(ctx)
            if frame.doctor:
                self.doctor = frame.doctor
                ctx.collect("врач", frame.doctor)
                if ctx.patient is None:
                    self.state = "identify"
                    return self.ident.start(ctx)
                return self._post_ident(ctx)
            return StepResult(label=self.label, reply=ph.ASK_SPECIALTY,
                              progress=False, expecting="специальность")

        if self.state == "identify":
            res = self.ident.handle(ctx, frame, text)
            if res is not None:
                return res
            return self._post_ident(ctx)

        if self.state == "relation":
            # звонит представитель ребёнка: «Вы мама Алисы?»
            rel = frame.relation
            if rel is None and frame.answer is True:
                rel = "mother"
            if rel is None:
                if frame.answer is False:
                    return StepResult(label=self.label,
                                      reply=ph.ASK_RELATION_FALLBACK,
                                      expecting="кем приходится")
                return StepResult(label=self.label,
                                  reply=self._relation_question(ctx),
                                  progress=False)
            ctx.caller_relation = rel
            ctx.collect("звонящий", rel)
            # после ответа о родстве продолжаем запись
            if ctx.patient is not None:
                return self._post_ident(ctx)
            self.state = "identify"
            return self.ident.start(ctx)

        if self.state == "attach_question":
            if frame.intent == Intent.DOCS:
                return self._attach_info(ctx)
            if frame.intent == Intent.PAID:
                return StepResult(label=self.label,
                                  reply=ph.TRANSFER_PAID,
                                  actions=[Transfer(
                                      target=Target.PAID, priority=3,
                                      card=ctx.card("Не прикреплён, "
                                                    "спрашивал про платный приём",
                                                    "платное"))],
                                  done=True, status="transferred")
            if frame.answer is False:
                return StepResult(label=self.label, reply=ph.ANYTHING_ELSE,
                                  done=True)
            return StepResult(label=self.label,
                              reply="Рассказать, как прикрепиться, или "
                                    "записать на платный приём?",
                              progress=False)

        if self.state == "dup":
            if frame.answer is True:
                from .appointments import AppointmentsFlow
                return StepResult(label=self.label, handoff=AppointmentsFlow(
                    action="reschedule", target_booking=self.pending_booking))
            if frame.answer is False:
                return self._offer(ctx)
            return StepResult(label=self.label, reply=ph.DUP_BOOKING.format(
                specialty=tu.specialty_nom(self.specialty), date=""),
                progress=False)

        if self.state == "offer":
            status, payload = self.chooser.choose(ctx, frame, text)
            if status == "picked":
                return self._confirm(ctx)
            if status in ("again", "none"):
                return payload
            return StepResult(label=self.label, reply=ph.OFFER_SLOTS.format(
                options=options_phrase(self.chooser.options, ctx.today)),
                progress=False)

        if self.state == "confirm":
            if frame.answer is True:
                return self._book(ctx)
            if frame.answer is False:
                return self._offer(ctx)
            return StepResult(label=self.label,
                              reply="Скажите «да» или «нет» — или назовите "
                                    "другое время.", progress=False)
        return StepResult(label=self.label, progress=False)

    # -- шаги --

    def _relation_question(self, ctx) -> str:
        return ph.ASK_RELATION_CHILD.format(name=ctx.patient.fio.first)

    def _attach_info(self, ctx) -> StepResult:
        return StepResult(
            label=self.label,
            reply=ph.ATTACH_HOWTO + " " + ph.ANYTHING_ELSE_SHORT,
            done=True, status="info")

    def _post_ident(self, ctx: DialogContext) -> StepResult:
        p = ctx.patient
        thanks = ph.THANKS_NAME.format(name=p.fio.name_patronymic()) + " "
        if not p.attached:
            self.state = "attach_question"
            return StepResult(label=self.label,
                              reply=thanks + ph.NOT_ATTACHED +
                              " Рассказать, как прикрепиться, или записать "
                              "на платный приём?")
        if p.pediatric or _under_18(p.dob, ctx.today):
            self.specialty = child_specialty(self.specialty)
            ctx.collect("специальность", tu.specialty_nom(self.specialty))
            if ctx.caller_relation is None:
                self.state = "relation"
                return StepResult(label=self.label,
                                  reply=self._relation_question(ctx),
                                  expecting="кем приходится")
        dup = [b for b in ctx.mis.active_bookings(p.id)
               if b.specialty in (self.specialty,
                                  self.specialty.removesuffix("_child"))]
        if dup and not self.gp_for_referral:
            self.pending_booking = dup[0]
            self.state = "dup"
            return StepResult(label=self.label,
                              reply=thanks + ph.DUP_BOOKING.format(
                                  specialty=tu.specialty_nom(self.specialty),
                                  date=tu.date_words(dup[0].date,
                                                     with_weekday=False)))
        return self._offer(ctx, prefix=thanks)

    def _offer(self, ctx: DialogContext, prefix: str = "") -> StepResult:
        doctor = self.doctor
        if self.specialty in ("gp", "pediatrician") and ctx.patient \
                and ctx.patient.gp_doctor:
            doctor = doctor or ctx.patient.gp_doctor
            if not self._gp_has_urgent_slot(ctx):
                # нет слота в пределах 24 часов — дежурный/другой врач
                duty = ctx.mis.free_slots(
                    self.specialty, date_from=ctx.today,
                    date_to=ctx.today + timedelta(days=1), today=ctx.today)
                duty = [s for s in duty if s.doctor != ctx.patient.gp_doctor]
                if duty:
                    self.chooser = SlotChooser(self.specialty)
                    self.chooser.all_slots = duty
                    self.chooser.options = duty[:3]
                    self.chooser.offset = 3
                    self.state = "offer"
                    return StepResult(
                        label=self.label,
                        reply=prefix + ph.NO_URGENT_GP_SLOT + " " +
                        ph.OFFER_SLOTS.format(
                            options=options_phrase(duty[:3], ctx.today)))
                summary = ("Нет слотов к участковому в пределах 24 часов: "
                           + self.specialty)
                return StepResult(label=self.label,
                                  reply=prefix + ph.NO_SLOTS,
                                  actions=waitlist_transfer(ctx, summary),
                                  done=True, status="transferred")
        self.chooser = SlotChooser(
            self.specialty, doctor=doctor,
            daypart=self.prefs.daypart if self.prefs else None,
            fixed_date=self.prefs.date if self.prefs else None,
            exclude_weekday=self.prefs.exclude_weekday
            if self.prefs else None)
        note = ""
        if self.specialty in ("gp", "pediatrician") and doctor:
            note = f"Ваш участковый — {doctor}. "
        self.state = "offer"
        return self.chooser.offer(ctx, prefix=prefix + note,
                                  relaxed_note="Именно в это время свободного "
                                               "нет. Ближайшее:")

    def _gp_has_urgent_slot(self, ctx: DialogContext) -> bool:
        slots = ctx.mis.free_slots(
            self.specialty, doctor=ctx.patient.gp_doctor,
            date_from=ctx.today, date_to=ctx.today + timedelta(days=1),
            today=ctx.today)
        return bool(slots)

    def _confirm(self, ctx: DialogContext) -> StepResult:
        slot = self.chooser.chosen
        self.state = "confirm"
        return StepResult(
            label=self.label,
            reply=ph.CONFIRM_BOOKING.format(
                specialty=tu.specialty_nom(self.specialty), doctor=slot.doctor,
                date=tu.date_words(slot.date), time=tu.time_words(slot.time),
                cabinet=tu.number_words(slot.cabinet)),
            expecting="подтверждение да/нет")

    def _book(self, ctx: DialogContext) -> StepResult:
        slot = self.chooser.chosen
        try:
            appt = ctx.mis.create_booking(
                ctx.patient.id, slot,
                referral_id=self.referral.id if self.referral else None)
        except SlotTaken:
            rest = [s for s in self.chooser.all_slots
                    if s.key() != slot.key()][:3]
            if not rest:
                return StepResult(
                    label=self.label,
                    reply="Извините, это время только что заняли. "
                          + ph.NO_SLOTS,
                    actions=waitlist_transfer(
                        ctx, "Слот занят при подтверждении: " + self.specialty),
                    done=True, status="transferred")
            self.chooser.options = rest
            self.state = "offer"
            return StepResult(label=self.label, reply=ph.SLOT_TAKEN.format(
                options=options_phrase(rest, ctx.today)))
        status = BOOKED_GP if self.gp_for_referral else BOOKED
        return StepResult(label=self.label, reply=ph.BOOKED_DONE,
                          actions=[booking_sms(ctx, appt, "booked")],
                          done=True, status=status)


class _ReferralBranch(BookingFlow):
    """Специальность из колонки «по направлению» (2.3): ищем действующее
    направление в МИС; нет направления — предложить терапевта (педиатра),
    чтобы его получить."""

    label = "booking_referral"

    def __init__(self, outer: BookingFlow):
        super().__init__(specialty=outer.specialty, doctor=outer.doctor)
        self.prefs = outer.prefs
        self.ctx_patient_known = True

    def _post_ident(self, ctx: DialogContext) -> StepResult:
        p = ctx.patient
        thanks = ph.THANKS_NAME.format(name=p.fio.name_patronymic()) + " "
        paper = self.last_frame is not None and self.last_frame.paper_referral
        if paper:
            return StepResult(
                label=self.label,
                reply=ph.PAPER_REFERRAL,
                actions=[Transfer(target=Target.OPERATOR, priority=3,
                                  card=ctx.card(
                                      "Бумажное направление из другой МО, "
                                      "требует ручной регистрации",
                                      "запись по направлению"))],
                done=True, status="transferred")
        if p.pediatric or _under_18(p.dob, ctx.today):
            self.specialty = child_specialty(self.specialty)
        ref = ctx.mis.find_referral(p.id, self.specialty)
        if ref is None:
            self.state = "ask_gp"
            return StepResult(label=self.label,
                              reply=thanks + ph.NO_REFERRAL.format(
                                  specialty=tu.specialty_dat(self.specialty)))
        self.referral = ref
        thanks = (f"{p.fio.name_patronymic()}, вижу ваше направление к "
                  f"{tu.specialty_dat(self.specialty)}. ")
        date_to = ref.issued + timedelta(days=ref.valid_days)
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
                    label=self.label,
                    reply=ph.PAPER_REFERRAL,
                    actions=[Transfer(
                        target=Target.OPERATOR, priority=3,
                        card=ctx.card("Бумажное направление из другой МО",
                                      "запись по направлению"))],
                    done=True, status="transferred")
            if frame.answer is True:
                self.specialty = default_specialty(
                    ctx.patient.pediatric if ctx.patient else False)
                self.gp_for_referral = True
                ctx.collect("специальность",
                            tu.specialty_nom(self.specialty)
                            + " (за направлением)")
                return self._offer(ctx)
            if frame.answer is False:
                return StepResult(label=self.label, reply=ph.ANYTHING_ELSE,
                                  done=True)
            return StepResult(label=self.label,
                              reply=ph.NO_REFERRAL.format(
                                  specialty=tu.specialty_dat(self.specialty)),
                              progress=False)
        return super().handle_utterance(ctx, frame, text)

    def _book(self, ctx: DialogContext) -> StepResult:
        res = super()._book(ctx)
        if res.status == BOOKED and not self.gp_for_referral:
            res.reply = ("Готово. Возьмите паспорт, полис и результаты "
                         "обследований, если они есть на руках. "
                         + ph.ANYTHING_ELSE_SHORT)
        return res
