# -*- coding: utf-8 -*-
"""Клиент МИС «Инфоклиника»: протокол и детерминированная заглушка.

Протокол — то, что нужно сценариям регламента (операции МИС из паспортов
С-01…С-11). Заглушка возвращает данные из примеров скрипта, даты —
абсолютные (пример скрипта: звонок в среду 30.09.2026), поэтому тесты
работают с FixedClock на эту дату.
"""
from __future__ import annotations

from datetime import date, time

from .schema import (Appointment, DoctorScheduleInfo, Fio, LabOrder,
                     MisPatient, Referral, Slot)


class MisUnavailable(Exception):
    """МИС не отвечает (> 5 секунд) — правило R-06."""


class SlotTaken(Exception):
    """Слот заняли, пока пациент выбирал (ветвление С-01)."""


DOCTORS = {
    "Кузнецова Анна Сергеевна": ("gp", 214),
    "Дежурина Мария Петровна": ("gp", 210),
    "Орлова Елена Павловна": ("cardiologist", 305),
    "Белова Ольга Игоревна": ("ent", 220),
    "Лебедев Дмитрий Петрович": ("ophthalmologist", 118),
    "Морозова Ирина Валерьевна": ("pediatrician", 320),
    "Волков Сергей Иванович": ("surgeon", 112),
    "Никитина Анна Львовна": ("surgeon", 112),
}

# Свободные слоты (до бронирования): врач → [(дата, время)]
_BASE_SLOTS: dict[str, list[tuple[str, str]]] = {
    "Кузнецова Анна Сергеевна": [
        ("2026-10-01", "08:45"), ("2026-10-02", "10:30"), ("2026-10-02", "15:15"),
        ("2026-10-05", "09:15"), ("2026-10-05", "16:00"), ("2026-10-06", "09:15"),
        ("2026-10-06", "12:00"), ("2026-10-07", "10:40"),
    ],
    "Дежурина Мария Петровна": [("2026-10-01", "13:00")],
    "Орлова Елена Павловна": [
        ("2026-10-06", "11:20"), ("2026-10-08", "14:40"),
    ],
    "Белова Ольга Игоревна": [
        ("2026-10-01", "09:00"), ("2026-10-01", "15:00"), ("2026-10-06", "12:00"),
        ("2026-10-07", "14:30"),
    ],
    "Лебедев Дмитрий Петрович": [
        ("2026-10-06", "10:00"), ("2026-10-07", "13:20"), ("2026-10-08", "09:40"),
    ],
    "Морозова Ирина Валерьевна": [
        ("2026-10-01", "09:00"), ("2026-10-02", "11:40"),
    ],
    "Волков Сергей Иванович": [
        ("2026-10-01", "10:00"), ("2026-10-02", "11:30"),
    ],
    "Никитина Анна Львовна": [("2026-10-01", "15:00")],
}


def _t(hhmm: str) -> time:
    hh, mm = hhmm.split(":")
    return time(int(hh), int(mm))


class StubMis:
    """Детерминированная заглушка МИС на данных примеров скрипта."""

    def __init__(self, no_gp_slots_24h: bool = False):
        self.patients: dict[str, MisPatient] = {
            "P-204518": MisPatient(
                id="P-204518",
                fio=Fio("Смирнова", "Ольга", "Викторовна"),
                dob=date(1978, 3, 12), phone="+79001234567",
                attached=True, district=14,
                gp_doctor="Кузнецова Анна Сергеевна"),
            "P-201119": MisPatient(
                id="P-201119",
                fio=Fio("Петров", "Игорь", "Николаевич"),
                dob=date(1962, 5, 5), phone="+79001234568",
                attached=True, district=7,
                gp_doctor="Кузнецова Анна Сергеевна"),
            "P-209001": MisPatient(
                id="P-209001",
                fio=Fio("Иванов", "Сергей", "Петрович"),
                dob=date(1990, 1, 3), phone="+79001234569",
                attached=True, district=14,
                gp_doctor="Кузнецова Анна Сергеевна"),
            "P-205519": MisPatient(
                id="P-205519",
                fio=Fio("Кузьмина", "Нина", "Андреевна"),
                dob=date(1955, 7, 20), phone="+79001234570",
                attached=True, district=3,
                gp_doctor="Кузнецова Анна Сергеевна"),
            "P-208511": MisPatient(
                id="P-208511",
                fio=Fio("Соколов", "Андрей", "Викторович"),
                dob=date(1985, 2, 9), phone="+79001234571",
                attached=True, district=9,
                gp_doctor="Дежурина Мария Петровна"),
            "P-201933": MisPatient(
                id="P-201933",
                fio=Fio("Смирнова", "Алиса", "Дмитриевна"),
                dob=date(2019, 4, 7), phone="+79001234567",
                attached=True, district=3, pediatric=True,
                gp_doctor="Морозова Ирина Валерьевна"),
            "P-208812": MisPatient(
                id="P-208812",
                fio=Fio("Орлова", "Мария", "Ивановна"),
                dob=date(1988, 11, 14), phone="+79001234572",
                attached=True, district=5,
                gp_doctor="Кузнецова Анна Сергеевна"),
            # не прикреплён — ветвление С-01
            "P-299900": MisPatient(
                id="P-299900",
                fio=Fio("Никитин", "Павел", "Олегович"),
                dob=date(1980, 6, 1), phone="+79001234573",
                attached=False, district=None, gp_doctor=""),
        }
        # номера полисов для каскада идентификации 3.5
        self.policies: dict[str, str] = {
            "P-204518": "1234567890123456",
            "P-201119": "2234567890123456",
        }
        self.referrals: dict[str, list[Referral]] = {
            # Петров: направление к кардиологу от 28.09, действует 30 дней
            "P-201119": [Referral(id="RF-77", specialty="cardiologist",
                                  issued=date(2026, 9, 28), valid_days=30)],
        }
        self.bookings: list[Appointment] = [
            Appointment("A-1", "P-209001", "ent", "Белова Ольга Игоревна",
                        date(2026, 10, 2), time(10, 40), 220),
            Appointment("A-2", "P-209001", "gp", "Кузнецова Анна Сергеевна",
                        date(2026, 10, 6), time(12, 0), 214),
            Appointment("A-3", "P-205519", "ophthalmologist",
                        "Лебедев Дмитрий Петрович",
                        date(2026, 10, 5), time(13, 20), 118),
            Appointment("A-4", "P-208511", "ent", "Белова Ольга Игоревна",
                        date(2026, 10, 2), time(10, 40), 220),
        ]
        self.lab_orders: dict[str, list[LabOrder]] = {
            "P-208812": [
                LabOrder("общий анализ крови", date(2026, 9, 28), "готов"),
                LabOrder("гормоны щитовидной железы", date(2026, 9, 28),
                         "в работе", expected=date(2026, 10, 2)),
            ],
        }
        self._free: dict[str, list[tuple[str, str]]] = {
            doc: list(slots) for doc, slots in _BASE_SLOTS.items()
        }
        self._next_booking_id = 100
        self._no_gp_slots_24h = no_gp_slots_24h
        if no_gp_slots_24h:
            # симуляция нарушения ПГГ: у участкового нет слота в пределах 24 ч
            self._free["Кузнецова Анна Сергеевна"] = [
                s for s in self._free["Кузнецова Анна Сергеевна"]
                if s[0] != "2026-10-01"]
        self._fail_next_calls = 0     # для теста R-06
        self.sent_sms: list[tuple[str, str]] = []
        self.waitlist: list[dict] = []
        self.notified_senior: list[str] = []
        self.cancelled: list[str] = []        # id отменённых записей
        self.doctor_on_leave: dict[str, date] = {}

    # -- управление состоянием заглушки (для тестов) --

    def fail_next(self, calls: int = 1) -> None:
        self._fail_next_calls = calls

    def steal_slot(self, doctor: str, day: str, hhmm: str) -> None:
        slot = (day, hhmm)
        if slot in self._free.get(doctor, []):
            self._free[doctor].remove(slot)

    def add_referral(self, patient_id: str, specialty: str, cito: bool = False,
                     expired: bool = False) -> None:
        issued = date(2026, 9, 28)
        self.referrals.setdefault(patient_id, []).append(
            Referral(id=f"RF-{len(self.referrals) + 1}", specialty=specialty,
                     issued=issued,
                     valid_days=1 if expired else 30, cito=cito))

    def add_booking(self, patient_id: str, specialty: str, doctor: str,
                    day: str, hhmm: str, cabinet: int) -> Appointment:
        appt = Appointment(f"A-{self._next_booking_id}", patient_id, specialty,
                           doctor, date.fromisoformat(day), _t(hhmm), cabinet)
        self._next_booking_id += 1
        self.bookings.append(appt)
        return appt

    # -- поиск --

    def _guard(self):
        if self._fail_next_calls > 0:
            self._fail_next_calls -= 1
            raise MisUnavailable("МИС не отвечает")

    def search_patient(self, fio: Fio | None = None, dob: date | None = None,
                       policy: str | None = None,
                       ani: str | None = None) -> list[MisPatient]:
        self._guard()
        if policy:
            for pid, pol in self.policies.items():
                if pol == policy:
                    return [self.patients[pid]]
            return []
        if fio is None or dob is None:
            return []
        found = []
        for p in self.patients.values():
            if p.fio.last.lower() == fio.last.lower() and p.dob == dob:
                found.append(p)
        return found

    # -- записи --

    def active_bookings(self, patient_id: str) -> list[Appointment]:
        self._guard()
        return [a for a in self.bookings
                if a.patient_id == patient_id and a.status == "Записан"]

    def find_referral(self, patient_id: str,
                      specialty: str) -> Referral | None:
        self._guard()
        best = None
        for ref in self.referrals.get(patient_id, []):
            if ref.specialty != specialty:
                continue
            expires = date.fromordinal(
                ref.issued.toordinal() + ref.valid_days)
            if expires < date(2026, 9, 30):
                continue                       # просрочено
            if ref.cito:
                return ref                     # cito приоритетнее
            best = ref
        return best

    def free_slots(self, specialty: str, doctor: str | None = None,
                   date_from: date | None = None, date_to: date | None = None,
                   daypart: str | None = None,
                   today: date | None = None) -> list[Slot]:
        self._guard()
        result: list[Slot] = []
        for doc, slots in self._free.items():
            spec = DOCTORS[doc][0]
            if specialty not in (spec, None):
                continue
            if doctor and not doc.startswith(doctor.split()[0]):
                continue
            for day, hhmm in slots:
                sd = date.fromisoformat(day)
                if date_from and sd < date_from:
                    continue
                if date_to and sd > date_to:
                    continue
                st = _t(hhmm)
                if daypart == "morning" and not (time(7, 30) <= st <= time(12, 0)):
                    continue
                if daypart == "afternoon" and not (time(12, 0) <= st <= time(16, 0)):
                    continue
                if daypart == "evening" and st <= time(16, 0):
                    continue
                if daypart == "early" and st > time(10, 0):
                    continue
                if daypart == "late" and st < time(15, 0):
                    continue
                result.append(Slot(doc, spec, sd, st, DOCTORS[doc][1]))
        result.sort(key=lambda s: (s.date, s.time))
        return result

    def create_booking(self, patient_id: str, slot: Slot,
                       referral_id: str | None = None) -> Appointment:
        self._guard()
        key = (slot.date.isoformat(), slot.time.strftime("%H:%M"))
        if key not in self._free.get(slot.doctor, []):
            raise SlotTaken(key)
        self._free[slot.doctor].remove(key)
        appt = Appointment(
            id=f"A-{self._next_booking_id}", patient_id=patient_id,
            specialty=slot.specialty, doctor=slot.doctor, date=slot.date,
            time=slot.time, cabinet=slot.cabinet, referral_id=referral_id)
        self._next_booking_id += 1
        self.bookings.append(appt)
        return appt

    def cancel_booking(self, booking_id: str, reason: str) -> Appointment:
        self._guard()
        for a in self.bookings:
            if a.id == booking_id:
                a.status = "Отменена"
                self.cancelled.append(booking_id)
                return a
        raise KeyError(booking_id)

    def reschedule_booking(self, booking_id: str, new_slot: Slot) -> Appointment:
        """Атомарно (С-03): новая запись создаётся, только если слот свободен."""
        self._guard()
        old = next(a for a in self.bookings if a.id == booking_id)
        key = (new_slot.date.isoformat(), new_slot.time.strftime("%H:%M"))
        if key not in self._free.get(new_slot.doctor, []):
            raise SlotTaken(key)
        self._free[new_slot.doctor].remove(key)
        # освободившийся слот уходит в лист ожидания
        old_key = (old.date.isoformat(), old.time.strftime("%H:%M"))
        self._free.setdefault(old.doctor, []).append(old_key)
        new = Appointment(
            id=f"A-{self._next_booking_id}", patient_id=old.patient_id,
            specialty=new_slot.specialty, doctor=new_slot.doctor,
            date=new_slot.date, time=new_slot.time,
            cabinet=new_slot.cabinet, referral_id=old.referral_id)
        self._next_booking_id += 1
        old.status = "Отменена"
        self.cancelled.append(old.id)
        self.bookings.append(new)
        return new

    def doctor_schedule(self, query: str) -> list[DoctorScheduleInfo]:
        """График по фамилии врача или специальности (С-07)."""
        self._guard()
        result = []
        for doc, (spec, cab) in DOCTORS.items():
            surname = doc.split()[0]
            if query and query.lower() not in doc.lower():
                continue
            if doc in self.doctor_on_leave:
                result.append(DoctorScheduleInfo(
                    doc, spec, "каждый будний день", "8:00–14:00", cab,
                    on_leave_until=self.doctor_on_leave[doc],
                    substitute="Дежурина Мария Петровна"))
                continue
            if spec == "surgeon" and surname == "Волков":
                result.append(DoctorScheduleInfo(doc, spec, "каждый будний день",
                                                 "с восьми до четырнадцати", cab))
            elif spec == "surgeon":
                result.append(DoctorScheduleInfo(doc, spec, "каждый будний день",
                                                 "с четырнадцати до двадцати", cab))
            else:
                result.append(DoctorScheduleInfo(doc, spec, "каждый будний день",
                                                 "по расписанию участка", cab))
        return result

    def lab_statuses(self, patient_id: str) -> list[LabOrder]:
        self._guard()
        return list(self.lab_orders.get(patient_id, []))

    def send_sms(self, phone: str, text: str) -> None:
        self.sent_sms.append((phone, text))

    def add_to_waitlist(self, patient_id: str, specialty: str,
                        prefs: str) -> None:
        self.waitlist.append({"patient": patient_id, "specialty": specialty,
                              "prefs": prefs})

    def notify_senior(self, text: str) -> None:
        self.notified_senior.append(text)
