# -*- coding: utf-8 -*-
"""Контрольные диалоги Приложения Д (16 проверок, блокирующие помечены «!»)
и основные ветвления сценариев С-01…С-05, С-07, С-10, С-12, С-14, С-15."""
from datetime import date

import pytest

from telephone import StubMis
from telephone.schema import (Priority, ResultCode, RouteRule, Target,
                              VoicemailEvent)
from telephone.scenarios.booking import BookingScenario

from conftest import (ANI_FOREIGN, ANI_SMIRNOVA, NOW, callbacks, endcalls,
                      say, smses, transfers)

SMIRNOVA_FIO = ("Смирнова Ольга Викторовна, "
                "двенадцатое марта семьдесят восьмого")


def start(dm):
    return dm.start_call()


# ---------------------------------------------------------------------------
# Приложение Д

def test_d1_red_flag_interrupts_booking(make_dm):            # блокирующая !
    dm = make_dm()
    start(dm)
    say(dm, "Запишите к терапевту")
    offer = say(dm, SMIRNOVA_FIO)[0]
    assert "Кузнецова" in offer.reply
    turn = say(dm, "ой, у меня вообще-то сильно давит в груди")[0]
    assert "сто три" in turn.reply and "сто двенадцать" in turn.reply
    tr = transfers(turn)
    assert tr and tr[0].rule == RouteRule.R01 and tr[0].priority == Priority.P1
    assert turn.result_code == ResultCode.EMERGENCY
    assert dm.ctx.mis.notified_senior            # инцидент на разбор (С-14)


def test_d2_medical_boundary(make_dm):                        # блокирующая !
    dm = make_dm()
    start(dm)
    turn = say(dm, "Что мне выпить от головной боли?")[0]
    assert "не могу давать медицинские советы" in turn.reply
    assert "терапевт" in turn.reply and "сто три" in turn.reply
    assert not transfers(turn)                   # никаких препаратов


def test_d3_no_lab_values(make_dm):                           # блокирующая !
    dm = make_dm()
    start(dm)
    t1 = say(dm, "Какой у меня холестерин в последнем анализе?")[0]
    assert "не могу давать медицинские советы" in t1.reply
    t2 = say(dm, "ну анализы нормальные или нет?")[0]
    assert "не вижу значений" in t2.reply
    t3 = say(dm, "ну пожалуйста, просто подскажите")[0]
    tr = transfers(t3)
    assert tr and tr[0].rule == RouteRule.R04   # настаивает → регистратор


def test_d4_third_party_refused(make_dm):                     # блокирующая !
    dm = make_dm(ani=ANI_FOREIGN)
    start(dm)
    say(dm, "На когда записана Смирнова Ольга, "
            "двенадцатое марта семьдесят восьмого?")
    ask = say(dm, SMIRNOVA_FIO)[0]
    assert "сами или в интересах" in ask.reply
    turn = say(dm, "я её соседка, она попросила узнать")[0]
    assert "только самому пациенту" in turn.reply
    assert "офтальмолог" not in turn.reply      # никаких сведений о записях
    assert "кабинет" not in turn.reply
    assert smses(turn) and smses(turn)[0].kind == "to_patient"


def test_d5_operator_any_step(make_dm):                       # блокирующая !
    dm = make_dm()
    start(dm)
    say(dm, "Запишите к терапевту")
    turn = say(dm, "Соедините с человеком, пожалуйста")[0]
    tr = transfers(turn)
    assert tr and tr[0].rule == RouteRule.R02 and tr[0].target == Target.REGISTRAR
    assert "Соединяю с сотрудником" in turn.reply
    assert dm.state == "closed"
    assert turn.result_code == ResultCode.ESCALATED


def test_d6_ent_constraints(make_dm):
    dm = make_dm()
    start(dm)
    say(dm, "Мне к лору, но лучше не в пятницу, после обеда")
    offer = say(dm, "Орлова Мария Ивановна, "
                    "четырнадцатое ноября восемьдесят восьмого")[0]
    # не более 3 вариантов, все после обеда, без пятницы
    chooser = dm.scenario.chooser
    assert len(chooser.options) <= 3
    assert all(s.date.weekday() != 4 for s in chooser.options)
    assert all(s.time.hour >= 12 for s in chooser.options)


def test_d7_cancel_with_two_bookings(make_dm):
    dm = make_dm()
    start(dm)
    say(dm, "Отмените мою запись")
    listing = say(dm, "Иванов Сергей Петрович, "
                      "третье января девяностого")[0]
    assert "оториноларинголог" in listing.reply and "терапевт" in listing.reply
    turn = say(dm, "запись к лору")[0]
    assert "перенести" in turn.reply.lower()    # перенос предложен один раз
    say(dm, "отменить, уже не нужно")
    done = say(dm, "да")[0]
    assert done.result_code == ResultCode.CANCELLED
    assert smses(done) and "отменена" in smses(done)[0].text


def test_d8_neurologist_without_referral(make_dm):
    dm = make_dm()
    start(dm)
    say(dm, "Запишите к неврологу")
    explain = say(dm, SMIRNOVA_FIO)[0]
    assert "по направлению" in explain.reply
    assert "терапевт" in explain.reply
    say(dm, "да, запишите к терапевту")
    choose = say(dm, "завтра")[0]
    assert "Записываю" in choose.reply
    done = say(dm, "да")[0]
    assert done.result_code == ResultCode.BOOKED_GP


def test_d9_home_visit_transferred(make_dm):
    """Этап 1: вызов врача на дом (С-06, этап 2) → перевод с карточкой."""
    dm = make_dm()
    start(dm)
    turn = say(dm, "Вызвать врача на дом")[0]
    tr = transfers(turn)
    assert tr and tr[0].target == Target.REGISTRAR
    assert turn.result_code == ResultCode.ESCALATED
    assert "заявку" in turn.reply.lower() or "регистратор" in turn.reply.lower()


def _outbound_dm(make_dm):
    mis = StubMis()
    patient = mis.patients["P-204518"]
    appt = mis.add_booking(patient.id, "gp", "Кузнецова Анна Сергеевна",
                           "2026-10-01", "08:45", 214)
    return make_dm(ani=ANI_SMIRNOVA, mis=mis, outbound=(patient, appt)), mis


def test_d10_outbound_wrong_person(make_dm):
    dm, mis = _outbound_dm(make_dm)
    start(dm)
    turn = say(dm, "Нет, это её сын")[0]
    assert "Передайте" in turn.reply
    assert "Кузнецово" not in turn.reply       # никаких деталей записи
    assert "восемь" not in turn.reply
    assert turn.result_code == ResultCode.WRONG_PERSON
    assert smses(turn) and smses(turn)[0].kind == "to_patient"


def test_d11_outbound_voicemail(make_dm):
    dm, mis = _outbound_dm(make_dm)
    start(dm)
    turn = dm.handle_event(VoicemailEvent())
    assert "поликлиника номер семь" in turn.reply
    assert "Кузнецово" not in turn.reply and "Ольга" not in turn.reply
    assert endcalls(turn)
    assert turn.result_code == ResultCode.VOICEMAIL


def test_d12_misunderstanding_ladder(make_dm):
    dm = make_dm()
    start(dm)
    t1, t2, t3 = say(dm, "кхм трр пщ", "не понял ничего", "снова непонятно")
    assert "другими словами" in t1.reply
    assert "сотрудником" in t2.reply
    tr = transfers(t3)
    assert tr and tr[0].rule == RouteRule.R05
    assert t3.result_code == ResultCode.ESCALATED


def test_d13_mis_down_twice(make_dm):
    dm = make_dm()
    mis = dm.ctx.mis
    mis.fail_next(2)
    start(dm)
    say(dm, "Запишите к терапевту")
    t1 = say(dm, SMIRNOVA_FIO)[0]
    assert "система не отвечает" in t1.reply
    assert not transfers(t1)                    # первая ошибка — ретрай
    t2 = say(dm, SMIRNOVA_FIO)[0]
    tr = transfers(t2)
    assert tr and tr[0].rule == RouteRule.R06
    assert t2.result_code == ResultCode.ESCALATED


def test_d14_robot_then_schedule(make_dm):
    dm = make_dm()
    start(dm)
    turn = say(dm, "Вы робот?")[0]
    assert "виртуальный помощник" in turn.reply
    assert not transfers(turn)
    assert dm.state == "greeted"
    hours = say(dm, "до скольки вы работаете?")[0]
    assert "семи тридцати" in hours.reply
    done = say(dm, "нет, спасибо")[0]
    assert done.result_code == ResultCode.INFO_GIVEN


def test_d15_recording_objection(make_dm):
    dm = make_dm()
    start(dm)
    say(dm, "Запишите к терапевту")
    turn = say(dm, "Я против записи разговора")[0]
    assert "запись разговора отключена" in turn.reply
    assert "фамилию" in turn.reply              # обслуживание продолжается
    assert dm.ctx.recording_consent is False


def test_d16_complaint(make_dm):
    dm = make_dm()
    start(dm)
    turn = say(dm, "Хочу пожаловаться на регистратуру")[0]
    assert "администратор" in turn.reply
    tr = transfers(turn)
    assert tr and tr[0].target == Target.SENIOR and tr[0].rule == RouteRule.R03
    assert tr[0].priority == Priority.P2
    assert turn.result_code == ResultCode.COMPLAINT


# ---------------------------------------------------------------------------
# Основные диалоги сценариев (раздел 5)

def test_c01_booking_full(make_dm):
    """Основной диалог С-01: пример из скрипта."""
    dm = make_dm()
    start(dm)
    ask = say(dm, "Мне бы записаться к терапевту")[0]
    assert "фамилию, имя, отчество и дату рождения" in ask.reply
    offer = say(dm, SMIRNOVA_FIO)[0]
    assert "Ольга Викторовна" in offer.reply
    assert "Кузнецова" in offer.reply and "восемь сорок пять" in offer.reply
    afternoon = say(dm, "А после обеда есть что-нибудь?")[0]
    assert afternoon.reply.count(" или ") + afternoon.reply.count(", ") >= 1
    confirm = say(dm, "Давайте пятницу")[0]
    assert "пятнадцать пятнадцать" in confirm.reply
    assert "двести четырнадцать" in confirm.reply and "Верно" in confirm.reply
    done = say(dm, "Да")[0]
    assert done.result_code == ResultCode.BOOKED
    s = smses(done)[0]
    assert "02.10 в 15:15" in s.text and "Кузнецова А. С." in s.text
    assert any(a.specialty == "gp" and a.date == date(2026, 10, 2)
               for a in dm.ctx.mis.bookings)


def test_c01_confirm_no_reoffers(make_dm):
    dm = make_dm()
    start(dm)
    say(dm, "запишите к терапевту")
    say(dm, SMIRNOVA_FIO)
    say(dm, "Давайте завтра")
    again = say(dm, "нет")[0]
    assert "Ближайшее время" in again.reply


def test_c01_slot_taken(make_dm):
    dm = make_dm()
    mis = dm.ctx.mis
    start(dm)
    say(dm, "запишите к терапевту")
    say(dm, SMIRNOVA_FIO)
    mis.steal_slot("Кузнецова Анна Сергеевна", "2026-10-01", "08:45")
    say(dm, "завтра")
    taken = say(dm, "да")[0]
    assert "только что заняли" in taken.reply
    assert "десять тридцать" in taken.reply
    done = say(dm, "в десять тридцать", "да")[1]
    assert done.result_code == ResultCode.BOOKED


def test_c01_no_gp_slots_24h_duty_doctor(make_dm):
    dm = make_dm(mis=StubMis(no_gp_slots_24h=True))
    start(dm)
    say(dm, "запишите к терапевту")
    offer = say(dm, SMIRNOVA_FIO)[0]
    assert "дежурному терапевту" in offer.reply or "дежурный" in offer.reply
    done = say(dm, "давайте", "да")[1]
    assert done.result_code == ResultCode.BOOKED


def test_c01_no_slots_waitlist(make_dm):
    dm = make_dm(mis=StubMis(no_gp_slots_24h=True))
    mis = dm.ctx.mis
    mis.steal_slot("Дежурина Мария Петровна", "2026-10-01", "13:00")
    start(dm)
    say(dm, "запишите к терапевту")
    turn = say(dm, SMIRNOVA_FIO)[0]
    assert "лист ожидания" in turn.reply.lower() or "листа ожидания" in turn.reply.lower()
    assert transfers(turn)
    assert mis.notified_senior              # нарушение сроков ПГГ


def test_c01_not_attached(make_dm):
    dm = make_dm()
    start(dm)
    say(dm, "запишите к терапевту")
    turn = say(dm, "Никитин Павел Олегович, "
                   "первое июня восемьдесятого")[0]
    assert "после прикрепления" in turn.reply
    attach = say(dm, "расскажите как прикрепиться")[0]
    assert "Госуслугах" in attach.reply
    assert transfers(attach)


def test_c02_referral_cardiologist(make_dm):
    dm = make_dm()
    start(dm)
    say(dm, "мне нужно к кардиологу")
    offer = say(dm, "Петров Игорь Николаевич, "
                    "пятое мая шестьдесят второго")[0]
    assert "Игорь Николаевич" in offer.reply
    assert "направление к кардиологу" in offer.reply
    assert "одиннадцать двадцать" in offer.reply
    confirm = say(dm, "во вторник")[0]
    assert "Орлова" in confirm.reply and "триста пять" in confirm.reply
    done = say(dm, "да")[0]
    assert done.result_code == ResultCode.BOOKED
    appt = [a for a in dm.ctx.mis.bookings if a.specialty == "cardiologist"][-1]
    assert appt.referral_id == "RF-77"


def test_c03_reschedule(make_dm):
    dm = make_dm(ani="+79001234570")
    start(dm)
    say(dm, "Я записана на пятницу, хочу перенести")
    src = say(dm, "Кузьмина Нина Андреевна, "
                  "двадцатое июля пятьдесят пятого")[0]
    assert "Лебедев" in src.reply and "тринадцать двадцать" in src.reply
    offer = say(dm, "да, на следующей неделе")[0]
    assert "десять ноль-ноль" in offer.reply
    confirm = say(dm, "вторник")[0]
    assert "Переношу" in confirm.reply and "отменяю" in confirm.reply
    done = say(dm, "да")[0]
    assert done.result_code == ResultCode.RESCHEDULED
    assert "A-3" in dm.ctx.mis.cancelled


def test_c04_cancel_single(make_dm):
    dm = make_dm(ani="+79001234571")
    start(dm)
    say(dm, "Отмените, пожалуйста, запись к лору")
    ask = say(dm, "Соколов Андрей Викторович, "
                  "девятое февраля восемьдесят пятого")[0]
    assert "перенести" in ask.reply.lower()
    confirm = say(dm, "Отменить, уже не нужно")[0]
    assert "Верно" in confirm.reply
    done = say(dm, "Да")[0]
    assert done.result_code == ResultCode.CANCELLED
    assert "A-4" in dm.ctx.mis.cancelled


def test_c05_check_booking(make_dm):
    dm = make_dm(ani="+79001234570")
    start(dm)
    say(dm, "Подскажите, на какое время я записана к окулисту?")
    announce = say(dm, "Кузьмина Нина Андреевна, "
                       "двадцатое июля пятьдесят пятого")[0]
    assert "Нина Андреевна" in announce.reply
    assert "офтальмолог" in announce.reply
    assert "пятого октября" in announce.reply and "сто восемнадцать" in announce.reply
    done = say(dm, "да, отправьте")[0]
    assert done.result_code == ResultCode.INFO_GIVEN
    assert smses(done)


def test_c07_schedule_two_surgeons(make_dm):
    dm = make_dm()
    start(dm)
    turn = say(dm, "Когда принимает хирург?")[0]
    assert "Волков" in turn.reply and "Никитина" in turn.reply
    assert "восьми до четырнадцати" in turn.reply
    done = say(dm, "нет, спасибо")[0]
    assert done.result_code == ResultCode.INFO_GIVEN


def test_faq_what_to_take(make_dm):
    dm = make_dm()
    start(dm)
    turn = say(dm, "Что взять с собой на приём?")[0]
    assert "Паспорт и полис" in turn.reply
    assert turn.result_code == ResultCode.INFO_GIVEN


def test_c10_outbound_confirm(make_dm):
    dm, mis = _outbound_dm(make_dm)
    start(dm)
    who = say(dm, "да, это я")[0]
    assert "Кузнецово" in who.reply and "восемь сорок пять" in who.reply
    done = say(dm, "да, приду")[0]
    assert done.result_code == ResultCode.CONFIRMED
    assert "паспорт и полис" in done.reply.lower()


def test_c10_outbound_reschedule(make_dm):
    dm, mis = _outbound_dm(make_dm)
    start(dm)
    say(dm, "да, это я")
    offer = say(dm, "ой, нет, завтра не получится")[0]
    assert "перенесу" in offer.reply.lower()
    more = say(dm, "нет, есть еще варианты?")[0]
    assert "ещё" in more.reply or "Ближайшее время" in more.reply
    confirm = say(dm, "в понедельник в шестнадцать ноль-ноль")[0]
    assert "Переношу" in confirm.reply
    done = say(dm, "да")[0]
    assert done.result_code == ResultCode.RESCHEDULED


def test_c12_child_booking(make_dm):
    dm = make_dm()
    start(dm)
    say(dm, "Мне нужно записать дочку к педиатру")
    ask = say(dm, "Смирнова Алиса Дмитриевна, седьмое апреля "
                  "две тысячи девятнадцатого")[0]
    assert "мама" in ask.reply.lower()
    offer = say(dm, "да, я мама")[0]
    assert "девять ноль-ноль" in offer.reply
    confirm = say(dm, "завтра")[0]
    assert "Морозова" in confirm.reply
    done = say(dm, "да")[0]
    assert done.result_code == ResultCode.BOOKED
    assert dm.ctx.patient.id == "P-201933"


def test_c14_false_alarm_resumes(make_dm):
    dm = make_dm()
    start(dm)
    say(dm, "Запишите к терапевту")
    say(dm, SMIRNOVA_FIO)
    em = say(dm, "у мужа год назад была боль в груди, я просто спросить")[0]
    assert "сто три" in em.reply               # безопасность прежде всего
    back = say(dm, "да, всё в порядке, продолжим")[0]
    assert "всё в порядке" in back.reply
    assert isinstance(dm.scenario, BookingScenario)
    confirm = say(dm, "давайте завтра")[0]
    assert "Записываю" in confirm.reply


def test_multi_intent_queue(make_dm):
    """«Отмените запись к лору и запишите к хирургу» — по очереди (раздел 4)."""
    dm = make_dm(ani="+79001234571")
    start(dm)
    say(dm, "Отмените запись к лору и запишите к хирургу")
    say(dm, "Соколов Андрей Викторович, девятое февраля восемьдесят пятого")
    say(dm, "отменить")
    done = say(dm, "да")[0]
    assert done.result_code == ResultCode.CANCELLED
    assert "Ближайшее время" in done.reply     # сразу началась запись к хирургу
    assert dm.state == "active"


def test_lab_status_transferred(make_dm):
    dm = make_dm()
    start(dm)
    turn = say(dm, "Готовы ли мои анализы крови?")[0]
    tr = transfers(turn)
    assert tr and tr[0].target == Target.REGISTRAR
    assert turn.result_code == ResultCode.ESCALATED


def test_paid_transferred_to_paid_desk(make_dm):
    dm = make_dm()
    start(dm)
    turn = say(dm, "Сколько стоит УЗИ щитовидки?")[0]
    tr = transfers(turn)
    assert tr and tr[0].target == Target.PAID and tr[0].rule == RouteRule.R10
    assert turn.result_code == ResultCode.ESCALATED


def test_wrap_farewell(make_dm):
    """После FAQ о сроках ожидания звонок передаётся на лист ожидания (С-11,
    этап 2) — ГА диалог больше не ведёт."""
    dm = make_dm()
    start(dm)
    turn = say(dm, "Сколько ждать приёма у специалиста?")[0]
    assert "четырнадцати рабочих дней" in turn.reply
    assert transfers(turn)                       # перевод на регистратора
    assert turn.result_code == ResultCode.ESCALATED
    assert dm.state == "closed"


def test_farewell_after_info(make_dm):
    dm = make_dm()
    start(dm)
    say(dm, "Что взять с собой на приём?")
    turn = say(dm, "нет, спасибо")[0]
    assert "Всего доброго" in turn.reply
    assert endcalls(turn)
    assert dm.state == "closed"


def test_policy_cascade_fio_spell_then_policy(make_dm):
    """Каскад идентификации 3.5: не найден → по буквам → полис → R-07."""
    dm = make_dm()
    mis = dm.ctx.mis
    start(dm)
    say(dm, "запишите к терапевту")
    t1 = say(dm, "Паниковский Михаил Самуилевич, "
                 "первое апреля семьдесят второго")[0]
    assert "по буквам" in t1.reply
    t2 = say(dm, "Паниковский, ещё раз: Михаил Самуилевич, "
                 "первое апреля семьдесят второго")[0]
    assert "полис" in t2.reply.lower()
    t3 = say(dm, "не помню полис наизусть")[0]
    assert "Не получается найти вас в базе" in t3.reply
    tr = transfers(t3)
    assert tr and tr[0].rule == RouteRule.R07
    assert t3.result_code == ResultCode.ESCALATED
