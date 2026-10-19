# -*- coding: utf-8 -*-
"""Surface realizer: фрейм -> текст.

Скелет — функция, собирающая фразу из компонентов. Компоненты
(формулировка просьбы, синоним специальности, временная фраза,
синонимы симптомов) перебираются заново на каждом вызове, поэтому
один фрейм даёт много разных текстов с одной и той же меткой.

STT-стилизация меняет только регистр и пунктуацию — опечаток нет
по решению проекта (текст приходит из speech-to-text).
"""

import random
import re

from vocab import (SPECIALTIES, SYMPTOMS, REQUESTS, GREETINGS, POLITENESS,
                   UNSUPPORTED_POOLS, STATEMENT_POOLS, UNLABELED_COMPLAINTS,
                   UNSUPPORTED_LEADS, UNSUPPORTED_LEADS_NO_GREET)
from frames import Frame

SKELETONS = {}

# доля текстов с неразмеченной жалобой-шумом (незнакомые симптомы
# должны игнорироваться, как незнакомое время уходит в <T_UNKNOWN>)
COMPLAINT_NOISE_PROBA = 0.15


def skeleton(sk_id, intents, ood=False, hard=False, structural=False):
    def deco(fn):
        SKELETONS[sk_id] = {"fn": fn, "intents": set(intents),
                            "ood": ood, "hard": hard,
                            "structural": structural}
        return fn
    return deco


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _symp_tail(p):
    if not p["symp"]:
        return ""
    return " " + " ".join(_cap(x) + "." for x in p["symp"])


# ---------------------------------------------------------------- скелеты

@skeleton("plain", ["book", "reschedule"])
def sk_plain(p, rng):
    parts = [f"{p['req']} {p['syn']['dat']}"]
    if p["temporal"]:
        parts.append(p["temporal"])
    return ", ".join(parts) + "." + _symp_tail(p)


@skeleton("symp_first", ["book", "reschedule"])
def sk_symp_first(p, rng):
    if not p["symp"]:
        return sk_plain(p, rng)
    head = " ".join(_cap(x) + "." for x in p["symp"])
    tail = f"{p['req']} {p['syn']['dat']}"
    if p["temporal"]:
        tail += f", {p['temporal']}"
    return f"{head} {tail}."


@skeleton("greet_polite", ["book", "reschedule"])
def sk_greet_polite(p, rng):
    parts = [f"{p['greet']}! {p['req']} {p['syn']['dat']}"]
    if p["temporal"]:
        parts.append(p["temporal"])
    parts.append(p["polite"])
    return ", ".join(parts) + "." + _symp_tail(p)


@skeleton("telegraph", ["book"])
def sk_telegraph(p, rng):
    s = _cap(p["syn"]["nom"])
    if p["temporal"]:
        s += f", {p['temporal']}"
    s += "."
    if p["symp"]:
        s += " " + ". ".join(p["symp"]) + "."
    return s


@skeleton("question", ["book", "reschedule"])
def sk_question(p, rng):
    parts = [f"{p['req']} {p['syn']['dat']}"]
    if p["temporal"]:
        parts.append(p["temporal"])
    return ", ".join(parts) + "?" + _symp_tail(p)


@skeleton("windows", ["book"])
def sk_windows(p, rng):
    s = f"есть свободные окна {p['syn']['ugen']}"
    if p["temporal"]:
        s += f" {p['temporal']}"
    return s + "?" + _symp_tail(p)


@skeleton("need_nom", ["book"])
def sk_need_nom(p, rng):
    s = f"нужен {p['syn']['nom']}"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("colloq", ["book", "reschedule"])
def sk_colloq(p, rng):
    lead = rng.choice(["слушай", "смотрите", "короче", "ребят"])
    parts = [f"{lead}, {p['req']} {p['syn']['dat']}"]
    if p["temporal"]:
        parts.append(p["temporal"])
    return ", ".join(parts) + "?" + _symp_tail(p)


@skeleton("short_please", ["book"])
def sk_short_please(p, rng):
    s = f"запишите, пожалуйста, {p['syn']['dat']}"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("negation", ["book", "reschedule"], hard=True)
def sk_negation(p, rng):
    if p["intent"] == "book":
        s = f"не отменяй, наоборот, хочу {p['syn']['dat']}"
        if p["temporal"]:
            s += f", {p['temporal']}"
    else:
        s = f"запись {p['syn']['dat']} не отменяйте, просто перенесите"
        if p["temporal"]:
            s += f", удобно {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("unknown_spec", ["book"], structural=True)
def sk_unknown_spec(p, rng):
    parts = [f"{p['req']} к врачу"]
    if rng.random() < 0.5:
        parts.append("не знаю к кому")
    if p["temporal"]:
        parts.append(p["temporal"])
    return ", ".join(parts) + "." + _symp_tail(p)


@skeleton("conflict", ["book"], structural=True)
def sk_conflict(p, rng):
    a, b = p["conflict"]
    if rng.random() < 0.5:
        s = f"{p['req']} {a['dat']} или {b['dat']}"
        if p["temporal"]:
            s += f", {p['temporal']}"
        s += "?"
    else:
        s = f"{p['req']} {a['dat']}, хотя лучше {b['dat']}"
        if p["temporal"]:
            s += f", {p['temporal']}"
        s += "."
    return s + _symp_tail(p)


@skeleton("resched_exist", ["reschedule"])
def sk_resched_exist(p, rng):
    s = f"у меня запись {p['syn']['dat']}, её надо перенести"
    if p["temporal"]:
        s += f", удобно {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("resched_unknown", ["reschedule"], structural=True)
def sk_resched_unknown(p, rng):
    s = rng.choice([
        "у меня есть запись, надо перенести",
        # Этап C ТЗ: косвенный перенос «не могу прийти» — не бессонница
        "есть запись, не могу прийти, перенесите",
        "запись есть, прийти не смогу, надо перенести",
        "у меня запись, прийти не получится, перенесите",
    ])
    if p["temporal"]:
        s += f", удобно {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("cancel_cannot", ["cancel"])
def sk_cancel_cannot(p, rng):
    s = "не смогу прийти, отмените запись"
    if p["syn"]:
        s += f" {p['syn']['dat']}"
    return s + "." + _symp_tail(p)


@skeleton("cancel_skip", ["cancel"], structural=True)
def sk_cancel_skip(p, rng):
    """«не приду, снимите запись» — косвенная отмена через «снять» + день.

    Этап C ТЗ (С-04): «не приду в пятницу, снимите запись» обязан быть
    CANCEL, а не SCHEDULE (иголка «в пятницу» тянула в график). Слот
    времени у cancel не заполняется — день относится к отменяемой записи.
    """
    verb = rng.choice(["снимите запись", "сними запись", "снимите мой приём",
                       "отмените запись", "отмените приём", "аннулируйте запись"])
    lead = rng.choice(["не приду", "не смогу прийти", "не приду на приём",
                       "прийти не получится"])
    day = rng.choice(["сегодня", "завтра", "в понедельник", "во вторник",
                      "в среду", "в четверг", "в пятницу", "в субботу",
                      "в воскресенье"])
    if rng.random() < 0.65:
        s = f"не приду {day}, {verb}"
    else:
        s = f"{lead}, {verb}"
    return _cap(s) + "." + _symp_tail(p)


@skeleton("cancel_direct", ["cancel"])
def sk_cancel_direct(p, rng):
    if p["syn"]:
        return f"{p['req']} {p['syn']['dat']}." + _symp_tail(p)
    return f"{p['req']}." + _symp_tail(p)


@skeleton("cancel_greet", ["cancel"])
def sk_cancel_greet(p, rng):
    s = f"{p['greet']}! {p['req']}"
    if p["syn"]:
        s += f" {p['syn']['dat']}"
    return f"{s}, {p['polite']}." + _symp_tail(p)


@skeleton("cancel_unknown", ["cancel"], structural=True)
def sk_cancel_unknown(p, rng):
    lead = rng.choice(["", "к сожалению, ",
                       "моя запись больше не нужна, ", ""])
    return f"{lead}не смогу прийти, отмените мою запись." + _symp_tail(p)


@skeleton("cancel_all", ["cancel"], structural=True)
def sk_cancel_all(p, rng):
    """«отмени все мои записи» — отмена без указания специальности."""
    verb = rng.choice(["отмени", "отмените", "сотри", "сотрите",
                       "сними", "снимите"])
    obj = rng.choice(["все мои записи", "все записи", "все мои приёмы",
                      "все приёмы", "все мои талоны"])
    s = f"{verb} {obj}"
    if rng.random() < 0.3:
        s = f"{rng.choice(GREETINGS)}! {s}"
    if rng.random() < 0.2:
        s += f", {rng.choice(POLITENESS)}"
    return _cap(s) + "." + _symp_tail(p)


# --- скелеты, намайненные из MASSIVE ru-RU calendar-фраз ---
# (data/massive_calendar_reference.jsonl; запросы расписания не взяты:
# в контракте нет intent=query, разметка была бы ошибочной)

@skeleton("mark_calendar", ["book"])
def sk_mark_calendar(p, rng):
    s = f"пометь в моём календаре приём {p['syn']['dat']}"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("add_calendar", ["book"])
def sk_add_calendar(p, rng):
    s = f"добавь в мой календарь приём {p['syn']['ugen']}"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("dont_forget", ["book"])
def sk_dont_forget(p, rng):
    s = f"не забудь записать меня {p['syn']['dat']}"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("check_spot", ["book"])
def sk_check_spot(p, rng):
    s = f"проверь, есть ли место {p['syn']['ugen']}"
    if p["temporal"]:
        s += f" {p['temporal']}"
    return s + "?" + _symp_tail(p)


@skeleton("move_simple", ["reschedule"])
def sk_move_simple(p, rng):
    s = f"перенеси мой приём {p['syn']['dat']}"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("remove_cancel", ["cancel"])
def sk_remove_cancel(p, rng):
    lead = rng.choice(["удали", "сотри"])
    s = f"{lead} мою запись"
    if p["syn"]:
        s += f" {p['syn']['dat']}"
    return s + "." + _symp_tail(p)


@skeleton("verb_join", ["book"])
def sk_verb_join(p, rng):
    """Живой порядок без запятой: «мне бы записаться к терапевту»
    (урок итерации: такая поверхность не покрывалась и уходила в OTHER)."""
    verb = rng.choice(["хочу записаться", "можно записаться",
                       "надо записаться", "хотел бы записаться",
                       "хотела бы записаться", "запишите меня",
                       "мне надо записаться", "мне нужно записаться",
                       "мне бы записаться", "хочу попасть",
                       "нужно попасть"])
    s = f"{_lead(rng)}{verb} {p['syn']['dat']}"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return _cap(s) + "?" + _symp_tail(p)


@skeleton("family", ["book"])
def sk_family(p, rng):
    """«записать дочку к педиатру» — запись в интересах ребёнка/родственника."""
    who = rng.choice(["дочку", "сына", "ребёнка", "ребенка", "маму",
                      "папу", "мужа", "бабушку"])
    verb = rng.choice(["записать", "запишите", "нужно записать",
                       "надо записать", "хочу записать"])
    s = f"{_lead(rng)}{verb} {who} {p['syn']['dat']}"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return _cap(s) + "." + _symp_tail(p)


@skeleton("family_need", ["book"])
def sk_family_need(p, rng):
    """«сыну нужен детский врач» — нужда родственника + специалист в им. падеже.

    Этап C ТЗ (С-12): «сыну нужен детский врач» обязан быть BOOK+PEDIATRICIAN,
    а не WAITLIST/UNKNOWN: «сыну» (датив родственника) + «нужен» + номинатив
    специальности — отдельная поверхность, которой не было.
    """
    who = rng.choice(["сыну", "дочке", "дочери", "мужу", "жене", "маме",
                      "папе", "бабушке", "дедушке", "ребёнку", "ребенку"])
    s = f"{_lead(rng)}{who} нужен {p['syn']['nom']}"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return _cap(s) + "." + _symp_tail(p)


@skeleton("referral", ["book"], structural=True)
def sk_referral(p, rng):
    """Этап B (С-02): запись по направлению — «у меня направление к X»."""
    verb = rng.choice(["запишите на приём", "хочу попасть на приём",
                       "запишите", "нужна запись", "хочу попасть"])
    if rng.random() < 0.5:
        s = f"{_lead(rng)}у меня направление {p['syn']['dat']}, {verb}"
    else:
        s = f"{_lead(rng)}направление {p['syn']['dat']} на руках, {verb}"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return _cap(s) + "." + _symp_tail(p)


@skeleton("cancel_notneed", ["cancel"])
def sk_cancel_notneed(p, rng):
    """Порядок слов живой речи: «отменить, уже не нужно»."""
    tail = rng.choice(["уже не нужно", "уже не надо", "не актуально",
                       "передумал", "передумала", "не смогу"])
    s = f"{_lead(rng)}отменить, {tail}"
    if p["syn"] and rng.random() < 0.5:
        s += f", запись {p['syn']['dat']}"
    return _cap(s) + "."


@skeleton("ood_a", ["book"], ood=True)
def sk_ood_a(p, rng):
    s = f"надо бы {p['syn']['dat']} попасть"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("ood_b", ["book"], ood=True)
def sk_ood_b(p, rng):
    if p["symp"]:
        s = ". ".join(_cap(x) for x in p["symp"]) + "."
        s += f" {_cap(p['syn']['nom'])} нужен срочно"
    else:
        s = f"{_cap(p['syn']['nom'])} нужен срочно"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return s + "."


@skeleton("ood_c", ["book"], ood=True)
def sk_ood_c(p, rng):
    verb = rng.choice(["записался бы", "записалась бы",
                       "хотел бы записаться", "хотела бы записаться"])
    s = f"{verb} {p['syn']['dat']}, если есть места"
    if p["temporal"]:
        s += f", {p['temporal']}"
    return s + "." + _symp_tail(p)


# --- телефонный каталог (раздел 4 РГ-ТЕЛ-01) ---

# Лиды живой речи — одни и те же для положительных интентов, чтобы
# «подскажите, » не становилось маркером unsupported-класса
# (урок итерации: «Подскажите, на когда я записана» уходило в OTHER).
_POSITIVE_LEADS = ["", "подскажите, ", "извините, ", "здравствуйте, ",
                   "добрый день, ", "будьте добры, ", "скажите "]


def _lead(rng) -> str:
    return rng.choice(_POSITIVE_LEADS)


@skeleton("check_q", ["check_booking"])
def sk_check_q(p, rng):
    s = f"{_lead(rng)}{p['req']}"
    if p["syn"]:
        s += f" {p['syn']['dat']}"
    return _cap(s) + "?"


@skeleton("check_talon", ["check_booking"])
def sk_check_talon(p, rng):
    if p["syn"]:
        return _cap(f"{_lead(rng)}мой талон {p['syn']['ugen']} на когда?")
    return _cap(f"{_lead(rng)}мой талон на когда?")


@skeleton("check_polite", ["check_booking"])
def sk_check_polite(p, rng):
    s = f"{p['greet']}! {p['req']}"
    if p["syn"]:
        s += f" {p['syn']['dat']}"
    return _cap(s) + f", {p['polite']}?"


@skeleton("sched_generic", ["schedule"])
def sk_sched_generic(p, rng):
    return _cap(f"{_lead(rng)}{p['req']}") + "?"


@skeleton("sched_cabinet", ["schedule"])
def sk_sched_cabinet(p, rng):
    """Этап B (С-07): «в каком кабинете принимает врач» — график/справка,
    не проверка своей записи («какой у меня кабинет» — check_booking)."""
    if p["syn"]:
        return _cap(f"{_lead(rng)}в каком кабинете принимает "
                    f"{p['syn']['nom']}") + "?"
    return _cap(f"{_lead(rng)}в каком кабинете приём") + "?"


@skeleton("sched_day", ["schedule"])
def sk_sched_day(p, rng):
    """Этап B (С-07): «вы в субботу работаете?» — вопрос о графике по
    дням недели (временные иголки на schedule не проверяются — слот
    времени у справочного интента не заполняется)."""
    day = rng.choice(["в понедельник", "во вторник", "в среду", "в четверг",
                      "в пятницу", "в субботу", "в воскресенье"])
    return rng.choice([
        f"вы {day} работаете",
        f"вы работаете {day}",
        f"{_lead(rng)}{day} вы работаете",
        f"вы принимаете {day}",
    ]) + "?"


@skeleton("sched_doctor", ["schedule"])
def sk_sched_doctor(p, rng):
    if p["syn"]:
        return f"когда принимает {p['syn']['nom']}?"
    return _cap(p["req"]) + "?"


@skeleton("sched_doctor_hours", ["schedule"])
def sk_sched_doctor_hours(p, rng):
    if p["syn"]:
        return f"в какие часы приём {p['syn']['ugen']}?"
    return "в какие часы приём?"


@skeleton("home_simple", ["home_visit"])
def sk_home_simple(p, rng):
    s = f"{_lead(rng)}{p['req']}"
    if rng.random() < 0.3:
        s += f", {p['polite']}"
    return _cap(s) + "."


@skeleton("home_ill", ["home_visit"])
def sk_home_ill(p, rng):
    return rng.choice([
        "я плохо себя чувствую, нужен врач на дом",
        "лежу с плохим самочувствием, врача на дом можно",
        "не могу прийти в поликлинику, нужен врач на дом",
        "выхожу из больницы, врач должен прийти домой",
        # Этап B (С-06): ребёнок + на дом
        "болеет ребёнок, врач нужен на дом",
        "ребёнку срочно нужен врач на дом",
    ])


@skeleton("home_relative", ["home_visit"], structural=True)
def sk_home_relative(p, rng):
    """Этап C ТЗ: симптом у родственника — «у мужа давит в груди».
    Фрейм несёт симптом (не голое «да»), красные флаги дорезает safety."""
    if not p["symp"]:
        return sk_home_ill(p, rng)
    who = rng.choice(["мужа", "жены", "мамы", "папы", "бабушки",
                      "дедушки", "ребёнка"])
    tail = rng.choice(["нужен врач на дом", "вызовите врача",
                       "врача на дом", "надо врача на дом"])
    return rng.choice([
        f"у {who} {p['symp'][0]}, {tail}",
        f"{_cap(p['symp'][0])} у {who}, {tail}",
    ]) + "."


@skeleton("home_spec", ["home_visit"], structural=True)
def sk_home_spec(p, rng):
    """«вызовите педиатра на дом» — врач конкретной специальности на дом
    (винительный падеж, форма acc в словаре специальностей)."""
    tmpl = rng.choice(["вызовите {} на дом", "пришлите {} на дом",
                       "можно вызвать {} на дом", "хочу вызвать {} на дом"])
    return _cap(tmpl.format(p["syn"]["acc"])) + "."


@skeleton("lab_q", ["lab_status"])
def sk_lab_q(p, rng):
    s = p["req"]
    if rng.random() < 0.25:
        s += ", сдавал на прошлой неделе"
    return _cap(s) + "?"


@skeleton("lab_variants", ["lab_status"])
def sk_lab_variants(p, rng):
    return rng.choice([
        "сдавала анализы, интересно готовы ли",
        "кровь сдача когда будет готова",
        "хочу узнать про результаты моих анализов",
        "анализы которые сдавал готовы уже",
        "проверьте пожалуйста мои анализы",
        "результаты крови можно узнать",
        "анализы после обследования готовы или нет",
        "мои результаты анализов посмотреть",
        "гормоны щитовидки готовы узнать бы",
        "когда забрать результаты анализов",
        "анализы готовые скажите",
        "напомните когда анализы будут готовы",
    ])


@skeleton("paid_q", ["paid"])
def sk_paid_q(p, rng):
    s = p["req"]
    if rng.random() < 0.3:
        s += ", без полиса"
    return _cap(s) + "?"


@skeleton("paid_want", ["paid"])
def sk_paid_want(p, rng):
    return rng.choice([
        "хочу платно обследоваться, сколько это стоит",
        "по полису не выйдет, посчитайте мне платно",
        "сколько будет стоить платный приём",
        "цена платного приёма изменилась",
    ])


@skeleton("paid_spec", ["paid"], structural=True)
def sk_paid_spec(p, rng):
    """Этап B (С-09): «сколько стоит приём у кардиолога» — платный приём
    конкретного специалиста, специальность остаётся в слоте."""
    if not p["syn"]:
        return sk_paid_q(p, rng)
    return rng.choice([
        f"сколько стоит приём {p['syn']['ugen']}",
        f"приём {p['syn']['ugen']} сколько стоит",
        f"сколько берёте за приём {p['syn']['ugen']}",
        f"консультация {p['syn']['ugen']} почём",
    ]) + "?"


@skeleton("waitlist_q", ["waitlist"])
def sk_waitlist_q(p, rng):
    s = p["req"]
    if rng.random() < 0.4:
        s += ", время не подошло"
    return _cap(s) + "."


@skeleton("waitlist_variants", ["waitlist"])
def sk_waitlist_variants(p, rng):
    return rng.choice([
        "времени нет, запишите меня в очередь",
        "свободных окон нет, хочу в лист ожидания",
        "поставьте в очередь на приём",
        "нет подходящего времени, подожду освобождения",
        "запишите в очередь, перезвоните когда освободится",
        "хочу встать в лист ожидания к врачу",
        "очередь на приём как попасть",
        "в лист ожидания меня добавьте",
        "буду ждать окно, запишите в очередь",
        "позвоните как только время появится",
    ])


@skeleton("docs_q", ["docs"])
def sk_docs_q(p, rng):
    s = p["req"]
    if rng.random() < 0.3:
        s = f"{p['greet']}! {s}"
    return _cap(s) + "?"


@skeleton("docs_first_visit", ["docs"])
def sk_docs_first_visit(p, rng):
    return rng.choice([
        "иду впервые, что взять с собой",
        "первый раз к вам, какие документы нужны",
        "впервые на приём, что приготовить из бумаг",
        "что нужно принести на первый приём",
    ])


@skeleton("complaint_plain", ["complaint"])
def sk_complaint_plain(p, rng):
    s = p["req"]
    if rng.random() < 0.35:
        s = f"{p['greet']}! {s}"
    return _cap(s) + "."


@skeleton("complaint_emotion", ["complaint"])
def sk_complaint_emotion(p, rng):
    return rng.choice([
        "это безобразие, хамят в регистратуре",
        "мне нагрубили на телефоне, куда жаловаться",
        "жалоба на вашего сотрудника, долго держали на линии",
        "хочу написать жалобу на регистратуру",
        "обслуживание ужасное, сообщите руководству",
    ])


@skeleton("complaint_person", ["complaint"])
def sk_complaint_person(p, rng):
    """Этап B (С-15): «врач нагрубил», «нахамил» — жалоба на персонал,
    а не запись по слову «врач»."""
    head, tail = rng.choice([
        ("врач нагрубил", "куда пожаловаться"),
        ("врач нахамил", "куда сообщить"),
        ("медсестра нахамила", "куда обратиться"),
        ("регистратор нагрубил", "хочу пожаловаться"),
        ("доктор был груб", "куда написать жалобу"),
        ("мне нахамили", "хочу оставить жалобу"),
    ])
    return _cap(f"{head}, {tail}") + "."


@skeleton("medical_q", ["medical"])
def sk_medical_q(p, rng):
    s = p["req"]
    if rng.random() < 0.3:
        s = f"{p['greet']}! {s}"
    return _cap(s) + "?"


@skeleton("medical_symp", ["medical"], structural=True)
def sk_medical_symp(p, rng):
    """Медвопрос с симптомом из метки: «кашель, что принимать»."""
    if not p["symp"]:
        return sk_medical_q(p, rng)
    return " ".join(_cap(x) + "." for x in p["symp"]) + " Что принимать?"


# --------------------------------------------------------------- выбор

HARD_PROBA = 0.25


def _choose_skeleton(frame: Frame, rng) -> str:
    if frame.conflict:
        return "conflict"
    if frame.intent == "book" and frame.needs_clarification:
        return "unknown_spec"
    if frame.intent == "book" and frame.aux.get("referral"):
        return "referral"
    if frame.intent == "medical" and frame.aux.get("symp_medical"):
        return "medical_symp"
    if frame.intent == "home_visit" and frame.aux.get("home_symp"):
        return "home_relative"
    if frame.intent == "home_visit" and frame.aux.get("home_spec"):
        return "home_spec"
    if frame.intent == "paid" and frame.aux.get("paid_spec"):
        return "paid_spec"
    if frame.intent == "schedule" and frame.aux.get("day_q"):
        return "sched_day"
    if frame.ood:
        return rng.choice(["ood_a", "ood_b", "ood_c"])
    if frame.intent == "cancel" and frame.specialty not in SPECIALTIES:
        if frame.aux.get("cancel_all"):
            return "cancel_all"
        # косвенная отмена «не приду в пятницу, снимите запись» (С-04) —
        # половина без-специальности отмен, иначе не покрывается формулировка
        return rng.choice(["cancel_unknown", "cancel_unknown", "cancel_skip"])
    if frame.intent == "reschedule" and frame.specialty not in SPECIALTIES:
        return "resched_unknown"

    pool = [sid for sid, m in SKELETONS.items()
            if frame.intent in m["intents"] and not m["ood"] and not m["hard"] and not m["structural"]]
    hard = [sid for sid, m in SKELETONS.items()
            if frame.intent in m["intents"] and not m["ood"] and m["hard"] and not m["structural"]]
    if hard and rng.random() < HARD_PROBA:
        return rng.choice(hard)
    return rng.choice(pool)


def _pick_syn(spec: str | None, ood: bool, rng):
    if spec not in SPECIALTIES:
        return None
    bucket = "ood" if ood else "train"
    return rng.choice(SPECIALTIES[spec][bucket])


def _pick_symptom_expr(sym: str, ood: bool, rng):
    entry = SYMPTOMS[sym]
    if ood and entry["ood"]:
        return rng.choice(entry["ood"])
    return rng.choice(entry["train"])


def stt_style(text: str, rng):
    """Вариации, характерные для вывода STT: регистр и пунктуация.
    Двоеточие в числовых слотах («в 15:15») не вырезается: вербатимная
    валидация слота времени и парсер опираются на него (GigaAM с
    включённой пунктуацией выдаёт «15:15»; бесконтекстные варианты
    «15 15»/«9.30» понимает парсер _SLOT_RE на инференсе)."""
    r = rng.random()
    if r < 0.35:
        return text, "as_is"
    if r < 0.75:
        t = re.sub(r"[.,!?;—-]", "", text.lower())
        return re.sub(r"\s+", " ", t).strip(), "lower_nopunct"
    return text.lower(), "lower"


def build_unsupported(rng, aux: dict) -> str:
    pool_name = aux["pool"]
    base = UNSUPPORTED_POOLS[pool_name][aux["variant"] % len(UNSUPPORTED_POOLS[pool_name])]
    if pool_name in STATEMENT_POOLS:
        lead = rng.choice(UNSUPPORTED_LEADS_NO_GREET)
        return f"{lead}{base}."
    lead = rng.choice(UNSUPPORTED_LEADS)
    return f"{lead}{base}?"


# тривиальные пути: голые подтверждения внутри сценариев
# (пулы живут в vocab.ANSWER_SURFACES — их же использует e5-резолвер)
from vocab import ANSWER_SURFACES  # noqa: E402

_YES_POOL = ANSWER_SURFACES["yes"]
_NO_POOL = ANSWER_SURFACES["no"]


def build_answer(rng, aux: dict, which: str) -> str:
    pool = _YES_POOL if which == "yes" else _NO_POOL
    return rng.choice(pool)


def _is_answer(frame) -> bool:
    from vocab import ANSWER_TOKENS
    return frame.intent in ANSWER_TOKENS


def realize(frame: Frame, rng) -> dict:
    """Один текст для фрейма. Повторные вызовы дают парафразы."""
    if _is_answer(frame):
        raw = build_answer(rng, frame.aux, frame.intent)
        text, stt = stt_style(raw, rng)
        return {"text": text, "skeleton": "answer", "stt": stt,
                "temporal": None}

    if frame.intent == "unsupported":
        raw = build_unsupported(rng, frame.aux)
        text, stt = stt_style(raw, rng)
        return {"text": text, "skeleton": "unsupported", "stt": stt,
                "temporal": None}

    syn = _pick_syn(frame.specialty, frame.ood, rng)
    p = {
        "intent": frame.intent,
        "req": rng.choice(REQUESTS[frame.intent]),
        "syn": syn,
        "symp": [_pick_symptom_expr(s, frame.ood, rng) for s in frame.symptoms],
        "temporal": frame.temporal,
        "greet": rng.choice(GREETINGS),
        "polite": rng.choice(POLITENESS),
        "conflict": None,
    }
    if frame.conflict:
        p["conflict"] = [_pick_syn(s, frame.ood, rng) for s in frame.extra_mentions]

    sk_id = _choose_skeleton(frame, rng)
    raw = SKELETONS[sk_id]["fn"](p, rng)
    # шум только при конкретной специальности: на фреймах-уточнениях
    # ("запишите к врачу, не знаю к кому") он путает NEEDS_CLARIFICATION
    if (not frame.conflict and not frame.needs_clarification
            and rng.random() < COMPLAINT_NOISE_PROBA):
        raw = f"{raw} {_cap(rng.choice(UNLABELED_COMPLAINTS))}."
    text, stt = stt_style(raw, rng)
    return {"text": text, "skeleton": sk_id, "stt": stt,
            "temporal": frame.temporal}
