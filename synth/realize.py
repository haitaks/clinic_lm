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
                   UNSUPPORTED_POOLS, STATEMENT_POOLS,
                   UNSUPPORTED_LEADS, UNSUPPORTED_LEADS_NO_GREET)
from frames import Frame

SKELETONS = {}


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
    s = "у меня есть запись, надо перенести"
    if p["temporal"]:
        s += f", удобно {p['temporal']}"
    return s + "." + _symp_tail(p)


@skeleton("cancel_cannot", ["cancel"])
def sk_cancel_cannot(p, rng):
    s = "не смогу прийти, отмените запись"
    if p["syn"]:
        s += f" {p['syn']['dat']}"
    return s + "." + _symp_tail(p)


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


# --------------------------------------------------------------- выбор

HARD_PROBA = 0.25


def _choose_skeleton(frame: Frame, rng) -> str:
    if frame.conflict:
        return "conflict"
    if frame.intent == "book" and frame.needs_clarification:
        return "unknown_spec"
    if frame.ood:
        return rng.choice(["ood_a", "ood_b", "ood_c"])
    if frame.intent == "cancel" and frame.specialty not in SPECIALTIES:
        return "cancel_unknown"
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
    """Вариации, характерные для вывода STT: регистр и пунктуация."""
    r = rng.random()
    if r < 0.35:
        return text, "as_is"
    if r < 0.75:
        t = re.sub(r"[.,!?;:—-]", "", text.lower())
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


def realize(frame: Frame, rng) -> dict:
    """Один текст для фрейма. Повторные вызовы дают парафразы."""
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
    text, stt = stt_style(raw, rng)
    return {"text": text, "skeleton": sk_id, "stt": stt,
            "temporal": frame.temporal}
