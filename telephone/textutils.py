# -*- coding: utf-8 -*-
"""Текстовые утилиты: нормализация реплик ASR и вербализация для TTS.

Правила вербализации — раздел 3.2 регламента: даты с днём недели
(«в четверг, первого октября»), время словами («в девять тридцать»),
номер кабинета словами («кабинет двести четырнадцать»).
"""
from __future__ import annotations

import re
from datetime import date, time, timedelta

# ---------------------------------------------------------------------------
# Нормализация

_PUNCT = re.compile(r"[^\w\s\-]", re.UNICODE)


def normalize(text: str) -> str:
    """Нижний регистр, ё→е, без пунктуации, одинарные пробелы."""
    text = text.lower().replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return normalize(text).split()


def stem(word: str) -> str:
    """Грубое усечение окончания для словарного матчинга.

    «пятницу»→«пятниц», «запишите»→«запиш». Для коротких слов обрезаем
    меньше, чтобы не получать осмысленных основ из 1–2 букв.
    """
    w = word
    cut = 1 if len(w) <= 5 else (2 if len(w) <= 7 else 3)
    return w[: max(3, len(w) - cut)]


def stems(text: str) -> list[str]:
    return [stem(w) for w in words(text)]


def contains_any(text: str, patterns: list[str]) -> bool:
    """Текст содержит любую из фраз (по нормализованной подстроке)."""
    n = normalize(text)
    return any(normalize(p) in n for p in patterns)


# ---------------------------------------------------------------------------
# Числительные

_UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь",
          "восемь", "девять", "десять", "одиннадцать", "двенадцать",
          "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
          "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = {20: "двадцать", 30: "тридцать", 40: "сорок", 50: "пятьдесят",
         60: "шестьдесят", 70: "семьдесят", 80: "восемьдесят", 90: "девяносто"}
_HUNDREDS = {100: "сто", 200: "двести", 300: "триста", 400: "четыреста",
             500: "пятьсот", 600: "шестьсот", 700: "семьсот",
             800: "восемьсот", 900: "девятьсот"}

# Порядковые в родительном и именительном падежах, 1…31 (для дат).
_ORD_GEN = {
    1: "первого", 2: "второго", 3: "третьего", 4: "четвертого", 5: "пятого",
    6: "шестого", 7: "седьмого", 8: "восьмого", 9: "девятого", 10: "десятого",
    11: "одиннадцатого", 12: "двенадцатого", 13: "тринадцатого",
    14: "четырнадцатого", 15: "пятнадцатого", 16: "шестнадцатого",
    17: "семнадцатого", 18: "восемнадцатого", 19: "девятнадцатого",
    20: "двадцатого", 21: "двадцать первого", 22: "двадцать второго",
    23: "двадцать третьего", 24: "двадцать четвертого", 25: "двадцать пятого",
    26: "двадцать шестого", 27: "двадцать седьмого", 28: "двадцать восьмого",
    29: "двадцать девятого", 30: "тридцатого", 31: "тридцать первого",
}
_ORD_NOM = {
    1: "первое", 2: "второе", 3: "третье", 4: "четвертое", 5: "пятое",
    6: "шестое", 7: "седьмое", 8: "восьмое", 9: "девятое", 10: "десятое",
    11: "одиннадцатое", 12: "двенадцатое", 13: "тринадцатое",
    14: "четырнадцатое", 15: "пятнадцатое", 16: "шестнадцатое",
    17: "семнадцатое", 18: "восемнадцатое", 19: "девятнадцатое",
    20: "двадцатое", 21: "двадцать первое", 22: "двадцать второе",
    23: "двадцать третье", 24: "двадцать четвертое", 25: "двадцать пятое",
    26: "двадцать шестое", 27: "двадцать седьмое", 28: "двадцать восьмое",
    29: "двадцать девятое", 30: "тридцатое", 31: "тридцать первое",
}

MONTHS_GEN = {1: "января", 2: "февраля", 3: "марта", 4: "апреля", 5: "мая",
              6: "июня", 7: "июля", 8: "августа", 9: "сентября",
              10: "октября", 11: "ноября", 12: "декабря"}
MONTHS_NOM = {1: "январь", 2: "февраль", 3: "март", 4: "апрель", 5: "май",
              6: "июнь", 7: "июль", 8: "август", 9: "сентябрь",
              10: "октябрь", 11: "ноябрь", 12: "декабрь"}

WEEKDAYS_ACC = {0: "в понедельник", 1: "во вторник", 2: "в среду",
                3: "в четверг", 4: "в пятницу", 5: "в субботу",
                6: "в воскресенье"}
WEEKDAYS_NOM = {0: "понедельник", 1: "вторник", 2: "среда", 3: "четверг",
                4: "пятница", 5: "суббота", 6: "воскресенье"}


def number_words(n: int) -> str:
    """Число словами: 214 → «двести четырнадцать»."""
    if n < 20:
        return _UNITS[n]
    parts: list[str] = []
    if n >= 1000:
        thousands = n // 1000
        n %= 1000
        if thousands == 1:
            parts.append("тысяча")
        elif thousands <= 4:
            parts.append(_UNITS[thousands] + " тысячи")
        else:
            parts.append(number_words(thousands) + " тысяч")
    if n >= 100:
        parts.append(_HUNDREDS[n // 100 * 100])
        n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10 * 10])
        n %= 10
    if n:
        parts.append(_UNITS[n])
    return " ".join(parts) or "ноль"


def time_words(t: time) -> str:
    """09:30 → «девять тридцать», 16:00 → «шестнадцать ноль-ноль»."""
    if t.minute == 0:
        mm = "ноль-ноль"
    elif t.minute < 10:
        mm = "ноль " + _UNITS[t.minute]
    else:
        mm = number_words(t.minute)
    return f"{number_words(t.hour)} {mm}"


def date_words(d: date, with_weekday: bool = True) -> str:
    """2026-10-02 → «в пятницу, второго октября»."""
    if with_weekday:
        return f"{WEEKDAYS_ACC[d.weekday()]}, {_ORD_GEN[d.day]} {MONTHS_GEN[d.month]}"
    return f"{_ORD_GEN[d.day]} {MONTHS_GEN[d.month]}"


def date_words_nom(d: date) -> str:
    """2026-10-02 → «второе октября» (перечисление вариантов)."""
    return f"{_ORD_NOM[d.day]} {MONTHS_GEN[d.month]}"


def date_words_move(d: date) -> str:
    """«вторник, шестое октября» — для «Переношу на …» (С-03, С-10)."""
    return f"{WEEKDAYS_NOM[d.weekday()]}, {date_words_nom(d)}"


def resolve_weekday(today: date, weekday: int) -> date:
    """Ближайший будущий такой день недели (строго позже сегодня)."""
    delta = (weekday - today.weekday()) % 7
    if delta == 0:
        delta = 7
    return today + timedelta(days=delta)


def next_monday(today: date) -> date:
    return resolve_weekday(today, 0)


# ---------------------------------------------------------------------------
# Разбор слов-числительных (обратная вербализация, для дат и полиса)

_WORD2NUM: dict[str, int] = {}
for _i, _w in enumerate(_UNITS):
    _WORD2NUM[_w] = _i
_WORD2NUM["две"] = 2
for _t, _w in _TENS.items():
    _WORD2NUM[_w] = _t
for _h, _w in _HUNDREDS.items():
    _WORD2NUM[_w] = _h
_WORD2NUM["тысяча"] = 1000
_WORD2NUM["тысячи"] = 1000
_WORD2NUM["тысяч"] = 1000
_WORD2NUM["ноль-ноль"] = 0

_GEN2ORD = {normalize(v): k for k, v in _ORD_GEN.items()}
_NOM2ORD = {normalize(v): k for k, v in _ORD_NOM.items()}
_GEN2ORD.update({"первого": 1, "второго": 2, "третьего": 3, "четвертого": 4,
                 "пятого": 5, "шестого": 6, "седьмого": 7, "восьмого": 8,
                 "девятого": 9, "десятого": 10, "одиннадцатого": 11,
                 "двенадцатого": 12, "тринадцатого": 13, "четырнадцатого": 14,
                 "пятнадцатого": 15, "шестнадцатого": 16, "семнадцатого": 17,
                 "восемнадцатого": 18, "девятнадцатого": 19, "двадцатого": 20,
                 "тридцатого": 30})
_NOM2ORD.update({"первое": 1, "второе": 2, "третье": 3, "четвертое": 4,
                 "пятое": 5, "шестое": 6, "седьмое": 7, "восьмое": 8,
                 "девятое": 9, "десятое": 10, "одиннадцатое": 11,
                 "двенадцатое": 12, "тринадцатое": 13, "четырнадцатое": 14,
                 "пятнадцатое": 15, "шестнадцатое": 16, "семнадцатое": 17,
                 "восемнадцатое": 18, "девятнадцатое": 19, "двадцатое": 20,
                 "тридцатое": 30})

_CARD_BY_STEM: dict[str, int] = {}
for _w, _v in _WORD2NUM.items():
    _CARD_BY_STEM.setdefault(stem(_w), _v)


def _ordinal_token_value(tok: str) -> int | None:
    """«восьмого»→8, «девятнадцатого»→19, «девяностого»→90
    (порядковые из слов-числительных)."""
    if tok in _GEN2ORD:
        return _GEN2ORD[tok]
    if tok in _NOM2ORD:
        return _NOM2ORD[tok]
    if tok.endswith("ого") or tok.endswith("ое") or tok.endswith("ий"):
        base = tok[:-3] if tok.endswith("ого") else tok[:-2]
        # ищем количественное, совпадающее по основе в любую сторону
        best: tuple[int, int] | None = None
        for w, v in _WORD2NUM.items():
            if w.startswith(base) or base.startswith(w):
                if best is None or len(w) > best[0]:
                    best = (len(w), v)
        return best[1] if best else _CARD_BY_STEM.get(stem(tok))
    return None


def parse_number_words(tokens: list[str]) -> list[int]:
    """Последовательность слов-числительных → список чисел.

    «двенадцать ноль три семьдесят восемь» → [12, 0, 3, 78];
    «две тысячи девятнадцатого» → [2019]; «ноль пять» → [0, 5].
    Правила: десятки+единицы — одно число; две единицы подряд — граница.
    """
    result: list[int] = []
    num: int | None = None     # сотни/тысячи текущего числа
    sub: int | None = None     # накопленное < 100

    def flush():
        nonlocal num, sub
        if num is not None or sub is not None:
            result.append((num or 0) + (sub or 0))
        num, sub = None, None

    for tok in tokens:
        v = _WORD2NUM.get(tok)
        if v is None:
            v = _ordinal_token_value(tok)
            if v is None:
                flush()
                continue
        if v == 1000:
            base = (num or 0) + (sub or 0)
            num = (base or 1) * 1000
            sub = None
        elif v >= 100:
            num = (num or 0) + v
        elif v >= 20:
            if sub is not None and sub < 20:
                flush()
            sub = v
        else:  # 0…19
            if sub is None:
                sub = v
            elif sub >= 20:
                sub = sub + v          # 70 + 8 = 78
            else:
                flush()                # 12 | 0 — граница числа
                sub = v
    flush()
    return result


# ---------------------------------------------------------------------------
# Названия месяцев и дней недели в репликах

_MONTH_BASES = {"январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5,
                "июн": 6, "июл": 7, "август": 8, "сентябр": 9,
                "октябр": 10, "ноябр": 11, "декабр": 12}
_WEEKDAY_BASES = {"понедельн": 0, "вторник": 1, "сред": 2, "четверг": 3,
                  "пятниц": 4, "суббот": 5, "воскресен": 6}
_SUFFIXES = ("", "а", "у", "е", "и", "ю", "я", "ой", "ем", "ём", "ью")


def month_of_token(token: str) -> int | None:
    for base, num in _MONTH_BASES.items():
        for suf in _SUFFIXES:
            if token == base + suf:
                return num
    return None


def weekday_of_token(token: str) -> int | None:
    for base, num in _WEEKDAY_BASES.items():
        for suf in _SUFFIXES:
            if token == base + suf:
                return num
    return None


# ---------------------------------------------------------------------------
# Склонения для реплик

def specialty_dat(specialty_key: str) -> str:
    """Ключ специальности → «к терапевту»."""
    from .clinic_data import SPECIALTIES
    return SPECIALTIES[specialty_key]["dat"]


def specialty_nom(specialty_key: str) -> str:
    from .clinic_data import SPECIALTIES
    return SPECIALTIES[specialty_key]["nom"]


def doctor_gen(doctor_full: str) -> str:
    """«Кузнецова Анна Сергеевна» → «Кузнецовой» (грубой родительный)."""
    last = doctor_full.split()[0]
    if last.endswith("а"):
        return last[:-1] + "ой"
    if last.endswith("я"):
        return last[:-1] + "ей"
    return last + "а"
