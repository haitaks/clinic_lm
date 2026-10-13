# -*- coding: utf-8 -*-
"""Парсеры слотов из реплики ASR: дата рождения, полис ОМС, желаемые
дата/время, ответы да/нет.

Извлечение ФИО из реплики сюда НЕ входит: его выполняет внешняя
NLU-модель, менеджер диалога получает имя готовым во фрейме (Frame.fio).

Правила разбора дат — Приложение А.3 регламента:
- «двенадцатое марта семьдесят восьмого» — словами;
- «12.03.78» — цифрами;
- «двенадцать ноль три семьдесят восемь» — цифры словами;
- год из двух цифр восстанавливается из условия «не старше 110 лет».
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from .schema import TimePrefs
from . import textutils as tu

# ---------------------------------------------------------------------------
# Да / нет / не знаю

_YES = {"да", "ага", "конечно", "верно", "точно", "подходит", "хорошо",
        "давайте", "угу", "именно", "согласен", "согласна", "записывайте",
        "подтверждаю", "подтвердите", "ладно", "ок", "оставляем", "отлично",
        "приду", "буду", "придем", "придем конечно"}
_NO = {"нет", "не надо", "не нужно", "неа", "неверно", "не верно",
       "неправильно", "не подходит", "не хочу", "не буду", "не требуется",
       "не стоит", "не приду", "не смогу", "никак"}
_NO_PREFIX = ("не приду", "не смогу", "не буду", "не хочу", "не надо",
              "не нужно", "не приеду", "не смогу прийти")
_MAYBE = ("не знаю", "может быть", "наверное", "возможно", "пожалуй",
          "подумаю", "посмотрю", "не уверен", "не уверена", "не факт")


def parse_yesno(text: str) -> tuple[bool | None, bool]:
    """→ (да/нет, флаг «не знаю»)."""
    n = tu.normalize(text).strip()
    if not n:
        return None, False
    for m in _MAYBE:
        if n.startswith(m) or f" {m}" in n:
            return None, True
    if n in _YES or n.split(" ")[0] in _YES:
        return True, False
    if n in _NO or n.split(" ")[0] in _NO or any(n.startswith(p) for p in _NO_PREFIX):
        return False, False
    if n.startswith("нет"):
        return False, False
    return None, False


def parse_relation(text: str) -> str | None:
    """«я мама» → mother; для сценария С-12."""
    n = tu.normalize(text)
    if any(w in n for w in ("мама", "мать", "мам ")):
        return "mother"
    if any(w in n for w in ("папа", "отец")):
        return "father"
    if "опекун" in n:
        return "guardian"
    if any(w in n for w in ("бабушк", "дедушк", "соседк", "родственник",
                            "тетя", "дядя")):
        return "relative"
    if " сам" in f" {n}" or n in ("сам", "сама", "я сам", "я сама"):
        return "self"
    if "представитель" in n:
        return "representative"
    return None


# ---------------------------------------------------------------------------
# Дата рождения

_DIGIT_DATE = re.compile(r"(\d{1,2})\s?[\.\-\s]\s?(\d{1,2})\s?[\.\-\s]\s?(\d{2,4})")


def _resolve_year(year: int, today: date) -> int | None:
    """Год из двух цифр → полный: пациенту не может быть больше 110 лет."""
    if 1000 <= year <= 2100:
        return year
    if year < 100:
        for century in (2000, 1900):
            full = century + year
            if 0 <= (today - date(full, 1, 1)).days / 365.25 <= 110:
                return full
    return None


def _try_dob_numbers(nums: list[int], today: date) -> date | None:
    """[12, 0, 3, 78] / [12, 3, 78] / [12, 3, 1978] → дата."""
    if len(nums) < 3:
        return None
    day, rest = nums[0], nums[1:]
    if rest[0] == 0 and len(rest) >= 2:
        month, year_nums = rest[1], rest[2:]
    else:
        month, year_nums = rest[0], rest[1:]
    if not (1 <= day <= 31 and 1 <= month <= 12) or not year_nums:
        return None
    year = 0
    for n in year_nums:
        year = year * 100 + n if n < 100 else year + n
    year = _resolve_year(year, today)
    if year is None:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _numeric_value(tok: str) -> int | None:
    if tok in tu._WORD2NUM:
        return tu._WORD2NUM[tok]
    return tu._ordinal_token_value(tok)


def _is_number_token(tok: str) -> bool:
    return _numeric_value(tok) is not None


def extract_dob(text: str, today: date) -> date | None:
    """Дата рождения из реплики (Приложение А.3): словами, цифрами или
    цифрами-словами. Извлечение ФИО из реплики — задача внешней NLU-модели."""
    raw = tu.normalize(text)
    # 1) цифровой формат «12.03.78» / «12 03 1978»
    m = _DIGIT_DATE.search(raw)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = _resolve_year(year, today)
        if year and 1 <= day <= 31 and 1 <= month <= 12:
            try:
                return date(year, month, day)
            except ValueError:
                pass
    # 2) словами: «двенадцатое марта семьдесят восьмого»
    toks = raw.split()
    for i, tok in enumerate(toks):
        month = tu.month_of_token(tok)
        if month is None:
            continue
        day_and_back = _day_before(toks, i)
        if day_and_back is None:
            continue
        day, back = day_and_back
        if not (1 <= day <= 31):
            continue
        year_nums: list[int] = []
        for j in range(i + 1, len(toks)):
            v = _numeric_value(toks[j])
            if v is None:
                break
            year_nums.append(v)
        year = _sum_year_tokens(year_nums)
        if year is None:
            continue
        year = _resolve_year(year, today)
        if year is None:
            continue
        try:
            return date(year, month, day)
        except ValueError:
            continue
    # 3) цифры словами: «двенадцать ноль три семьдесят восемь»
    return _try_dob_numbers(tu.parse_number_words(toks), today)


def _day_before(toks: list[str], month_idx: int) -> tuple[int, int] | None:
    """Число месяца из токенов перед названием месяца
    («двенадцатое», «двадцать первого»). → (число, сколько токенов занято)."""
    if month_idx == 0:
        return None
    prev = toks[month_idx - 1]
    v = _numeric_value(prev)
    if v is None:
        return None
    if month_idx >= 2:
        prev2 = toks[month_idx - 2]
        v2 = _numeric_value(prev2)
        if v2 is not None and 20 <= v2 <= 90 and v2 % 10 == 0:
            return v2 + v, 2          # «двадцать первого»
    return v, 1


def _sum_year_tokens(nums: list[int]) -> int | None:
    """[70, 8] → 78; [2, 1000, 19] → 2019; [1978] → 1978."""
    if not nums:
        return None
    total = 0
    for v in nums:
        if v == 1000:
            total = (total or 1) * 1000
        else:
            total += v
    return total


# ---------------------------------------------------------------------------
# Полис ОМС

def parse_policy_no(text: str) -> str | None:
    """16 цифр из реплики (цифрами или словами)."""
    digits = re.sub(r"\D", "", text)
    if len(digits) == 16:
        return digits
    toks = tu.normalize(text).split()
    if toks and all(_is_number_token(t) for t in toks):
        digit_str = "".join(str(n) for n in tu.parse_number_words(toks))
        if len(digit_str) == 16:
            return digit_str
    return None


# ---------------------------------------------------------------------------
# Желаемые дата/время (А.3)

_DAYPARTS = [
    ("morning", ("утром", "с утра", "утро")),
    ("afternoon", ("после обеда", "днем", "день", "в обед")),
    ("evening", ("вечером", "вечер", "после работы")),
    ("early", ("пораньше", "по раньше", "с самого утра")),
    ("late", ("попозже", "по позже", "позднее")),
]


def parse_time_prefs(text: str, today: date) -> TimePrefs | None:
    n = tu.normalize(text)
    prefs = TimePrefs()
    found = False
    # исключения: «не в пятницу», «кроме субботы» — день не выбираем
    excluded_forms: set[str] = set()
    for base, wd in tu._WEEKDAY_BASES.items():
        for suf in ("", "а", "у", "е", "и", "ю", "я"):
            tok = base + suf
            if f"не {tok}" in n or f"не в {tok}" in n or f"кроме {tok}" in n:
                prefs.exclude_weekday = wd
                excluded_forms.add(tok)
    if "послезавтра" in n:
        prefs.date = today + timedelta(days=2)
        found = True
    elif "завтра" in n:
        prefs.date = today + timedelta(days=1)
        found = True
    elif "сегодня" in n:
        prefs.date = today
        found = True
    if "на следующей неделе" in n or "на будущей неделе" in n:
        prefs.next_week = True
        found = True
    if re.search(r"через недел", n):
        prefs.in_days = 7
        found = True
    if not found or prefs.date is None:
        for tok in n.split():
            if tok in excluded_forms:
                continue
            wd = tu.weekday_of_token(tok)
            if wd is not None:
                prefs.weekday = wd
                found = True
                break
    for key, variants in _DAYPARTS:
        if any(v in n for v in variants):
            prefs.daypart = key
            found = True
            break
    if not found:
        return None
    if prefs.next_week:
        monday = tu.next_monday(today)
        prefs.date = monday + timedelta(days=prefs.weekday or 0)
    elif prefs.in_days == 7:
        prefs.date = today + timedelta(days=7)
    elif prefs.weekday is not None and prefs.date is None:
        prefs.date = tu.resolve_weekday(today, prefs.weekday)
    return prefs


def daypart_of(text: str) -> str | None:
    """Только часть суток из реплики («а после обеда есть?»)."""
    n = tu.normalize(text)
    for key, variants in _DAYPARTS:
        if any(v in n for v in variants):
            return key
    return None


# ---------------------------------------------------------------------------
# Сопоставление реплики с предложенными вариантами (≤3 слота, правило 3.2)

def option_stems(slot_date: date, slot_time,
                 today: date | None = None,
                 specialty: str | None = None) -> set[str]:
    """Ключи-основы варианта: специальность, день недели, число месяца,
    час, минуты, «завтра»/«послезавтра» для близких дат."""
    st = {tu.stem(tu.WEEKDAYS_NOM[slot_date.weekday()]),
          tu.stem(tu._ORD_NOM[slot_date.day]),
          tu.stem(tu._ORD_GEN[slot_date.day])}
    if specialty is not None:
        from .clinic_data import SPECIALTIES
        st.add(tu.stem(SPECIALTIES[specialty]["nom"]))
        st.add(tu.stem(SPECIALTIES[specialty]["dat"]))
    if today is not None:
        delta = (slot_date - today).days
        if delta == 1:
            st.add(tu.stem("завтра"))
        elif delta == 2:
            st.add(tu.stem("послезавтра"))
    for w in (tu.number_words(slot_time.hour), tu.number_words(slot_time.minute)):
        st.add(tu.stem(w))
    return st


def match_option(options: list, text: str,
                 today: date | None = None) -> int | None:
    """Индекс выбранного варианта или None. «давайте пятницу» → вариант
    с пятницей; «завтра» → вариант на завтра; «второй» → позиция в списке
    (если не совпала дата)."""
    ws = tu.stems(text)
    date_hit = _date_option_hit(options, ws, today)
    if date_hit is not None:
        return date_hit
    # порядковый номер варианта («второй, пожалуйста»)
    ords = {"перв": 0, "втор": 1, "трет": 2}
    if "вариант" in tu.normalize(text) or len(ws) <= 2:
        for prefix, idx in ords.items():
            if any(w.startswith(prefix) for w in ws) and idx < len(options):
                return idx
    return None


def _date_option_hit(options: list, ws: list[str],
                     today: date | None) -> int | None:
    wset = set(ws)
    best, best_score = None, 0
    for i, opt in enumerate(options):
        keys = option_stems(opt.date, opt.time, today,
                           specialty=getattr(opt, "specialty", None))
        score = len(keys & wset)
        if score > best_score:
            best, best_score = i, score
    return best if best_score >= 1 else None
