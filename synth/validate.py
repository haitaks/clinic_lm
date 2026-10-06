# -*- coding: utf-8 -*-
"""Round-trip валидация: текст обязан соответствовать своему фрейму.

Проверяет согласованность сгенерированного (или пришедшего от LLM)
текста с меткой:
- каждая помеченная специальность упомянута в тексте;
- чужие специальности/симптомы не упомянуты (кроме extra_mentions —
  например, конфликт "к терапевту или к дерматологу");
- временная фраза слота <TIME> присутствует в тексте вербатим, а при
  <TIME_UNKNOWN> в тексте нет фраз из TEMPORALS.

Также self_check() ловит пересечения "иголок" в словаре: если одна
метка — подстрока другой, валидация начнёт ложно срабатывать.
"""

import re

from vocab import (SPECIALTIES, SYMPTOMS, TEMPORALS, REQUESTS,
                   GREETINGS, POLITENESS, UNSUPPORTED_POOLS,
                   UNSUPPORTED_LEADS, UNSUPPORTED_LEADS_NO_GREET,
                   TEMPORAL_ENTRIES, TIME_FACTORS, TIME_SCOPE, TIME_DAY)

_spec_needles = {
    spec: {form for syn in entry["train"] + entry["ood"] for form in syn.values()}
    for spec, entry in SPECIALTIES.items()
}
_sym_needles = {
    sym: set(entry["train"] + entry["ood"])
    for sym, entry in SYMPTOMS.items()
}


def normalize(text: str) -> str:
    # фолдинг ё->е: STT и машинный перевод обычно выдают "е",
    # а словари могут содержать "ё" (например, "днём")
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def _contains(needle: str, normalized_text: str) -> bool:
    """Поиск иголки с границами слов: "днем" не должно матчиться
    внутри "последнем". Для однословных иголок разрешён хвост-склонение
    до 3 букв ("стоматолог" находит "с стоматологом"), приставки
    ("последнем") по-прежнему запрещены lookbehind'ом."""
    n = needle.replace("ё", "е")
    esc = re.escape(n)
    if " " in n:
        pattern = r"(?<![а-яёa-z])" + esc + r"(?![а-яёa-z])"
    else:
        pattern = r"(?<![а-яёa-z])" + esc + r"[а-яё]{0,3}(?![а-яёa-z])"
    return re.search(pattern, normalized_text) is not None


def validate_text(frame, text: str) -> list[str]:
    """Список причин несоответствия (пустой = всё хорошо)."""
    t = normalize(text)
    reasons = []

    allowed_specs = set(frame.extra_mentions)
    if frame.specialty in SPECIALTIES:
        allowed_specs.add(frame.specialty)

    for spec, needles in _spec_needles.items():
        present = any(_contains(n, t) for n in needles)
        if spec in allowed_specs:
            if spec == frame.specialty and not present:
                reasons.append(f"specialty {spec} не упомянута")
        elif present:
            reasons.append(f"чужая специальность {spec} в тексте")

    for sym, exprs in _sym_needles.items():
        present = any(_contains(n, t) for n in exprs)
        if sym in frame.symptoms and not present:
            reasons.append(f"симптом {sym} не упомянут")
        elif sym not in frame.symptoms and present:
            reasons.append(f"чужой симптом {sym} в тексте")

    if frame.intent in ("book", "reschedule"):
        if frame.temporal:
            if not _contains(frame.temporal, t):
                reasons.append("временная фраза не скопирована вербатим")
        else:
            found = [c for c in TEMPORALS if _contains(c, t)]
            if found:
                reasons.append(f"временная фраза без метки: {found[0]}")

    return reasons


def self_check():
    """Пересечения "иголок" между разными метками и чужими строками словаря.

    Пример реальной ловушки: короткое "лор" как подстрока в другом слове,
    или "зубной" внутри "зубная боль". Ловится до генерации датасета.
    Также проверяет фактор-аннотации TEMPORAL_ENTRIES: токен существует,
    порядок категорий scope -> day -> time, без дублей категорий.
    """
    problems = []

    rank = {"scope": 0, "day": 1, "time": 2}

    def _cat(tok):
        if tok in TIME_SCOPE:
            return "scope"
        if tok in TIME_DAY:
            return "day"
        return "time"

    for e in TEMPORAL_ENTRIES:
        facs = e["factors"]
        if not facs:
            problems.append(f"temporal '{e['text']}': пустые факторы")
            continue
        bad = [t for t in facs if t not in TIME_FACTORS]
        if bad:
            problems.append(f"temporal '{e['text']}': неизвестные факторы {bad}")
            continue
        cats = [_cat(t) for t in facs]
        if cats != sorted(cats, key=lambda c: rank[c]):
            problems.append(
                f"temporal '{e['text']}': нарушен порядок категорий {facs}")
        if len(set(cats)) != len(cats):
            problems.append(
                f"temporal '{e['text']}': дубликат категории {facs}")

    labelled = [(("spec", s), needles) for s, needles in _spec_needles.items()]
    labelled += [(("sym", s), needles) for s, needles in _sym_needles.items()]

    for i in range(len(labelled)):
        for j in range(i + 1, len(labelled)):
            (ka, na), (kb, nb) = labelled[i], labelled[j]
            if ka == kb:
                continue  # синонимы одной метки могут пересекаться
            for x in na:
                for y in nb:
                    if x == y or x in y or y in x:
                        problems.append(f"пересечение {ka}/{kb}: '{x}' vs '{y}'")

    all_needles = [(k, n) for k, needles in labelled for n in needles]
    aux_strings = []
    for phrases in REQUESTS.values():
        aux_strings += phrases
    aux_strings += GREETINGS + POLITENESS + TEMPORALS
    aux_strings += list(UNSUPPORTED_LEADS) + list(UNSUPPORTED_LEADS_NO_GREET)
    for pool in UNSUPPORTED_POOLS.values():
        aux_strings += pool

    for key, needle in all_needles:
        for aux in aux_strings:
            if needle != aux and needle in aux:
                problems.append(
                    f"иголка {key} '{needle}' — подстрока служебной строки '{aux}'")

    return problems
