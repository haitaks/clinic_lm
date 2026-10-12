# -*- coding: utf-8 -*-
"""Routing «симптомы -> врач»: рекомендация специальности, когда в
запросе она не названа.

Инверсия PLAUSIBLE (synth/vocab.py — какие симптомы уместны у какого
врача; эта же матрица управляет генерацией, так что рекомендация
согласована с обучающими данными). Правила:
- несколько симптомов -> пересечение их списков (боль в глазу + голова
  -> офтальмолог);
- пустое пересечение -> голосование по объединению;
- порядок кандидатов: от узкого специалиста к терапевту
  (специфичность = чем меньше симптомов в профиле специальности, тем
  она специализированнее; gp — широкий профиль, всегда последний);
- без симптомов -> gp (первичное звено).

Фрейм-строка НЕ меняется: routing — надстройка для UI (primary +
alternatives + причина), вызывается при specialty == UNKNOWN_SPECIALTY.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "synth"))

from vocab import SPECIALTIES, PLAUSIBLE  # noqa: E402

# симптом -> специальности, у которых он в профиле (train-матрица)
SYM_TO_SPECS = {}
for spec, symptoms in PLAUSIBLE.items():
    for sym in symptoms:
        SYM_TO_SPECS.setdefault(sym, []).append(spec)

# Приоритет врача для МНОГОЗНАЧНЫХ симптомов (порядок = порядок
# предложения в UI). Автоматика (специфичность) даёт медицински
# спорные ответы вроде «головная боль -> гинеколог по алфавиту»,
# поэтому многозначные случаи настраиваются руками; однозначные
# (toothache -> dentist) покрываются пересечением без таблицы.
# Это точка настройки под конкретную клинику.
SYM_PRIORITY = {
    "headache": ["neurologist", "gp", "ophthalmologist", "gynecologist"],
    "stomach_pain": ["gp", "surgeon", "gynecologist"],
    "back_pain": ["neurologist", "surgeon", "gp"],
    "eye_pain": ["ophthalmologist", "gp"],
    "rash": ["dermatologist", "gp"],
    "allergy": ["dermatologist", "ent", "gp"],
    "sore_throat": ["ent", "gp"],
    "cough": ["ent", "gp"],
    "fever": ["gp"],
    "dizziness": ["neurologist", "gp"],
}


def _order(sym, specs):
    """Порядок кандидатов: таблица приоритетов симптома главная
    (в ней gp может быть и первым — «живот -> терапевт»); вне таблицы
    кандидаты идут в конец, среди них gp последним, затем по
    специфичности профиля."""
    prio = SYM_PRIORITY.get(sym, [])

    def key(s):
        if s in prio:
            return (0, prio.index(s))
        return (1, 1 if s == "gp" else 0, len(PLAUSIBLE[s]))

    return sorted(specs, key=key)


def route_specialty(sym_keys: list[str]) -> dict:
    """-> {primary, alternatives, reason}; sym_keys — канон-ключи
    SYMPTOMS из сборки фрейма."""
    if not sym_keys:
        return {"primary": "gp", "alternatives": [],
                "reason": "no_symptoms"}

    sets = [set(SYM_TO_SPECS.get(s, ())) for s in sym_keys]
    inter = set.intersection(*sets) if sets else set()
    if inter:
        ordered = _order(sym_keys[0], inter)
        return {"primary": ordered[0],
                "alternatives": ordered[1:],
                "reason": "intersection"}
    union = set().union(*sets) if sets else set()
    if union:
        ordered = _order(sym_keys[0], union)
        return {"primary": ordered[0],
                "alternatives": ordered[1:],
                "reason": "vote"}
    # симптомы вне матрицы (OOD-пары): дефолт первичного звена
    return {"primary": "gp", "alternatives": [],
            "reason": "unknown_symptoms"}


def route_from_frame(frame: str) -> dict | None:
    """Фрейм-строка -> рекомендация, если специальность не названа.
    Только для book/reschedule: у cancel без специальности («отмени
    все мои записи») рекомендация врача неуместна."""
    toks = frame.split()
    if len(toks) < 2 or toks[0] not in ("<BOOK>", "<RESCHEDULE>"):
        return None
    if toks[1] != "<UNKNOWN_SPECIALTY>":
        return None
    from vocab import SYMPTOMS
    sym_key_by_token = {e["token"]: k for k, e in SYMPTOMS.items()}
    syms = [t for t in toks if t.startswith("<SYMPTOM_")]
    keys = [sym_key_by_token[t] for t in syms if t in sym_key_by_token]
    if not keys:
        return None                      # без симптомов — дефолт UI
    r = route_specialty(keys)
    return {"symptoms": syms, **r,
            "primary_token": SPECIALTIES[r["primary"]]["token"]}


def _demo():
    cases = [
        (["toothache"], "однознозначно"),
        (["anxiety"], None),
        (["eye_pain", "headache"], "пересечение -> офтальмолог"),
        (["back_pain"], None),
        (["headache"], None),
        (["stomach_pain"], None),
        (["cough", "sore_throat"], "пересечение -> ent/gp"),
        ([], "без симптомов"),
    ]
    for syms, note in cases:
        r = route_specialty(syms)
        print(f"{str(syms):38s} -> {r['primary']:16s} "
              f"alt={r['alternatives']} {r['reason']}"
              + (f"   # {note}" if note else ""))


if __name__ == "__main__":
    _demo()
