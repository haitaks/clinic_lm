# -*- coding: utf-8 -*-
"""Семантические фреймы и их сэмплирование.

Фрейм — канонический смысл заявления. Метка (target) строится ТОЛЬКО
из фрейма, поэтому она точна по построению. Схема target:

  <INTENT> <SPECIALTY>
  <TIME> [T_SCOPE] [T_DAY] [T_TIME] </TIME | <T_UNKNOWN>
  <SYMPTOM_*>... [<NEEDS_CLARIFICATION>] <END_FRAME>

Время выводится фактор-токенами (таблица TEMPORAL_FACTORS в vocab.py);
сырая фраза (frame.temporal) остаётся в записи как якорь валидации
(вербатимная проверка) и материал для UI-подтверждения.
"""

from dataclasses import dataclass, field

from vocab import (SPECIALTIES, SYMPTOMS, PLAUSIBLE, OOD_PAIRS,
                   INTENT_TOKENS, UNSUPPORTED_POOLS, TEMPORALS,
                   TEMPORAL_FACTORS, temporal_factors_of,
                   AMBIG_DAY_SURFACES, SLOT_SURFACES)

SPEC_TOKEN_UNKNOWN = "<UNKNOWN_SPECIALTY>"
TIME_OPEN, TIME_CLOSE, TIME_UNKNOWN = "<TIME>", "</TIME>", "<T_UNKNOWN>"
# зарезервирован для конфликтов вида "завтра или в пятницу" (в синтетике v1
# не генерируется, нужен constrained decoding'у и разметке реальных данных)
TIME_AMBIG = "<T_AMBIG>"
CLARIF_TOKEN, END_TOKEN = "<NEEDS_CLARIFICATION>", "<END_FRAME>"

# 60% фреймов с временной фразой — реалистичная доля для заявок.
TEMPORAL_PROBA = 0.6

# Доля «или»-альтернатив среди временных фраз («завтра в 15:00 или
# в 15:30» -> факторы дня + T_AMBIG: конкретный слот выбирает UI).
AMBIG_PROBA = 0.08


def _sample_temporal(rng, proba: float = TEMPORAL_PROBA) -> str | None:
    if rng.random() >= proba:
        return None
    if rng.random() < AMBIG_PROBA:
        i = rng.randrange(len(SLOT_SURFACES) - 2)
        j = i + 1 + rng.randrange(2)
        day = rng.choice(AMBIG_DAY_SURFACES + [None])
        s = f"{SLOT_SURFACES[i]} или {SLOT_SURFACES[j]}"
        return f"{day} {s}" if day else s
    return rng.choice(TEMPORALS)


@dataclass
class Frame:
    intent: str                    # book | reschedule | cancel | unsupported
    specialty: str | None          # ключ SPECIALTIES | "unknown" | None
    symptoms: tuple                # ключи SYMPTOMS
    needs_clarification: bool
    temporal: str | None = None    # сырая фраза (поверхность)
    ood: bool = False              # использует только hold-out поверхности
    extra_mentions: tuple = ()     # спец-ключи, разрешённые в тексте сверх метки
    aux: dict = field(default_factory=dict)  # подсказки рилайзеру (напр. пул unsupported)

    @property
    def temporal_factors(self) -> tuple:
        """Фактор-токены слота времени: точная таблица или «или»-парсер
        (день + T_AMBIG)."""
        if not self.temporal:
            return ()
        return temporal_factors_of(self.temporal)

    @property
    def conflict(self) -> bool:
        """Выбор между двумя специальностями -> UNKNOWN + уточнение."""
        return bool(self.extra_mentions) and self.intent != "unsupported"


def spec_token(spec: str | None) -> str:
    if spec in SPECIALTIES:
        return SPECIALTIES[spec]["token"]
    return SPEC_TOKEN_UNKNOWN


def target_string(frame: Frame) -> str:
    if frame.intent == "unsupported":
        return f"{INTENT_TOKENS['unsupported']} {END_TOKEN}"
    toks = [INTENT_TOKENS[frame.intent], spec_token(frame.specialty)]
    if frame.temporal:
        factors = frame.temporal_factors
        if not factors:
            raise ValueError(
                f"у временной фразы {frame.temporal!r} нет фактор-аннотации "
                f"в TEMPORAL_FACTORS")
        toks += [TIME_OPEN, *factors, TIME_CLOSE]
    else:
        toks.append(TIME_UNKNOWN)
    toks += [SYMPTOMS[s]["token"] for s in frame.symptoms]
    if frame.needs_clarification:
        toks.append(CLARIF_TOKEN)
    toks.append(END_TOKEN)
    return " ".join(toks)


def frame_key(frame: Frame):
    """Ключ группировки парафраз: один фрейм -> один сплит (без утечки).

    Для unsupported в ключ входит пул и вариант, иначе все "часы работы"
    схлопнутся в одну группу и целиком уедут в один сплит.
    """
    if frame.intent == "unsupported":
        return ("unsupported", frame.aux.get("pool"),
                frame.aux.get("variant"))
    return (frame.intent, frame.specialty, frame.symptoms,
            frame.needs_clarification, tuple(sorted(frame.extra_mentions)))


def _sample_symptoms(rng, spec: str, p_none: float, p_two: float):
    pool = PLAUSIBLE[spec]
    r = rng.random()
    if r < p_none or not pool:
        return ()
    if r < p_none + p_two and len(pool) >= 2:
        return tuple(rng.sample(pool, 2))
    return (rng.choice(pool),)


def _sample_book(rng, ood: bool) -> Frame:
    weights = [2.0 if s == "gp" else 1.0 for s in SPECIALTIES]
    spec = rng.choices(list(SPECIALTIES), weights=weights)[0]
    if ood and any(s == spec for s, _ in OOD_PAIRS) and rng.random() < 0.6:
        symptoms = (rng.choice([sym for s, sym in OOD_PAIRS if s == spec]),)
    else:
        symptoms = _sample_symptoms(rng, spec, p_none=0.40, p_two=0.15)
    temporal = _sample_temporal(rng)
    return Frame("book", spec, symptoms, False, temporal, ood)


def _sample_clarification(rng) -> Frame:
    if rng.random() < 0.55:
        # "запишите к врачу, не знаю к кому" -> UNKNOWN + уточнение
        symptoms = (rng.choice(list(SYMPTOMS)),) if rng.random() < 0.6 else ()
        temporal = _sample_temporal(rng)
        return Frame("book", "unknown", symptoms, True, temporal)
    # конфликт двух специальностей: "к терапевту или к дерматологу?"
    a, b = rng.sample(list(SPECIALTIES), 2)
    symptoms = ()
    temporal = _sample_temporal(rng)
    return Frame("book", "unknown", symptoms, True, temporal,
                 extra_mentions=(a, b))


def _sample_reschedule(rng) -> Frame:
    spec = rng.choices(list(SPECIALTIES) + ["unknown"],
                       weights=[1.0] * len(SPECIALTIES) + [0.35])[0]
    symptoms = _sample_symptoms(rng, spec, p_none=0.65, p_two=0.0) \
        if spec in SPECIALTIES else ()
    temporal = _sample_temporal(rng, proba=0.75)
    return Frame("reschedule", spec, symptoms, False, temporal)


def _sample_cancel(rng) -> Frame:
    if rng.random() < 0.3:
        # «отмени все мои записи» — отмена вообще, без специальности
        # (раньше дыра: модель не знала формулировку и говорила UNSUPPORTED)
        return Frame("cancel", "unknown", (), False, None,
                     aux={"cancel_all": True})
    spec = rng.choices(list(SPECIALTIES) + ["unknown"],
                       weights=[1.0] * len(SPECIALTIES) + [0.6])[0]
    symptoms = _sample_symptoms(rng, spec, p_none=0.75, p_two=0.0) \
        if spec in SPECIALTIES else ()
    # у cancel слот времени в v1 не заполняем: "завтра" тут относится
    # к отменяемой записи, а не к желаемому времени
    return Frame("cancel", spec, symptoms, False, None)


def _sample_unsupported(rng) -> Frame:
    pool = rng.choice(list(UNSUPPORTED_POOLS))
    variant = rng.randrange(len(UNSUPPORTED_POOLS[pool]))
    return Frame("unsupported", None, (), False, None,
                 aux={"pool": pool, "variant": variant})


# распределение интентов: BOOK доминирует, отрицательный класс заметный
_INTENT_PLAN = [
    ("book", 0.50, _sample_book),
    ("clar", 0.08, _sample_clarification),
    ("reschedule", 0.12, _sample_reschedule),
    ("cancel", 0.10, _sample_cancel),
    ("unsupported", 0.10, _sample_unsupported),
]


def sample_frame(rng, ood: bool = False) -> Frame:
    if ood:
        return _sample_book(rng, ood=True)
    r = rng.random()
    acc = 0.0
    for _, w, fn in _INTENT_PLAN:
        acc += w
        if r < acc:
            return fn(rng) if fn is not _sample_book else _sample_book(rng, False)
    return _sample_book(rng, False)
