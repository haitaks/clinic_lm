# -*- coding: utf-8 -*-
"""Детерминированная сборка semantic frame из предсказаний теггера.

Вход: интент-класс + последовательность (bpe_id, tag). Правила выведены
из данных генератора и проверены round-trip (см. round_trip_check):
- unsupported -> <UNSUPPORTED> <END_FRAME>
- 2+ специальности -> <UNKNOWN_SPECIALTY> + NEEDS_CLARIFICATION
- 0 специальностей -> <UNKNOWN_SPECIALTY> (+ NEEDS_CLARIFICATION для
  book/reschedule; cancel без специальности уточнения не требует)
- время: текст TIME-спана -> факторы по TEMPORAL_FACTORS; нет спана
  или не распознан -> <T_UNKNOWN>
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "synth"))

from vocab import (SPECIALTIES, SYMPTOMS, TEMPORAL_FACTORS,
                   PLAUSIBLE)  # noqa: E402
from annotate import TAGS, TAG_IX, INTENTS, fold, needle_re  # noqa: E402

SPEC_TOKEN = {k: v["token"] for k, v in SPECIALTIES.items()}
SYM_TOKEN = {k: v["token"] for k, v in SYMPTOMS.items()}
TIME_RE = {fold(t): needle_re(t) for t in TEMPORAL_FACTORS}


def _match_in(surface: str, needles_by_key: dict) -> str | None:
    f = fold(surface)
    for key, surfaces in needles_by_key.items():
        for n in surfaces:
            if needle_re(n).search(f):
                return key
    return None


# канонизация спанов: словари поверхностей (train+ood, все падежи)
_SPEC_SURF = {
    k: {f.replace("ё", "е") for syn in v["train"] + v["ood"]
        for f in syn.values()} for k, v in SPECIALTIES.items()}
_SYM_SURF = {
    k: {e.replace("ё", "е") for e in v["train"] + v["ood"]}
    for k, v in SYMPTOMS.items()}


_TIME_RANK = {"scope": 0, "day": 1, "time": 2}


def _time_cat(tok: str) -> str:
    from vocab import TIME_SCOPE, TIME_DAY
    if tok in TIME_SCOPE:
        return "scope"
    if tok in TIME_DAY:
        return "day"
    return "time"


def time_factors(surface: str) -> list[str] | None:
    """Текст TIME-спана -> фактор-токены. Точный матч по таблице;
    если спан чуть шире (тег протянут) — ищем известные фразы внутри
    и объединяем факторы по правилу категорий scope -> day -> time."""
    key = fold(surface.strip())
    if key in TEMPORAL_FACTORS:
        return TEMPORAL_FACTORS[key]
    f = fold(surface)
    found = []
    for phrase in sorted(TEMPORAL_FACTORS, key=len, reverse=True):
        m = needle_re(phrase).search(f)
        if m and all(not (m.start() < e2 and s2 < m.end()) for s2, e2 in found):
            found.append((m.start(), m.end()))
            f = f[:m.start()] + " " * (m.end() - m.start()) + f[m.end():]
    found.sort()
    factors, seen_cat = [], set()
    for s, e in found:
        facs = TEMPORAL_FACTORS[fold(surface[s:e].strip())]
        for t in facs:
            if _time_cat(t) not in seen_cat:
                factors.append(t)
                seen_cat.add(_time_cat(t))
    return factors or None


def assemble(intent: str, spec_keys: list[str], sym_keys: list[str],
             time_text: str | None) -> str:
    if intent == "unsupported":
        return "<UNSUPPORTED> <END_FRAME>"
    parts = [f"<{intent.upper()}>"]
    clarif = False
    if len(spec_keys) >= 2:                      # конфликт "или"
        parts.append("<UNKNOWN_SPECIALTY>")
        clarif = True
    elif len(spec_keys) == 1:
        parts.append(SPEC_TOKEN[spec_keys[0]])
    else:
        parts.append("<UNKNOWN_SPECIALTY>")
        # эмпирика train: book без специальности всегда с уточнением,
        # reschedule/cancel без специальности — никогда
        if intent == "book":
            clarif = True
    factors = time_factors(time_text) if time_text else None
    if factors:
        parts.append("<TIME> " + " ".join(factors) + " </TIME>")
    else:
        parts.append("<T_UNKNOWN>")
    for s in sym_keys:
        parts.append(SYM_TOKEN[s])
    if clarif:
        parts.append("<NEEDS_CLARIFICATION>")
    parts.append("<END_FRAME>")
    return " ".join(parts)


def canon_span(surface: str, tag: str) -> list[str]:
    """Канон-ключи внутри спана: словарный лево-жадный скан. Подряд
    идущие симптомы без O-разрыва ("голова раскалывается аллергия")
    режутся на отдельные ключи, а не схлопываются в первый матч."""
    from annotate import find_spans
    keys = []
    for _, _, t, canon in find_spans(surface, with_time=False):
        if t == tag and canon not in keys:
            keys.append(canon)
    return keys


def frame_from_tags(text: str, ids, tags, intent_ix: int,
                    tok=None, spans=None, resolver=None):
    """Полный инференс-путь: (bpe, tag) -> спаны -> фрейм-строка.

    spans можно передать готовыми (gold-разметка для round-trip),
    иначе спаны восстанавливаются из тегов по границам слов текста.
    resolver: callable(text_span, tag) -> canon | None — e5-резолвер
    для OOV-спанов (подключается на этапе 5)."""
    intent = INTENTS[intent_ix]

    if spans is None:
        # инференс: спаны из тегов модели + словарный скан. Словарь
        # добавляет SPEC/SYM, которые сеть не узнала (невидимые
        # синонимы: "кожник" — иголки включают ood-формы). TIME из
        # словаря НЕ берём: во временных фразах негативов ("какая
        # завтра погода") он ложноположителен — время только по тегу.
        model_spans = spans_from_tags(text, tags, tok)
        from annotate import find_spans
        dict_spans = find_spans(text, with_time=False)
        merged = {(s, e, t) for s, e, t, _ in dict_spans}
        merged |= {(s, e, t) for s, e, t, _ in model_spans}
        spans = sorted((s, e, t,
                        next((c for ds, de, dt, c in dict_spans
                              if (ds, de, dt) == (s, e, t)), None))
                       for s, e, t in merged)
    specs, syms, times = [], [], []
    for s, e, t, canon in spans:
        if t in ("SPEC", "SYM"):
            keys = ([canon] if canon else canon_span(text[s:e], t))
            if not keys and resolver:
                key = resolver(text[s:e], t)
                keys = [key] if key else []
            target = specs if t == "SPEC" else syms
            for k in keys:
                if k and k not in target:
                    target.append(k)
        elif t == "TIME":
            times.append(text[s:e])
    # противоречие: unsupported, но в тексте есть специальность/симптом
    # (настоящие негативы — пулы UNSUPPORTED и CLINC — иголок не
    # содержат, что гарантирует валидатор генератора); невидимая
    # формулировка запроса не может сделать из "к кожнику" чепуху
    if intent == "unsupported" and (specs or syms):
        intent = "book"
    # после ревизии интента словарное время безопасно: у негативов
    # ("какая завтра погода") интент остался unsupported и время
    # не мержится; для book/reschedule/cancel фразы времени настоящие
    if intent != "unsupported" and not times:
        from vocab import TEMPORAL_FACTORS as _TF
        for phrase in sorted(_TF, key=len, reverse=True):
            m = needle_re(phrase).search(fold(text))
            if m:
                times.append(text[m.start():m.end()])
                break
    time_text = times[0] if times else None
    return assemble(intent, specs, syms, time_text)


def spans_from_tags(text: str, tags, tok):
    """Восстановление спанов из тегов BPE-токенов: подряд идущие токены
    с одним тегом сливаются в спан (offsets из токенизатора)."""
    if tok is None:
        return []
    enc = tok.encode(text)
    if len(enc.ids) != len(tags):
        return []
    spans, cur = [], None
    for (start, end), t in zip(enc.offsets, tags):
        name = TAGS[t] if isinstance(t, int) else t
        if name == "O":
            continue
        if cur and cur[2] == name and start <= cur[1]:
            cur = (cur[0], max(cur[1], end), name, None)
        else:
            if cur:
                spans.append(cur)
            cur = (start, end, name, None)
    if cur:
        spans.append(cur)
    return [(s, e, t, c) for s, e, t, c in spans]


def round_trip_check():
    """Сборка из gold-разметки обязана совпадать с target датасета."""
    data_dir = Path(__file__).resolve().parent / "data"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    for path in sorted(data_dir.glob("*.tags.jsonl")):
        rows = [json.loads(l) for l in
                path.read_text(encoding="utf-8").splitlines() if l.strip()]
        bad = []
        for r in rows:
            got = frame_from_tags(r["text"], r["ids"], r["tags"],
                                  r["intent"], tok=tok,
                                  spans=[tuple(s) for s in r["spans"]])
            if got != r["target"]:
                bad.append((r["text"], r["target"], got))
        name = path.stem.replace(".tags", "")
        print(f"{name:20s} round-trip: {len(rows) - len(bad)}/{len(rows)}"
              f"{'' if not bad else '  ПРИМЕРЫ РАСХОЖДЕНИЙ:'}")
        for t, gold, got in bad[:3]:
            print(f"   text: {t}\n   gold: {gold}\n   got : {got}")


if __name__ == "__main__":
    round_trip_check()
