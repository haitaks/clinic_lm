# -*- coding: utf-8 -*-
"""Авторазметка датасета для тегированной архитектуры NLU.

Слова запроса размечаются тегами {O, SPEC, SYM, TIME} по словарям
иголок (синонимы специальностей во всех падежах, выражения симптомов,
вербатим-спан времени из frame.temporal). Интент — отдельная метка
фразы: он классифицируется по высказыванию целиком, не по словам
(интент часто не выражен отдельным словом, есть отрицания).

Разметка проецируется на BPE-токены (offsets из HF tokenizers).
Round-trip контроль: assemble.assemble_from_tags обязан воспроизводить
target исходной записи.

Запуск:  .venv-tr/Scripts/python tagger/annotate.py
Выход:   tagger/data/{split}.tags.jsonl + статистика в консоль.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "synth"))

from vocab import SPECIALTIES, SYMPTOMS  # noqa: E402

TAGS = ["O", "SPEC", "SYM", "TIME"]
TAG_IX = {t: i for i, t in enumerate(TAGS)}
INTENTS = ["book", "reschedule", "cancel", "unsupported"]
INTENT_IX = {t: i for i, t in enumerate(INTENTS)}

# иголки: канон-ключ -> множество поверхностей (с ё-фолдингом)
SPEC_NEEDLES = {
    spec: {f.replace("ё", "е") for syn in entry["train"] + entry["ood"]
           for f in syn.values()}
    for spec, entry in SPECIALTIES.items()}
SYM_NEEDLES = {
    sym: {e.replace("ё", "е") for e in entry["train"] + entry["ood"]}
    for sym, entry in SYMPTOMS.items()}


def needle_re(needle: str) -> re.Pattern:
    """Границы слов + хвост-склонение до 3 букв для однословных иголок
    (повторяет логику synth/validate.py:_contains)."""
    esc = re.escape(needle)
    if " " in needle:
        return re.compile(r"(?<![а-яёa-z])" + esc + r"(?![а-яёa-z])")
    return re.compile(r"(?<![а-яёa-z])" + esc + r"[а-яё]{0,3}(?![а-яёa-z])")


SPEC_RES = [(spec, needle_re(n)) for spec, ns in SPEC_NEEDLES.items() for n in ns]
SYM_RES = [(sym, needle_re(n)) for sym, ns in SYM_NEEDLES.items() for n in ns]


def fold(text: str) -> str:
    """Нижний регистр + ё->е, длина сохраняется (offsets остаются валидны)."""
    return text.lower().replace("ё", "е")


def find_spans(text: str, with_time=True, temporal=None):
    """Спаны (start, end, tag, canon) по позициям СИМВОЛОВ текста.
    Лево-жадный скан: слева направо, в каждой позиции берётся самый
    длинный матч, скан прыгает за его конец. Так перечисления
    "живот болит голова болит" не теряют первый симптом из-за
    перекрывающихся двухсловных иголок ("болит голова" начинается
    внутри "живот болит").
    with_time=False — инференс-режим без знания gold temporal."""
    f = fold(text)
    raw = []
    for canon, rx in SPEC_RES:
        for m in rx.finditer(f):
            raw.append((m.start(), m.end(), "SPEC", canon))
    for canon, rx in SYM_RES:
        for m in rx.finditer(f):
            raw.append((m.start(), m.end(), "SYM", canon))
    if with_time and temporal:
        n = fold(temporal)
        m = needle_re(n).search(f)
        if m:
            raw.append((m.start(), m.end(), "TIME", None))
    by_start = {}
    for s in raw:
        by_start.setdefault(s[0], []).append(s)
    for s in by_start.values():
        s.sort(key=lambda x: -(x[1] - x[0]))
    spans, pos, n = [], 0, len(f)
    while pos < n:
        cands = by_start.get(pos)
        if cands:
            best = cands[0]
            spans.append(best)
            pos = best[1]
        else:
            pos += 1
    return spans


def annotate(text: str, frame: dict, tok: Tokenizer):
    """-> dict(ids, tags, intent, spans) для обучения теггера."""
    enc = tok.encode(text)
    spans = find_spans(text, with_time=True, temporal=frame.get("temporal"))
    tags = []
    for start, end in enc.offsets:
        hit = next((t for s, e, t, _ in spans if start < e and s < end), "O")
        tags.append(TAG_IX[hit])
    return {"ids": enc.ids, "tags": tags,
            "intent": INTENT_IX[frame["intent"]],
            "spans": [[s, e, t, c] for s, e, t, c in spans]}


def load_rows(split: str):
    src = ROOT / "data" / f"{split}.jsonl"
    rows = [json.loads(l) for l in
            src.read_text(encoding="utf-8").splitlines()]
    if split in ("hasd_ru", "hasd_unsupported_ru", "clinc_ru"):
        rows = [r for r in rows if r.get("split") == "ext_test"]
    return rows


def _clinc(part: str):
    src = ROOT / "data" / "clinc_ru.jsonl"
    return [json.loads(l) for l in
            src.read_text(encoding="utf-8").splitlines()
            if l and json.loads(l).get("split") == part]


def main():
    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(exist_ok=True)

    total = 0
    tag_counter, intent_counter = Counter(), Counter()
    for split in ("train", "val", "test_iid", "test_ood", "manual_gold",
                  "hasd_ru", "hasd_unsupported_ru", "clinc_ext_test",
                  "clinc_train_aug"):
        if split == "clinc_ext_test":
            rows = _clinc("ext_test")
        elif split == "clinc_train_aug":
            rows = _clinc("train_aug")
        else:
            rows = load_rows(split)
        out = []
        for r in rows:
            ann = annotate(r["text"], r["frame"], tok)
            out.append({"text": r["text"], "target": r["target"], **ann})
            total += 1
            tag_counter.update(TAGS[t] for t in ann["tags"])
            intent_counter[INTENTS[ann["intent"]]] += 1
        path = out_dir / f"{split}.tags.jsonl"
        path.write_text("\n".join(json.dumps(o, ensure_ascii=False)
                                  for o in out) + "\n", encoding="utf-8")
        print(f"{split:20s} {len(out):6d} записей -> {path.name}")

    print(f"\nвсего: {total}")
    print("теги:", dict(tag_counter))
    print("интенты:", dict(intent_counter))


if __name__ == "__main__":
    main()
