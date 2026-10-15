# -*- coding: utf-8 -*-
"""CLINC150: негативы для класса <UNSUPPORTED>.

Запуск (в venv с NLLB):
    .venv-tr/Scripts/python synth/import_clinc.py

Источник: DeepPavlov/clinc150 (24 100 фраз, 150 интентов; OOS-порции в
этой версии нет). Логика: для клинического домена ЛЮБОЙ запрос CLINC150 —
не запись к врачу, поэтому стратифицированная подвыборка по всем интентам
даёт разнообразные негативы: рестораны, погода, будильники, банковские
фразы. По рекомендации обзора (PDF) — не заливать десятки тысяч,
а взять умеренную долю, чтобы negative-класс не задавил booking.

Перевод: тот же детерминированный NLLB-600M (greedy на DirectML),
кэш в data/clinc_translations.json. Round-trip валидатор отсекает
переводы со словами специальностей/симптомов -> в needs_review.

Выход:
    data/clinc_ru.jsonl          — записи source="clinc150": 80% split=train_aug,
                                   20% split=ext_test, target <UNSUPPORTED> <END_FRAME>
    data/clinc_needs_review.jsonl
    data/clinc_stats.json
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
import pandas as pd

from frames import Frame
from validate import validate_text, normalize
from import_hasd import build_translator, translate_batch, apply_glossary

PER_INTENT = 12          # максимум фраз на интент до фильтров
TRAIN_AUG_QUOTA = 1200   # сколько отдать в train-аугментацию

# интенты, ставшие позитивными классами каталога телефонии (после
# расширения словаря из открытых источников): держать их негативами —
# семантический конфликт (например, «yes» размечался UNSUPPORTED)
EXCLUDE_INTENTS = {
    "calendar", "calendar_update", "confirm_reservation",
    "cancel", "cancel_reservation",
    "order_status", "application_status",
    "bill_balance", "bill_due", "min_payment", "pay_bill",
    "report_fraud", "report_lost_card",
    "international_visa", "vaccines",
    "yes", "no", "maybe",
}


def load_clinc():
    parts = []
    for f in ("data/train-00000-of-00001.parquet",
              "data/validation-00000-of-00001.parquet",
              "data/test-00000-of-00001.parquet"):
        p = hf_hub_download("DeepPavlov/clinc150", f, repo_type="dataset")
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    intents = pd.read_parquet(hf_hub_download(
        "DeepPavlov/clinc150", "intents/intents-00000-of-00001.parquet",
        repo_type="dataset"))
    id2name = dict(zip(intents["id"], intents["name"]))
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["label"]):
            continue
        rows.append({"text_en": r["utterance"],
                     "intent": id2name.get(int(r["label"]), "?")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()
    out = Path(args.out)
    rng = random.Random(args.seed)

    rows = load_clinc()
    print(f"CLINC150 загружен: {len(rows)} фраз")

    by_intent = {}
    for r in rows:
        by_intent.setdefault(r["intent"], []).append(r)
    sample = []
    for intent, items in sorted(by_intent.items()):
        if intent in EXCLUDE_INTENTS:
            continue
        rng.shuffle(items)
        sample += items[:PER_INTENT]
    print(f"стратифицированная подвыборка: {len(sample)} фраз "
          f"({PER_INTENT} на интент x {len(by_intent) - len(EXCLUDE_INTENTS)}, "
          f"исключено {len(EXCLUDE_INTENTS)} теперь-позитивных интентов)")

    cache_path = out / "clinc_translations.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [r for r in sample if r["text_en"] not in cache]
    if todo:
        print(f"переводим {len(todo)} фраз (NLLB-600M, greedy)...")
        import time
        t0 = time.time()
        import torch
        tok, model, device, bos, beams = build_translator()
        translated = translate_batch(torch, tok, model, device, bos,
                                     [r["text_en"] for r in todo], 32, beams)
        for r, tr in zip(todo, translated):
            cache[r["text_en"]] = apply_glossary(tr.strip())
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        print(f"перевод занял {time.time() - t0:.0f}s")

    frame = Frame("unsupported", None, (), False, None)
    records, review, seen = [], [], set()
    for r in sample:
        text = cache.get(r["text_en"])
        if not text:
            continue
        norm = normalize(text)
        if norm in seen or len(norm) < 3:
            continue
        seen.add(norm)
        problems = validate_text(frame, text)
        rec = {
            "id": hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12],
            "split": "train_aug",  # пересчитается ниже
            "source": "clinc150",
            "text": text,
            "target": "<UNSUPPORTED> <END_FRAME>",
            "frame": {"intent": "unsupported", "specialty": None,
                      "symptoms": [], "temporal": None,
                      "temporal_factors": [], "needs_clarification": False,
                      "extra_mentions": []},
            "provenance": {"skeleton": "clinc", "stt_variant": "nllb_greedy",
                           "orig_intent": r["intent"]},
        }
        if problems:
            rec["review_reason"] = "; ".join(problems)
            review.append(rec)
        else:
            records.append(rec)

    rng.shuffle(records)
    for i, rec in enumerate(records):
        rec["split"] = "train_aug" if i < TRAIN_AUG_QUOTA else "ext_test"

    _write(out / "clinc_ru.jsonl", records)
    _write(out / "clinc_needs_review.jsonl", review)
    stats = {
        "translated_total": len(cache),
        "kept": len(records),
        "train_aug": sum(1 for r in records if r["split"] == "train_aug"),
        "ext_test": sum(1 for r in records if r["split"] == "ext_test"),
        "needs_review": len(review),
        "review_reasons": dict(Counter(r["review_reason"].split(";")[0].strip()
                                       for r in review).most_common(5)),
        "top_intents": dict(Counter(r["provenance"]["orig_intent"] for r in records)
                            .most_common(8)),
    }
    (out / "clinc_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n--- примеры ---")
    for r in records[:8]:
        print(f"  [{r['provenance']['orig_intent']}] {r['text']}")


def _write(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
