# -*- coding: utf-8 -*-
"""Сборка синтетического датасета заявок к врачу.

Запуск:
    python synth/build_dataset.py                 # объёмы по умолчанию
    python synth/build_dataset.py --train 50000   # свой объём train

Выход: data/train.jsonl, data/val.jsonl, data/test_iid.jsonl,
       data/test_ood.jsonl, data/manual_gold.jsonl, data/stats.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from validate import self_check
from split import build


# Ручной набор "как пишут живые люди, не видевшие генератор".
# Это затравка true-OOD пробы — пополняйте фразами реальных пользователей.
# Формат: (текст, интент, специальность, сырая временная фраза,
#          фактор-токены времени, симптомы, needs_clarification)
MANUAL_GOLD = [
    ("зуб разрывает, чё есть вечером после смены?",
     "book", "dentist", "вечером после смены", ["T_EVENING", "T_AFTER_SHIFT"],
     ["toothache"], False),
    ("надо проверить зрение, сыну восемь лет",
     "book", "ophthalmologist", None, [], [], False),
    ("по записи к неврологу приходить или в живую очередь?",
     "schedule", "neurologist", None, [], [], False),
    ("отмените всё, уехал из города",
     "cancel", "unknown", None, [], [], False),
    ("перенесите мой приём к лору на завтра",
     "reschedule", "ent", "на завтра", ["T_TOMORROW"], [], False),
    ("не к терапевту, а к лору, если можно",
     "book", "ent", None, [], [], False),
    ("больно наступать на пятку, кто этим занимается?",
     "book", "unknown", None, [], [], True),
    ("сколько стоит приём с рентгеном",
     "paid", "unknown", None, [], [], False),
    ("третий день температура за 38, запишите к кому надо",
     "book", "unknown", None, [], ["fever"], True),
    ("здравствуйте! можно попасть к дерматологу сегодня?",
     "book", "dermatologist", "сегодня", ["T_TODAY"], [], False),
    ("спасибо, доктор помог",
     "unsupported", None, None, [], [], False),
    ("жена просит записать её к гинекологу на следующей неделе",
     "book", "gynecologist", "на следующей неделе", ["T_NEXT_WEEK"], [], False),
    ("приём у окулиста в среду больше не нужен",
     "cancel", "ophthalmologist", None, [], [], False),
    ("панические атаки участились, нужен специалист",
     "book", "psychologist", None, [], ["anxiety"], False),
    ("во сколько открываетесь",
     "schedule", "unknown", None, [], [], False),
    # --- доработка NLU по ТЗ (Этапы A/B/C): затравки новых поверхностей ---
    ("нужен талон к окулисту на завтра",
     "book", "ophthalmologist", "на завтра", ["T_TOMORROW"], [], False),
    ("перезапишите меня на пятницу вечером",
     "reschedule", "unknown", "на пятницу вечером",
     ["T_FRI", "T_EVENING"], [], False),
    ("у меня направление к кардиологу, запишите",
     "book", "cardiologist", None, [], [], False),
    ("ребёнок заболел, нужен врач на дом",
     "home_visit", "unknown", None, [], [], False),
    ("вы в субботу работаете?",
     "schedule", "unknown", None, [], [], False),
    ("в каком кабинете принимает невролог?",
     "schedule", "neurologist", None, [], [], False),
    ("врач нагрубил, куда жаловаться",
     "complaint", "unknown", None, [], [], False),
    ("у мужа давит в груди",
     "home_visit", "unknown", None, [], ["chest_pain"], False),
    ("не могу прийти в четверг, перенесите",
     "reschedule", "unknown", "в четверг", ["T_THU"], [], False),
    ("запишите сына к детскому врачу",
     "book", "pediatrician", None, [], [], False),
]


def _gold_records():
    from frames import (INTENT_TOKENS, SPEC_TOKEN_UNKNOWN, TIME_OPEN,
                        TIME_CLOSE, TIME_UNKNOWN, CLARIF_TOKEN, END_TOKEN)
    from vocab import SPECIALTIES, SYMPTOMS, TIME_FACTORS
    recs = []
    for i, (text, intent, spec, temporal, factors, symptoms, clar) in enumerate(MANUAL_GOLD, 1):
        if intent == "unsupported":
            target = f"{INTENT_TOKENS['unsupported']} {END_TOKEN}"
        else:
            toks = [INTENT_TOKENS[intent],
                    SPECIALTIES[spec]["token"] if spec in SPECIALTIES else SPEC_TOKEN_UNKNOWN]
            if temporal:
                assert factors and all(f in TIME_FACTORS for f in factors), \
                    f"gold #{i}: некорректные факторы {factors}"
                toks += [TIME_OPEN, *factors, TIME_CLOSE]
            else:
                toks.append(TIME_UNKNOWN)
            toks += [SYMPTOMS[s]["token"] for s in symptoms]
            if clar:
                toks.append(CLARIF_TOKEN)
            toks.append(END_TOKEN)
            target = " ".join(toks)
        recs.append({"id": f"gold{i:03d}", "split": "manual_gold",
                     "source": "manual", "text": text,
                     "target": target,
                     "frame": {"intent": intent, "specialty": spec,
                               "symptoms": symptoms, "temporal": temporal,
                               "temporal_factors": factors,
                               "needs_clarification": clar,
                               "extra_mentions": []},
                     "provenance": {"skeleton": "manual", "stt_variant": "as_is"}})
    return recs


def _write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _distro(records, key):
    c = Counter()
    for r in records:
        v = r["frame"][key]
        c[v if not isinstance(v, list) else ",".join(v) or "none"] += 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser(description="Сборка синтетического датасета")
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--iid", type=int, default=1000)
    ap.add_argument("--ood", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=str, default="data")
    args = ap.parse_args()

    problems = self_check()
    if problems:
        print("СЛОВАРЬ НЕ ПРОШЁЛ САМОПРОВЕРКУ:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("self_check: пересечений иголок нет")

    buckets, gen_stats = build({"train": args.train, "val": args.val,
                                "iid": args.iid, "ood": args.ood},
                               seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = {"train": "train.jsonl", "val": "val.jsonl",
             "iid": "test_iid.jsonl", "ood": "test_ood.jsonl"}
    for q, fname in names.items():
        _write_jsonl(out / fname, buckets[q])
    _write_jsonl(out / "manual_gold.jsonl", _gold_records())

    stats = {
        "sizes": {q: len(buckets[q]) for q in buckets},
        "manual_gold": len(MANUAL_GOLD),
        "drops": gen_stats["drops"],
        "frames_used": gen_stats["frames_used"],
        "intent_distro": {q: _distro(buckets[q], "intent") for q in buckets},
        "specialty_distro_train": _distro(buckets["train"], "specialty"),
        "symptom_distro_train": _distro(buckets["train"], "symptoms"),
        "temporal_known_train": sum(1 for r in buckets["train"]
                                    if r["frame"]["temporal"]) / max(len(buckets["train"]), 1),
        "temporal_factor_distro_train": dict(Counter(
            f for r in buckets["train"]
            for f in r["frame"]["temporal_factors"]).most_common()),
        "clarification_train": sum(1 for r in buckets["train"]
                                   if r["frame"]["needs_clarification"]) / max(len(buckets["train"]), 1),
        "avg_target_tokens_train": sum(len(r["target"].split()) for r in buckets["train"])
                                   / max(len(buckets["train"]), 1),
    }
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    for q in buckets:
        print(f"\n=== {q}: {len(buckets[q])} примеров, примеры ===")
        for r in buckets[q][:4]:
            print(f"  text:   {r['text']}")
            print(f"  target: {r['target']}")
    print(f"\nГотово. Файлы в {out}/, статистика в {out}/stats.json")
    if any(len(buckets[q]) < n for q, n in
           (("train", args.train), ("val", args.val),
            ("iid", args.iid), ("ood", args.ood))):
        print("ВНИМАНИЕ: некоторые сплиты не добрали объём (см. frames_used/drops).")


if __name__ == "__main__":
    main()
