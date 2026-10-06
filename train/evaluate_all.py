# -*- coding: utf-8 -*-
"""Полный прогон всех тестов + отчёт о необработанных кейсах.

Запуск:  python train/evaluate_all.py
Выход:   консольная таблица метрик + models/failures_report.md
"""

import io
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from model import TinyGPT
from constrained import generate_frame
from train import parse_target

SETS = [
    ("val", "val.jsonl", None),
    ("test_iid", "test_iid.jsonl", None),
    ("test_ood", "test_ood.jsonl", 200),
    ("manual_gold", "manual_gold.jsonl", None),
    ("hasd", "hasd_ru.jsonl", 250),
    ("hasd_unsup", "hasd_unsupported_ru.jsonl", 250),
    ("clinc", "clinc_ru.jsonl", "ext_test"),
]


def load(name, only_split=None):
    p = ROOT / "data" / name
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    if only_split:
        rows = [r for r in rows if r["split"] == only_split]
    return rows


def main():
    ckpt_path = ROOT / "models" / (sys.argv[1] if len(sys.argv) > 1 else "model.pt")
    print(f"чекпоинт: {ckpt_path.name}")
    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TinyGPT(tok.get_vocab_size(), **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    ids = {t: tok.token_to_id(t) for t in
           ("<PAD>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>")}

    def predict(text):
        return generate_frame(model, tok, text, torch.device("cpu"),
                              bos=ids["<BOS>"], user=ids["<USER>"],
                              assistant=ids["<ASSISTANT>"],
                              eos=ids["<EOS>"], pad=ids["<PAD>"])[0]

    table, failures = [], []
    for label, fname, extra in SETS:
        rows = load(fname, extra if isinstance(extra, str) else None)
        if isinstance(extra, int):
            rows = rows[:extra]
        st = Counter()
        for r in rows:
            pred = predict(r["text"])
            gold = parse_target(r["target"])
            p = parse_target(pred)
            st["n"] += 1
            st["exact"] += pred == r["target"]
            st["intent"] += p["intent"] == gold["intent"]
            if gold["intent"] != "<UNSUPPORTED>":
                st["spec_n"] += 1
                st["spec"] += p["specialty"] == gold["specialty"]
                st["time"] += p["time"] == gold["time"]
            st["symp"] += set(p["symptoms"]) == set(gold["symptoms"])
            if pred != r["target"]:
                fails = []
                if p["intent"] != gold["intent"]:
                    fails.append("intent")
                if p["specialty"] != gold["specialty"]:
                    fails.append("specialty")
                if p["time"] != gold["time"]:
                    fails.append("time")
                if set(p["symptoms"]) != set(gold["symptoms"]):
                    fails.append("symptoms")
                failures.append({"set": label, "fields": fails,
                                 "text": r["text"], "gold": r["target"],
                                 "pred": pred})
        n = max(st["n"], 1)
        nspec = max(st.get("spec_n", 0), 1)
        table.append((label, st["n"], st["exact"] / n, st["intent"] / n,
                      st["spec"] / nspec, st["time"] / nspec,
                      st["symp"] / n))

    print(f"{'набор':14s} {'n':>5s} {'exact':>6s} {'intent':>7s} "
          f"{'spec':>6s} {'time':>6s} {'symp':>6s}")
    for label, n, ex, it, sp, tm, sy in table:
        print(f"{label:14s} {n:5d} {ex:6.3f} {it:7.3f} {sp:6.3f} "
              f"{tm:6.3f} {sy:6.3f}")

    # отчёт о провалах
    by_set = Counter(f["set"] for f in failures)
    by_field = Counter(fl for f in failures for fl in f["fields"])
    lines = ["# Кейсы, которые модель не обработала", "",
             f"Всего промахов: {len(failures)}", "",
             "## По наборам", ""]
    lines += [f"- {k}: {v}" for k, v in by_set.most_common()]
    lines += ["", "## По полям", ""]
    lines += [f"- {k}: {v}" for k, v in by_field.most_common()]
    for label in ("test_ood", "manual_gold", "hasd"):
        subset = [f for f in failures if f["set"] == label]
        if not subset:
            continue
        lines += ["", f"## Примеры: {label} ({len(subset)})", ""]
        for f in subset[:12]:
            lines += [f"- **{'+'.join(f['fields'])}**",
                      f"  - текст: `{f['text']}`",
                      f"  - надо: `{f['gold']}`",
                      f"  - вышло: `{f['pred']}`"]
    (ROOT / "models" / "failures_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\nотчёт о провалах: models/failures_report.md "
          f"({len(failures)} кейсов)")


if __name__ == "__main__":
    main()
