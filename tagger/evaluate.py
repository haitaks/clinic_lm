# -*- coding: utf-8 -*-
"""Оценка тегированной архитектуры: теги+интент -> спаны -> фрейм.

Метрики те же, что у train/evaluate_all.py (parse_target), чтобы
таблицы были сравнимы напрямую. Внизу — вшитая таблица текущей
GPT-модели (модели/чекпоинт cee5e51, data/eval_final.log) для
сравнения без её перезапуска.

Запуск: .venv-tr/Scripts/python tagger/evaluate.py
"""

import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Tagger  # noqa: E402
from annotate import TAGS, INTENTS  # noqa: E402
from assemble import frame_from_tags  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"

# цифры GPT-модели (models/model.pt, коммит cee5e51) из data/eval_final.log
GPT_REF = {
    "val": (0.958, 0.996, 1.000, 1.000, 0.964),
    "test_iid": (0.928, 0.997, 1.000, 1.000, 0.932),
    "test_ood": (0.095, 0.995, 0.125, 0.995, 0.595),
    "manual_gold": (0.400, 0.600, 0.273, 0.455, 0.867),
    "hasd": (0.140, 0.152, 0.164, 0.164, 1.000),
    "hasd_unsup": (0.960, 0.960, 0.000, 0.000, 1.000),
    "clinc": (0.986, 0.986, 0.000, 0.000, 1.000),
}
LIMITS = {"test_ood": 200, "hasd": 250, "hasd_unsup": 250}
FILES = {"hasd": "hasd_ru", "hasd_unsup": "hasd_unsupported_ru",
         "clinc": "clinc_ext_test"}


def parse_target(target):
    toks = target.split()
    out = {"intent": None, "specialty": None, "time": None, "symptoms": [],
           "clarification": False}
    if not toks:
        return out
    out["intent"] = toks[0]
    if out["intent"] == "<UNSUPPORTED>":
        return out
    out["specialty"] = toks[1] if len(toks) > 1 else None
    rest = toks[2:]
    if rest and rest[0] == "<TIME>":
        end = rest.index("</TIME>") if "</TIME>" in rest else len(rest)
        out["time"] = " ".join(rest[1:end])
        rest = rest[end + 1:]
    elif rest and rest[0] == "<T_UNKNOWN>":
        out["time"] = "<T_UNKNOWN>"
        rest = rest[1:]
    out["symptoms"] = [t for t in rest if t.startswith("<SYMPTOM_")]
    out["clarification"] = "<NEEDS_CLARIFICATION>" in rest
    return out


def predict_frame(model, tok, text, device, resolver=None):
    enc = tok.encode(text)
    ids = torch.tensor([[tok.token_to_id("<BOS>")] + enc.ids], device=device)
    pad = torch.zeros_like(ids, dtype=torch.bool)
    with torch.no_grad():
        ilog, tlog = model(ids, pad)
    intent = int(ilog.argmax(-1))
    tags = tlog.argmax(-1)[0, 1:].tolist()      # без BOS-позиции
    return frame_from_tags(text, enc.ids, tags, intent, tok=tok,
                           resolver=resolver)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolver", action="store_true",
                    help="включить e5-резолвер OOV-спанов (нужен transformers)")
    args = ap.parse_args()

    resolver = None
    if args.resolver:
        from oov_resolver import E5Resolver
        base = E5Resolver(threshold=0.85)
        # строгий режим: только SYM-спаны. SPEC словарь закрывает
        # полностью, а на ood-скелетах сеть шумит SPEC-тегами и e5
        # послушно канонизирует мусор (spec 1.000 -> 0.960)
        resolver = (lambda span, tag: base(span, tag) if tag == "SYM"
                    else None)
        print("e5-резолвер включён (SYM, порог 0.85)\n")

    ckpt = torch.load(ROOT / "tagger" / "tagger.pt", map_location="cpu",
                      weights_only=False)
    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    model = Tagger(tok.get_vocab_size())
    model.load_state_dict(ckpt["model"])
    model.eval()
    dev = torch.device("cpu")

    print(f"{'набор':14s} {'n':>5s}  {'exact':>6s} {'intent':>6s} "
          f"{'spec':>6s} {'time':>6s} {'symp':>6s}   GPT (exact/intent/spec)")
    for name in ("val", "test_iid", "test_ood", "manual_gold",
                 "hasd", "hasd_unsup", "clinc"):
        rows = [json.loads(l) for l in
                (DATA / f"{FILES.get(name, name)}.tags.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        if name in LIMITS:
            rows = rows[:LIMITS[name]]
        stats = {"n": 0, "exact": 0, "intent": 0, "specialty": 0, "time": 0,
                 "symptoms_f1": 0.0}
        for r in rows:
            pred = predict_frame(model, tok, r["text"], dev, resolver)
            gold = parse_target(r["target"])
            p = parse_target(pred)
            stats["n"] += 1
            stats["exact"] += pred == r["target"]
            stats["intent"] += p["intent"] == gold["intent"]
            stats["specialty"] += p["specialty"] == gold["specialty"]
            stats["time"] += p["time"] == gold["time"]
            gs, ps = set(gold["symptoms"]), set(p["symptoms"])
            prec = len(gs & ps) / len(ps) if ps else 1.0
            rec = len(gs & ps) / len(gs) if gs else 1.0
            stats["symptoms_f1"] += (2 * prec * rec / (prec + rec)
                                     if prec + rec else 0.0)
        n = max(stats["n"], 1)
        ref = GPT_REF[name]
        print(f"{name:14s} {stats['n']:5d}  "
              f"{stats['exact'] / n:6.3f} {stats['intent'] / n:6.3f} "
              f"{stats['specialty'] / n:6.3f} {stats['time'] / n:6.3f} "
              f"{stats['symptoms_f1'] / n:6.3f}   "
              f"{ref[0]:.3f}/{ref[1]:.3f}/{ref[2]:.3f}")

    print("\nдемо-фразы:")
    for text in ("отмени все мои записи",
                 "запишите меня к зубному в следующую пятницу вечером, зуб ноет",
                 "у меня болит голова",
                 "надо к кожнику попасть, чешется всё",
                 "какая завтра погода в Москве"):
        print(f"  {text[:52]:54s} -> "
              f"{predict_frame(model, tok, text, dev, resolver)}")


if __name__ == "__main__":
    main()
