# -*- coding: utf-8 -*-
"""Обучение decoder-only модели: текст заявки -> semantic frame.

Формат примера:  <BOS> <USER> {text} <ASSISTANT> {target} <EOS>
Loss считается ТОЛЬКО по target-токенам и <EOS> (остальное -100).

Запуск:
    python train/train.py --epochs 3                # cpu, nano-конфиг
    python train/train.py --device dml --epochs 5   # DirectML (Radeon)
    python train/train.py --smoke                   # 300 шагов для проверки
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from model import TinyGPT, count_params, build_batch  # noqa: E402
from constrained import generate_frame  # noqa: E402

PAD, UNK, BOS, EOS, USER, ASSISTANT = ("<PAD>", "<UNK>", "<BOS>", "<EOS>",
                                       "<USER>", "<ASSISTANT>")


def load_split(name, with_clinc=False):
    rows = [json.loads(l) for l in
            (ROOT / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    if with_clinc and name == "train":
        clinc = [json.loads(l) for l in
                 (ROOT / "data" / "clinc_ru.jsonl").read_text(encoding="utf-8").splitlines()]
        rows += [r for r in clinc if r["split"] == "train_aug"]
    return rows


def encode_example(row, tok, ids):
    """(последовательность, индекс старта target) — loss с этого индекса.

    Target состоит только из спецтокенов; ByteLevel-энкодер вставляет между
    ними служебный 'Ġ' (пробел). Вырезаем его: иначе constrained decoding
    без 'Ġ' даёт распределение, которого не было в train.
    """
    text_ids = tok.encode(row["text"]).ids
    target_ids = [i for i in tok.encode(row["target"]).ids
                  if i != ids["G"]]
    budget = 128 - len(target_ids) - 4          # контекст: BOS USER .. ASSISTANT .. EOS
    prefix = [ids[BOS], ids[USER]] + text_ids[:budget] + [ids[ASSISTANT]]
    seq = prefix + target_ids + [ids[EOS]]
    return seq, len(prefix)


class Examples:
    def __init__(self, rows, tok, ids, limit=0):
        self.data = []
        for row in rows:
            seq, start = encode_example(row, tok, ids)
            if len(seq) <= tok_limit:
                self.data.append((seq, start))
        if limit:
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)


tok_limit = 128


def train(model, examples, ids, device, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    steps_total = math.ceil(len(examples) / args.batch) * args.epochs
    warmup = min(100, steps_total // 10)

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, steps_total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    model.train()
    rng = random.Random(args.seed)
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        totals = []
        for i in range(0, len(order), args.batch):
            batch = [examples.data[j] for j in order[i:i + args.batch]]
            seqs = [b[0] for b in batch]
            idx, pad_mask = build_batch(seqs, ids[PAD], device)
            logits = model(idx, pad_mask)
            # logits[t] предсказывает seq[t+1]: labels стоят со сдвигом влево,
            # первый предсказываемый токен — seq[start] по позиции start-1 (<ASSISTANT>)
            labels = torch.full_like(idx, -100)
            for bi, (seq, start) in enumerate(batch):
                labels[bi, start - 1:len(seq) - 1] = torch.tensor(
                    seq[start:], device=device)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.step()
            totals.append(loss.item())
            step += 1
            if args.smoke and step >= args.smoke_steps:
                break
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{steps_total} "
                      f"loss {sum(totals[-50:]) / len(totals[-50:]):.4f} "
                      f"({(time.time() - t0):.0f}s)", flush=True)
        if args.smoke and step >= args.smoke_steps:
            break
    return step


def parse_target(target):
    """target-строка -> словарь полей для метрик."""
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


def evaluate(model, tok, ids, device, rows, limit=0, verbose=False):
    model.eval()
    stats = {"n": 0, "exact": 0, "intent": 0, "specialty": 0, "time": 0,
             "symptoms_f1": 0.0, "samples": []}
    rows = rows[:limit] if limit else rows
    for row in rows:
        pred_str, _ = generate_frame(
            model, tok, row["text"], device,
            bos=ids[BOS], user=ids[USER], assistant=ids[ASSISTANT],
            eos=ids[EOS], pad=ids[PAD])
        gold = parse_target(row["target"])
        pred = parse_target(pred_str)
        stats["n"] += 1
        stats["exact"] += pred_str == row["target"]
        stats["intent"] += pred["intent"] == gold["intent"]
        stats["specialty"] += pred["specialty"] == gold["specialty"]
        stats["time"] += pred["time"] == gold["time"]
        gs, ps = set(gold["symptoms"]), set(pred["symptoms"])
        prec = len(gs & ps) / len(ps) if ps else 1.0
        rec = len(gs & ps) / len(gs) if gs else 1.0
        stats["symptoms_f1"] += 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if verbose and len(stats["samples"]) < 5 and pred_str != row["target"]:
            stats["samples"].append({"text": row["text"], "gold": row["target"],
                                     "pred": pred_str})
    n = max(stats["n"], 1)
    return {k: (v / n if k not in ("n", "samples") else v)
            for k, v in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu", choices=["cpu", "dml"])
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=300)
    ap.add_argument("--save", default="models/model.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    if args.device == "dml":
        import torch_directml
        device = torch_directml.device()

    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    ids = {t: tok.token_to_id(t) for t in (PAD, UNK, BOS, EOS, USER, ASSISTANT)}
    ids["G"] = tok.token_to_id("Ġ")
    assert all(v is not None for v in ids.values()), ids

    train_rows = load_split("train", with_clinc=True)
    val_rows = load_split("val")
    train_ex = Examples(train_rows, tok, ids, args.limit)
    val_ex_meta = val_rows  # метрики считаем генерацией, не маской
    print(f"train: {len(train_ex)} примеров, val: {len(val_rows)}, "
          f"device: {device}, vocab: {tok.get_vocab_size()}")

    model = TinyGPT(tok.get_vocab_size(), d_model=args.d_model,
                    n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    print(f"параметров: {count_params(model) / 1e6:.2f}M")

    steps = train(model, train_ex, ids, device, args)
    out = ROOT / args.save
    torch.save({"model": model.state_dict(),
                "config": {"d_model": args.d_model, "n_layers": args.n_layers,
                           "n_heads": args.n_heads}}, out)
    print(f"сохранено: {out} (шагов: {steps})")

    val_limit = 300 if (args.smoke or args.limit) else 0
    metrics = evaluate(model, tok, ids, device, val_ex_meta, limit=val_limit,
                       verbose=True)
    print("\n=== val метрики ===")
    for k in ("exact", "intent", "specialty", "time", "symptoms_f1"):
        print(f"  {k:12s} {metrics[k]:.3f}")
    print(f"  n = {metrics['n']}")
    for s in metrics["samples"]:
        print(f"  MISS: {s['text']}\n     gold: {s['gold']}\n     pred: {s['pred']}")
    (ROOT / "models" / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
