# -*- coding: utf-8 -*-
"""Обучение joint intent+tagging энкодера на DirectML.

Запуск:
    .venv-tr/Scripts/python tagger/train_tagger.py --device dml --epochs 20
Выход: tagger/tagger.pt + метрики тегов/интента на val в консоли.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Tagger, count_params  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
PAD, BOS = "<PAD>", "<BOS>"


def load_tags(name):
    return [json.loads(l) for l in
            (DATA / f"{name}.tags.jsonl").read_text(encoding="utf-8").splitlines()]


def build_batch(batch, pad_id, bos_id, device):
    seqs = [[bos_id] + r["ids"] for r in batch]
    maxlen = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    tags = torch.full((len(seqs), maxlen), -100, dtype=torch.long)
    for i, (s, r) in enumerate(zip(seqs, batch)):
        idx[i, :len(s)] = torch.tensor(s)
        tags[i, 1:len(s)] = torch.tensor(r["tags"])
    pad_mask = idx == pad_id
    intents = torch.tensor([r["intent"] for r in batch])
    return (idx.to(device), pad_mask.to(device),
            intents.to(device), tags.to(device))


def evaluate_tags(model, rows, pad_id, bos_id, device):
    model.eval()
    n_int = n_tok = hit_int = hit_tok = 0
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            batch = rows[i:i + 64]
            idx, pad_mask, intents, tags = build_batch(
                batch, pad_id, bos_id, device)
            ilog, tlog = model(idx, pad_mask)
            n_int += len(batch)
            hit_int += (ilog.argmax(-1) == intents).sum().item()
            m = tags != -100
            n_tok += m.sum().item()
            hit_tok += (tlog.argmax(-1)[m] == tags[m]).sum().item()
    return hit_int / n_int, hit_tok / n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu", choices=["cpu", "dml"])
    ap.add_argument("--save", default="tagger/tagger.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    if args.device == "dml":
        import torch_directml
        device = torch_directml.device()

    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    pad_id, bos_id = tok.token_to_id(PAD), tok.token_to_id(BOS)

    train_rows = load_tags("train") + load_tags("clinc_train_aug")
    val_rows = load_tags("val")
    print(f"train: {len(train_rows)}, val: {len(val_rows)}, "
          f"device: {device}, vocab: {tok.get_vocab_size()}")

    model = Tagger(tok.get_vocab_size()).to(device)
    print(f"параметров: {count_params(model) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    steps_total = math.ceil(len(train_rows) / args.batch) * args.epochs
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
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        totals = []
        for i in range(0, len(order), args.batch):
            batch = [train_rows[j] for j in order[i:i + args.batch]]
            idx, pad_mask, intents, tags = build_batch(
                batch, pad_id, bos_id, device)
            ilog, tlog = model(idx, pad_mask)
            loss = F.cross_entropy(ilog, intents) + \
                F.cross_entropy(tlog.reshape(-1, tlog.size(-1)),
                                tags.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.step()
            totals.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(f"  epoch {epoch + 1} step {step}/{steps_total} "
                      f"loss {sum(totals[-100:]) / min(len(totals), 100):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    # снятие весов CPU-копиями (урок torch-directml: см. train/train.py)
    sd = {k: v.detach().to("cpu").clone()
          for k, v in model.state_dict().items()}
    out = Path(args.save)
    torch.save({"model": sd, "config": {}}, out)
    print(f"сохранено: {out}")

    # самопроверка: файл обязан повторять живую модель на val-батчах
    back = Tagger(tok.get_vocab_size())
    back.load_state_dict(torch.load(out, map_location="cpu",
                                    weights_only=False)["model"])
    back.eval()
    live_int, live_tok = evaluate_tags(back, val_rows[:256],
                                       pad_id, bos_id, torch.device("cpu"))
    print(f"val (из файла): intent acc {live_int:.4f}, "
          f"token acc {live_tok:.4f}")
    assert live_int > 0.9 and live_tok > 0.9, \
        "чекпойнт не похож на обученную модель — снятие весов нечестно"


if __name__ == "__main__":
    main()
