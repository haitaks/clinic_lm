# -*- coding: utf-8 -*-
"""Майнинг синонимов специальностей внешними эмбеддингами (rubert-tiny2).

Запуск (в venv с рабочей парой transformers/torch):
    .venv-tr/Scripts/python synth/mine_synonyms.py

Идея: собрать слова и частые биграммы из наших корпусов (MASSIVE ru-RU,
синтетика, переводы HASD/CLINC), векторизовать rubert-tiny2 (29 МБ) и
найти ближайших соседей канонических названий специальностей. Кандидаты
печатаются с косинусной близостью — дальше ручная курификация и добавка
в vocab.py (train-наборы; hold-out слова OOD-теста не трогаем).

Sanity-проверка внутри: hold-out синонимы («дантист», «окулист», ...)
должны быть ближе к своему якорю, чем к чужим.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "synth"))
from vocab import SPECIALTIES  # noqa: E402

MODEL = "cointegrated/rubert-tiny2"
WORD_RE = re.compile(r"[а-яё]{3,}")


def collect_candidates():
    """Уникальные слова (freq>=3) и биграммы (freq>=5) из корпусов."""
    texts = []

    massive = ROOT / "data" / "_massive" / "1.1" / "data" / "ru-RU.jsonl"
    if massive.exists():
        texts += [json.loads(l)["utt"] for l in
                  massive.read_text(encoding="utf-8").splitlines()]
    for name in ("train", "val", "test_iid", "hasd_ru", "hasd_unsupported_ru",
                 "clinc_ru", "manual_gold"):
        p = ROOT / "data" / f"{name}.jsonl"
        if p.exists():
            texts += [json.loads(r)["text"] for r in
                      p.read_text(encoding="utf-8").splitlines()]

    words, bigrams = Counter(), Counter()
    for t in texts:
        toks = WORD_RE.findall(t.lower())
        words.update(toks)
        bigrams.update(zip(toks, toks[1:]))
    cand = {w for w, c in words.items() if c >= 3 and len(w) <= 20}
    cand |= {" ".join(bg) for bg, c in bigrams.items() if c >= 5}
    return sorted(cand)


def embed_phrases(phrases, batch=256):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(phrases), batch):
            enc = tok(phrases[i:i + batch], padding=True, truncation=True,
                      max_length=16, return_tensors="pt")
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"][..., None].float()
            v = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            v = torch.nn.functional.normalize(v, dim=1)
            vecs.append(v)
    return torch.cat(vecs)


def main():
    cand = collect_candidates()
    print(f"кандидатов из корпусов: {len(cand)} (слова + биграммы)")

    # якоря: train-синонимы (номинатив) и hold-out для sanity
    anchors_train, anchors_ood, anchor_owner = [], [], {}
    for spec, entry in SPECIALTIES.items():
        for s in entry["train"]:
            anchors_train.append(s["nom"])
            anchor_owner[s["nom"]] = spec
        for s in entry["ood"]:
            anchors_ood.append(s["nom"])
            anchor_owner[s["nom"]] = spec

    phrases = cand + anchors_train + anchors_ood
    print(f"векторизуем {len(phrases)} фраз...")
    vecs = embed_phrases(phrases)
    cand_v = vecs[:len(cand)]
    tr_v = vecs[len(cand):len(cand) + len(anchors_train)]
    ood_v = vecs[len(cand) + len(anchors_train):]

    print("\n=== SANITY: близость hold-out синонима к якорям своей специальности ===")
    name2vec = {n: v for n, v in
                zip(anchors_train + anchors_ood, list(tr_v) + list(ood_v))}
    for ood_name in anchors_ood:
        owner = anchor_owner[ood_name]
        own = max(float(torch.cosine_similarity(name2vec[ood_name],
                                                name2vec[n], dim=0))
                  for n in anchors_train if anchor_owner[n] == owner)
        print(f"  {ood_name:18s} -> {owner:16s} sim={own:.3f}")

    print("\n=== СОСЕДИ (топ-15 на специальность) ===")
    used = {w for entry in SPECIALTIES.values()
            for s in entry["train"] + entry["ood"] for w in s.values()}
    for spec, entry in SPECIALTIES.items():
        anchor = entry["train"][0]["nom"]
        av = tr_v[anchors_train.index(anchor)]
        sims = torch.cosine_similarity(cand_v, av[None, :], dim=1)
        top = sims.argsort(descending=True)[:40]
        shown = 0
        print(f"\n[{spec}] якорь: {anchor}")
        for i in top:
            w = cand[i]
            if w in used or w == anchor:
                continue
            print(f"   {float(sims[i]):.3f}  {w}")
            shown += 1
            if shown >= 15:
                break


if __name__ == "__main__":
    main()
