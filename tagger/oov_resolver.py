# -*- coding: utf-8 -*-
"""e5-резолвер OOV-спанов: незнакомое слово -> ближайший канонический
синоним по косинусной близости multilingual-e5-small.

Индексы строятся по словарям иголок synth/vocab.py (train+ood формы
специальностей во всех падежах и выражения симптомов). Порог подбирается
на живой речи manual_gold + контроль ложных срабатываний на негативах.

Запуск:  .venv-tf/Scripts/python tagger/oov_resolver.py
Выход:   tagger/models/e5_index.npz + surfaces.json, отчёт в консоль.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vocab import SPECIALTIES, SYMPTOMS  # noqa: E402

OUT = Path(__file__).resolve().parent / "models"
DEVICE = torch.device("cpu")


def load_e5():
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
    model.to(DEVICE).eval()
    return tok, model


@torch.no_grad()
def embed(tok, model, texts, batch=64):
    """Среднее по last_hidden_state с маской внимания (контракт e5)."""
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(DEVICE)
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out)


def build_surfaces():
    surfaces = []          # (текст, тег, канон-ключ)
    for key, entry in SPECIALTIES.items():
        for syn in entry["train"] + entry["ood"]:
            for form in syn.values():
                surfaces.append((form, "SPEC", key))
    for key, entry in SYMPTOMS.items():
        for e in entry["train"] + entry["ood"]:
            surfaces.append((e, "SYM", key))
    return surfaces


class E5Resolver:
    """Резолвер для assemble.frame_from_tags(resolver=...): спан,
    помеченный тегом модели, но не канонизированный словарём, -> ключ.

    Порог по косинусу НЕ отделяет мусор от истины (погода -> fever
    0.89) — поэтому применяется только к спанам, которые сеть сама
    пометила SPEC/SYM; порог отсекает лишь совсем чужие строки."""

    def __init__(self, index_dir=None, threshold=0.75):
        index_dir = Path(index_dir) if index_dir else OUT
        z = np.load(index_dir / "e5_index.npz", allow_pickle=False)
        self.vecs = z["vecs"]
        self.tags = list(z["tags"])
        self.keys = list(z["keys"])
        self.threshold = threshold
        self.tok, self.model = load_e5()

    def __call__(self, span_text: str, tag: str):
        v = embed(self.tok, self.model, ["query: " + span_text])[0]
        m = np.array([t == tag for t in self.tags])
        sims = self.vecs[m] @ v
        best = int(sims.argmax())
        if sims[best] < self.threshold:
            return None
        return self.keys[[i for i, ok in enumerate(m) if ok][best]]


def main():
    OUT.mkdir(exist_ok=True)
    tok, model = load_e5()
    surfaces = build_surfaces()
    texts = ["passage: " + s[0] for s in surfaces]
    vecs = embed(tok, model, texts)
    np.savez_compressed(OUT / "e5_index.npz",
                        vecs=vecs.astype(np.float32),
                        tags=np.array([s[1] for s in surfaces]),
                        keys=np.array([s[2] for s in surfaces]))
    (OUT / "surfaces.json").write_text(
        json.dumps([{"text": t, "tag": g, "key": k} for t, g, k in surfaces],
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"индекс: {len(surfaces)} поверхностей, dim {vecs.shape[1]}")

    # --- проверка на живой речи и ложных срабатываниях ---
    cases = [
        ("зуб разрывает", "SYM", "toothache"),
        ("мигрень", "SYM", "headache"),
        ("чешется всё", "SYM", "rash"),
        ("болит спина", "SYM", "back_pain"),
        ("зрение проверить", "SPEC", "ophthalmologist"),
        ("лор", "SPEC", "ent"),
        ("кожник", "SPEC", "dermatologist"),
        ("зубной врач", "SPEC", "dentist"),
        ("нервный врач", "SPEC", "neurologist"),
        # ложные: резолвер обязан отказать (None)
        ("погода", "SYM", None),
        ("пицца", "SPEC", None),
        ("будильник", "SYM", None),
        ("записи", "SPEC", None),
    ]
    q = embed(tok, model, ["query: " + c[0] for c in cases])
    for (text, tag, want), v in zip(cases, q):
        m = vecs[np.array([s[1] == tag for s in surfaces])]
        keys = [s[2] for s in surfaces if s[1] == tag]
        sims = m @ v
        best = int(sims.argmax())
        print(f"{text!r:24s} [{tag}] -> {keys[best]:18s} "
              f"sim={sims[best]:.3f}  (хотим {want})")


if __name__ == "__main__":
    main()
