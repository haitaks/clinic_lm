# -*- coding: utf-8 -*-
"""Генерация датасета: фреймы -> тексты -> валидация -> дедупликация -> сплиты.

Защита от утечки:
- все парафразы одного фрейма попадают строго в один сплит
  (назначение закрепляется при первом появлении ключа фрейма);
- OOD-пул генерируется из отдельного пространства поверхностей
  (hold-out синонимы, hold-out скелеты, hold-out пары
  специальность+симптом) и живёт в отдельном пространстве ключей.
"""

import hashlib
import random
from collections import defaultdict

from frames import Frame, sample_frame, frame_key, target_string
from realize import realize
from validate import validate_text, normalize
from vocab import SPECIALTIES

CAP_PER_FRAME = 60          # максимум текстов на один фрейм
PARAPHRASE_MIN, PARAPHRASE_MAX = 3, 7


def _record(frame: Frame, text: str, sk: str, stt: str, split: str) -> dict:
    return {
        "id": hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:12],
        "split": split,
        "source": "synthetic",
        "text": text,
        "target": target_string(frame),
        "frame": {
            "intent": frame.intent,
            "specialty": frame.specialty if frame.specialty in SPECIALTIES
            else "unknown" if frame.specialty else None,
            "symptoms": list(frame.symptoms),
            "temporal": frame.temporal,
            "temporal_factors": list(frame.temporal_factors),
            "needs_clarification": frame.needs_clarification,
            "extra_mentions": list(frame.extra_mentions),
        },
        "provenance": {"skeleton": sk, "stt_variant": stt, **frame.aux},
    }


def build(quotas: dict, seed: int = 13):
    rng = random.Random(seed)
    buckets = {name: [] for name in quotas}
    key_bucket = {}          # frame_key -> назначенный сплит (целостность групп)
    key_bucket_ood = {}      # отдельное пространство ключей для OOD-пула
    paraphrase_count = defaultdict(int)
    seen_texts = set()
    drops = defaultdict(int)

    order = [q for q in quotas if quotas[q] > 0]
    guard = 0
    max_iter = sum(quotas.values()) * 60

    while any(len(buckets[q]) < quotas[q] for q in order) and guard < max_iter:
        guard += 1
        pending = [q for q in order if len(buckets[q]) < quotas[q]]
        q = rng.choices(pending,
                        weights=[quotas[x] - len(buckets[x]) for x in pending])[0]

        frame = sample_frame(rng, ood=(q == "ood"))
        key = frame_key(frame)
        registry = key_bucket_ood if q == "ood" else key_bucket
        if key in registry and registry[key] != q:
            continue                      # группа уже закреплена за другим сплитом
        if paraphrase_count[(q == "ood", key)] >= CAP_PER_FRAME:
            continue

        got = 0
        for _ in range(rng.randint(PARAPHRASE_MIN, PARAPHRASE_MAX)):
            if len(buckets[q]) >= quotas[q]:
                break
            piece = realize(frame, rng)
            norm = normalize(piece["text"])
            if norm in seen_texts:
                drops["duplicate"] += 1
                continue
            reasons = validate_text(frame, piece["text"])
            if reasons:
                drops["validator"] += 1
                continue
            seen_texts.add(norm)
            paraphrase_count[(q == "ood", key)] += 1
            registry.setdefault(key, q)
            buckets[q].append(_record(frame, piece["text"], piece["skeleton"],
                                      piece["stt"], q))
            got += 1
        if got == 0:
            drops["frame_exhausted"] += 1

    stats = {
        "drops": dict(drops),
        "sizes": {q: len(buckets[q]) for q in buckets},
        "frames_used": {q: sum(1 for k, v in key_bucket.items() if v == q)
                        + sum(1 for v in key_bucket_ood.values() if v == q)
                        for q in buckets},
    }
    return buckets, stats
