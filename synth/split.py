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
    paraphrase_count = defaultdict(int)   # (сплит, frame_key) -> n
    seen_texts = set()                    # глобальная уникальность текстов
    drops = defaultdict(int)

    # Зашита от утечки здесь текстовая, не фреймовая: normalize(text)
    # уникален глобально (кроме тривиальных да/нет — они атомарные
    # реплики и могут повторяться между сплитами). Фрейм может
    # встретиться в нескольких сплитах с разными поверхностями —
    # иначе тонкие интенты (check_booking, да/нет) не добирают объём
    # ни в один сплит: их ключ-пространство мало, а блокировка
    # «фрейм закреплён за первым сплитом» отдавала его целиком train.
    # Сплиты заполняются последовательно (val -> iid -> ood -> train):
    # каждый получает своё план-распределение интентов, тонкие классы
    # (waitlist, complaint) не вытесняются дефицит-взвешенной жеребьёвкой,
    # где train с квотой 20000 забирал почти все итерации
    order = [q for q in ("val", "iid", "ood", "train") if quotas.get(q)]
    guard = 0
    max_iter = sum(quotas.values()) * 60

    for q in order:
        drops_split = 0
        while len(buckets[q]) < quotas[q] and guard < max_iter:
            guard += 1
            if drops_split > quotas[q] * 40:
                break                    # сплит упёрся в текст-пространство
            frame = sample_frame(rng, ood=(q == "ood"))
            key = frame_key(frame)
            if paraphrase_count[(q, key)] >= CAP_PER_FRAME:
                drops_split += 1
                continue

            got = 0
            for _ in range(rng.randint(PARAPHRASE_MIN, PARAPHRASE_MAX)):
                if len(buckets[q]) >= quotas[q]:
                    break
                piece = realize(frame, rng)
                norm = normalize(piece["text"])
                dedup = (q, norm) if frame.is_answer else norm
                if dedup in seen_texts:
                    drops["duplicate"] += 1
                    continue
                reasons = validate_text(frame, piece["text"])
                if reasons:
                    drops["validator"] += 1
                    continue
                seen_texts.add(dedup)
                paraphrase_count[(q, key)] += 1
                buckets[q].append(_record(frame, piece["text"],
                                          piece["skeleton"], piece["stt"], q))
                got += 1
            if got == 0:
                drops["frame_exhausted"] += 1
                drops_split += 1

    stats = {
        "drops": dict(drops),
        "sizes": {q: len(buckets[q]) for q in buckets},
        "frames_used": {q: sum(1 for qq, _ in paraphrase_count if qq == q)
                        for q in buckets},
    }
    return buckets, stats
