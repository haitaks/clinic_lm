# -*- coding: utf-8 -*-
"""Импорт HASD (andreaceto/hasd): детерминированный перевод + автосборка фреймов.

Запуск (из venv с совместимой парой transformers/torch, см. README):
    .venv-tr/Scripts/python synth/import_hasd.py

Перевод: facebook/nllb-200-distilled-600M, beam=4, sampling выключен —
результат воспроизводим бит в бит. Устройство — DirectML (Radeon) с
автоматическим откатом на CPU.

Схема работы:
1. HASD -> фильтр интентов (booking-четвёрка / unsupported-пятёрка);
   фразы с practitioner_name / appointment_id дропаются (нет контракта),
   с appointment_type уходят в ручную разметку.
2. Перевод en->ru + глоссарий послепереводных фиксов ("назначение"->"запись").
3. Автосборка фрейма по словарям-иголкам (специальность/симптомы) и
   регэкспам TEMPORALS (вербатимный спан для слота <TIME>).
4. Round-trip валидация; всё сомнительное — в needs_review, не в обучение.

Выход в data/:
    hasd_ru.jsonl             — booking-фразы с фреймом (split=ext_test)
    hasd_unsupported_ru.jsonl — чистые негативы (split=ext_test)
    hasd_needs_review.jsonl   — ручная разметка (в обучение не идут)
    hasd_translations.json    — кэш переводов (перезапуск не переводит снова)
    hasd_stats.json
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
import pandas as pd

from vocab import SPECIALTIES, SYMPTOMS, TEMPORALS
from frames import Frame, target_string
from validate import _spec_needles, _sym_needles, normalize, validate_text, _contains

INTENT_MAP = {
    "schedule": "book",
    "query_avail": "book",
    "reschedule": "reschedule",
    "cancel": "cancel",
    "bye": "unsupported",
    "greeting": "unsupported",
    "oos": "unsupported",
    "positive_reply": "unsupported",
    "negative_reply": "unsupported",
}
DROP_ENTITY = {"practitioner_name", "appointment_id"}
REVIEW_ENTITY = {"appointment_type"}

# Машинный перевод календарит: чиним самые частые кальки пословно.
WORD_MAP = {
    "назначение": "запись", "назначения": "записи", "назначению": "записи",
    "назначением": "записью", "назначений": "записей", "назначении": "записи",
    "назначениях": "записях", "назначениям": "записям",
    "назначениями": "записями",
    "открытия": "свободные окна", "открытий": "свободных окон",
    "открытие": "свободное окно",
}

# намёки на время, которых может не быть в TEMPORALS -> ручная разметка
TEMPORAL_HINT = re.compile(
    r"\b(сегодня|завтра|послезавтра|понедельник|вторник|сред[аыуе]|четверг|"
    r"пятниц[аыуе]|суббот|воскресень|недел|месяц|час|минут|утра|утром|"
    r"вечера|вечером|днем|днём|полдень|полудн)\w*")
GREET_RE = re.compile(r"добр(ый|ое)\s+(вечер|утро|день)")


def apply_glossary(text: str) -> str:
    t = re.sub(r"[а-яё]+", lambda m: WORD_MAP.get(m.group(0), m.group(0)), text)
    t = re.sub(r"\s+([.,!?;:])", "\\1", t)    # " ." -> "."
    return re.sub(r"\s{2,}", " ", t).strip()


def load_hasd():
    rows = []
    for split in ("train", "validation", "test"):
        p = hf_hub_download("andreaceto/hasd",
                            f"data/{split}-00000-of-00001.parquet",
                            repo_type="dataset")
        df = pd.read_parquet(p)
        names = ["bye", "cancel", "greeting", "negative_reply", "oos",
                 "positive_reply", "query_avail", "reschedule", "schedule"]
        for _, r in df.iterrows():
            rows.append({"orig_split": split, "orig_intent": names[r["intent"]],
                         "text_en": r["text"],
                         "entity_labels": sorted({e["label"] for e in r["entities"]})})
    return rows


def build_translator():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
                                        src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    device = "cpu"
    try:
        import torch_directml
        if torch_directml.is_available():
            model = model.to(torch_directml.device())
            device = torch_directml.device()
    except Exception as e:  # noqa: BLE001
        print("DirectML недоступен, работаем на CPU:", e)
    model.eval()
    bos = tok.convert_tokens_to_ids("rus_Cyrl")
    # beam>1 на DirectML склеивает фразы батча (кросс-контаминация) —
    # поэтому на DML всегда greedy; beam=4 остаётся для CPU
    beams = 1 if device != "cpu" else 4
    return tok, model, device, bos, beams


def translate_batch(torch, tok, model, device, bos, texts, batch_size=32, beams=1):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos,
                                 num_beams=beams, do_sample=False,
                                 max_new_tokens=96)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        if (i // batch_size) % 10 == 0:
            print(f"  переведено {i + len(chunk)}/{len(texts)}", flush=True)
    return out


# ------------------------------------------------------------- автосборка

def detect_specialties(text):
    t = normalize(text)
    return [spec for spec, needles in _spec_needles.items()
            if any(_contains(n, t) for n in needles)]


def detect_symptoms(text):
    t = normalize(text)
    hits = []
    for sym, exprs in _sym_needles.items():
        for e in exprs:
            m = re.search(r"(?<![а-яёa-z])" + re.escape(e.replace("ё", "е"))
                          + r"(?![а-яёa-z])", t)
            if m:
                hits.append((m.start(), sym))
                break
    return [sym for _, sym in sorted(hits)]


def extract_temporal(text):
    """Вербатимный спан для слота <TIME>: самый длинший chunk из TEMPORALS."""
    low = text.lower()
    for chunk in sorted(TEMPORALS, key=len, reverse=True):
        pattern = (r"(?<![а-яёa-z])" + re.escape(chunk).replace("ё", "[её]")
                   + r"(?![а-яёa-z])")
        m = re.search(pattern, low)
        if m:
            return m.group(0)
    return None


def has_temporal_hint(text):
    cleaned = GREET_RE.sub("", normalize(text))
    return bool(TEMPORAL_HINT.search(cleaned))


def auto_frame(intent, specs, symptoms, temporal):
    conflict = len(specs) >= 2 and intent != "unsupported"
    if intent == "unsupported":
        return Frame("unsupported", None, (), False, None)
    specialty = specs[0] if len(specs) == 1 else "unknown"
    return Frame(intent, specialty, tuple(symptoms), conflict, temporal,
                 extra_mentions=tuple(specs) if conflict else ())


def make_record(frame, text, row, extra=None):
    return {
        "id": hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:12],
        "split": "ext_test",
        "source": "hasd",
        "text": text,
        "target": target_string(frame),
        "frame": {
            "intent": frame.intent,
            "specialty": frame.specialty if frame.specialty != "unknown" else "unknown",
            "symptoms": list(frame.symptoms),
            "temporal": frame.temporal,
            "temporal_factors": list(frame.temporal_factors),
            "needs_clarification": frame.needs_clarification,
            "extra_mentions": list(frame.extra_mentions),
        },
        "provenance": {
            "skeleton": "hasd", "stt_variant": "nllb_greedy",
            "orig_intent": row["orig_intent"], "orig_split": row["orig_split"],
            "entity_labels": row["entity_labels"],
            **(extra or {}),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="перевести только N фраз (отладка)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_hasd()
    print(f"HASD загружен: {len(rows)} фраз")

    # фильтрация по сущностям и интентам
    candidates, dropped = [], Counter()
    for r in rows:
        if set(r["entity_labels"]) & DROP_ENTITY:
            dropped["entity_drop"] += 1
            continue
        if set(r["entity_labels"]) & REVIEW_ENTITY:
            r["review_reason"] = "appointment_type: возможен скрытый тип приёма"
        candidates.append(r)
    print("дроп по сущностям:", dict(dropped))

    cache_path = out / "hasd_translations.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [r for r in candidates if r["text_en"] not in cache]
    if args.limit:
        todo = todo[:args.limit]
    if todo:
        print(f"переводим {len(todo)} фраз (NLLB-600M, beams={beams})...")
        t0 = time.time()
        import torch
        tok, model, device, bos, beams = build_translator()
        translated = translate_batch(torch, tok, model, device, bos,
                                     [r["text_en"] for r in todo], args.batch, beams)
        for r, tr in zip(todo, translated):
            cache[r["text_en"]] = apply_glossary(tr.strip())
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        print(f"перевод занял {time.time() - t0:.0f}s")

    # автосборка + валидация
    framed, unsupported, review = [], [], []
    seen = set()
    for r in candidates:
        text = cache.get(r["text_en"])
        if not text:
            continue
        norm = normalize(text)
        if norm in seen:
            continue
        seen.add(norm)
        if len(text) > 300 or len(norm) < 3:
            r2 = {**r, "text_ru": text}
            r2["review_reason"] = "длина"
            review.append(r2)
            continue

        intent = INTENT_MAP[r["orig_intent"]]
        specs = detect_specialties(text)
        symptoms = detect_symptoms(text)
        temporal = extract_temporal(text)
        reason = r.get("review_reason")

        if intent in ("book", "reschedule", "cancel"):
            if temporal is None and has_temporal_hint(text):
                reason = reason or "временной намёк вне словаря TEMPORALS"
        if intent == "unsupported" and (specs or symptoms):
            reason = reason or f"unsupported, но найдены иголки: {specs or symptoms}"

        frame = auto_frame(intent, specs, symptoms, temporal)
        problems = validate_text(frame, text)

        rec = make_record(frame, text, r)
        if reason or problems:
            rec["review_reason"] = "; ".join(filter(None, [reason, *problems]))
            review.append(rec)
        elif intent == "unsupported":
            unsupported.append(rec)
        else:
            framed.append(rec)

    _write(out / "hasd_ru.jsonl", framed)
    _write(out / "hasd_unsupported_ru.jsonl", unsupported)
    _write(out / "hasd_needs_review.jsonl", review)

    stats = {
        "translated_total": len(cache),
        "framed": len(framed),
        "unsupported": len(unsupported),
        "needs_review": len(review),
        "framed_by_intent": dict(Counter(r["frame"]["intent"] for r in framed)),
        "framed_by_specialty": dict(Counter(r["frame"]["specialty"] for r in framed)),
        "with_temporal": sum(1 for r in framed if r["frame"]["temporal"]),
        "review_reasons": dict(Counter(r["review_reason"].split(";")[0].strip()
                                       for r in review)),
    }
    (out / "hasd_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\n--- примеры framed ---")
    for r in framed[:8]:
        print(" ", r["text"], "\n   ->", r["target"])
    print("\n--- примеры needs_review ---")
    for r in review[:6]:
        print(" ", r["text"], "\n   ??", r["review_reason"])


def _write(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
