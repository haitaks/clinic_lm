# -*- coding: utf-8 -*-
"""JSON-словарь для Kotlin-сборки фрейма теггера.

Всё, что нужно Android-стороне для детерминированной сборки: иголки
специальностей/симптомов (канонизация спанов), таблица времени
(фраза -> факторы), категории факторов (порядок в слоте), поверхности
дней для «или»-альтернатив. Единый источник правды — synth/vocab.py;
поверхности с ё-фолдингом (текст нормализуется так же перед матчингом).

Запуск: .venv-tr/Scripts/python tagger/gen_kotlin_vocab.py
Выход:  tagger/models/vocab_kotlin.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "synth"))

from vocab import (SPECIALTIES, SYMPTOMS, TEMPORAL_ENTRIES,  # noqa: E402
                   TIME_SCOPE, TIME_DAY, TIME_TIME, AMBIG_DAY_SURFACES,
                   PLAUSIBLE)

OUT = Path(__file__).resolve().parent / "models"

# порядок предложения врачей для неоднозначных симптомов (настройка
# под клинику; дублирует tagger/routing.py:SYM_PRIORITY)
SYM_PRIORITY = {
    "headache": ["neurologist", "gp", "ophthalmologist", "gynecologist"],
    "stomach_pain": ["gp", "surgeon", "gynecologist"],
    "back_pain": ["neurologist", "surgeon", "gp"],
    "eye_pain": ["ophthalmologist", "gp"],
    "rash": ["dermatologist", "gp"],
    "allergy": ["dermatologist", "ent", "gp"],
    "sore_throat": ["ent", "gp"],
    "cough": ["ent", "gp"],
    "fever": ["gp"],
    "dizziness": ["neurologist", "gp"],
}


def _invert(plausible):
    out = {}
    for spec, symptoms in plausible.items():
        for sym in symptoms:
            out.setdefault(sym, []).append(spec)
    return out


def main():
    spec = {}
    for key, entry in SPECIALTIES.items():
        surfaces = sorted({f.replace("ё", "е")
                           for syn in entry["train"] + entry["ood"]
                           for f in syn.values()})
        spec[key] = {"token": entry["token"], "surfaces": surfaces}
    sym = {}
    for key, entry in SYMPTOMS.items():
        surfaces = sorted({e.replace("ё", "е")
                           for e in entry["train"] + entry["ood"]})
        sym[key] = {"token": entry["token"], "surfaces": surfaces}
    temporal = {e["text"].replace("ё", "е"): e["factors"]
                for e in TEMPORAL_ENTRIES}
    out = {
        # тег-классы: ключ -> токен фрейма + поверхности (матчинг по
        # границам слов, однословные иголки ловят хвост-склонение до
        # 3 букв — Kotlin-реализация в ANDROID_INTEGRATION.md)
        "spec": spec,
        "sym": sym,
        "temporal": temporal,
        "time_categories": {
            "scope": sorted(TIME_SCOPE),
            "day": sorted(TIME_DAY),
            "time": sorted(TIME_TIME),
        },
        "ambig_day_surfaces": AMBIG_DAY_SURFACES,
        # routing «симптом -> врач»: матрица (симптом -> специальности,
        # у которых он в профиле) + приоритеты неоднозначных случаев;
        # первичный кандидат = первый по таблице приоритетов, gp —
        # дефолт. Референс-логика: tagger/routing.py
        "routing": {
            "sym_to_specs": {sym: sorted(specs)
                             for sym, specs in _invert(PLAUSIBLE).items()},
            "sym_priority": SYM_PRIORITY,
            "spec_tokens": {k: v["token"] for k, v in SPECIALTIES.items()},
        },
        "tags": ["O", "SPEC", "SYM", "TIME"],
        "intents": ["book", "reschedule", "cancel", "unsupported"],
    }
    path = OUT / "vocab_kotlin.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    n_spec = sum(len(v["surfaces"]) for v in spec.values())
    n_sym = sum(len(v["surfaces"]) for v in sym.values())
    print(f"записано: {path}")
    print(f"специальностей: {len(spec)} ({n_spec} поверхностей), "
          f"симптомов: {len(sym)} ({n_sym}), временных фраз: {len(temporal)}")


if __name__ == "__main__":
    main()
