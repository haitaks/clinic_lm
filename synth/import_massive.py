# -*- coding: utf-8 -*-
"""MASSIVE ru-RU: майнинг временных фраз и референс calendar-фраз.

Источник: amazon-massive-dataset-1.1.tar.gz -> 1.1/data/ru-RU.jsonl
(16 521 профессионально переведённая фраза). Скрипт:

1. извлекает значения слотов date/time/date_time (формат "[time : пять утра]")
   и пишет ранжированный по частоте список кандидатов в
   data/massive_temporal_candidates.json;
2. сохраняет полные фразы calendar/datetime-сценариев в
   data/massive_calendar_reference.jsonl — референс для будущего майнинга
   скелетов;
3. сверяет курируемый список CURATED (уже добавлен в vocab.TEMPORALS)
   с частотами источника — отчёт о происхождении.

Кураторские правила (почему не всё подряд):
- только словесные фразы без цифр и знаков — правило вербатимного слота
  <TIME> при STT-нормализации;
- прошедшее время ("вчера", "прошлой неделе") — не желаемое время визита;
- длительности ("полчаса", "два часа", "десять минут") — это не время записи;
- голые числа и числа без "утра/вечера/дня" — неоднозначны;
- месяцы/числа месяцев ("первое апреля") — отложено до v2 хардкод-парсера.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

SLOT_RE = re.compile(r"\[(\w+)\s*:\s*([^\]]+)\]")
TEMPORAL_SLOT_TYPES = {"date", "time", "date_time"}
CALENDAR_SCENARIOS = {"calendar", "datetime"}

# Курируемый список, добавленный в vocab.TEMPORALS (блок "из MASSIVE").
# Проверяется против частот источника для отчёта.
CURATED = [
    # абсолютное время словами
    "в восемь утра", "в девять утра", "в десять утра", "в одиннадцать утра",
    "в полдень", "в час дня", "в два часа дня", "в три часа дня",
    "в четыре часа дня", "в пять вечера", "в шесть вечера", "в семь вечера",
    "в восемь вечера", "в девять вечера", "в десять вечера",
    "в восемь тридцать утра", "в девять тридцать утра",
    "в пять тридцать вечера", "в шесть тридцать вечера",
    # комбинации день+время
    "завтра утром", "завтра вечером", "послезавтра утром",
    "в пятницу вечером", "в субботу утром",
    "в эти выходные", "на выходных", "на этих выходных",
    # относительные
    "через три дня", "через месяц", "в следующем месяце",
]


def load_rows(path: Path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/_massive/1.1/data/ru-RU.jsonl")
    out = Path("data")
    rows = load_rows(src)
    print(f"MASSIVE ru-RU: {len(rows)} фраз")

    freq = Counter()
    for r in rows:
        for typ, val in SLOT_RE.findall(r["annot_utt"]):
            if typ in TEMPORAL_SLOT_TYPES:
                v = val.strip().lower()
                if re.fullmatch(r"[а-яёa-z ]{3,30}", v):
                    freq[v] += 1

    (out / "massive_temporal_candidates.json").write_text(
        json.dumps([{"phrase": p, "freq": c} for p, c in freq.most_common()],
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"уникальных словесных временных фраз: {len(freq)} "
          f"-> data/massive_temporal_candidates.json")

    # референс полных фраз для майнинга скелетов
    ref = [{"intent": r["intent"], "utt": r["utt"]}
           for r in rows if r["scenario"] in CALENDAR_SCENARIOS]
    with open(out / "massive_calendar_reference.jsonl", "w", encoding="utf-8") as f:
        for r in ref:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"calendar/datetime фраз для майнинга скелетов: {len(ref)} "
          f"-> data/massive_calendar_reference.jsonl")

    # отчёт по курируемому списку: подтверждены ли фразы источником
    print("\nкурируемый список против MASSIVE (freq в источнике):")
    missing = []
    for c in CURATED:
        # слот хранит значение без предлога: "в девять утра" -> "девять утра"
        stripped = re.sub(r"^(в|во|на|через)\s+", "", c)
        f1, f2 = freq.get(c, 0), freq.get(stripped, 0)
        if f1 == 0 and f2 == 0:
            missing.append(c)
        print(f"  {c:32s} прямых={f1:3d} без предлога={f2:3d}")
    if missing:
        print("\nНЕ подтверждены источником:", missing)


if __name__ == "__main__":
    main()
