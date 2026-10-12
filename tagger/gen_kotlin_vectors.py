# -*- coding: utf-8 -*-
"""Тест-векторы Kotlin для тегированной архитектуры.

Для контрольных фраз: BPE-ids (проверка токенизатора), ожидаемые
intent-класс и теги токенов (проверка инференса tagger_int8.tflite)
и итоговый semantic frame (проверка сборки на Kotlin).

Запуск: .venv-tf/Scripts/python tagger/gen_kotlin_vectors.py
Выход:  tagger/models/kotlin_vectors.json
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tagger"))

from tokenizers import Tokenizer  # noqa: E402
from annotate import TAGS, INTENTS  # noqa: E402
from assemble import frame_from_tags, spans_from_tags  # noqa: E402
from export_tflite import TfliteTagger  # noqa: E402

TEXTS = [
    "запишите меня к зубному в следующую пятницу вечером, зуб ноет",
    "запишите меня к терапевту завтра в 15:00",
    "запиши к неврологу в 9:45",
    "хочу к дерматологу в четверг в 17:15",
    "завтра в 15:00 или в 15:30 к хирургу, колено ноет",
    "отмени все мои записи",
    "сотрите все мои приёмы, пожалуйста",
    "можно к терапевту завтра утром, голова болит",
    "хочу отменить свою запись к окуристу",
    "перенесите приём к дерматологу на послезавтра после работы",
    "мне бы к хирургу, колено опухло и болит",
    "нужно попасть к неврологу как можно скорее, спину прострелило",
    "надо к кожнику попасть, чешется всё",
    "зуб разрывает, чё есть вечером после смены?",
    "у меня болит голова",
    "какая завтра погода в Москве",
    "поставь будильник на семь утра",
    "закажи пиццу на вечер",
    "запиши к невропатологу на завтра, мигрень",
]


def main():
    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    m = TfliteTagger(ROOT / "tagger" / "models" / "tagger_int8.tflite")

    out = {"tags": TAGS, "intents": INTENTS, "special_ids":
           {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
            "<USER>": 4, "<ASSISTANT>": 5}, "cases": []}
    for text in TEXTS:
        enc = tok.encode(text)
        x = torch.tensor([[2] + enc.ids])
        iv, tv = m(x)
        intent = int(iv.argmax())
        tags = tv.argmax(-1)[1:].tolist()           # без BOS
        frame = frame_from_tags(text, enc.ids, tags, intent, tok=tok)
        out["cases"].append({
            "text": text,
            "bpe_tokens": enc.tokens,
            "bpe_ids": enc.ids,
            "expected_intent": INTENTS[intent],
            "expected_tags": [TAGS[t] for t in tags],
            "expected_frame": frame,
        })
        print(f"{text[:46]:48s} -> {frame}")

    path = ROOT / "tagger" / "models" / "kotlin_vectors.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\nзаписано: {path} ({len(out['cases'])} кейсов)")


if __name__ == "__main__":
    main()
