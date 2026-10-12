# -*- coding: utf-8 -*-
"""Тест-векторы для проверки Kotlin-реализации (Android).

Генерирует models/kotlin_testvectors.json: для контрольных фраз —
BPE-токены (проверка токенизатора), ожидаемый semantic frame от
tflite-модели int8 (проверка инференса + автомата). На устройстве:
текст -> ids обязаны совпасть; ids -> фрейм обязан совпасть.

Запуск:  .venv-tf/Scripts/python train/gen_kotlin_testvectors.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from tokenizers import Tokenizer  # noqa: E402

from constrained import generate_frame  # noqa: E402
from export_tflite import TfliteModel  # noqa: E402

TEXTS = [
    "запишите меня к зубному в следующую пятницу вечером, зуб ноет",
    "запишите меня к терапевту завтра в 15:00",
    "запиши к неврологу в 9:45",
    "хочу к дерматологу в четверг в 17:15",
    "завтра в 15:00 или в 15:30 к хирургу",
    "отмени все мои записи",
    "можно к терапевту завтра утром, голова болит",
    "хочу отменить свою запись к окуристу",
    "перенесите приём к дерматологу на послезавтра после работы",
    "мне бы к хирургу, колено опухло и болит",
    "запишите меня к психотерапевту в среду в десять тридцать",
    "нужно попасть к неврологу как можно скорее, спину прострелило",
    "зуб разрывает, чё есть вечером после смены?",
    "какая завтра погода в Москве",
    "поставь будильник на семь утра",
    "закажи пиццу на вечер",
    "к дантисту можно записаться?",
]


def main():
    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    ids = {t: tok.token_to_id(t) for t in
           ("<PAD>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>")}

    m8 = TfliteModel(ROOT / "models" / "model_int8.tflite",
                     tok.get_vocab_size(), ids["<PAD>"])

    out = {"special_ids": ids, "cases": []}
    for text in TEXTS:
        enc = tok.encode(text)
        frame, _ = generate_frame(
            m8, tok, text, torch.device("cpu"),
            bos=ids["<BOS>"], user=ids["<USER>"],
            assistant=ids["<ASSISTANT>"], eos=ids["<EOS>"],
            pad=ids["<PAD>"])
        out["cases"].append({
            "text": text,
            "bpe_tokens": enc.tokens,
            "bpe_ids": enc.ids,
            "expected_frame": frame,
        })
        print(f"{text[:48]:50s} -> {frame}")

    path = ROOT / "models" / "kotlin_testvectors.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\nзаписано: {path} ({len(out['cases'])} кейсов)")


if __name__ == "__main__":
    main()
