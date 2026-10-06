# -*- coding: utf-8 -*-
"""Локальный тест модели: файл со строками-запросами → semantic frame.

Использование:
    python predict.py examples.txt               # человекочитаемый вывод
    python predict.py examples.txt --tsv         # TSV: текст<TAB>фрейм
    python predict.py examples.txt -o ответ.txt  # результат в файл
    echo "запиши к терапевту" | python predict.py   # stdin

Каждая строка файла — один запрос. Пустые строки пропускаются с пометкой.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "train"))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from model import TinyGPT  # noqa: E402
from constrained import generate_frame  # noqa: E402

MODEL_PATH = ROOT / "models" / "model.pt"
TOKENIZER_PATH = ROOT / "models" / "tokenizer.json"

DEMO = [
    "запишите меня к зубному в следующую пятницу вечером, зуб ноет",
    "можно к терапевту завтра утром",
    "перенеси мой приём к кардиологу",
    "какая завтра погода в Москве",
]


def load():
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = TinyGPT(tok.get_vocab_size(), **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, dict(
        bos=tok.token_to_id("<BOS>"),
        user=tok.token_to_id("<USER>"),
        assistant=tok.token_to_id("<ASSISTANT>"),
        eos=tok.token_to_id("<EOS>"),
        pad=tok.token_to_id("<PAD>"),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?",
                    help="файл: одна строка = один запрос "
                         "(без аргумента — stdin; пустой stdin — демо)")
    ap.add_argument("-o", "--output", help="записать результат в файл")
    ap.add_argument("--tsv", action="store_true",
                    help="формат TSV: текст<TAB>фрейм")
    args = ap.parse_args()

    # кириллица в консоли Windows
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.input:
        lines = Path(args.input).read_text(encoding="utf-8-sig").splitlines()
    elif not sys.stdin.isatty():
        lines = sys.stdin.read().splitlines()
    else:
        lines = []
    if not lines:
        print("(вход не передан — прогоняю демо-примеры)", file=sys.stderr)
        lines = DEMO

    model, tok, special_ids = load()

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for i, raw in enumerate(lines, 1):
            text = raw.strip()
            if not text:
                result, ms = "(пустая строка — пропущена)", 0.0
            else:
                t0 = time.perf_counter()
                frame, _ = generate_frame(model, tok, text,
                                          torch.device("cpu"), **special_ids)
                result, ms = frame, (time.perf_counter() - t0) * 1000
            if args.tsv:
                print(f"{text}\t{result}", file=out)
            else:
                print(f"[{i}] {text}", file=out)
                print(f"    → {result}   ({ms:.0f} мс)", file=out)
    finally:
        if out is not sys.stdout:
            out.close()
            print(f"сохранено: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
