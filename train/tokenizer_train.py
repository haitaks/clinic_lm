# -*- coding: utf-8 -*-
"""Обучение BPE-токенизатора на корпусе заявок.

byte-level BPE (без [UNK] вообще: любые опечатки/эмодзи бьются по байтам),
семантические токены добавляются как неделимые special tokens.

Запуск:  python train/tokenizer_train.py
Выход:   models/tokenizer.json
"""

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "synth"))

from vocab import (SPECIALTIES, SYMPTOMS, TEMPORAL_ENTRIES,
                   TIME_SCOPE, TIME_DAY, TIME_TIME)  # noqa: E402

SPECIAL_TOKENS = [
    "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>",
    "<TIME>", "</TIME>", "<NEEDS_CLARIFICATION>", "<END_FRAME>",
    "<BOOK>", "<RESCHEDULE>", "<CANCEL>", "<UNSUPPORTED>",
    "<UNKNOWN_SPECIALTY>", "<T_UNKNOWN>", "T_AMBIG",
]
for entry in SPECIALTIES.values():
    SPECIAL_TOKENS.append(entry["token"])
for entry in SYMPTOMS.values():
    SPECIAL_TOKENS.append(entry["token"])
SPECIAL_TOKENS += sorted(TIME_SCOPE | TIME_DAY | TIME_TIME)

VOCAB_SIZE = 3000


def corpus_texts():
    """Тексты и target-последовательности: BPE учится на обеих частях."""
    lines = []
    for name in ("train", "val", "test_iid", "test_ood", "manual_gold"):
        p = ROOT / "data" / f"{name}.jsonl"
        if not p.exists():
            continue
        for row in map(json.loads, p.read_text(encoding="utf-8").splitlines()):
            lines.append(row["text"])
            lines.append(row["target"])
    clinc = ROOT / "data" / "clinc_ru.jsonl"
    if clinc.exists():
        for row in map(json.loads, clinc.read_text(encoding="utf-8").splitlines()):
            if row["split"] == "train_aug":
                lines.append(row["text"])
    return lines


def main():
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    texts = corpus_texts()
    print(f"обучаем BPE на {len(texts)} строках, "
          f"спецтокенов: {len(SPECIAL_TOKENS)}, vocab: {VOCAB_SIZE}")
    tok.train_from_iterator(texts, trainer=trainer)
    # trainer только резервирует id; atomicity при encode даёт
    # add_special_tokens (AddedVocabulary матчит их до BPE)
    tok.add_special_tokens(SPECIAL_TOKENS)

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    tok.save(str(out / "tokenizer.json"))

    # sanity: спецтокены неделимы, русский бьётся без UNK
    enc = tok.encode("хочу записаться к стоматологу в следующую пятницу вечером")
    assert tok.token_to_id("<TIME>") is not None
    assert tok.token_to_id("T_NEXT_WEEK") is not None
    assert tok.token_to_id("<SYMPTOM_TOOTHACHE>") is not None
    for t in ("<BOOK>", "<DENTIST>", "T_FRI", "<END_FRAME>"):
        assert tok.token_to_id(t) is not None, t
    seq = "<BOOK> <DENTIST> <TIME> T_NEXT_WEEK T_FRI </TIME> <END_FRAME>"
    ids = [tok.token_to_id(t) for t in
           ("<BOOK>", "<DENTIST>", "T_NEXT_WEEK", "T_FRI", "<END_FRAME>")]
    enc_special = tok.encode(seq)
    assert all(i in enc_special.ids for i in ids), "спецтокен разбился на части!"
    print(f"токенов в словаре: {tok.get_vocab_size()}")
    print(f"пример: '{enc.tokens[:12]}'")
    print(f"сохранено: {out / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
