# -*- coding: utf-8 -*-
"""Constrained decoding: конечный автомат поверх слотов target.

Модель выбирает ЗНАЧЕНИЯ, автомат гарантирует синтаксис:
порядок INTENT -> SPECIALTY -> TIME -> SYMPTOMS -> [CLARIF] -> END.
Нейросеть физически не может сгенерировать невалидную последовательность.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "synth"))

from vocab import SPECIALTIES, SYMPTOMS, TIME_SCOPE, TIME_DAY  # noqa: E402

INTENTS = ["<BOOK>", "<RESCHEDULE>", "<CANCEL>", "<UNSUPPORTED>"]
SPECIALTY_TOKENS = [e["token"] for e in SPECIALTIES.values()] + ["<UNKNOWN_SPECIALTY>"]
SYMPTOM_TOKENS = [e["token"] for e in SYMPTOMS.values()]
ALL_TIME_FACTORS = TIME_SCOPE | TIME_DAY | {
    "T_MORNING", "T_AFTERNOON", "T_EVENING", "T_EARLY", "T_MIDDAY",
    "T_BEFORE_NOON", "T_AFTER_NOON", "T_AFTER_18", "T_AFTER_WORK",
    "T_AFTER_SHIFT", "T_SOON",
    "T_H08_00", "T_H08_30", "T_H09_00", "T_H09_30", "T_H10_00", "T_H11_00",
    "T_H13_00", "T_H14_00", "T_H15_00", "T_H16_00", "T_H17_00", "T_H17_30",
    "T_H18_00", "T_H18_30", "T_H19_00", "T_H20_00", "T_H21_00", "T_H22_00",
}

RANK = {"scope": 0, "day": 1, "time": 2}


def _cat(token):
    if token in TIME_SCOPE:
        return "scope"
    if token in TIME_DAY:
        return "day"
    return "time"


class FrameAutomaton:
    """Шаг за шаго возвращает допустимое множество токенов."""

    def __init__(self, tokenizer):
        self.id_of = tokenizer.token_to_id
        self.initialized = False
        self.state = "intent"
        self.last_factor_rank = -1
        self.factors_emitted = 0
        self.emitted_symptoms = set()

    def _ids(self, tokens):
        return {self.id_of(t) for t in tokens if self.id_of(t) is not None}

    def allowed(self):
        if self.state == "intent":
            return self._ids(INTENTS)
        if self.state == "end_frame":
            return self._ids(["<END_FRAME>"])
        if self.state == "specialty":
            return self._ids(SPECIALTY_TOKENS)
        if self.state == "time":
            return self._ids(["<T_UNKNOWN>", "<TIME>"])
        if self.state == "time_factors":
            allowed = {t for t in ALL_TIME_FACTORS
                       if RANK[_cat(t)] > self.last_factor_rank}
            if self.factors_emitted:
                allowed.add("</TIME>")
            return self._ids(allowed)
        if self.state == "symptoms":
            symps = {t for t in SYMPTOM_TOKENS if t not in self.emitted_symptoms}
            return self._ids(symps | {"<NEEDS_CLARIFICATION>", "<END_FRAME>"})
        if self.state == "clarification":
            return self._ids(["<END_FRAME>"])
        if self.state == "done":
            return {self.id_of("<EOS>")}
        return set()

    def step(self, token_id):
        tok = None
        for t in INTENTS + SPECIALTY_TOKENS + SYMPTOM_TOKENS + [
                "<T_UNKNOWN>", "<TIME>", "</TIME>", "<T_AMBIG>",
                "<NEEDS_CLARIFICATION>", "<END_FRAME>", "<EOS>"] + list(ALL_TIME_FACTORS):
            if self.id_of(t) == token_id:
                tok = t
                break
        if tok is None:
            self.state = "done"
            return

        if self.state == "intent":
            self.state = "end_frame" if tok == "<UNSUPPORTED>" else "specialty"
        elif self.state == "specialty":
            self.state = "time"
        elif self.state == "time":
            self.state = "symptoms" if tok == "<T_UNKNOWN>" else "time_factors"
        elif self.state == "time_factors":
            if tok == "</TIME>":
                self.state = "symptoms"
            else:
                self.last_factor_rank = RANK[_cat(tok)]
                self.factors_emitted += 1
        elif self.state == "symptoms":
            if tok in self.emitted_symptoms:
                return
            if tok in SYMPTOM_TOKENS:
                self.emitted_symptoms.add(tok)
            elif tok == "<NEEDS_CLARIFICATION>":
                self.state = "clarification"
            elif tok == "<END_FRAME>":
                self.state = "done"
        elif self.state == "clarification":
            if tok == "<END_FRAME>":
                self.state = "done"
        elif self.state == "end_frame":
            if tok == "<END_FRAME>":
                self.state = "done"


@torch.no_grad()
def generate_frame(model, tokenizer, text, device, max_steps=24,
                   bos=None, user=None, assistant=None, eos=None, pad=None):
    """Жадная генерация target с ограничением автоматом."""
    prefix = [bos, user] + tokenizer.encode(text).ids + [assistant]

    autom = FrameAutomaton(tokenizer)
    seq = list(prefix)
    for _ in range(max_steps):
        ctx = seq[-model.context:]
        logits = model(torch.tensor([ctx], device=device))[0, -1]
        allowed = autom.allowed()
        if not allowed:
            break
        mask = torch.full_like(logits, float("-inf"))
        mask[list(allowed)] = 0.0
        nxt = int((logits + mask).argmax())
        if nxt == eos:
            break
        seq.append(nxt)
        autom.step(nxt)
        if autom.state == "done":
            seq.append(eos)
            break

    special_ids = {bos, user, assistant, eos, pad}
    out_tokens = [tokenizer.id_to_token(i) for i in seq[len(prefix):]]
    semantic = [t for t in out_tokens
                if t is not None and t != "<EOS>" and t != "<PAD>"]
    return " ".join(semantic), semantic
