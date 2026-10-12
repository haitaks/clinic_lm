# -*- coding: utf-8 -*-
"""Двунаправленный энкодер для joint intent+tagging NLU.

Те же блоки, что у train/model.py TinyGPT, но без каузальной маски
(tagging видит весь текст) и с двумя головами:
- intent: mean-pool по не-паддингу -> Linear -> 4 класса
- tags:   по токенам -> Linear -> {O, SPEC, SYM, TIME}

Дублирование train/model.py сознательное: эксперимент изолирован
в tagger/ и должен читаться без прыжков по каталогам.
"""

import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        B, T, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]              # [B, h, T, dh]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        NEG = -1e9
        if pad_mask is not None:                       # паддинг не виден
            att = att.masked_fill(pad_mask[:, None, None, :], NEG)
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        x = x + self.drop(self.attn(self.ln1(x), pad_mask))
        return x + self.mlp(self.ln2(x))


class Tagger(nn.Module):
    def __init__(self, vocab_size, n_intents=4, n_tags=4,
                 d_model=256, n_layers=8, n_heads=8, ffn_dim=1024,
                 context=128, dropout=0.1):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, ffn_dim, dropout)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.intent_head = nn.Linear(d_model, n_intents)
        self.tag_head = nn.Linear(d_model, n_tags)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, pad_mask=None):
        """idx [B, T] (BOS + bpe-токены, хвост <PAD>), pad_mask [B, T]
        True на паддинге. -> (intent_logits [B, n_intents],
                              tag_logits [B, T, n_tags])"""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None])
        for blk in self.blocks:
            x = blk(x, pad_mask)
        x = self.ln_f(x)
        if pad_mask is None:
            pooled = x.mean(dim=1)
        else:
            keep = (~pad_mask).unsqueeze(-1).float()
            pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.intent_head(pooled), self.tag_head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
