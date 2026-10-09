# -*- coding: utf-8 -*-
"""Decoder-only Transformer (GPT-style) для semantic parsing заявок.

Чистый torch без зависимостей от HF: pre-LN, causal attention,
tied embeddings. Конфиг по умолчанию ~3.5M параметров ("nano" из
обзора); флагами поднимается до base/strong.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadSelfAttention(nn.Module):
    """Собственная реализация: nn.MultiheadAttention плохо экспортируется
    в ONNX с динамикой, а трассировка ловит fast-path с запеченными
    формами. Здесь только matmul/softmax — экспорт детерминирован."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal, pad_mask):
        B, T, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]          # [B, h, T, dh]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        # -1e9 вместо -inf: exp(-1e9) == 0.0 в fp32, ключи-паддинг получают
        # нулевое внимание, но в графе нет inf и нет nan_to_num. Полностью
        # замаскированных строк не бывает (правый паддинг, BOS первый),
        # softmax конечен всегда. Семантика 1:1 с TF-зеркалом export_tflite.
        NEG_MASK = -1e9
        att = att.masked_fill(causal, NEG_MASK)
        if pad_mask is not None:                   # ключи-паддинг не видны
            att = att.masked_fill(pad_mask[:, None, None, :], NEG_MASK)
        att = att.softmax(dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask, pad_mask):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, attn_mask, pad_mask))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_layers=6, n_heads=6,
                 ffn_dim=768, context=128, dropout=0.1):
        super().__init__()
        self.context = context
        # Каузальная маска — константа контекста. persistent=False:
        # в state_dict не попадает (старые чекпойнты грузятся как раньше),
        # а экспортёру не нужно строить Trilu из torch.triu — некоторые
        # ONNX->TF конвертеры на нём спотыкаются.
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context, context, dtype=torch.bool),
                       diagonal=1),
            persistent=False)
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, pad_mask=None):
        # idx: [B, T]; pad_mask: [B, T] True на паддинге
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None])
        causal = self.causal_mask[:T, :T]
        for blk in self.blocks:
            x = blk(x, causal, pad_mask)
        return self.head(self.ln_f(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_batch(sequences, pad_id, device):
    """sequences: list[list[int]] -> idx/labels/pad_mask на device."""
    maxlen = max(len(s) for s in sequences)
    idx = torch.full((len(sequences), maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(sequences):
        idx[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    pad_mask = idx == pad_id
    return idx.to(device), pad_mask.to(device)
