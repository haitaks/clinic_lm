# -*- coding: utf-8 -*-
"""Экспорт теггера в TFLite для Android (LiteRT).

Контракт (проще, чем у GPT-варианта: ОДИН forward вместо 7-15 шагов
генерации):
    input_ids : int64[1, 128]          // BOS + bpe, хвост <PAD>
    intent    : float32[1, 4]
    tags      : float32[1, 128, 4]     // логиты; тег i-го токена = tags[0, i]

Граф — TF-зеркало tagger/model.py (математика 1:1: без каузальной
маски, pad-ключи скрыты -1e9, mean-pool по не-паддингу, tied нет).
Проверки: парность intent/tags логитов torch vs tflite (<1e-3) и
сквозная сборка фрейма (python-сборка поверх tflite-выхода == torch).

Запуск: .venv-tf/Scripts/python tagger/export_tflite.py
Выход:  tagger/models/tagger.tflite (+ tagger_int8.tflite).
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tagger"))

from tokenizers import Tokenizer  # noqa: E402
from model import Tagger  # noqa: E402

CONTEXT = 128
NEG = -1e9


def build_tf_mirror(sd, cfg):
    import tensorflow as tf

    d, layers, heads = cfg["d_model"], cfg["n_layers"], cfg["n_heads"]
    dh = d // heads

    tok = tf.constant(sd["tok.weight"].numpy())
    pos = tf.constant(sd["pos.weight"].numpy())

    blocks = []
    for i in range(layers):
        p = f"blocks.{i}."
        blocks.append({k: tf.constant(sd[p + k].numpy()) for k in (
            "ln1.weight", "ln1.bias",
            "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias",
            "ln2.weight", "ln2.bias",
            "mlp.0.weight", "mlp.0.bias",
            "mlp.2.weight", "mlp.2.bias",
        )})
    ln_f = {k: tf.constant(sd[f"ln_f.{k}"].numpy()) for k in ("weight", "bias")}
    w_int = (tf.constant(sd["intent_head.weight"].numpy()),
             tf.constant(sd["intent_head.bias"].numpy()))
    w_tag = (tf.constant(sd["tag_head.weight"].numpy()),
             tf.constant(sd["tag_head.bias"].numpy()))

    def ln(x, w, b, eps=1e-5):
        mu = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mu), axis=-1, keepdims=True)
        return (x - mu) * tf.math.rsqrt(var + eps) * w + b

    def block(x, w, pad):
        h = ln(x, w["ln1.weight"], w["ln1.bias"])
        qkv = tf.matmul(h, w["attn.qkv.weight"], transpose_b=True) \
            + w["attn.qkv.bias"]
        qkv = tf.transpose(tf.reshape(qkv, [1, CONTEXT, 3, heads, dh]),
                           [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = tf.matmul(q, k, transpose_b=True) / np.sqrt(dh)
        att = tf.where(pad[:, None, None, :], NEG, att)
        att = tf.nn.softmax(att, axis=-1)
        o = tf.matmul(att, v)
        o = tf.reshape(tf.transpose(o, [0, 2, 1, 3]), [1, CONTEXT, d])
        x = x + tf.matmul(o, w["attn.proj.weight"], transpose_b=True) \
            + w["attn.proj.bias"]
        h = ln(x, w["ln2.weight"], w["ln2.bias"])
        h = tf.matmul(h, w["mlp.0.weight"], transpose_b=True) \
            + w["mlp.0.bias"]
        h = 0.5 * h * (1.0 + tf.math.erf(h / np.sqrt(2.0)))
        return x + tf.matmul(h, w["mlp.2.weight"], transpose_b=True) \
            + w["mlp.2.bias"]

    class Mirror(tf.Module):

        @tf.function(input_signature=[
            tf.TensorSpec([1, CONTEXT], tf.int64)])
        def __call__(self, ids):
            pad = tf.cast(tf.equal(ids, 0), tf.float32)      # <PAD> id = 0
            x = tf.gather(tok, ids) + pos[None]
            for w in blocks:
                x = block(x, w, tf.cast(pad, tf.bool))
            x = ln(x, ln_f["weight"], ln_f["bias"])
            keep = 1.0 - pad
            pooled = tf.reduce_sum(x * keep[..., None], axis=1) / \
                tf.maximum(tf.reduce_sum(keep, axis=1, keepdims=True), 1.0)
            intent = tf.matmul(pooled, w_int[0], transpose_b=True) + w_int[1]
            tags = tf.matmul(x, w_tag[0], transpose_b=True) + w_tag[1]
            return {"intent": intent, "tags": tags}

    return Mirror()


class TfliteTagger:
    """Обёртка: ids torch [1, T] -> (intent [4], tags [T, 4]) torch."""

    def __init__(self, path):
        import tensorflow as tf
        self.s = tf.lite.Interpreter(model_path=str(path), num_threads=1)
        self.s.allocate_tensors()
        self.i = self.s.get_input_details()[0]
        # выходы различаем по форме: intent [1,4], tags [1,128,4]
        # (в графе имена Identity/Identity_1, порядок не гарантирован)
        outs = self.s.get_output_details()
        assert len(outs) == 2, outs
        if len(outs[0]["shape"]) == 2:
            self.intent_ix, self.tags_ix = outs[0]["index"], outs[1]["index"]
        else:
            self.intent_ix, self.tags_ix = outs[1]["index"], outs[0]["index"]

    def __call__(self, idx):
        T = idx.shape[1]
        full = np.zeros((1, CONTEXT), dtype=np.int64)
        full[0, :T] = idx.detach().cpu().numpy()
        self.s.set_tensor(self.i["index"], full)
        self.s.invoke()
        intent = self.s.get_tensor(self.intent_ix)
        tags = self.s.get_tensor(self.tags_ix)
        return (torch.from_numpy(intent[0].copy()),
                torch.from_numpy(tags[0, :T].copy()))


def main():
    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    ckpt = torch.load(ROOT / "tagger" / "tagger.pt", map_location="cpu",
                      weights_only=False)
    model = Tagger(tok.get_vocab_size())
    model.load_state_dict(ckpt["model"])
    model.eval()

    import tensorflow as tf
    mirror = build_tf_mirror(ckpt["model"],
                             {"d_model": 256, "n_layers": 8, "n_heads": 8})

    def convert(optimizations):
        conv = tf.lite.TFLiteConverter.from_concrete_functions(
            [mirror.__call__.get_concrete_function(
                tf.TensorSpec([1, CONTEXT], tf.int64))], mirror)
        conv.optimizations = optimizations
        return conv.convert()

    out_dir = ROOT / "tagger" / "models"
    out_dir.mkdir(exist_ok=True)
    fp32 = out_dir / "tagger.tflite"
    int8 = out_dir / "tagger_int8.tflite"
    fp32.write_bytes(convert([]))
    int8.write_bytes(convert([tf.lite.Optimize.DEFAULT]))
    for p in (fp32, int8):
        print(f"экспортировано: {p} ({p.stat().st_size / 1e6:.1f} МБ)")

    # --- парность логитов ---
    rng = np.random.default_rng(0)
    m32 = TfliteTagger(fp32)
    m8 = TfliteTagger(int8)
    for T in (5, 17, 100):
        ids = rng.integers(10, tok.get_vocab_size(), size=(1, T))
        ids = np.concatenate([np.full((1, 1), 2), ids], axis=1)  # BOS
        x = torch.from_numpy(ids)
        pad = x == 0
        with torch.no_grad():
            ri, rt = model(x, pad)
        ti, tt = m32(x)
        d1 = (ri - ti).abs().max().item()
        d2 = (rt - tt).abs().max().item()
        print(f"  fp32 T={T:3d}: intent {d1:.2e}, tags {d2:.2e}")
        assert d1 < 1e-3 and d2 < 1e-3, "fp32 разошлась"
        i8i, i8t = m8(x)
        same = (ri.argmax() == i8i.argmax()).item() and \
            bool((rt.argmax(-1) == i8t.argmax(-1)).all())
        print(f"  int8 T={T:3d}: argmax совпал: {same}")

    # --- сквозная сборка фрейма через tflite ---
    from assemble import frame_from_tags
    from annotate import INTENTS
    text = "запишите меня к зубному в следующую пятницу вечером, зуб ноет"
    enc = tok.encode(text)
    x = torch.tensor([[2] + enc.ids])
    with torch.no_grad():
        ri, rt = model(x, torch.zeros_like(x, dtype=torch.bool))
    ti, tt = m32(x)
    for label, (iv, tv) in (("torch", (ri, rt)), ("tflite", (ti, tt))):
        tags = tv.argmax(-1)[1:].tolist()          # без BOS
        frame = frame_from_tags(text, enc.ids, tags,
                                int(iv.argmax()), tok=tok)
        print(f"  {label}: {frame}")


if __name__ == "__main__":
    main()
