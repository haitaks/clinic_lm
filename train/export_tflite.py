# -*- coding: utf-8 -*-
"""Экспорт модели в TFLite для Android (TensorFlow Lite / LiteRT).

Запуск:  python train/export_tflite.py
Выход:   models/model.tflite        (fp32, ~27 МБ)
         models/model_int8.tflite   (weight-quant int8, ~8 МБ)

Зачем TFLite, если уже есть model.onnx: Java-обёртка ONNX Runtime тянет
в APK собственную libonnxruntime.so, которая конфликтует с QNN-сборкой
sherpa-onnx (версионированные vs неверсионированные символы OrtGetApiBase,
один процесс — одна библиотека с таким именем). TFLite — независимый
рантайм без onnxruntime-символов: в APK остаётся только sherpa, конфликт
снят без пересборки нативных библиотек.

Пайплайн: model.pt --TF-зеркало--> tflite. Граф собирается из чистых
tf-операций с весами из чекпойнта (математика 1:1 с train/model.py,
включая tied embeddings и точный GELU через erf), конвертация штатным
TFLiteConverter. Квантизация int8 — dynamic range (веса int8, активации
fp32), калибровочный датасет не нужен. Путь через onnx2tf пробовали:
fp32 работает, но int8-квантование этой версии выдаёт NaN — потому
граф и конвертация здесь собственные. fp16-вариант тоже отброшен:
ядро MEAN не готовит тендрвы в fp16 на CPU.

Проверки: парность логитов torch vs tflite (fp32 < 1e-3) и сквозная
генерация constrained decoding — фреймы должны совпасть посимвольно
для fp32 и int8.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from tokenizers import Tokenizer  # noqa: E402

from model import TinyGPT  # noqa: E402
from constrained import generate_frame  # noqa: E402

CONTEXT = 128
NEG_MASK = -1e9   # вместо -inf: exp(-1e9) == 0.0 в fp32, но без inf в графе


def build_tf_mirror(sd, cfg):
    """tf.Module, повторяющий TinyGPT.forward(idx) при T=128 без паддинга.

    Веса — tf.constant из state_dict: конвертеру не нужны переменные,
    tied head использует ту же матрицу tok.weight (assert ниже).
    """
    import tensorflow as tf

    tok = tf.constant(sd["tok.weight"].numpy())            # [V, d]
    pos = tf.constant(sd["pos.weight"].numpy())            # [128, d]
    assert np.allclose(sd["head.weight"].numpy(), sd["tok.weight"].numpy())
    causal = tf.constant(np.triu(np.ones((CONTEXT, CONTEXT), bool), 1))

    d, layers, heads = cfg["d_model"], cfg["n_layers"], cfg["n_heads"]
    dh = d // heads
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

    def ln(x, w, b, eps=1e-5):
        mu = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mu), axis=-1, keepdims=True)
        return (x - mu) * tf.math.rsqrt(var + eps) * w + b

    def block(x, w):
        h = ln(x, w["ln1.weight"], w["ln1.bias"])
        qkv = tf.matmul(h, w["attn.qkv.weight"], transpose_b=True) \
            + w["attn.qkv.bias"]
        qkv = tf.transpose(tf.reshape(qkv, [1, CONTEXT, 3, heads, dh]),
                           [2, 0, 3, 1, 4])                # [3,1,H,T,dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = tf.matmul(q, k, transpose_b=True) / np.sqrt(dh)
        att = tf.where(causal, NEG_MASK, att)
        att = tf.nn.softmax(att, axis=-1)
        o = tf.matmul(att, v)                               # [1,H,T,dh]
        o = tf.reshape(tf.transpose(o, [0, 2, 1, 3]), [1, CONTEXT, d])
        x = x + tf.matmul(o, w["attn.proj.weight"], transpose_b=True) \
            + w["attn.proj.bias"]
        h = ln(x, w["ln2.weight"], w["ln2.bias"])
        h = tf.matmul(h, w["mlp.0.weight"], transpose_b=True) \
            + w["mlp.0.bias"]
        h = 0.5 * h * (1.0 + tf.math.erf(h / np.sqrt(2.0)))  # GELU, exact
        return x + tf.matmul(h, w["mlp.2.weight"], transpose_b=True) \
            + w["mlp.2.bias"]

    class Mirror(tf.Module):

        @tf.function(input_signature=[
            tf.TensorSpec([1, CONTEXT], tf.int64)])
        def __call__(self, ids):
            x = tf.gather(tok, ids) + pos[None]
            for w in blocks:
                x = block(x, w)
            return tf.matmul(ln(x, ln_f["weight"], ln_f["bias"]),
                             tok, transpose_b=True)          # tied head

    return Mirror()


class TfliteModel:
    """Обёртка для generate_frame: контракт как у OnnxModel в
    export_onnx.py — torch [1, T] на вход, torch [1, T, V] на выход."""

    def __init__(self, model_path, vocab, pad_id):
        import tensorflow as tf
        self.sess = tf.lite.Interpreter(model_path=str(model_path),
                                        num_threads=1)
        self.sess.allocate_tensors()
        self.inp = self.sess.get_input_details()[0]
        self.out = self.sess.get_output_details()[0]
        self.vocab, self.pad_id = vocab, pad_id
        self.context = CONTEXT
        assert list(self.inp["shape"]) == [1, CONTEXT], self.inp["shape"]
        assert list(self.out["shape"]) == [1, CONTEXT, vocab], self.out["shape"]
        assert np.dtype(self.inp["dtype"]).kind == "i", self.inp["dtype"]

    def __call__(self, idx):
        T = idx.shape[1]
        full = np.full((1, CONTEXT), self.pad_id)
        full[0, :T] = idx.detach().cpu().numpy()
        self.sess.set_tensor(self.inp["index"],
                             full.astype(self.inp["dtype"]))
        self.sess.invoke()
        logits = self.sess.get_tensor(self.out["index"])
        return torch.from_numpy(logits[:, :T].copy())


def main():
    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    ckpt = torch.load(ROOT / "models" / "model.pt", map_location="cpu",
                      weights_only=False)
    model = TinyGPT(tok.get_vocab_size(), **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    import tensorflow as tf
    mirror = build_tf_mirror(ckpt["model"], ckpt["config"])

    def convert(optimizations):
        conv = tf.lite.TFLiteConverter.from_concrete_functions(
            [mirror.__call__.get_concrete_function()], mirror)
        conv.optimizations = optimizations
        return conv.convert()

    out_fp32 = ROOT / "models" / "model.tflite"
    out_int8 = ROOT / "models" / "model_int8.tflite"
    out_fp32.write_bytes(convert([]))
    out_int8.write_bytes(convert([tf.lite.Optimize.DEFAULT]))  # dynamic range
    for p in (out_fp32, out_int8):
        print(f"экспортировано: {p} ({p.stat().st_size / 1e6:.1f} МБ)")

    # --- парность логитов (те же входы, что в export_onnx.py) ---
    pad_id = tok.token_to_id("<PAD>")
    fp32_m = TfliteModel(out_fp32, tok.get_vocab_size(), pad_id)
    int8_m = TfliteModel(out_int8, tok.get_vocab_size(), pad_id)
    rng = np.random.default_rng(0)
    for T in (5, 17, 100):
        ids = rng.integers(0, tok.get_vocab_size(), size=(1, T)).astype(np.int64)
        with torch.no_grad():
            ref = model(torch.from_numpy(ids)).numpy()
        got = fp32_m(torch.from_numpy(ids)).numpy()
        diff = float(np.abs(ref[0] - got[0]).max())
        print(f"  fp32 T={T:3d}: max|torch-tflite| = {diff:.2e}")
        assert diff < 1e-3, "fp32 разошлась сильнее 1e-3"
        got8 = int8_m(torch.from_numpy(ids)).numpy()
        diff8 = float(np.abs(ref[0] - got8[0]).max())
        am = ref[0, T - 1].argmax() == got8[0, T - 1].argmax()
        print(f"  int8 T={T:3d}: max|torch-int8|  = {diff8:.2e}, "
              f"argmax последней позиции совпал: {am}")

    # --- сквозная проверка: генерация через tflite == torch ---
    ids = {t: tok.token_to_id(t) for t in
           ("<PAD>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>")}
    text = "запишите меня к зубному в следующую пятницу вечером, зуб ноет"
    kw = dict(bos=ids["<BOS>"], user=ids["<USER>"],
              assistant=ids["<ASSISTANT>"], eos=ids["<EOS>"],
              pad=ids["<PAD>"])
    dev = torch.device("cpu")
    pred_torch, _ = generate_frame(model, tok, text, dev, **kw)
    pred_fp32, _ = generate_frame(fp32_m, tok, text, dev, **kw)
    pred_int8, _ = generate_frame(int8_m, tok, text, dev, **kw)
    print(f"  torch: {pred_torch}")
    print(f"  fp32:  {pred_fp32}")
    print(f"  int8:  {pred_int8}")
    assert pred_fp32 == pred_torch, "fp32-генерация разошлась"
    assert pred_int8 == pred_torch, "int8-генерация разошлась"
    print("парность подтверждена")


if __name__ == "__main__":
    main()
