# -*- coding: utf-8 -*-
"""Экспорт модели в ONNX для Android (ONNX Runtime Mobile).

Запуск:  python train/export_onnx.py
Выход:   models/model.onnx (+ проверка парности с torch до 1e-4)

Для разработчика Android:
- вход:  int64 [1, T] — id токенов последовательности
         <BOS> <USER> {текст} <ASSISTANT> [сгенерированные токены]
- выход: float32 [1, T, V] — логиты; для следующего токена брать [0, T-1, :]
- дальше: маска по автомату слотов (train/constrained.py — референс)
  и argmax; повторять до <END_FRAME>/<EOS>.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from tokenizers import Tokenizer
from model import TinyGPT
from constrained import generate_frame


class ExportWrapper(torch.nn.Module):
    """Модель без pad_mask: на инференсе батч = 1 без паддинга."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx):
        return self.model(idx)


def main():
    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer.json"))
    ckpt = torch.load(ROOT / "models" / "model.pt", map_location="cpu")
    model = TinyGPT(tok.get_vocab_size(), **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapper = ExportWrapper(model)

    # nn.MultiheadAttention плохо экспортируется с динамической осью:
    # фиксируем длину 128, инференс всегда паддит PAD-ом (каузальная маска
    # гарантирует, что хвост не влияет на позиции до него)
    dummy = torch.randint(0, tok.get_vocab_size(), (1, 128), dtype=torch.long)
    out_path = ROOT / "models" / "model.onnx"
    (ROOT / "models" / "model.onnx.data").unlink(missing_ok=True)  # сайдкар
    # один файл: веса инлайном, без model.onnx.data. external_data появился
    # в torch 2.5; в 2.4 сайдкар и так не создаётся для моделей < 2 ГБ
    export_kwargs = dict(
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
    )
    if "external_data" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["external_data"] = False
    torch.onnx.export(wrapper, dummy, str(out_path), **export_kwargs)
    print(f"экспортировано: {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} МБ, вход [1,128])")
    # torch.onnx.export может оставить модуль в train-режиме -> dropout
    # зашумит torch-референс в проверке парности
    model.eval()

    # --- проверка парности torch vs onnxruntime ---
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path),
                                providers=["CPUExecutionProvider"])
    pad_id = tok.token_to_id("<PAD>")
    rng = np.random.default_rng(0)
    for T in (5, 17, 100):
        ids = rng.integers(0, tok.get_vocab_size(), size=(1, T)).astype(np.int64)
        full = np.full((1, 128), pad_id, dtype=np.int64)
        full[0, :T] = ids
        with torch.no_grad():
            ref = model(torch.from_numpy(ids)).numpy()
        got = sess.run(None, {"input_ids": full})[0][0, :T]
        diff = float(np.abs(ref[0] - got).max())
        print(f"  T={T:3d}: max|torch-onnx| = {diff:.2e}")
        assert diff < 1e-4, "расхождение больше 1e-4"

    # --- сквозная проверка: генерация через ONNX совпадает с torch ---
    class OnnxModel:
        def __init__(self, sess, vocab, pad_id):
            self.sess, self.vocab, self.pad_id = sess, vocab, pad_id
            self.context = 128

        def __call__(self, idx):
            T = idx.shape[1]
            full = torch.full((1, 128), self.pad_id, dtype=torch.long)
            full[0, :T] = idx[0]
            logits = self.sess.run(None, {"input_ids":
                                          full.numpy().astype(np.int64)})[0]
            return torch.from_numpy(logits[:, :T])

    ids = {t: tok.token_to_id(t) for t in
           ("<PAD>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>")}
    text = "запишите меня к зубному в следующую пятницу вечером, зуб ноет"
    onnx_model = OnnxModel(sess, tok.get_vocab_size(), tok.token_to_id("<PAD>"))
    pred_onnx, _ = generate_frame(
        onnx_model, tok, text, torch.device("cpu"),
        bos=ids["<BOS>"], user=ids["<USER>"], assistant=ids["<ASSISTANT>"],
        eos=ids["<EOS>"], pad=ids["<PAD>"])
    pred_torch, _ = generate_frame(
        model, tok, text, torch.device("cpu"),
        bos=ids["<BOS>"], user=ids["<USER>"], assistant=ids["<ASSISTANT>"],
        eos=ids["<EOS>"], pad=ids["<PAD>"])
    print(f"  torch: {pred_torch}")
    print(f"  onnx:  {pred_onnx}")
    assert pred_onnx == pred_torch, "генерация разошлась"
    print("парность подтверждена")


if __name__ == "__main__":
    main()
