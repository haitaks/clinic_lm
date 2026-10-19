# -*- coding: utf-8 -*-
"""Экспорт мультилейбл-модели для Java: линейный граф + спецификация признаков.

Внутренний токенизатор skl2onnx (TfIdfVectorizer op) не повторяет sklearn
на кириллице (零-признаки / потеря n-грамм), поэтому контракт повторяет
проверенную схему frame-модели: препроцессинг на клиенте, в граф уходит
готовый вектор.

Артефакты:
  models/multilabel_model.onnx           float32[None, F] -> вероятности тегов
  models/multilabel_features.json  словарь признаков + idf + алгоритм
  models/multilabel_testvectors.json  тексты -> ожидаемые решения тегов
                                    (юнит-тест для Java-реализации)

Парность: эталонная реализация векторизации (та же, что описана для
Java) обязана совпадать со sklearn до машинного нуля, а ONNX-граф — до
1e-4 (fp32 матmul). Проверяется на контрольных фразах автоматически.

Запуск:  python train/export_multilabel_onnx.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

import joblib  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

PROBES = [
    "запишите меня к стоматологу завтра утром",
    "не могу прийти в четверг, перенесите",
    "хочу к глазнику попасть на послезавтра",
    "вызовите врача на дом маме",
    "сколько стоит приём кардиолога",
    "готовы ли анализы",
    "который час",
    "да",
    "здравствуйте, можно к дантисту на этой неделе?",
    "какие документы нужны для прикрепления",
]


def tokenize(text: str) -> list:
    """Референс токенизации = sklearn token_pattern по умолчанию."""
    import re
    return re.findall(r"\w\w+", text.lower())


def vectorize(text: str, vocab: dict, idf: np.ndarray) -> np.ndarray:
    """Референс TF-IDF для Java: word(1,2), sublinear, l2."""
    tokens = tokenize(text)
    counts = {}
    for i in range(len(tokens)):
        counts[tokens[i]] = counts.get(tokens[i], 0) + 1
        if i + 1 < len(tokens):
            bg = tokens[i] + " " + tokens[i + 1]
            counts[bg] = counts.get(bg, 0) + 1
    vec = np.zeros(len(vocab), dtype=np.float64)
    for term, cnt in counts.items():
        j = vocab.get(term)
        if j is not None and cnt > 0:
            vec[j] = (1.0 + np.log(cnt)) * idf[j]
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def main():
    bundle = joblib.load(ROOT / "models" / "multilabel_export.joblib")
    pipeline = bundle["pipeline"]
    columns = bundle["columns"]
    vect = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    vocab = vect.vocabulary_
    idf = vect.idf_

    # --- эталон векторизации против sklearn: машинный нуль
    for p in PROBES:
        ref = vectorize(p, vocab, idf)
        sk = vect.transform([p]).toarray()[0]
        d = np.abs(ref - sk).max()
        assert d < 1e-9, f"референс векторизации разошлся со sklearn: {d}"

    # --- линейный граф: float вектор -> вероятности тегов
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    n_feat = len(vocab)
    onx = to_onnx(clf, initial_types=[("features", FloatTensorType([None, n_feat]))],
                  options={id(clf): {"zipmap": False}})
    out = ROOT / "models" / "multilabel_model.onnx"
    out.write_bytes(onx.SerializeToString())
    print(f"сохранено: {out} ({out.stat().st_size / 1e6:.1f} МБ, "
          f"признаков {n_feat}, тегов {len(columns)})")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    assert out_names == ["label", "probabilities"], out_names

    feats = np.array([vectorize(p, vocab, idf) for p in PROBES],
                     dtype=np.float32)
    # выходы LinearClassifier: [label (int64), probabilities (float32)]
    ort_out = np.asarray(sess.run(None, {"features": feats})[1])
    sk_out = pipeline.predict_proba(PROBES)
    diff = np.abs(ort_out - sk_out).max()
    same = (ort_out.argmax(axis=1) == np.asarray(sk_out).argmax(axis=1)).all()
    print(f"макс. |sklearn - onnx|: {diff:.2e}, argmax совпадает: {same}")
    assert diff < 1e-4 and same, "парность нарушена"

    # --- спецификация признаков для Java
    terms = [None] * n_feat
    for term, j in vocab.items():
        terms[j] = term
    (ROOT / "models" / "multilabel_features.json").write_text(
        json.dumps({
            "algorithm": ("lowercase -> tokens=regex \\w\\w+ (2+ симв.) -> "
                          "unigram + биграмма соседних токенов через пробел -> "
                          "tf=1+ln(count) -> tf*idf -> L2-норма"),
            "vocabulary": terms,
            "idf": [round(float(x), 8) for x in idf],
            "threshold": bundle["threshold"],
        }, ensure_ascii=False), encoding="utf-8")
    print("спецификация: models/multilabel_features.json")

    # --- порядок тегов выхода (для Java)
    (ROOT / "models" / "multilabel_tags.json").write_text(
        json.dumps([{"group": g, "tag": t} for g, t in columns],
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("порядок тегов: models/multilabel_tags.json")

    # --- тест-векторы для юнит-теста Java
    th = bundle["threshold"]
    vectors = []
    for p in PROBES:
        proba = pipeline.predict_proba([p])[0]
        intent = max((i for i, c in enumerate(columns) if c[0] == "intent"),
                     key=lambda i: proba[i])
        vectors.append({
            "text": p,
            "expect_intent": columns[intent][1],
            "expect_tags": [columns[i][1] for i in range(len(columns))
                            if proba[i] >= th],
        })
    (ROOT / "models" / "multilabel_testvectors.json").write_text(
        json.dumps(vectors, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"тест-векторы: models/multilabel_testvectors.json ({len(vectors)} фраз)")


if __name__ == "__main__":
    main()
