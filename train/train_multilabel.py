# -*- coding: utf-8 -*-
"""Мультилейбл-классификатор фрейма: TF-IDF -> OneVsRest LogReg -> теги.

По предложению из PDF (TF-IDF + бинарный классификатор на каждый тег),
адаптировано под групповую грамматику фрейма: независимые классификаторы
на теги + групповое декодирование
    intent (15)     -> argmax
    specialty       -> argmax, подавляется для да/нет/unsupported
    симптомы/уточн. -> порог (0.5, свип по val в отчёт)
    время           -> НЕ классифицируется: вербатим-матчинг TEMPORAL-
                       таблицы (проектное решение: время = хардкод-парсер)

Защита от незнакомых слов — сравниваются четыре ветки:
    (а) word        — word(1,2) TF-IDF, как в PDF
    (б) lemma       — + лемматизация pymorphy2 (внешний морфологический
                      словарь) корпуса и входа
    (в) lemma+dict  — + замены по models/oov_dict.json (словарь синонимов
                      из synth/vocab.py; применяется только к словам вне
                      словаря модели)
    (г) lemma+dict+char — + char_wb(2,5) n-граммы (экспортируются внутрь
                      ONNX-графа)

Выход: models/multilabel.joblib, models/multilabel_report.txt,
        models/oov_dict.json, models/multilabel_model.onnx (ветка-победитель).

Запуск:  python train/train_multilabel.py          # все ветки + экспорт
         python train/train_multilabel.py --quick  # без свипа порога
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "synth"))

from train import load_split  # noqa: E402  (корпус + негативы CLINC/HASD)
from vocab import (SPECIALTIES, SYMPTOMS, TEMPORAL_ENTRIES, REQUESTS,
                   GREETINGS, POLITENESS, UNSUPPORTED_POOLS, TEMPORALS,
                   UNSUPPORTED_LEADS, UNSUPPORTED_LEADS_NO_GREET,
                   ANSWER_SURFACES)  # noqa: E402

TRIVIAL_INTENTS = {"<YES>", "<NO>", "<UNSUPPORTED>"}
END = "<END_FRAME>"
THRESHOLD = 0.5

# ---------------------------------------------------------------- теги/фрейм

def parse_tags(target: str):
    """target-строка -> (intent, specialty, symptoms, clarification)."""
    toks = target.split()
    intent = toks[0]
    if intent in TRIVIAL_INTENTS:
        return intent, None, [], False
    specialty = toks[1] if len(toks) > 1 else "<UNKNOWN_SPECIALTY>"
    rest = toks[2:]
    if rest and rest[0] == "<TIME>":
        end = rest.index("</TIME>") if "</TIME>" in rest else len(rest)
        rest = rest[end + 1:]
    elif rest and rest[0] == "<T_UNKNOWN>":
        rest = rest[1:]
    symptoms = [t for t in rest if t.startswith("<SYMPTOM_")]
    return intent, specialty, symptoms, "<NEEDS_CLARIFICATION>" in rest


def gold_time_factors(target: str) -> list:
    toks = target.split()
    rest = toks[2:] if len(toks) > 2 and toks[0] not in TRIVIAL_INTENTS else []
    if rest and rest[0] == "<TIME>":
        end = rest.index("</TIME>") if "</TIME>" in rest else len(rest)
        return rest[1:end]
    if rest and rest[0] == "<T_UNKNOWN>":
        return ["<T_UNKNOWN>"]
    return ["<T_UNKNOWN>"] if toks[0] not in TRIVIAL_INTENTS else []


_TEMPORALS_SORTED = sorted(
    ((e["text"].replace("ё", "е"), e["factors"]) for e in TEMPORAL_ENTRIES),
    key=lambda x: -len(x[0]))


def match_time(text: str) -> list:
    """Вербатим-матчинг временной фразы: самый длинный TEMPORAL-вход."""
    t = text.lower().replace("ё", "е")
    for phrase, factors in _TEMPORALS_SORTED:
        esc = re.escape(phrase)
        if re.search(r"(?<![а-яa-z0-9])" + esc + r"(?![а-яa-z0-9])", t):
            return list(factors)
    return ["<T_UNKNOWN>"]


def build_frame(intent, specialty, symptoms, clarification, factors):
    if intent in TRIVIAL_INTENTS:
        return f"{intent} {END}"
    parts = [intent, specialty or "<UNKNOWN_SPECIALTY>"]
    if factors and factors != ["<T_UNKNOWN>"]:
        parts += ["<TIME>"] + factors + ["</TIME>"]
    else:
        parts.append("<T_UNKNOWN>")
    parts += sorted(symptoms)
    if clarification:
        parts.append("<NEEDS_CLARIFICATION>")
    parts.append(END)
    return " ".join(parts)


# ------------------------------------------------------------ препроцессинг

_morph = None


def _lemmatize(text: str) -> str:
    global _morph
    if _morph is None:
        # pymorphy2 0.9.1 использует inspect.getargspec, удалённый в
        # Python 3.12; getfullargspec возвращает супермножество — режем
        import inspect
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = lambda f: inspect.getfullargspec(f)[:4]
        import pymorphy2
        _morph = pymorphy2.MorphAnalyzer()
    out = []
    for tok in re.findall(r"[а-яёА-ЯЁa-zA-Z0-9:.,!?-]+|\S", text):
        if re.fullmatch(r"[а-яёА-ЯЁ-]+", tok) and len(tok) > 1:
            out.append(_morph.parse(tok)[0].normal_form)
        else:
            out.append(tok)
    return " ".join(out)


def _vocab_words() -> set:
    """Служебная лексика генератора: замены не касаются её."""
    words = set()
    for phrases in REQUESTS.values():
        words.update(phrases)
    words.update(GREETINGS)
    words.update(POLITENESS)
    words.update(TEMPORALS)
    words.update(UNSUPPORTED_LEADS)
    words.update(UNSUPPORTED_LEADS_NO_GREET)
    for pool in UNSUPPORTED_POOLS.values():
        words.update(pool)
    for pool in ANSWER_SURFACES.values():
        words.update(pool)
    out = set()
    for phrase in words:
        out.update(w.lower() for w in re.findall(r"[а-яё0-9:-]+", phrase))
    return out


def build_oov_dict() -> dict:
    """Синоним специальностей/симптомов -> каноническое слово.

    Ключ — слово из синонимичной поверхности, значение — главное слово
    канонической поверхности того же класса. Применять только к словам,
    которых нет в словаре модели (защита от порчи известных фраз).
    """
    mapping = {}
    groups = []
    # специальности: синоним = словарь падежных форм {nom, dat, ugen...}
    for entry in SPECIALTIES.values():
        groups.append([form for syn in entry["train"] + entry["ood"]
                       for form in syn.values()])
    # симптомы: поверхности — строки
    for entry in SYMPTOMS.values():
        groups.append(entry["train"] + entry.get("ood", []))
    for surfaces in groups:
        canon_word = next(
            (w for w in re.findall(r"[а-яё0-9:-]+", surfaces[0])
             if len(w) > 3), None)
        if not canon_word:
            continue
        for surface in surfaces[1:]:
            for w in re.findall(r"[а-яё0-9:-]+", surface):
                if len(w) > 3 and w != canon_word:
                    mapping.setdefault(w, canon_word)
    return mapping


def apply_oov_dict(text: str, mapping: dict, known: set,
                   service: set) -> str:
    out = []
    for tok in re.findall(r"[а-яёА-ЯЁ0-9:.,!?-]+|\S", text):
        low = tok.lower().replace("ё", "е")
        if (low in mapping and low not in known
                and low not in service and len(low) > 3):
            out.append(mapping[low])
        else:
            out.append(tok)
    return " ".join(out)


# ----------------------------------------------------------------- обучение

def tag_columns(rows):
    """Список (группа, тег) по корпусу — стабильный порядок."""
    intents, specs, syms = [], set(), set()
    for r in rows:
        intent, spec, symptoms, _ = parse_tags(r["target"])
        if intent not in intents:
            intents.append(intent)
        if spec and spec != "<UNKNOWN_SPECIALTY>":
            specs.add(spec)
        syms.update(symptoms)
    syms.discard("<NEEDS_CLARIFICATION>")
    columns = ([("intent", t) for t in sorted(intents)]
               + [("spec", t) for t in sorted(specs)]
               # явный тег «специальность не названа»: без него argmax
               # тычет в случайную именованную на репликах без специальности
               + [("spec", "<UNKNOWN_SPECIALTY>")]
               + [("sym", t) for t in sorted(syms)]
               + [("clar", "<NEEDS_CLARIFICATION>")])
    return columns


def make_y(rows, columns):
    idx = {c: i for i, c in enumerate(columns)}
    y = np.zeros((len(rows), len(columns)), dtype=np.int8)
    for ri, r in enumerate(rows):
        intent, spec, symptoms, clar = parse_tags(r["target"])
        y[ri, idx[("intent", intent)]] = 1
        if spec and spec != "<UNKNOWN_SPECIALTY>":
            y[ri, idx[("spec", spec)]] = 1
        else:
            y[ri, idx[("spec", "<UNKNOWN_SPECIALTY>")]] = 1
        for s in symptoms:
            y[ri, idx[("sym", s)]] = 1
        if clar:
            y[ri, idx[("clar", "<NEEDS_CLARIFICATION>")]] = 1
    return y


def make_model(char_features: bool) -> Pipeline:
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    if char_features:
        feats = FeatureUnion([
            ("w", word),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                  sublinear_tf=True, min_df=2)),
        ])
    else:
        feats = word
    return Pipeline([
        ("tfidf", feats),
        ("clf", OneVsRestClassifier(
            LogisticRegression(C=1.0, max_iter=1000, solver="liblinear",
                               random_state=7),
            # n_jobs=1: параллельный joblib + scipy/numpy 2.x падает на
            # csr_sort_indices (WRITEBACKIFCOPY base is read-only)
            n_jobs=1)),
    ])


class Branch:
    """Ветка препроцессинга + обученная модель + предиктор фрейма."""

    def __init__(self, name, lemma, use_dict, char):
        self.name, self.lemma, self.use_dict, self.char = name, lemma, use_dict, char
        self.model = None
        self.columns = None
        self.known = None
        self.oov = None
        self.service = _vocab_words()

    def prep(self, text, fit_vocabulary=None):
        t = text
        if self.lemma:
            t = _lemmatize(t)
        if self.use_dict and self.oov:
            known = fit_vocabulary if fit_vocabulary is not None else self.known
            t = apply_oov_dict(t, self.oov, known, self.service)
        return t

    def fit(self, rows, columns):
        self.columns = columns
        self.oov = build_oov_dict() if self.use_dict else {}
        texts_raw = [r["text"] for r in rows]
        # словарь модели нужен для правила «заменяем только незнакомое»;
        # строим по word-признакам сырого текста (лемматизация не меняет
        # состав лексем настолько, чтобы это повлияло на правило)
        base_vect = TfidfVectorizer(ngram_range=(1, 2)).fit(texts_raw)
        self.known = {w.replace("ё", "е") for w in base_vect.vocabulary_}
        texts = [self.prep(t) for t in texts_raw]
        self.model = make_model(self.char)
        self.model.fit(texts, make_y(rows, columns))

    def predict_tags(self, texts):
        """-> list[(intent, spec, symptoms, clar, probas_by_group)]."""
        prepped = [self.prep(t) for t in texts]
        proba = self.model.predict_proba(prepped)
        idx = {c: i for i, c in enumerate(self.columns)}
        out = []
        for pi in range(len(texts)):
            intent = max((c for c in self.columns if c[0] == "intent"),
                         key=lambda c: proba[pi, idx[c]])[1]
            spec, symptoms, clar = None, [], False
            if intent not in TRIVIAL_INTENTS:
                spec = max((c for c in self.columns if c[0] == "spec"),
                           key=lambda c: proba[pi, idx[c]],
                           default=("<UNKNOWN_SPECIALTY>",))[1]
                symptoms = [c[1] for c in self.columns
                            if c[0] == "sym" and proba[pi, idx[c]] >= THRESHOLD]
                clar = proba[pi, idx[("clar", "<NEEDS_CLARIFICATION>")]] >= THRESHOLD
            out.append((intent, spec, symptoms, clar))
        return out

    def predict(self, texts):
        frames = []
        for text, (intent, spec, sym, clar) in zip(
                texts, self.predict_tags(texts)):
            frames.append(build_frame(intent, spec, sym, clar,
                                      match_time(text)))
        return frames


# ------------------------------------------------------------------ метрики

def evaluate(branch, rows, limit=0):
    rows = rows[:limit] if limit else rows
    texts = [r["text"] for r in rows]
    preds = branch.predict(texts)
    st = {"n": 0, "exact": 0, "intent": 0, "spec_n": 0, "spec": 0,
          "time_n": 0, "time": 0, "symp_n": 0, "symp": 0}
    for r, p in zip(rows, preds):
        g_intent, g_spec, g_sym, _ = parse_tags(r["target"])
        p_intent, p_spec, p_sym, _ = parse_tags(p)
        st["n"] += 1
        st["exact"] += p == r["target"]
        st["intent"] += p_intent == g_intent
        if g_intent not in TRIVIAL_INTENTS:
            st["spec_n"] += 1
            st["spec"] += p_spec == g_spec
            st["time_n"] += 1
            st["time"] += (match_time(r["text"]) == gold_time_factors(r["target"]))
        if g_sym or p_sym:
            st["symp_n"] += 1
            st["symp"] += set(g_sym) == set(p_sym)
    n = max(st["n"], 1)
    return {"exact": st["exact"] / n, "intent": st["intent"] / n,
            "spec": st["spec"] / max(st["spec_n"], 1),
            "time": st["time"] / max(st["time_n"], 1),
            "symp": st["symp"] / max(st["symp_n"], 1)}


def acceptance(branch):
    """40 регламентных реплик из test_nlu_acceptance (без torch)."""
    from test_nlu_acceptance import ROWS
    fails = []
    for scenario, text, intent, spec, mode, symptoms in ROWS:
        pred = branch.predict([text])[0]
        p = parse_tags(pred)
        got = p[0]
        problems = []
        if mode == "any":
            if got in ("<YES>", "<NO>"):
                problems.append("ложное да/нет")
        elif mode == "symptom":
            if got in ("<YES>", "<NO>"):
                problems.append("ложное да/нет")
            if symptoms and not symptoms.issubset(set(p[2])):
                problems.append(f"нет {sorted(symptoms)}")
        else:
            if got != intent:
                problems.append(f"интент {got} != {intent}")
            if spec is not None and p[1] != spec:
                problems.append(f"спец {p[1]} != {spec}")
            if got == "<UNSUPPORTED>" and intent is not None:
                problems.append("уход в UNSUPPORTED")
        if problems:
            fails.append(f"[{scenario}] «{text}» -> {pred}: "
                         + "; ".join(problems))
    return 40 - len(fails), fails


DISTORTED = [
    "запишитесь пожалуста к стамотологу на завтра",
    "хочу к глазнику попасть",
    "надо к зубнику срочно",
    "костоправ нужен на послезавтра",
    "перепишите мне приём к лору пожалста",
    "записана хочу проверить к неврапатологу",
    "отмените мой визит к кардиалогу",
    "вызавите врача на дом маме",
    "талончик к терапевту можно",
    "болит горло чем полоскать дома подскажите",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="без свипа порога симптомов")
    args = ap.parse_args()

    t0 = time.time()
    train_rows = load_split("train", with_clinc=True)
    sets = {}
    for name, fname, extra in [
            ("val", "val.jsonl", None), ("iid", "test_iid.jsonl", None),
            ("ood", "test_ood.jsonl", 200),
            ("manual", "manual_gold.jsonl", None),
            ("hasd", "hasd_ru.jsonl", 250)]:
        rows = [json.loads(l) for l in
                (ROOT / "data" / fname).read_text(encoding="utf-8").splitlines()]
        sets[name] = rows[:extra] if extra else rows
    clinc = [json.loads(l) for l in
             (ROOT / "data" / "clinc_ru.jsonl").read_text(encoding="utf-8").splitlines()]
    hasd_unsup = [json.loads(l) for l in
                  (ROOT / "data" / "hasd_unsupported_ru.jsonl").read_text(
                      encoding="utf-8").splitlines()]
    sets["hasd_unsup"] = [r for r in hasd_unsup if r["split"] == "ext_test"]
    sets["clinc"] = [r for r in clinc if r["split"] == "ext_test"]
    print(f"train: {len(train_rows)}, наборы: "
          + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))

    # словарь замен — артефакт для Java
    (ROOT / "models").mkdir(exist_ok=True)
    oov_dict = build_oov_dict()
    (ROOT / "models" / "oov_dict.json").write_text(
        json.dumps({"rule": "применять только к словам вне словаря модели",
                    "mapping": oov_dict}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"oov_dict.json: {len(oov_dict)} замен")

    columns = tag_columns(train_rows)
    print(f"тегов: {len(columns)} "
          f"(intent={sum(1 for c in columns if c[0]=='intent')}, "
          f"spec={sum(1 for c in columns if c[0]=='spec')}, "
          f"sym={sum(1 for c in columns if c[0]=='sym')})")

    branches = [
        Branch("(а) word", lemma=False, use_dict=False, char=False),
        Branch("(б) lemma", lemma=True, use_dict=False, char=False),
        Branch("(в) lemma+dict", lemma=True, use_dict=True, char=False),
        Branch("(г) lemma+dict+char", lemma=True, use_dict=True, char=True),
        # экспортопригодная: skl2onnx не конвертирует char_wb-анализатор,
        # а лемматизация pymorphy не переносится в граф. Ветка (д) — то,
        # что уезжает в multilabel_model.onnx: словарь замен применяет Java.
        Branch("(д) dict (экспорт)", lemma=False, use_dict=True, char=False),
    ]
    report = []
    for br in branches:
        t1 = time.time()
        br.fit(train_rows, columns)
        lines = [f"=== {br.name}  (обучение {time.time()-t1:.0f}s) ==="]
        for name, rows in sets.items():
            m = evaluate(br, rows)
            lines.append(f"  {name:10s} n={len(rows):4d} "
                         f"exact={m['exact']:.3f} intent={m['intent']:.3f} "
                         f"spec={m['spec']:.3f} time={m['time']:.3f} "
                         f"symp={m['symp']:.3f}")
        acc, fails = acceptance(br)
        lines.append(f"  приёмка: {acc}/40"
                     + ("" if acc == 40 else f"  промахи: {fails[:3]}"))
        block = "\n".join(lines)
        print(block)
        report.append(block)

    # исковерканные фразы: разница веток видна наглядно
    lines = ["=== исковерканные фразы ==="]
    for text in DISTORTED:
        lines.append(f"  «{text}»")
        for br in branches:
            lines.append(f"    {br.name:20s} {br.predict([text])[0]}")
    print("\n".join(lines))
    report.append("\n".join(lines))

    if not args.quick:
        lines = ["=== свип порога симптомов (val, микро-F1 симптомов) ==="]
        best, best_f1 = THRESHOLD, -1.0
        val_rows = sets["val"]
        gold_sets = [set(parse_tags(r["target"])[2]) for r in val_rows]
        br = branches[-1]
        prepped = [br.prep(r["text"]) for r in val_rows]
        proba = br.model.predict_proba(prepped)
        sym_idx = [i for i, c in enumerate(br.columns) if c[0] == "sym"]
        for th in (0.3, 0.4, 0.5, 0.6, 0.7):
            y_true = np.array([[1 if br.columns[i][1] in g else 0
                                for i in sym_idx] for g in gold_sets])
            y_pred = (proba[:, sym_idx] >= th).astype(int)
            f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
            lines.append(f"  th={th:.1f}: {f1:.3f}")
            if f1 > best_f1:
                best, best_f1 = th, f1
        lines.append(f"  лучший порог: {best} (F1 {best_f1:.3f})")
        print("\n".join(lines))
        report.append("\n".join(lines))

    winner = max(branches, key=lambda b: (evaluate(b, sets["ood"])["intent"],
                                          evaluate(b, sets["val"])["exact"]))
    print(f"ветка-победитель: {winner.name}")

    out = ROOT / "models" / "multilabel.joblib"
    joblib.dump({"pipeline": winner.model, "columns": winner.columns,
                 "threshold": THRESHOLD, "branch": winner.name,
                 "lemma": winner.lemma, "use_dict": winner.use_dict,
                 "oov_dict": winner.oov, "known_vocab": sorted(winner.known)},
                out)
    print(f"сохранено: {out}")

    export_br = next((b for b in branches if "экспорт" in b.name), branches[0])
    out_exp = ROOT / "models" / "multilabel_export.joblib"
    joblib.dump({"pipeline": export_br.model, "columns": export_br.columns,
                 "threshold": THRESHOLD, "branch": export_br.name,
                 "lemma": export_br.lemma, "use_dict": export_br.use_dict,
                 "oov_dict": export_br.oov,
                 "known_vocab": sorted(export_br.known)}, out_exp)
    print(f"для ONNX-экспорта: {out_exp} ({export_br.name})")

    (ROOT / "models" / "multilabel_report.txt").write_text(
        f"мультилейбл-классификатор, ветка-победитель: {winner.name}\n"
        f"train={len(train_rows)} (с негативами CLINC+HASD)\n\n"
        + "\n\n".join(report) + "\n", encoding="utf-8")
    print(f"отчёт: models/multilabel_report.txt "
          f"(всего {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
