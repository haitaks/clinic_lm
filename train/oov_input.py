# -*- coding: utf-8 -*-
"""e5-резолвер ВХОДА для frame-модели: незнакомое слово реплики → ближайший
знакомый синоним из словаря (specialty/symptom поверхности).

Отличие от tagger/oov_resolver.py (он канонизирует ВЫХОД по тегам модели
и выключен): здесь подмена происходит ДО инференса — модель видит только
слова из обучающего словаря. Индекс и эмбеддер общие (e5_index.npz,
multilingual-e5-small), поэтому косинус согласован с построением индекса.

Защита от ложных подмен (главный риск, см. tagger/README: порог не отделяет
мусор «погода»→fever 0.89):
- заменяются только слова, которых нет ни в одной известной поверхности
  и нет в «служебном» словаре генератора (REQUESTS/GREETINGS/ПОЛИТНОСТЬ/
  временные фразы/пулы unsupported) — служебные слова неприкосновенны;
- порог косинуса;
- заменяется максимум ОДНО слово в реплике (самый близкий кандидат).

Запуск автономной проверки:  python train/oov_input.py
Использование: resolver = E5SynonymResolver(); text2 = resolver(text)
"""

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tagger"))
sys.path.insert(0, str(ROOT / "synth"))

from oov_resolver import load_e5, embed  # noqa: E402

THRESHOLD = 0.80


def _service_words() -> set[str]:
    """Слова служебных словарей генератора: их резолвер не трогает."""
    from vocab import (REQUESTS, GREETINGS, POLITENESS, UNSUPPORTED_POOLS,
                       UNSUPPORTED_LEADS, UNSUPPORTED_LEADS_NO_GREET,
                       TEMPORALS, ANSWER_SURFACES)
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
        out.update(_tokens(phrase))
    return out


def _tokens(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9:]+", text.lower().replace("ё", "е"))


class E5SynonymResolver:
    def __init__(self, index_path: str | None = None,
                 threshold: float = THRESHOLD):
        # Индекс строится по TRAIN-поверхностям словаря (не по e5_index.npz
        # теггера: там и hold-out синонимы, а цель — заменить незнакомое
        # слово ЗНАКОМЫМ из обучения, «дантист» -> «стоматолог»).
        from vocab import SPECIALTIES, SYMPTOMS
        surfaces = []
        for entry in SPECIALTIES.values():
            for syn in entry["train"]:
                surfaces.extend(syn.values())
        for entry in SYMPTOMS.values():
            surfaces.extend(entry["train"])
        self.surface_texts = surfaces
        self.tok, self.model = load_e5()
        self.vecs = embed(self.tok, self.model,
                          ["passage: " + s for s in surfaces])
        self.service = _service_words()
        # слова, уже известные обучающему словарю, подменять не нужно
        self.known = {t for s in surfaces for t in _tokens(s)}
        self.threshold = threshold

    def _candidates(self, text: str) -> list[str]:
        """Незнакомые содержательные слова: не служебные, не известные,
        не стоп-слова русского языка (короткая эвристика длины)."""
        words = _tokens(text)
        seen = set()
        out = []
        for w in words:
            if (len(w) >= 4 and w not in seen
                    and w not in self.service and w not in self.known):
                out.append(w)
                seen.add(w)
        return out

    def __call__(self, text: str) -> str:
        cands = self._candidates(text)
        if not cands:
            return text
        q = embed(self.tok, self.model, ["query: " + c for c in cands])
        sims = q @ self.vecs.T                     # [C, N]
        best_c, best_n = np.unravel_index(int(sims.argmax()), sims.shape)
        if sims[best_c, best_n] < self.threshold:
            return text
        word = cands[best_c]
        surface = self.surface_texts[best_n]
        pattern = (r"(?<![а-яёa-z])" + re.escape(word)
                   + r"[а-яё]{0,3}(?![а-яёa-z0-9])")
        return re.sub(pattern, surface, text, count=1)


def _demo():
    resolver = E5SynonymResolver()
    cases = [
        "запишите меня к глазнику на завтра",         # глазник → окулист? (ood-словарь знает)
        "хочу к кожнику",                              # кожник (ood) — известен
        "запишите к вертебрологу пожалуйста",          # незнакомое слово
        "мне бы к мануальщику",                        # незнакомое
        "запишите меня к терапевту завтра утром",      # всё знакомо — без изменений
        "да, давайте завтра",                          # служебные слова — без изменений
    ]
    for t in cases:
        print(f"{t!r:50} -> {resolver(t)!r}")


if __name__ == "__main__":
    _demo()
