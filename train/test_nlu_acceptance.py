# -*- coding: utf-8 -*-
"""Приёмочный тест NLU по регламенту РГ-ТЕЛ-01 (ТЗ, раздел 9): 40 реплик
сценариев С-01…С-15, все зелёные после пересборки модели.

Модель тестируется напрямую (constrained.generate_frame), без NLU-адаптера
телефонии и без маршрутизатора. Правила приёмки:
- С-01…С-13, С-15: интент совпадает с эталоном; если в реплике назван
  врач — специальность совпадает (значит, не <UNSUPPORTED> и не
  <UNKNOWN_SPECIALTY>);
- С-14 (экстренные): <UNSUPPORTED> допустим — их держит safety-слой
  (PolicyRules/regex) осознанно; проверяется только отсутствие ложного
  «да/нет» (Этап C ТЗ);
- Этап C: «не могу прийти» — перенос без <SYMPTOM_INSOMNIA>; «у мужа
  давит в груди» — фрейм с <SYMPTOM_CHEST_PAIN>, не «да».

Запуск: pytest train/test_nlu_acceptance.py -q
До переобучения тест пропускается: старый tokenizer_v2.json не знает
новых токенов Этапа A — сначала выполните runbook пересборки (README).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

MODEL = ROOT / "models" / "model.pt"
TOKENIZER = ROOT / "models" / "tokenizer_v2.json"

# (сценарий, реплика, интент, специальность|None, режим, симптомы|None)
# Режимы: None — точное совпадение интента/специальности;
#         "any" — только не да/нет (С-14, Этап C);
#         "symptom" — не да/нет + обязательные симптом-токены.
ROWS = [
    # С-01 запись: «окулист»/«глазной»/«дантист» → специальность, «талон» → BOOK
    ("C-01", "здравствуйте, хочу записаться к окулисту",
     "<BOOK>", "<OPHTHALMOLOGIST>", None, None),
    ("C-01", "нужен талон к терапевту",
     "<BOOK>", "<GP>", None, None),
    ("C-01", "запишите меня к глазному на завтра утром",
     "<BOOK>", "<OPHTHALMOLOGIST>", None, None),
    ("C-01", "можно к дантисту на этой неделе?",
     "<BOOK>", "<DENTIST>", None, None),
    # С-02 по направлению: «направление к X» → BOOK + специальность
    ("C-02", "у меня направление к кардиологу, запишите на приём",
     "<BOOK>", "<CARDIOLOGIST>", None, None),
    ("C-02", "запишите по направлению к невропатологу",
     "<BOOK>", "<NEUROLOGIST>", None, None),
    ("C-02", "направление к эндокринологу на руках, хочу попасть",
     "<BOOK>", "<ENDOCRINOLOGIST>", None, None),
    # С-03 перенос: «перезапишите» → RESCHEDULE
    ("C-03", "перезапишите меня на другой день",
     "<RESCHEDULE>", None, None, None),
    ("C-03", "не могу прийти в четверг, перенесите приём",
     "<RESCHEDULE>", None, None, None),
    ("C-03", "хочу сдвинуть приём к урологу",
     "<RESCHEDULE>", "<UROLOGIST>", None, None),
    # С-04 отмена
    ("C-04", "отмените мою запись к стоматологу",
     "<CANCEL>", "<DENTIST>", None, None),
    ("C-04", "не приду в пятницу, снимите запись",
     "<CANCEL>", None, None, None),
    ("C-04", "отменить приём у кардиолога",
     "<CANCEL>", "<CARDIOLOGIST>", None, None),
    # С-05 проверка: «мой талон на когда» остаётся check_booking
    ("C-05", "мой талон на когда?",
     "<CHECK_BOOKING>", None, None, None),
    ("C-05", "на когда я записан к лору?",
     "<CHECK_BOOKING>", "<ENT>", None, None),
    # С-06 вызов на дом: «ребёнок заболел» → HOME_VISIT
    ("C-06", "ребёнок заболел, нужен врач на дом",
     "<HOME_VISIT>", None, None, None),
    ("C-06", "вызовите педиатра на дом",
     "<HOME_VISIT>", None, None, None),
    ("C-06", "мама лежачая, нужен врач домой",
     "<HOME_VISIT>", None, None, None),
    # С-07 график: «работаете в субботу», «кабинет» → SCHEDULE
    ("C-07", "вы в субботу работаете?",
     "<SCHEDULE>", None, None, None),
    ("C-07", "в каком кабинете принимает терапевт?",
     "<SCHEDULE>", "<GP>", None, None),
    ("C-07", "работаете в субботу?",
     "<SCHEDULE>", None, None, None),
    # С-08 анализы
    ("C-08", "готовы ли анализы?",
     "<LAB_STATUS>", None, None, None),
    ("C-08", "когда будут результаты анализов?",
     "<LAB_STATUS>", None, None, None),
    # С-09 платно: «кардиолог» → специальность в слоте платного приёма
    ("C-09", "сколько стоит приём кардиолога?",
     "<PAID>", "<CARDIOLOGIST>", None, None),
    ("C-09", "сколько стоит приём у эндокринолога?",
     "<PAID>", "<ENDOCRINOLOGIST>", None, None),
    ("C-09", "можно платно записаться?",
     "<PAID>", None, None, None),
    # С-11 лист ожидания
    ("C-11", "запишите меня в лист ожидания",
     "<WAITLIST>", None, None, None),
    ("C-11", "позвоните, когда появится окно",
     "<WAITLIST>", None, None, None),
    # С-12 ребёнок: «педиатр»/«детский врач» → PEDIATRICIAN
    ("C-12", "запишите ребёнка к педиатру",
     "<BOOK>", "<PEDIATRICIAN>", None, None),
    ("C-12", "сыну нужен детский врач",
     "<BOOK>", "<PEDIATRICIAN>", None, None),
    ("C-12", "ребёнка записать к кожнику",
     "<BOOK>", "<DERMATOLOGIST>", None, None),
    # С-13 прикрепление
    ("C-13", "как прикрепиться к поликлинике?",
     "<DOCS>", None, None, None),
    ("C-13", "какие документы нужны для прикрепления?",
     "<DOCS>", None, None, None),
    # С-14 экстренно: остаётся на safety-слое, <UNSUPPORTED> допустим
    ("C-14", "вызовите скорую пожалуйста",
     None, None, "any", None),
    ("C-14", "мужу плохо с сердцем, что делать",
     None, None, "any", None),
    # С-15 жалоба: «врач нагрубил»/«нахамил» → COMPLAINT, не BOOK
    ("C-15", "врач нагрубил, куда жаловаться?",
     "<COMPLAINT>", None, None, None),
    ("C-15", "мне нахамили в регистратуре",
     "<COMPLAINT>", None, None, None),
    ("C-15", "хочу оставить жалобу на врача",
     "<COMPLAINT>", None, None, None),
    # Этап C ТЗ: ложные срабатывания
    ("C-этап", "у мужа давит в груди",
     None, None, "symptom", {"<SYMPTOM_CHEST_PAIN>"}),
    ("C-этап", "не могу прийти завтра",
     "<RESCHEDULE>", None, None, set()),
]

assert len(ROWS) == 40, f"в приёмке должно быть 40 реплик, есть {len(ROWS)}"


class NluAcceptanceTest(unittest.TestCase):
    """40 регламентных реплик: все зелёные (ТЗ, п. 9.3)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from tokenizers import Tokenizer
        except ImportError as e:  # прагматики: нет тяжёлых зависимостей
            raise unittest.SkipTest(f"нет зависимостей инференса: {e}")
        from model import TinyGPT
        from constrained import generate_frame
        from train import parse_target

        tok = Tokenizer.from_file(str(TOKENIZER))
        missing = [t for t in ("<CARDIOLOGIST>", "<PEDIATRICIAN>",
                               "<SYMPTOM_CHEST_PAIN>", "<SYMPTOM_NAUSEA>")
                   if tok.token_to_id(t) is None]
        if missing:
            raise unittest.SkipTest(
                "models/tokenizer_v2.json не знает токены Этапа A "
                f"({', '.join(missing)}) — модель не переобучена под "
                "новый словарь. Выполните runbook пересборки из README "
                "(build_dataset → tokenizer_train → train).")
        ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
        model = TinyGPT(tok.get_vocab_size(), **ckpt["config"])
        model.load_state_dict(ckpt["model"])
        model.eval()
        cls.tok, cls.model = tok, model
        cls.generate_frame = staticmethod(generate_frame)
        cls.parse_target = staticmethod(parse_target)
        cls.ids = {t: tok.token_to_id(t) for t in
                   ("<PAD>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>")}
        cls.torch = torch

    def _predict(self, text):
        device = self.torch.device("cpu")
        pred, _ = self.generate_frame(
            self.model, self.tok, text, device,
            bos=self.ids["<BOS>"], user=self.ids["<USER>"],
            assistant=self.ids["<ASSISTANT>"], eos=self.ids["<EOS>"],
            pad=self.ids["<PAD>"])
        return pred

    def test_40_regulation_utterances(self):
        fails = []
        for scenario, text, intent, spec, mode, symptoms in ROWS:
            with self.subTest(scenario=scenario, text=text):
                pred = self._predict(text)
                parsed = self.parse_target(pred)
                got_intent = parsed["intent"]
                problems = []
                if mode == "any":
                    if got_intent in ("<YES>", "<NO>"):
                        problems.append(f"ложное да/нет: {pred}")
                elif mode == "symptom":
                    if got_intent in ("<YES>", "<NO>"):
                        problems.append(f"ложное да/нет: {pred}")
                    if symptoms and not symptoms.issubset(set(parsed["symptoms"])):
                        problems.append(
                            f"нет симптомов {sorted(symptoms)}: {pred}")
                else:
                    if got_intent != intent:
                        problems.append(f"интент {got_intent} != {intent}")
                    if spec is not None and parsed["specialty"] != spec:
                        problems.append(
                            f"специальность {parsed['specialty']} != {spec}")
                    if symptoms is not None and not (
                            symptoms.issubset(set(parsed["symptoms"]))
                            if symptoms else not parsed["symptoms"]):
                        problems.append(f"симптомы {parsed['symptoms']}"
                                        f" != {sorted(symptoms or [])}")
                    if got_intent == "<UNSUPPORTED>":
                        problems.append("уход в <UNSUPPORTED>")
                if problems:
                    fails.append(f"[{scenario}] «{text}» -> {pred}: "
                                 + "; ".join(problems))
        self.assertEqual([], fails, "\n" + "\n".join(fails))


if __name__ == "__main__":
    unittest.main(verbosity=2)
