# -*- coding: utf-8 -*-
"""Батарейка новых интентов телефонии (каталог РГ-ТЕЛ-01) для frame-модели.

Модель тестируется напрямую (constrained.generate_frame), без NLU-адаптера
телефонии и без маршрутизатора. Три среза на каждый интент:

- dict  — дословные поверхности из synth/vocab.py (санити: видно в train);
- para  — парафразы живой речи, написанные руками НЕ по словарю
          (проверка генерализации; волна 2: фразы итерации 1, вошедшие
          в словарь после расширения датасета, заменены свежими;
          совпадения с train по тексту подсвечиваются отдельно);
- плюс  bare да/нет и контроль старых интентов/негативов (не выели ли
          новые классы старые).

Запуск: .venv-tr/Scripts/python train/test_new_intents.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from model import TinyGPT  # noqa: E402
from constrained import generate_frame  # noqa: E402
from train import parse_target  # noqa: E402

# gold-интент -> [(срез, фраза)]
BATTERY = {
    "<CHECK_BOOKING>": [
        ("dict", "мой талон на когда?"),
        ("dict", "какой у меня кабинет?"),
        ("dict", "есть ли у меня запись?"),
        ("dict", "что у меня по записям"),
        ("para", "я не помню, записан я или нет"),
        ("para", "посмотрите, есть ли что-то на моё имя"),
        ("para", "мне говорили, что я в списке, подтвердите"),
        ("para", "на какое число меня записывали"),
        ("para", "напомните, когда мне к врачу"),
    ],
    "<HOME_VISIT>": [
        ("dict", "вызовите врача на дом."),
        ("dict", "можно врача на дом?"),
        ("dict", "мама лежачая, нужен врач домой."),
        ("para", "дедушке плохо, доктор нужен на дом"),
        ("para", "не можем прийти в клинику, нужен выезд врача"),
        ("para", "оформите вызов врача на дом"),
        ("para", "кто-нибудь приезжает на дом?"),
        ("para", "лежачего больного посмотрят дома?"),
    ],
    "<SCHEDULE>": [
        ("dict", "график работы?"),
        ("dict", "по субботам вы работаете?"),
        ("dict", "в какое время у вас обед?"),
        ("para", "в праздники вы работаете?"),
        ("para", "как у вас со субботним приёмом"),
        ("para", "приём по вечерам есть?"),
        ("para", "в какой час закрывается поликлиника"),
        ("para", "когда начинается приём врачей"),
    ],
    "<LAB_STATUS>": [
        ("dict", "готовы ли анализы?"),
        ("dict", "когда будут результаты?"),
        ("dict", "статус моих анализов?"),
        ("para", "анализы сдали, когда забирать"),
        ("para", "кровь уже проверили"),
        ("para", "можно узнать про готовность анализов"),
        ("para", "из лаборатории что-нибудь приходило"),
        ("para", "мочи уже готовы?"),
    ],
    "<PAID>": [
        ("dict", "сколько стоит приём?"),
        ("dict", "сколько берёте за приём?"),
        ("dict", "сколько с меня выходит?"),
        ("para", "за приём сколько платить"),
        ("para", "приём по платным услугам почём"),
        ("para", "сколько нужно на консультацию"),
        ("para", "консультация по деньгам сколько выходит"),
        ("para", "прайс на приём подскажите"),
    ],
    "<WAITLIST>": [
        ("dict", "запишите в лист ожидания."),
        ("dict", "позвоните когда будет окно?"),
        ("dict", "дайте знать, если появится окошко."),
        ("para", "если кто-то отменит запись, мне позвоните"),
        ("para", "в очередь на ближайшее время поставьте"),
        ("para", "хочу на замену, если кто не придёт"),
        ("para", "добавьте меня в список ожидания"),
    ],
    "<DOCS>": [
        ("dict", "какие документы нужны?"),
        ("dict", "нужно ли направление?"),
        ("dict", "без направления примут?"),
        ("para", "что с документами на приём"),
        ("para", "с собой брать что-нибудь надо"),
        ("para", "справки какие-то нужны для приёма"),
        ("para", "полис с собой приносить?"),
    ],
    "<COMPLAINT>": [
        ("dict", "хочу пожаловаться."),
        ("dict", "куда писать жалобу?"),
        ("dict", "медсестра нахамила, куда сообщить?"),
        ("para", "хочу выразить недовольство врачом"),
        ("para", "куда можно обратиться с претензией"),
        ("para", "обслуживание оставляет желать лучшего"),
        ("para", "прошу соединить с руководителем для жалобы"),
    ],
    "<MEDICAL>": [
        ("dict", "что принимать при простуде?"),
        ("dict", "какое лекарство купить?"),
        ("dict", "чем сбить давление?"),
        ("para", "чем обезболить зуб до приёма"),
        ("para", "как справиться с бессонницей"),
        ("para", "народные средства от кашля есть?"),
        ("para", "можно ли парацетамол при температуре"),
    ],
    "<YES>": [
        ("dict", "да"),
        ("dict", "верно"),
        ("dict", "давайте"),
        ("dict", "так и есть"),
        ("para", "конечно да"),
        ("para", "точно так"),
        ("para", "абсолютно верно"),
    ],
    "<NO>": [
        ("dict", "не надо"),
        ("dict", "нет, не подходит"),
        ("dict", "не требуется"),
        ("para", "не стоит"),
        ("para", "ни в коем разе"),
        ("para", "не буду"),
    ],
    # контроль: старые интенты и негативы (проверка, что новые классы
    # не перетянули формулировки записи)
    "<BOOK>": [
        ("ctrl", "хочу попасть к терапевту"),
        ("ctrl", "мне надо к неврологу"),
        ("ctrl", "запишите меня к стоматологу завтра утром"),
        ("ctrl", "мне бы к хирургу на этой неделе"),
    ],
    "<RESCHEDULE>": [
        ("ctrl", "не могу в пятницу, перенесите"),
        ("ctrl", "хочу выбрать другой день для визита"),
        ("ctrl", "сдвиньте моё время"),
    ],
    "<CANCEL>": [
        ("ctrl", "не приду, снимите запись"),
        ("ctrl", "приём в среду не актуален"),
        ("ctrl", "отмена визита к неврологу"),
    ],
    "<UNSUPPORTED>": [
        ("ctrl", "какая завтра погода в москве"),
        ("ctrl", "подскажите номер такси"),
        ("ctrl", "вы кредиты оформляете?"),
        ("ctrl", "который час"),
    ],
}


def main():
    ckpt = torch.load(ROOT / "models" / "model.pt", map_location="cpu",
                      weights_only=False)
    tok = Tokenizer.from_file(str(ROOT / "models" / "tokenizer_v2.json"))
    model = TinyGPT(tok.get_vocab_size(), **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    ids = {t: tok.token_to_id(t) for t in
           ("<PAD>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>")}

    def predict(text):
        return generate_frame(model, tok, text, torch.device("cpu"),
                              bos=ids["<BOS>"], user=ids["<USER>"],
                              assistant=ids["<ASSISTANT>"],
                              eos=ids["<EOS>"], pad=ids["<PAD>"])[0]

    train_texts = {json.loads(l)["text"].lower() for l in
                   (ROOT / "data" / "train.jsonl").read_text(
                       encoding="utf-8").splitlines() if l.strip()}
    val_texts = {json.loads(l)["text"].lower() for l in
                 (ROOT / "data" / "val.jsonl").read_text(
                     encoding="utf-8").splitlines() if l.strip()}
    seen_ds = train_texts | val_texts

    results, confusion = [], Counter()
    for gold, probes in BATTERY.items():
        for sl, text in probes:
            pred = predict(text)
            p_intent = parse_target(pred)["intent"]
            ok = p_intent == gold
            results.append((gold, sl, text, ok, pred, p_intent))
            if not ok:
                confusion[(gold, p_intent)] += 1

    # таблица по интентам
    print(f"{'интент':17s} {'dict':>11s} {'para':>11s} {'ctrl':>11s}")
    for gold in BATTERY:
        row = {}
        for sl in ("dict", "para", "ctrl"):
            sub = [r for r in results if r[0] == gold and r[1] == sl]
            if sub:
                acc = sum(r[3] for r in sub) / len(sub)
                row[sl] = f"{acc:.2f} ({sum(r[3] for r in sub)}/{len(sub)})"
            else:
                row[sl] = "-"
        print(f"{gold:17s} {row['dict']:>11s} {row['para']:>11s} "
              f"{row.get('ctrl', '-'):>11s}")
    n = len(results)
    n_ok = sum(r[3] for r in results)
    new_ok = sum(r[3] for r in results
                 if r[1] in ("dict", "para") and r[0] not in
                 ("<BOOK>", "<RESCHEDULE>", "<CANCEL>", "<UNSUPPORTED>"))
    new_n = sum(1 for r in results
                if r[1] in ("dict", "para") and r[0] not in
                ("<BOOK>", "<RESCHEDULE>", "<CANCEL>", "<UNSUPPORTED>"))
    print(f"\nвсего: {n_ok}/{n} | новые интенты (dict+para): {new_ok}/{new_n}")

    leaks = [r for r in results if r[1] == "para" and r[2].lower() in seen_ds]
    if leaks:
        print(f"парафразы, совпавшие с train/val дословно ({len(leaks)}): "
              + "; ".join(f"«{r[2]}»" for r in leaks))

    if confusion:
        print("\nконфьюжн (gold -> pred):")
        for (g, p), c in confusion.most_common():
            print(f"  {g} -> {p}: {c}")

    fails = [r for r in results if not r[3]]
    if fails:
        print(f"\nпромахи ({len(fails)}):")
        for gold, sl, text, _, pred, _ in fails:
            print(f"  [{sl}] «{text}»\n      надо {gold}\n      вышло {pred}")


if __name__ == "__main__":
    main()
