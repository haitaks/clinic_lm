# -*- coding: utf-8 -*-
"""Майнинг поверхностей интентов телефонии из открытых датасетов.

Открытые источники (лицензии допускают переиспользование):
- CLINC150 (DeepPavlov/clinc150, HF-кэш) — EN, интенты календаря/статусов/
  счетов/жалоб/подтверждений;
- MASSIVE ru-RU (amazon-massive-dataset-1.1, локально) — профессиональный
  русский перевод, календарь/заказы/дата-время;
- SGD (dstc8-schema-guided-dialogue, GitHub) — EN, часы работы/цены/лист
  ожидания ресторанов;
- MedQuAD (lavita/MedQuAD, HF) — EN, вопросы о лечении.

Фразы чужих доменов адаптируются под регистратуру поликлиники локальной
LLM (llama.cpp, qwen2.5-1.5b-instruct): сервер на 127.0.0.1:8010.
Результат кэшируется в data/expand_llm_cache.json — перезапуск бесплатен.

Выход: data/expand_surfaces.json — кандидаты по интентам для ревизии
(ручная проверка перед переносом в vocab.REQUESTS/ANSWER_SURFACES).

Запуск: .venv-tr/Scripts/python synth/expand_mining.py [--limit N]
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vocab import (REQUESTS, ANSWER_SURFACES, UNSUPPORTED_POOLS, TEMPORALS)
from validate import _spec_needles, _sym_needles, _contains, normalize

LLM_URL = "http://127.0.0.1:8010/v1/chat/completions"
CACHE = ROOT / "data" / "expand_llm_cache.json"
OUT = ROOT / "data" / "expand_surfaces.json"
PER_SOURCE = 40                     # кандидатов на источник до фильтров

# ---- CLINC150: интент источника -> наш интент --------------------------
CLINC_MAP = {
    "check_booking": ["calendar", "confirm_reservation"],
    "reschedule": ["calendar_update"],
    "cancel": ["cancel", "cancel_reservation"],
    "lab_status": ["order_status", "application_status"],
    "paid": ["bill_balance", "bill_due", "min_payment", "pay_bill"],
    "complaint": ["report_fraud", "report_lost_card"],
    "docs": ["international_visa"],
    "medical": ["vaccines"],
    # тривиальные пути — отдельным потоком, без доменной адаптации
    "@yes": ["yes"], "@no": ["no", "maybe"],
}

# ---- MASSIVE ru-RU ------------------------------------------------------
MASSIVE_MAP = {
    "check_booking": ["calendar_query"],
    "cancel": ["calendar_remove"],
    "lab_status": ["takeaway_query"],
    "@unsupported_datetime": ["datetime_query"],
}
# календарные вопросы про детали события — не про запись, выкидываем
MASSIVE_DROP = re.compile(r"дресс|парков|меню|facebook|трансляц|повестк|"
                          r"который час|сколько времени|место проведен")

# ---- SGD ----------------------------------------------------------------
SGD_PATTERNS = {
    "schedule": re.compile(
        r"\b(what time do (you|u|the) (open|close)|how late.*open|"
        r"are (you|u) open|open on|opening time|closing time|"
        r"hours of operation|what are (your|the) hours|till what time|"
        r"until what time)\b", re.I),
    "paid": re.compile(
        r"\b(how much (does|is|do|would)|price|cost|expensive|"
        r"any (fees|charge))\b", re.I),
    "waitlist": re.compile(
        r"\b(wait ?list|waiting list|how long is the wait|"
        r"wait for a (table|spot))\b", re.I),
}

# ---- MedQuAD ------------------------------------------------------------
MEDQ_SELECT = re.compile(
    r"^(what|how|can|should|is it|which)\b.*(treat|cure|medicine|"
    r"medication|relieve|manage|remedy|take for|therap)", re.I)

# ---- фильтры ------------------------------------------------------------
LATIN = re.compile(r"[a-z0-9]")
DIGITS = re.compile(r"[0-9]")
# утечки чужих доменов и артефакты LLM
DOMAIN_LEAKS = [
    "виз", "банк", "карт", "счёт", "счет", "ресторан", "заказ", "доставк",
    "мошенн", "полёт", "полет", "самолёт", "самолет", "такси", "убер",
    "столик", "брон", "встреч", "мероприяти", "событи", "календар",
    "собеседован", "отпуск", "работодатель", "напоминани", "будильник",
    "песн", "музык", "погод", "подароч", "бонус", "балл", "кэшбэк",
    "кешбэк", "покупк", "товар", "вопрос№", "pm", "am",
]
# «когда у вас обед» — валидная фраза про график: белый список исключений
LEAK_WHITELIST = re.compile(r"обед|перерыв")

PROMPT_EN = """Ты переводишь реплики звонящих на английском на русский для датасета телефонной регистратуры поликлиники. Это не подстрочник: адаптируй под звонок в регистратуру — убери слова чужих доменов и все названия (страны, банки, магазины, компании), замени на клинические: запись, приём, врач, анализы, поликлиника, документы, жалоба. Правила:
- нижний регистр, живая разговорная речь
- выведи ТОЛЬКО переведённую фразу, без кавычек и пояснений
- конкретные даты и время (сегодня, пятница, три часа) убери из фразы
Примеры:
what time is my reservation on friday -> на какое время у меня запись
where is my order it was supposed to be here -> где мои результаты, они уже должны были прийти
how much do I owe on my bill -> сколько с меня за приём
can you check if my appointment is still on -> проверьте, действует ли моя запись"""

PROMPT_RU = """Ты переформулируешь реплики для звонка в телефонную регистратуру поликлиники: голосовой ассистент, календарь, доставка, ресторан → клиника. Замены: «встреча/событие/мероприятие» → «запись/приём», «заказ» → «анализы/результаты». Правила:
- нижний регистр, живая разговорная речь
- выведи ТОЛЬКО переформулированную фразу
- конкретные даты и время убери из фразы
Примеры:
отмени мою встречу в пятницу -> отмени мою запись
когда моё следующее событие -> когда моя следующая запись
когда я могу ждать свой заказ -> когда ждать результаты анализов"""

# подсказка целевого интента: приклеивается к промпту, лечит буквализм
# при трансплантации домена (виза/банк -> документы/жалоба)
TARGET_HINTS = {
    "check_booking": ("Вопрос пациента о ЕГО СУЩЕСТВУЮЩЕЙ записи: когда, во "
                      "сколько, куда приходить, действительна ли она.\n"
                      "what's on my agenda -> что у меня по записям\n"
                      "is my spot still held -> моя запись ещё действует"),
    "cancel": ("Просьба ОТМЕНИТЬ запись или приём у врача.\n"
               "scratch that from my plans -> уберите эту запись\n"
               "delete the event -> отмените приём"),
    "reschedule": ("Просьба ПЕРЕНЕСТИ запись на другое время.\n"
                   "move my appointment -> перенесите мою запись\n"
                   "can we do another day -> можно записаться на другой день"),
    "lab_status": ("Вопрос о ГОТОВНОСТИ анализов и результатов (кровь, "
                   "моча, гормоны).\n"
                   "any update on my stuff -> есть новости по моим анализам\n"
                   "is it done yet -> уже готово"),
    "paid": ("Вопрос о ЦЕНЕ и СТОИМОСТИ приёма, консультации.\n"
             "how much is it -> сколько это стоит\n"
             "any fees -> какие-то сборы есть"),
    "complaint": ("ЖАЛОБА на работу персонала, врача или регистратуры.\n"
                  "I want to report a problem -> хочу пожаловаться\n"
                  "this is unacceptable -> такое обслуживание недопустимо\n"
                  "I need to report fraud -> хочу сообщить о нарушении"),
    "docs": ("Вопрос о ДОКУМЕНТАХ, справках, направлении для приёма "
             "(полис, паспорт, направление).\n"
             "do I need a visa for the trip -> какие документы нужны\n"
             "what papers should I bring -> какие справки принести"),
    "medical": ("ВОПРОС-СОВЕТ про лечение: чем лечить, что принимать, можно "
                "ли что-то.\n"
                "what should I take for it -> что мне принимать\n"
                "how to treat it at home -> как лечить дома"),
    "schedule": ("Вопрос о ЧАСАХ и ГРАФИКЕ работы клиники, регистратуры, "
                 "врача.\n"
                 "are you open sundays -> вы по воскресеньям работаете\n"
                 "what time do you close -> до скольки вы работаете"),
    "waitlist": ("Просьба добавить в ЛИСТ ОЖИДАНИЯ: позвонить, когда "
                 "освободится время.\n"
                 "put me on the waitlist -> поставьте меня в лист ожидания\n"
                 "call me if something opens up -> позвоните, если освободится время"),
    "yes": "", "no": "",
    "unsupported_datetime": ("Вопрос о текущем времени и дате — НЕ о клинике.\n"
                             "what time is it -> который час"),
}

PROMPT_ANSWER = """Ты переводишь короткие реплики-подтверждения с английского на русский для датасета телефонного диалога с регистратурой. Варианты ответов «да» переводи как подтверждение, «нет/может быть» — как отказ или сомнение. Правила: нижний регистр, 1–3 слова, живая речь, выведи ТОЛЬКО перевод.
Примеры:
yeap -> ага
that is correct -> всё верно
sounds good -> давайте
definitely not -> точно нет
not really -> не совсем"""


def llm(prompt: str, text: str) -> str:
    body = {"model": "x", "temperature": 0, "max_tokens": 80,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": text}]}
    req = urllib.request.Request(LLM_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=180))
    return r["choices"][0]["message"]["content"].strip()


def build_prompt(target: str, base: str) -> str:
    hint = TARGET_HINTS.get(target, "")
    return f"{base}\n\nЗадача: {hint}" if hint else base


# ---- фаза 2: парафразы для интентов, которых нет в открытых наборах ----
# (schedule/waitlist/home_visit/docs/complaint — регистратурная специфика;
# источники: словарные поверхности + живые формулировки из батареи провалов
# test_new_intents.py — там модель ошибалась, значит покрытие узкое)
PARAPHRASE_SEEDS = {
    "schedule": [
        "вы в воскресенье работаете", "до какого часа принимает регистратура",
        "по каким дням можно попасть к врачу", "когда у вас обед",
        "часы приёма подскажите", "график работы поликлиники",
        "до скольки вы работаете", "во сколько открываетесь",
    ],
    "waitlist": [
        "если место освободится, позвоните мне",
        "поставьте на очередь, если кто отменит",
        "хочу стоять в очереди к врачу", "дайте знать, если появится окошко",
        "запишите в лист ожидания", "позвоните когда будет окно",
    ],
    "home_visit": [
        "врача бы на дом вызвать, бабушке плохо", "нужен вызов терапевта на дом",
        "мама лежачая, нужен врач домой", "можно вызвать участкового на дом",
        "пришлите врача домой к ребёнку", "можно врача на дом",
    ],
    "docs": [
        "что нужно принести на приём", "какие бумаги готовить",
        "с собой что-то брать нужно", "как прикрепиться к поликлинике",
        "без направления примут", "какие документы нужны",
    ],
    "complaint": [
        "я недоволен обслуживанием", "хочу написать претензию на врача",
        "как подать жалобу на регистратуру", "медсестра нахамила, куда сообщить",
        "хочу пожаловаться", "куда писать жалобу",
    ],
    "reschedule": [
        "не смогу прийти, перенесите на другой день",
        "запись надо переделать на другое время",
        "перенесите мою запись", "поменяйте время записи",
    ],
    "check_booking": [
        "на какое число мне приходить", "во сколько мне подойти к врачу",
        "я записан, но не помню на когда", "запись ещё действует",
        "мой талон на когда", "куда мне прийти",
    ],
    "yes": [
        "именно так", "всё правильно", "так и есть", "верно",
        "да, давайте", "подтверждаю",
    ],
    "no": [
        "ни за что", "откажусь", "не требуется", "не надо",
        "нет, не подходит", "не нужно",
    ],
}

PARAPHRASE_PROMPT = """Дай 6 разговорных парафраз фразы звонящего в регистратуру поликлиники. Смысл должен остаться тем же, формулировки — другие, как говорят по телефону живые люди. Правила:
- нижний регистр, без дат, времени и цифр
- без кавычек и нумерации, каждая фраза на своей строке
- больше ничего не выводи
Фраза: """

PARAPHRASE_PROMPT_ANSWER = """Дай 6 разговорных вариантов ответа звонящего (подтверждение или отказ — как в исходной фразе) в диалоге с регистратурой поликлиники. Правила:
- 1–3 слова, нижний регистр, без кавычек и нумерации
- каждая фраза на своей строке, больше ничего не выводи
Фраза: """


def llm_paraphrase(prompt: str, seed: str) -> list:
    body = {"model": "x", "temperature": 0.8, "max_tokens": 160,
            "messages": [{"role": "user",
                          "content": prompt + seed}]}
    req = urllib.request.Request(LLM_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=180))
    out = []
    for ln in r["choices"][0]["message"]["content"].splitlines():
        ln = ln.strip().lstrip("-•*0123456789. ").strip()
        if ln:
            out.append(ln)
    return out


def load_clinc():
    from huggingface_hub import hf_hub_download
    import pandas as pd
    parts = []
    for f in ("data/train-00000-of-00001.parquet",
              "data/validation-00000-of-00001.parquet",
              "data/test-00000-of-00001.parquet"):
        parts.append(pd.read_parquet(hf_hub_download(
            "DeepPavlov/clinc150", f, repo_type="dataset")))
    df = pd.concat(parts, ignore_index=True)
    intents = pd.read_parquet(hf_hub_download(
        "DeepPavlov/clinc150", "intents/intents-00000-of-00001.parquet",
        repo_type="dataset"))
    id2name = dict(zip(intents["id"], intents["name"]))
    rows = []
    for _, r in df.iterrows():
        if not pd.isna(r["label"]):
            rows.append((id2name.get(int(r["label"]), "?"),
                         str(r["utterance"]).strip()))
    return rows


def load_medquad(limit=PER_SOURCE):
    import pandas as pd
    df = pd.read_parquet(ROOT / "data" / "_sgd" / "medquad.parquet")
    col = "Question" if "Question" in df.columns else df.columns[0]
    out = []
    for q in df[col].dropna().astype(str):
        q = " ".join(q.split())
        if len(q) <= 65 and MEDQ_SELECT.search(q):
            out.append(q)
        if len(out) >= limit * 3:
            break
    return out


def load_sgd():
    out = {"schedule": [], "paid": [], "waitlist": []}
    for p in sorted((ROOT / "data" / "_sgd").glob("dialogues_*.json")):
        for dlg in json.loads(p.read_text(encoding="utf-8")):
            for turn in dlg["turns"]:
                if turn["speaker"] != "USER":
                    continue
                u = " ".join(turn["utterance"].split())
                if not (8 <= len(u) <= 70):
                    continue
                for intent, pat in SGD_PATTERNS.items():
                    if pat.search(u):
                        out[intent].append(u)
    return out


def load_massive():
    src = ROOT / "data" / "_massive" / "1.1" / "data" / "ru-RU.jsonl"
    rows = [json.loads(l) for l in
            src.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for intent, massive_intents in MASSIVE_MAP.items():
        pool = [r["utt"] for r in rows if r["intent"] in massive_intents]
        pool = [u for u in pool if not MASSIVE_DROP.search(u.lower())
                and 8 <= len(u) <= 70 and not DIGITS.search(u)]
        out[intent] = pool
    return out


def ok_surface(s: str, is_answer: bool = False) -> bool:
    s = s.strip().lower()
    s = re.sub(r"^[«»\"'-]+|[«»\"'.!?-]+$", "", s).strip()
    if not s:
        return False
    if LATIN.search(s):
        return False
    if not (4 <= len(s) <= 70):
        return False
    if not is_answer:
        if not LEAK_WHITELIST.search(s):
            for leak in DOMAIN_LEAKS:
                if leak in s:
                    return False
        n = normalize(s)
        for group in (_spec_needles, _sym_needles):
            for ndls in group.values():
                for ndl in ndls:
                    if _contains(ndl, n):
                        return False
        for t in TEMPORALS:
            if _contains(t, n):
                return False
    if re.search(r"^(ну|э|ээ|мм)\b", s):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=PER_SOURCE)
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text(encoding="utf-8")) \
        if CACHE.exists() else {}

    # 1. кандидаты: (intent, текст, промпт, origin)
    cands = []

    clinc = load_clinc()
    by_intent = {}
    for name, utt in clinc:
        by_intent.setdefault(name, []).append(utt)
    for target, src_intents in CLINC_MAP.items():
        is_answer = target.startswith("@")
        prompt = PROMPT_ANSWER if is_answer else PROMPT_EN
        pool = []
        for si in src_intents:
            pool += [u for u in by_intent.get(si, [])
                     if 4 <= len(u) <= 80][:args.limit]
        seen = set()
        for u in pool:
            if u.lower() in seen:
                continue
            seen.add(u.lower())
            cands.append((target.lstrip("@"), u, prompt,
                          "clinc:" + si, is_answer))

    massive = load_massive()
    for target, pool in massive.items():
        if target.startswith("@"):
            for u in pool[:args.limit]:
                cands.append((target.lstrip("@"), u, PROMPT_RU,
                              "massive:datetime", True))
        else:
            for u in pool[:args.limit]:
                cands.append((target, u, PROMPT_RU, "massive", False))

    sgd = load_sgd()
    for target, pool in sgd.items():
        for u in pool[:args.limit]:
            cands.append((target, u, PROMPT_EN, "sgd", False))

    for q in load_medquad(args.limit)[:args.limit]:
        cands.append(("medical", q, PROMPT_EN, "medquad", False))

    # 2. перевод с кэшем (ключ включает интент — он определяет промпт)
    todo = [c for c in cands if f"{c[0]}|{c[3]}|{c[1]}" not in cache]
    print(f"кандидатов: {len(cands)}, переводить: {len(todo)}")
    t0 = time.time()
    for i, (target, text, prompt, origin, is_answer) in enumerate(todo):
        try:
            cache[f"{target}|{origin}|{text}"] = \
                llm(build_prompt(target, prompt), text)
        except Exception as e:                       # noqa: BLE001
            print(f"  LLM error на «{text[:40]}»: {e}")
            break
        if (i + 1) % 25 == 0:
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                             encoding="utf-8")
            print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                     encoding="utf-8")

    # 2b. парафразы для интентов без открытых источников (фаза LLM)
    para_todo = [(intent, seed) for intent, seeds in PARAPHRASE_SEEDS.items()
                 for seed in seeds if f"para|{intent}|{seed}" not in cache]
    print(f"парафраз-затравок: {len(para_todo)}")
    for i, (intent, seed) in enumerate(para_todo):
        prompt = (PARAPHRASE_PROMPT_ANSWER if intent in ("yes", "no")
                  else PARAPHRASE_PROMPT)
        try:
            cache[f"para|{intent}|{seed}"] = llm_paraphrase(prompt, seed)
        except Exception as e:                   # noqa: BLE001
            print(f"  LLM error на «{seed[:40]}»: {e}")
            break
        if (i + 1) % 10 == 0:
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                     encoding="utf-8")

    # 3. пост-фильтрация и дедупликация
    existing = set()
    for phrases in REQUESTS.values():
        existing |= {normalize(p) for p in phrases}
    for which in ANSWER_SURFACES:
        existing |= {normalize(p) for p in ANSWER_SURFACES[which]}
    for pool in UNSUPPORTED_POOLS.values():
        existing |= {normalize(p) for p in pool}

    result, seen = {}, set()
    dropped = Counter()
    for target, text, prompt, origin, is_answer in cands:
        ru = cache.get(f"{target}|{origin}|{text}", "")
        ru = " ".join(ru.split())
        # модель иногда добавляет префикс «перевод:»
        ru = re.sub(r"^(перевод|ответ)\s*:\s*", "", ru, flags=re.I).strip()
        if not ok_surface(ru, is_answer):
            dropped[f"{target}:filter"] += 1
            continue
        n = normalize(ru)
        if n in existing or n in seen:
            dropped[f"{target}:dup"] += 1
            continue
        seen.add(n)
        result.setdefault(target, []).append(ru)

    # 3b. парафразы через те же фильтры (кэш хранит списки)
    para_dropped = Counter()
    for intent, seeds in PARAPHRASE_SEEDS.items():
        is_answer = intent in ("yes", "no")
        for seed in seeds:
            for ru in cache.get(f"para|{intent}|{seed}", []):
                ru = " ".join(str(ru).split())
                if not ok_surface(ru, is_answer):
                    para_dropped[f"{intent}:filter"] += 1
                    continue
                n = normalize(ru)
                if n in existing or n in seen:
                    para_dropped[f"{intent}:dup"] += 1
                    continue
                seen.add(n)
                result.setdefault(intent, []).append(ru)
    dropped.update(para_dropped)

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in result.items()},
                     ensure_ascii=False, indent=1))
    print("отброшено:", dict(dropped))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
