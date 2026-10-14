# -*- coding: utf-8 -*-
"""Консольная демка менеджера диалога.

Запуск из корня clinic_lm::

    python -m telephone.demo                        # frame-модель + Natasha + e5
    .venv-tr/Scripts/python -m telephone.demo       # то же, но e5-резолвер живой
    python -m telephone.demo --nlu fixture          # словарная заглушка
    python -m telephone.demo --outbound Смирнова

Команды: /silence — событие тишины, /dtmf 1234… — набор цифр,
/quit — выход. Дата звонка фиксирована как в примерах скрипта
(среда, 30.09.2026), чтобы «завтра» и расписание были детерминированы.
"""
from __future__ import annotations

import argparse
from datetime import datetime

from . import DialogManager, FixtureNlu, StubMis
from .mis import DOCTORS
from .schema import DtmfEvent, SilenceEvent, VoicemailEvent
from .scenarios.common import render_card


def _outbound(name: str, mis: StubMis):
    """Запись на завтра для исходящего звонка-напоминания (С-10)."""
    for p in mis.patients.values():
        if p.fio.last.lower().startswith(name.lower()):
            appt = mis.add_booking(
                p.id, "gp", p.gp_doctor or "Кузнецова Анна Сергеевна",
                "2026-10-01", "08:45", 214)
            return p, appt
    raise SystemExit(f"Пациент с фамилией {name} не найден в Заглушке МИС")


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо менеджера диалога")
    parser.add_argument("--outbound", metavar="ФАМИЛИЯ",
                        help="исходящий звонок-напоминание (С-10)")
    parser.add_argument("--ani", default="+79001234567",
                        help="ANI звонящего")
    parser.add_argument("--nlu", choices=["frame", "fixture"],
                        default="frame",
                        help="NLU-контур: frame-модель (по умолчанию) или "
                             "словарная заглушка")
    args = parser.parse_args()

    mis = StubMis()
    now = datetime(2026, 9, 30, 12, 0)
    outbound = _outbound(args.outbound, mis) if args.outbound else None
    if args.nlu == "frame":
        from .nlu import FrameModelNlu
        nlu = FrameModelNlu()          # + names_backstop (Natasha) внутри
    else:
        nlu = FixtureNlu([p.fio for p in mis.patients.values()])
    dm = DialogManager(nlu=nlu, mis=mis, now=now, ani=args.ani,
                       outbound=outbound)

    print("=" * 72)
    print("Демо ГА регистратуры. Врачи в Заглушке МИС:",
          ", ".join(sorted(d.split()[0] for d in DOCTORS)))
    print("Пациенты: " + "; ".join(
        f"{p.fio.full()} ({p.dob.isoformat()})"
        for p in mis.patients.values()))
    print("=" * 72)

    turn = dm.start_call()
    _print(turn)
    while True:
        try:
            line = input("Пациент> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/silence":
            turn = dm.handle_event(SilenceEvent())
        elif line == "/voicemail":
            turn = dm.handle_event(VoicemailEvent())
        elif line.startswith("/dtmf"):
            turn = dm.handle_event(DtmfEvent(digits=line.split()[1]))
        else:
            turn = dm.handle_utterance(line)
        _print(turn)
        if dm.state == "closed":
            break

    print("-" * 72)
    summary = dm.summary()
    print(f"Итог звонка: {summary['status']}")
    print(f"Собрано: {summary['collected']}")
    print(f"SMS: {mis.sent_sms}")
    if mis.notified_senior:
        print(f"Старшему администратору: {mis.notified_senior}")


def _print(turn) -> None:
    print(f"ГА> {turn.reply}")
    for a in turn.actions:
        kind = type(a).__name__
        if hasattr(a, "card"):
            print(f"    [{kind}] {render_card(a.card)}")
        else:
            print(f"    [{kind}] {a}")


if __name__ == "__main__":
    main()
