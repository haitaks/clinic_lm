# -*- coding: utf-8 -*-
import sys
from datetime import datetime
from pathlib import Path

# корень clinic_lm — чтобы `import telephone` работал из любой директории
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from telephone import DialogManager, FixtureNlu, StubMis  # noqa: E402
from telephone.schema import (CallbackTask, DmTurn, EndCall, SmsRequest,  # noqa: E402
                              Transfer)

# Дата примера скрипта: среда, 30 сентября 2026, обед
NOW = datetime(2026, 9, 30, 12, 0)

# телефоны пациентов Заглушки МИС (см. telephone/mis.py)
ANI_SMIRNOVA = "+79001234567"
ANI_FOREIGN = "+79990000001"


@pytest.fixture
def make_dm():
    def factory(ani: str = ANI_SMIRNOVA, mis: StubMis | None = None,
                outbound=None) -> DialogManager:
        mis = mis or StubMis()
        nlu = FixtureNlu([p.fio for p in mis.patients.values()])
        return DialogManager(nlu=nlu, mis=mis, now=NOW, ani=ani,
                             outbound=outbound)
    return factory


def say(dm: DialogManager, *lines: str) -> list[DmTurn]:
    """Прогнать реплики пациента, вернуть ответы."""
    return [dm.handle_utterance(line) for line in lines]


def transfers(turn: DmTurn) -> list[Transfer]:
    return [a for a in turn.actions if isinstance(a, Transfer)]


def callbacks(turn: DmTurn) -> list[CallbackTask]:
    return [a for a in turn.actions if isinstance(a, CallbackTask)]


def smses(turn: DmTurn) -> list[SmsRequest]:
    return [a for a in turn.actions if isinstance(a, SmsRequest)]


def endcalls(turn: DmTurn) -> list[EndCall]:
    return [a for a in turn.actions if isinstance(a, EndCall)]
