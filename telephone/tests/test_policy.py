# -*- coding: utf-8 -*-
"""Сквозные правила (раздел 7.3, словарь 7.4)."""
from telephone import policy
from telephone.schema import RouteRule


def hit(text):
    return policy.check_utterance(text)


class TestRedFlagsA:
    def test_chest_pain(self):
        h = hit("ой, у меня вообще-то сильно давит в груди")
        assert h is not None and h.kind == "red_a" and h.rule == RouteRule.R01

    def test_breathing(self):
        assert hit("муж не может дышать").kind == "red_a"

    def test_unconscious(self):
        assert hit("отец потерял сознание").kind == "red_a"

    def test_self_harm(self):
        assert hit("я не хочу жить").kind == "red_a"

    def test_child(self):
        assert hit("ребенок проглотил батарейку").kind == "red_a"

    def test_negated_history_is_false_positive_allowed(self):
        # «год назад была боль в груди» — срабатывание допустимо (7.1),
        # сценарий С-14 обрабатывает ложное срабатывание
        assert hit("год назад была боль в груди").kind == "red_a"


class TestRedFlagsB:
    def test_fever(self):
        h = hit("температура сорок и не сбивается")
        assert h.kind == "red_b" and h.rule == RouteRule.R11

    def test_severe_pain(self):
        assert hit("невыносимо болит живот").kind == "red_b"


class TestOperator:
    def test_request(self):
        h = hit("соедините с человеком")
        assert h.kind == "operator" and h.rule == RouteRule.R02

    def test_short(self):
        assert hit("оператора!").kind == "operator"

    def test_third_party_phrase_is_not_operator(self):
        # «в интересах другого человека» — не просьба об операторе
        assert hit("я звоню в интересах другого человека") is None


class TestOtherRules:
    def test_complaint(self):
        h = hit("хочу пожаловаться на регистратуру")
        assert h.kind == "complaint" and h.rule == RouteRule.R03

    def test_robot(self):
        assert hit("вы робот?").kind == "robot"

    def test_recording_objection(self):
        assert hit("я против записи разговора").kind == "recording_objection"

    def test_medical(self):
        assert hit("что мне выпить от головной боли?").kind == "medical"

    def test_cholesterol(self):
        h = hit("какой у меня холестерин в последнем анализе?")
        assert h.kind == "medical_values"

    def test_tooth(self):
        h = hit("у меня острая зубная боль, щека распухла")
        assert h.kind == "tooth" and h.rule == RouteRule.R09

    def test_booking_not_flagged(self):
        assert hit("мне бы записаться к терапевту") is None

    def test_name_not_organization(self):
        # «Смирнова» не содержит «СМИ» как отдельное слово
        assert hit("запишите Смирнову к терапевту") is None
