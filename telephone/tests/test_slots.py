# -*- coding: utf-8 -*-
"""Парсеры слотов: Приложение А.3 (даты рождения), полис, да/нет, время.

Извлечение ФИО — контракт внешней модели (Frame.fio), детерминированных
тестов на него нет; в диалоговых тестах работает FixtureNlu."""
from datetime import date

from telephone import slots

TODAY = date(2026, 9, 30)


class TestDob:
    def test_words_full(self):
        assert slots.extract_dob(
            "Смирнова Ольга Викторовна, двенадцатое марта семьдесят "
            "восьмого", TODAY) == date(1978, 3, 12)

    def test_words_genitive_day(self):
        assert slots.extract_dob(
            "Петров Игорь Николаевич, пятое мая шестьдесят второго",
            TODAY) == date(1962, 5, 5)

    def test_words_2000s(self):
        assert slots.extract_dob(
            "Смирнова Алиса Дмитриевна, седьмое апреля две тысячи "
            "девятнадцатого", TODAY) == date(2019, 4, 7)

    def test_digits_dotted(self):
        assert slots.extract_dob("Смирнова Ольга Викторовна 12.03.78",
                                 TODAY) == date(1978, 3, 12)

    def test_digits_words(self):
        assert slots.extract_dob(
            "двенадцать ноль три семьдесят восемь", TODAY) == date(1978, 3, 12)

    def test_two_digit_year_child(self):
        # обе границы ≤110 лет → выбираем младший век (ребёнок 2019 г.р.)
        assert slots.extract_dob("первое января девятнадцатого",
                                 TODAY) == date(2019, 1, 1)

    def test_kuzmina_example(self):
        assert slots.extract_dob(
            "Кузьмина Нина Андреевна, двадцатое июля пятьдесят пятого",
            TODAY) == date(1955, 7, 20)

    def test_compound_day(self):
        assert slots.extract_dob("двадцать первого октября "
                                 "девяносто шестого",
                                 TODAY) == date(1996, 10, 21)

    def test_no_dob(self):
        assert slots.extract_dob("Смирнова Ольга Викторовна", TODAY) is None


class TestPolicyNo:
    def test_digits(self):
        assert slots.parse_policy_no("1234567890123456") == "1234567890123456"

    def test_words(self):
        # одна тысяча двести тридцать четыре… — по цифре на слово
        toks = "один два три четыре пять шесть семь восемь девять ноль " \
               "один два три четыре пять шесть".split()
        assert slots.parse_policy_no(" ".join(toks)) == "1234567890123456"


class TestYesNo:
    def test_yes(self):
        assert slots.parse_yesno("да") == (True, False)
        assert slots.parse_yesno("Да, верно") == (True, False)
        assert slots.parse_yesno("давайте") == (True, False)

    def test_no(self):
        assert slots.parse_yesno("нет") == (False, False)
        assert slots.parse_yesno("Нет, завтра не получится") == (False, False)
        assert slots.parse_yesno("не приду") == (False, False)

    def test_maybe(self):
        assert slots.parse_yesno("не знаю") == (None, True)
        assert slots.parse_yesno("может быть") == (None, True)

    def test_silence_like(self):
        assert slots.parse_yesno("") == (None, False)


class TestTimePrefs:
    def test_tomorrow_morning(self):
        p = slots.parse_time_prefs("завтра утром", TODAY)
        assert p.date == date(2026, 10, 1)
        assert p.daypart == "morning"

    def test_weekday(self):
        p = slots.parse_time_prefs("в пятницу", TODAY)
        assert p.date == date(2026, 10, 2)

    def test_afternoon(self):
        p = slots.parse_time_prefs("а после обеда есть что-нибудь?", TODAY)
        assert p.daypart == "afternoon"

    def test_next_week_monday(self):
        p = slots.parse_time_prefs("на следующей неделе в понедельник", TODAY)
        assert p.date == date(2026, 10, 5)

    def test_not_friday(self):
        p = slots.parse_time_prefs("мне к лору, но лучше не в пятницу, "
                                   "после обеда", TODAY)
        assert p.exclude_weekday == 4        # пятница
        assert p.weekday is None
        assert p.daypart == "afternoon"


class TestRelation:
    def test_mother(self):
        assert slots.parse_relation("да, я мама") == "mother"

    def test_neighbor(self):
        assert slots.parse_relation("я её соседка") == "relative"

    def test_self(self):
        assert slots.parse_relation("я сам") == "self"
