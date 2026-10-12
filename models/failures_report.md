# Кейсы, которые модель не обработала

Всего промахов: 608

## По наборам

- hasd: 227
- test_ood: 168
- val: 107
- test_iid: 84
- clinc: 9
- hasd_unsup: 7
- manual_gold: 6

## По полям

- specialty: 406
- time: 263
- intent: 260
- symptoms: 221

## Примеры: test_ood (168)

- **specialty**
  - текст: `надо бы к дантисту попасть зуб стреляет`
  - надо: `<BOOK> <DENTIST> <T_UNKNOWN> <SYMPTOM_TOOTHACHE> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_TOOTHACHE> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty+symptoms**
  - текст: `Зуб стреляет. Дантист нужен срочно.`
  - надо: `<BOOK> <DENTIST> <T_UNKNOWN> <SYMPTOM_TOOTHACHE> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
- **specialty**
  - текст: `надо бы к дантисту попасть. Зуб стреляет.`
  - надо: `<BOOK> <DENTIST> <T_UNKNOWN> <SYMPTOM_TOOTHACHE> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_TOOTHACHE> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty**
  - текст: `женский врач нужен срочно в три часа дня`
  - надо: `<BOOK> <GYNECOLOGIST> <TIME> T_H15_00 </TIME> <END_FRAME>`
  - вышло: `<BOOK> <NEUROLOGIST> <TIME> T_H15_00 </TIME> <END_FRAME>`
- **specialty**
  - текст: `надо бы к женскому врачу попасть, в три часа дня.`
  - надо: `<BOOK> <GYNECOLOGIST> <TIME> T_H15_00 </TIME> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_H15_00 </TIME> <END_FRAME>`
- **specialty**
  - текст: `Женский врач нужен срочно, в три часа дня.`
  - надо: `<BOOK> <GYNECOLOGIST> <TIME> T_H15_00 </TIME> <END_FRAME>`
  - вышло: `<BOOK> <NEUROLOGIST> <TIME> T_H15_00 </TIME> <END_FRAME>`
- **specialty**
  - текст: `надо бы к невропатологу попасть, в четверг.`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_THU </TIME> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_THU </TIME> <END_FRAME>`
- **specialty**
  - текст: `Невропатолог нужен срочно, в четверг.`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_THU </TIME> <END_FRAME>`
  - вышло: `<BOOK> <DENTIST> <TIME> T_THU </TIME> <END_FRAME>`
- **specialty**
  - текст: `надо бы к невропатологу попасть в четверг`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_THU </TIME> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_THU </TIME> <END_FRAME>`
- **specialty**
  - текст: `надо бы к невропатологу попасть в четверг вены ноют`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_THU </TIME> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_THU </TIME> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `Шатает при ходьбе. Невропатолог нужен срочно, в среду.`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_WED </TIME> <SYMPTOM_DIZZINESS> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty+symptoms**
  - текст: `шатает при ходьбе невропатолог нужен срочно в среду`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_WED </TIME> <SYMPTOM_DIZZINESS> <END_FRAME>`
  - вышло: `<BOOK> <DENTIST> <TIME> T_WED </TIME> <END_FRAME>`

## Примеры: manual_gold (6)

- **time+symptoms**
  - текст: `зуб разрывает, чё есть вечером после смены?`
  - надо: `<BOOK> <DENTIST> <TIME> T_EVENING T_AFTER_SHIFT </TIME> <SYMPTOM_TOOTHACHE> <END_FRAME>`
  - вышло: `<BOOK> <DENTIST> <TIME> T_AFTER_SHIFT </TIME> <END_FRAME>`
- **intent+specialty+time**
  - текст: `надо проверить зрение, сыну восемь лет`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `больно наступать на пятку, кто этим занимается?`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <NEEDS_CLARIFICATION> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `третий день температура за 38, запишите к кому надо`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_FEVER> <NEEDS_CLARIFICATION> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `приём у окулиста в среду больше не нужен`
  - надо: `<CANCEL> <OPHTHALMOLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<BOOK> <NEUROLOGIST> <TIME> T_WED </TIME> <END_FRAME>`
- **specialty**
  - текст: `панические атаки участились, нужен специалист`
  - надо: `<BOOK> <PSYCHOLOGIST> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_ANXIETY> <NEEDS_CLARIFICATION> <END_FRAME>`

## Примеры: hasd (227)

- **intent+specialty+time**
  - текст: `Доступность для лаборатории.`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Проверьте календарь бронирования.`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Измени запись.`
  - надо: `<RESCHEDULE> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty**
  - текст: `Мне нужно отменить встречу по поводу его диссертации.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<CANCEL> <DERMATOLOGIST> <T_UNKNOWN> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Нам нужно что-то перепланировать.`
  - надо: `<RESCHEDULE> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `У меня конфликт в планировании.`
  - надо: `<RESCHEDULE> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Список открыт.`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Я не буду там, просто отмени встречу.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Я посещаю мой уроки.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty**
  - текст: `Запиши меня.`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<BOOK> <DENTIST> <T_UNKNOWN> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Мне нужно удалить запланированный.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Я не могу записаться на встречу, пожалуйста, отмените ее.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`