# Кейсы, которые модель не обработала

Всего промахов: 547

## По наборам

- hasd: 229
- test_ood: 191
- test_iid: 75
- val: 37
- manual_gold: 9
- hasd_unsup: 4
- clinc: 2

## По полям

- specialty: 440
- intent: 293
- time: 291
- symptoms: 92

## Примеры: test_ood (191)

- **intent+specialty+time+symptoms**
  - текст: `шатает при ходьбе невропатолог нужен срочно на субботу`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_SAT </TIME> <SYMPTOM_DIZZINESS> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `Шатает при ходьбе. Невропатолог нужен срочно, на субботу.`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_SAT </TIME> <SYMPTOM_DIZZINESS> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty+symptoms**
  - текст: `надо бы к невропатологу попасть, на субботу. Шатает при ходьбе.`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_SAT </TIME> <SYMPTOM_DIZZINESS> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_SAT </TIME> <END_FRAME>`
- **specialty+symptoms**
  - текст: `хотела бы записаться к невропатологу, если есть места, на субботу. Шатает при ходьбе.`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_SAT </TIME> <SYMPTOM_DIZZINESS> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_SAT </TIME> <END_FRAME>`
- **specialty+symptoms**
  - текст: `записалась бы к невропатологу, если есть места, на субботу. Шатает при ходьбе.`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_SAT </TIME> <SYMPTOM_DIZZINESS> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_SAT </TIME> <END_FRAME>`
- **specialty+symptoms**
  - текст: `надо бы к отоларингологу попасть, днём. обсыпало.`
  - надо: `<BOOK> <ENT> <TIME> T_AFTERNOON </TIME> <SYMPTOM_RASH> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_AFTERNOON </TIME> <NEEDS_CLARIFICATION> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `обсыпало отоларинголог нужен срочно днём`
  - надо: `<BOOK> <ENT> <TIME> T_AFTERNOON </TIME> <SYMPTOM_RASH> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `обсыпало отоларинголог нужен срочно днём шею заклинило`
  - надо: `<BOOK> <ENT> <TIME> T_AFTERNOON </TIME> <SYMPTOM_RASH> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `Обсыпало. Отоларинголог нужен срочно, в 9:00.`
  - надо: `<BOOK> <ENT> <TIME> T_H09_00 </TIME> <SYMPTOM_RASH> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `обсыпало отоларинголог нужен срочно в 9:00`
  - надо: `<BOOK> <ENT> <TIME> T_H09_00 </TIME> <SYMPTOM_RASH> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty+symptoms**
  - текст: `надо бы к отоларингологу попасть в 9:00 обсыпало`
  - надо: `<BOOK> <ENT> <TIME> T_H09_00 </TIME> <SYMPTOM_RASH> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_H09_00 </TIME> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty+symptoms**
  - текст: `хотел бы записаться к отоларингологу, если есть места, в 9:00. Обсыпало.`
  - надо: `<BOOK> <ENT> <TIME> T_H09_00 </TIME> <SYMPTOM_RASH> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_H09_00 </TIME> <END_FRAME>`

## Примеры: manual_gold (9)

- **intent+specialty+time+symptoms**
  - текст: `зуб разрывает, чё есть вечером после смены?`
  - надо: `<BOOK> <DENTIST> <TIME> T_EVENING T_AFTER_SHIFT </TIME> <SYMPTOM_TOOTHACHE> <END_FRAME>`
  - вышло: `<RESCHEDULE> <UNKNOWN_SPECIALTY> <TIME> T_AFTER_SHIFT </TIME> <END_FRAME>`
- **intent+specialty+time**
  - текст: `надо проверить зрение, сыну восемь лет`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `по записи к неврологу приходить или в живую очередь?`
  - надо: `<SCHEDULE> <NEUROLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<RESCHEDULE> <UNKNOWN_SPECIALTY> <TIME> T_SOON </TIME> <END_FRAME>`
- **specialty**
  - текст: `отмените всё, уехал из города`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<CANCEL> <PSYCHOLOGIST> <T_UNKNOWN> <END_FRAME>`
- **specialty**
  - текст: `не к терапевту, а к лору, если можно`
  - надо: `<BOOK> <ENT> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<BOOK> <GP> <T_UNKNOWN> <END_FRAME>`
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
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty**
  - текст: `панические атаки участились, нужен специалист`
  - надо: `<BOOK> <PSYCHOLOGIST> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`
  - вышло: `<CHECK_BOOKING> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`

## Примеры: hasd (229)

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
- **intent+specialty+time**
  - текст: `Запиши меня.`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Мне нужно удалить запланированный.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Я не могу записаться на встречу, пожалуйста, отмените ее.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`