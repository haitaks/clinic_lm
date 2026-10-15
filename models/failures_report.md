# Кейсы, которые модель не обработала

Всего промахов: 459

## По наборам

- hasd: 198
- test_ood: 182
- hasd_unsup: 38
- clinc: 17
- val: 8
- test_iid: 8
- manual_gold: 8

## По полям

- specialty: 412
- intent: 270
- time: 236
- symptoms: 96

## Примеры: test_ood (182)

- **specialty**
  - текст: `надо бы к травматологу попасть ноет живот`
  - надо: `<BOOK> <SURGEON> <T_UNKNOWN> <SYMPTOM_STOMACH_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_STOMACH_PAIN> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty**
  - текст: `Ноет живот. Травматолог нужен срочно.`
  - надо: `<BOOK> <SURGEON> <T_UNKNOWN> <SYMPTOM_STOMACH_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <ENT> <T_UNKNOWN> <SYMPTOM_STOMACH_PAIN> <END_FRAME>`
- **specialty**
  - текст: `надо бы к травматологу попасть. Ноет живот.`
  - надо: `<BOOK> <SURGEON> <T_UNKNOWN> <SYMPTOM_STOMACH_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_STOMACH_PAIN> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty**
  - текст: `ноет живот травматолог нужен срочно`
  - надо: `<BOOK> <SURGEON> <T_UNKNOWN> <SYMPTOM_STOMACH_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_STOMACH_PAIN> <END_FRAME>`
- **specialty+symptoms**
  - текст: `записался бы к травматологу если есть места ноет живот`
  - надо: `<BOOK> <SURGEON> <T_UNKNOWN> <SYMPTOM_STOMACH_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
- **specialty+symptoms**
  - текст: `надо бы к окулисту попасть. Мигрень. Болят глаза.`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_HEADACHE> <SYMPTOM_EYE_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_EYE_PAIN> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty+symptoms**
  - текст: `записалась бы к окулисту, если есть места. мигрень. болят глаза.`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_HEADACHE> <SYMPTOM_EYE_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_EYE_PAIN> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty+symptoms**
  - текст: `записался бы к окулисту, если есть места. мигрень. болят глаза.`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_HEADACHE> <SYMPTOM_EYE_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_COUGH> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty**
  - текст: `надо бы к дантисту попасть`
  - надо: `<BOOK> <DENTIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
- **specialty**
  - текст: `записался бы к дантисту, если есть места.`
  - надо: `<BOOK> <DENTIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
- **intent+specialty+time**
  - текст: `дантист нужен срочно.`
  - надо: `<BOOK> <DENTIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty**
  - текст: `Дантист нужен срочно. Локоть ноет.`
  - надо: `<BOOK> <DENTIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<LAB_STATUS> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`

## Примеры: manual_gold (8)

- **time+symptoms**
  - текст: `зуб разрывает, чё есть вечером после смены?`
  - надо: `<BOOK> <DENTIST> <TIME> T_EVENING T_AFTER_SHIFT </TIME> <SYMPTOM_TOOTHACHE> <END_FRAME>`
  - вышло: `<BOOK> <DENTIST> <TIME> T_TOMORROW T_EVENING </TIME> <END_FRAME>`
- **intent+specialty+time**
  - текст: `надо проверить зрение, сыну восемь лет`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty**
  - текст: `по записи к неврологу приходить или в живую очередь?`
  - надо: `<SCHEDULE> <NEUROLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<CHECK_BOOKING> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
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
- **intent+specialty+time+symptoms**
  - текст: `панические атаки участились, нужен специалист`
  - надо: `<BOOK> <PSYCHOLOGIST> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`

## Примеры: hasd (198)

- **intent+specialty+time**
  - текст: `Доступность для лаборатории.`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Измени запись.`
  - надо: `<RESCHEDULE> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent**
  - текст: `Нам нужно что-то перепланировать.`
  - надо: `<RESCHEDULE> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<CHECK_BOOKING> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
- **intent+specialty+time**
  - текст: `У меня конфликт в планировании.`
  - надо: `<RESCHEDULE> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent**
  - текст: `Список открыт.`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<SCHEDULE> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Я посещаю мой уроки.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent**
  - текст: `Запиши меня.`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<CHECK_BOOKING> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Я не могу записаться на встречу, пожалуйста, отмените ее.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Мне нужно что-то переместить на запястье.`
  - надо: `<RESCHEDULE> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent**
  - текст: `Нужно записаться на работу`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<SCHEDULE> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Написать мой устный экзамен.`
  - надо: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `принял мою запись.`
  - надо: `<RESCHEDULE> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`