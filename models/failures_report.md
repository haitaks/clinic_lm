# Кейсы, которые модель не обработала

Всего промахов: 427

## По наборам

- hasd: 247
- test_ood: 170
- manual_gold: 9
- val: 1

## По полям

- specialty: 420
- time: 302
- intent: 301
- symptoms: 97

## Примеры: test_ood (170)

- **specialty**
  - текст: `надо бы к кожнику попасть, в четыре часа дня. Аллергия на что-то.`
  - надо: `<BOOK> <DERMATOLOGIST> <TIME> T_H16_00 </TIME> <SYMPTOM_ALLERGY> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_H16_00 </TIME> <SYMPTOM_ALLERGY> <NEEDS_CLARIFICATION> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `аллергия на чтото кожник нужен срочно в четыре часа дня`
  - надо: `<BOOK> <DERMATOLOGIST> <TIME> T_H16_00 </TIME> <SYMPTOM_ALLERGY> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `Аллергия на что-то. Кожник нужен срочно, в четыре часа дня.`
  - надо: `<BOOK> <DERMATOLOGIST> <TIME> T_H16_00 </TIME> <SYMPTOM_ALLERGY> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty**
  - текст: `записалась бы к кожнику, если есть места, в четыре часа дня. Аллергия на что-то.`
  - надо: `<BOOK> <DERMATOLOGIST> <TIME> T_H16_00 </TIME> <SYMPTOM_ALLERGY> <END_FRAME>`
  - вышло: `<BOOK> <PSYCHOLOGIST> <TIME> T_H16_00 </TIME> <SYMPTOM_ALLERGY> <END_FRAME>`
- **specialty**
  - текст: `надо бы к кожнику попасть в четыре часа дня аллергия на чтото`
  - надо: `<BOOK> <DERMATOLOGIST> <TIME> T_H16_00 </TIME> <SYMPTOM_ALLERGY> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_H16_00 </TIME> <SYMPTOM_ALLERGY> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty**
  - текст: `записалась бы к невропатологу, если есть места, послезавтра утром.`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_DAY_AFTER_TOMORROW T_MORNING </TIME> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <TIME> T_DAY_AFTER_TOMORROW T_MORNING </TIME> <END_FRAME>`
- **intent+specialty+time**
  - текст: `Невропатолог нужен срочно, послезавтра утром.`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_DAY_AFTER_TOMORROW T_MORNING </TIME> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty**
  - текст: `записалась бы к невропатологу если есть места послезавтра утром`
  - надо: `<BOOK> <NEUROLOGIST> <TIME> T_DAY_AFTER_TOMORROW T_MORNING </TIME> <END_FRAME>`
  - вышло: `<BOOK> <DENTIST> <TIME> T_DAY_AFTER_TOMORROW T_MORNING </TIME> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `часто накрывает тревога психоаналитик нужен срочно`
  - надо: `<BOOK> <PSYCHOLOGIST> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `часто накрывает тревога. психоаналитик нужен срочно.`
  - надо: `<BOOK> <PSYCHOLOGIST> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty**
  - текст: `надо бы к психоаналитику попасть часто накрывает тревога`
  - надо: `<BOOK> <PSYCHOLOGIST> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_ANXIETY> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty**
  - текст: `Семейный врач нужен срочно.`
  - надо: `<BOOK> <GP> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<BOOK> <PSYCHOLOGIST> <T_UNKNOWN> <END_FRAME>`

## Примеры: manual_gold (9)

- **intent+specialty+time+symptoms**
  - текст: `зуб разрывает, чё есть вечером после смены?`
  - надо: `<BOOK> <DENTIST> <TIME> T_EVENING T_AFTER_SHIFT </TIME> <SYMPTOM_TOOTHACHE> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `надо проверить зрение, сыну восемь лет`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty+symptoms**
  - текст: `отмените всё, уехал из города`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<CANCEL> <DENTIST> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`
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
  - текст: `жена просит записать её к гинекологу на следующей неделе`
  - надо: `<BOOK> <GYNECOLOGIST> <TIME> T_NEXT_WEEK </TIME> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time**
  - текст: `приём у окулиста в среду больше не нужен`
  - надо: `<CANCEL> <OPHTHALMOLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `панические атаки участились, нужен специалист`
  - надо: `<BOOK> <PSYCHOLOGIST> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`

## Примеры: hasd (247)

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
- **intent+specialty+time**
  - текст: `Мне нужно отменить встречу по поводу его диссертации.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
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