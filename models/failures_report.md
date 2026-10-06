# Кейсы, которые модель не обработала

Всего промахов: 423

## По наборам

- hasd: 248
- test_ood: 167
- manual_gold: 8

## По полям

- specialty: 411
- time: 285
- intent: 277
- symptoms: 135

## Примеры: test_ood (167)

- **intent+specialty+time**
  - текст: `женский врач нужен срочно`
  - надо: `<BOOK> <GYNECOLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty**
  - текст: `надо бы к женскому врачу попасть.`
  - надо: `<BOOK> <GYNECOLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <NEEDS_CLARIFICATION> <END_FRAME>`
- **specialty+symptoms**
  - текст: `записалась бы к женскому врачу, если есть места.`
  - надо: `<BOOK> <GYNECOLOGIST> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_TOOTHACHE> <NEEDS_CLARIFICATION> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `мигрень болят глаза окулист нужен срочно`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_HEADACHE> <SYMPTOM_EYE_PAIN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **symptoms**
  - текст: `надо бы к окулисту попасть мигрень болят глаза`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_HEADACHE> <SYMPTOM_EYE_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_EYE_PAIN> <END_FRAME>`
- **symptoms**
  - текст: `надо бы к окулисту попасть. мигрень. болят глаза.`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_HEADACHE> <SYMPTOM_EYE_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_EYE_PAIN> <END_FRAME>`
- **specialty+symptoms**
  - текст: `хотел бы записаться к окулисту, если есть места. Мигрень. Болят глаза.`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_HEADACHE> <SYMPTOM_EYE_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_EYE_PAIN> <NEEDS_CLARIFICATION> <END_FRAME>`
- **symptoms**
  - текст: `мигрень. болят глаза. окулист нужен срочно.`
  - надо: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_HEADACHE> <SYMPTOM_EYE_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <OPHTHALMOLOGIST> <T_UNKNOWN> <SYMPTOM_EYE_PAIN> <END_FRAME>`
- **specialty+symptoms**
  - текст: `хотел бы записаться к невропатологу, если есть места. Спину скрутило.`
  - надо: `<BOOK> <NEUROLOGIST> <T_UNKNOWN> <SYMPTOM_BACK_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <SYMPTOM_ANXIETY> <NEEDS_CLARIFICATION> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `спину скрутило невропатолог нужен срочно`
  - надо: `<BOOK> <NEUROLOGIST> <T_UNKNOWN> <SYMPTOM_BACK_PAIN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`
- **specialty+symptoms**
  - текст: `надо бы к невропатологу попасть. Спину скрутило.`
  - надо: `<BOOK> <NEUROLOGIST> <T_UNKNOWN> <SYMPTOM_BACK_PAIN> <END_FRAME>`
  - вышло: `<BOOK> <DENTIST> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`
- **intent+specialty+time+symptoms**
  - текст: `спину скрутило. невропатолог нужен срочно.`
  - надо: `<BOOK> <NEUROLOGIST> <T_UNKNOWN> <SYMPTOM_BACK_PAIN> <END_FRAME>`
  - вышло: `<UNSUPPORTED> <END_FRAME>`

## Примеры: manual_gold (8)

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

## Примеры: hasd (248)

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
- **specialty+time+symptoms**
  - текст: `Мне нужно отменить встречу по поводу его диссертации.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<CANCEL> <DENTIST> <TIME> T_SOON </TIME> <SYMPTOM_ANXIETY> <END_FRAME>`
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
- **specialty+symptoms**
  - текст: `Я не могу записаться на встречу, пожалуйста, отмените ее.`
  - надо: `<CANCEL> <UNKNOWN_SPECIALTY> <T_UNKNOWN> <END_FRAME>`
  - вышло: `<CANCEL> <PSYCHOLOGIST> <T_UNKNOWN> <SYMPTOM_ANXIETY> <END_FRAME>`