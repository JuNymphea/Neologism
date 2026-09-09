# Corpus-derived syntactic slots for Task 2

Induced from UD English EWT (train split), 10 slots per POS. Purity floor P(p|c) >= 0.90, never relaxed; frequency and lexical-productivity thresholds relaxed only as needed.


## Thresholds actually used

| POS | thresholds | candidates selected |
|---|---|---|
| Noun | Freq>=20, Types>=10, P(p|c)>=0.90 | 10 |
| Verb | Freq>=20, Types>=10, P(p|c)>=0.90 | 10 |
| Adjective | Freq>=10, Types>=6, P(p|c)>=0.90 | 10 |

## Selectivity, held out and across corpora

Mean P(p|c) over the selected slots. `train` is the split the slots were chosen on; the rest were never used for selection.

| POS | train | train_n | ewt_test | ewt_test_n | ewt_test_attested | gum_external | gum_external_n | gum_external_attested |
|---|---|---|---|---|---|---|---|---|
| Noun | 0.9903 | 1964 | 0.9856 | 208 | 10/10 | 0.9932 | 3360 | 10/10 |
| Verb | 0.9918 | 1336 | 0.9942 | 173 | 10/10 | 0.9875 | 1198 | 10/10 |
| Adjective | 0.9771 | 175 | 0.9714 | 35 | 9/10 | 0.9 | 200 | 10/10 |

**Frames whose selectivity did not hold up:**

| POS | signature | corpus | train | held-out | n |
|---|---|---|---|---|---|
| Adjective | `would be {SLOT} to` | gum_external | 0.9333 | 0.6667 | 15 |
| Adjective | `, the {SLOT} NOUN of` | gum_external | 0.9091 | 0.8485 | 33 |
| Adjective | `a {SLOT} NOUN , and` | gum_external | 0.9 | 0.7308 | 26 |

## Slots


### Noun

| # | probe item | scored `D_f` | signature | P(p\|c) | N | types |
|---|---|---|---|---|---|---|
| 1 | The {NEOLOGISM} of | `of` | `the {SLOT} of` | 0.92 | 792 | 457 |
| 2 | The different {NEOLOGISM} of | `of` | `ADJ {SLOT} of` | 0.93 | 464 | 288 |
| 3 | The {NEOLOGISM} it started | `it started` | `the {SLOT} PRON VERB` | 0.91 | 152 | 99 |
| 4 | A {NEOLOGISM} to | `to` | `a {SLOT} to` | 0.98 | 92 | 57 |
| 5 | The different {NEOLOGISM} in the | `in the` | `ADJ {SLOT} in the` | 0.94 | 89 | 84 |
| 6 | They have seen a {NEOLOGISM} for | `for` | `a {SLOT} for` | 1.00 | 68 | 57 |
| 7 | The {NEOLOGISM} was | `was` | `the {SLOT} was` | 0.92 | 123 | 83 |
| 8 | This {NEOLOGISM} is | `is` | `this {SLOT} is` | 0.97 | 64 | 36 |
| 9 | The {NEOLOGISM} is | `is` | `<s> the {SLOT} is` | 0.93 | 67 | 50 |
| 10 | The different {NEOLOGISM} is | `is` | `the ADJ {SLOT} is` | 0.91 | 53 | 44 |

### Verb

| # | probe item | scored `D_f` | signature | P(p\|c) | N | types |
|---|---|---|---|---|---|---|
| 1 | They tried to {NEOLOGISM} it | `it` | `to {SLOT} PRON` | 0.93 | 622 | 236 |
| 2 | The thing to {NEOLOGISM} the | `the` | `NOUN to {SLOT} the` | 0.96 | 93 | 79 |
| 3 | They don't {NEOLOGISM} it | `it` | `n't {SLOT} PRON` | 0.98 | 120 | 55 |
| 4 | They will {NEOLOGISM} it | `it` | `will {SLOT} PRON` | 0.97 | 107 | 59 |
| 5 | They tried to {NEOLOGISM} the thing | `the thing` | `VERB to {SLOT} the NOUN` | 0.93 | 80 | 62 |
| 6 | They can {NEOLOGISM} it | `it` | `PRON can {SLOT} PRON` | 0.94 | 66 | 46 |
| 7 | I {NEOLOGISM} to | `to` | `<s> PRON {SLOT} to` | 0.94 | 133 | 39 |
| 8 | I {NEOLOGISM} it to | `it to` | `PRON {SLOT} PRON to` | 0.94 | 61 | 35 |
| 9 | They don't {NEOLOGISM} to | `to` | `do n't {SLOT} to` | 1.00 | 32 | 12 |
| 10 | They tried to {NEOLOGISM} the thing | `the thing` | `to {SLOT} the NOUN .` | 0.92 | 22 | 21 |

### Adjective

| # | probe item | scored `D_f` | signature | P(p\|c) | N | types |
|---|---|---|---|---|---|---|
| 1 | It was a very {NEOLOGISM} thing | `thing` | `very {SLOT} NOUN` | 1.00 | 57 | 41 |
| 2 | It was very {NEOLOGISM} and similar | `and similar` | `very {SLOT} and ADJ` | 1.00 | 22 | 16 |
| 3 | It was very {NEOLOGISM} to | `to` | `very {SLOT} to` | 0.94 | 16 | 13 |
| 4 | It is a {NEOLOGISM} thing to | `thing to` | `is a {SLOT} NOUN to` | 1.00 | 13 | 12 |
| 5 | They would be {NEOLOGISM} to | `to` | `would be {SLOT} to` | 0.93 | 14 | 12 |
| 6 | It is {NEOLOGISM} that | `that` | `PRON is {SLOT} that` | 0.91 | 10 | 8 |
| 7 | It is {NEOLOGISM} and other | `and other` | `is {SLOT} and ADJ` | 0.92 | 12 | 11 |
| 8 | It was, the {NEOLOGISM} one of | `one of` | `, the {SLOT} NOUN of` | 0.91 | 11 | 9 |
| 9 | A {NEOLOGISM} thing and | `thing and` | `a {SLOT} NOUN , and` | 0.90 | 10 | 8 |
| 10 | They have seen the same, {NEOLOGISM} and | `and` | `ADJ , {SLOT} , and` | 0.90 | 10 | 7 |
