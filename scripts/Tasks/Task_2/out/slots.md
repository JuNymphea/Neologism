# Corpus-derived syntactic slots for Task 2

Induced from UD English EWT (train split), 10 slots per POS. Purity floor P(p|c) >= 0.90, never relaxed; frequency and lexical-productivity thresholds relaxed only as needed.


## Thresholds actually used

| POS | thresholds | candidates selected |
|---|---|---|
| Noun | Freq>=20, Types>=10, P(p|c)>=0.90 | 10 |
| Verb | Freq>=20, Types>=10, P(p|c)>=0.90 | 10 |
| Adjective | Freq>=8, Types>=5, P(p|c)>=0.90 | 10 |

## Selectivity, held out and across corpora

Mean P(p|c) over the selected slots. `train` is the split the slots were chosen on; the rest were never used for selection.

| POS | train | train_n | ewt_test | ewt_test_n | ewt_test_attested | gum_external | gum_external_n | gum_external_attested |
|---|---|---|---|---|---|---|---|---|
| Noun | 0.9903 | 1849 | 0.9747 | 198 | 10/10 | 0.9931 | 3197 | 10/10 |
| Verb | 0.9944 | 1246 | 0.9935 | 153 | 9/10 | 0.9898 | 1076 | 10/10 |
| Adjective | 0.9941 | 170 | 1.0 | 38 | 7/10 | 0.9379 | 145 | 10/10 |

**Frames whose selectivity did not hold up:**

| POS | signature | corpus | train | held-out | n |
|---|---|---|---|---|---|
| Adjective | `would be {SLOT} to VERB` | gum_external | 0.9231 | 0.6667 | 9 |
| Adjective | `a more {SLOT} NOUN` | gum_external | 1.0 | 0.8462 | 13 |

## Slots


### Noun

| # | probe item | scored `D_f` | signature | P(p\|c) | N | types |
|---|---|---|---|---|---|---|
| 1 | The {NEOLOGISM} of | `of` | `the {SLOT} of` | 0.92 | 792 | 457 |
| 2 | The different {NEOLOGISM} of | `of` | `ADJ {SLOT} of` | 0.93 | 464 | 288 |
| 3 | A {NEOLOGISM} to | `to` | `a {SLOT} to` | 0.98 | 92 | 57 |
| 4 | The {NEOLOGISM} was | `was` | `the {SLOT} was` | 0.92 | 123 | 83 |
| 5 | They did the {NEOLOGISM} to | `to` | `VERB the {SLOT} to` | 0.91 | 100 | 66 |
| 6 | The {NEOLOGISM} is | `is` | `<s> the {SLOT} is` | 0.93 | 67 | 50 |
| 7 | They have seen the different {NEOLOGISM} it happened | `it happened` | `the ADJ {SLOT} PRON VERB` | 0.95 | 62 | 35 |
| 8 | The different {NEOLOGISM} is | `is` | `the ADJ {SLOT} is` | 0.91 | 53 | 44 |
| 9 | A different {NEOLOGISM} in | `in` | `a ADJ {SLOT} in` | 1.00 | 42 | 38 |
| 10 | A different {NEOLOGISM} to | `to` | `a ADJ {SLOT} to` | 0.95 | 54 | 36 |

### Verb

| # | probe item | scored `D_f` | signature | P(p\|c) | N | types |
|---|---|---|---|---|---|---|
| 1 | They tried to {NEOLOGISM} it | `it` | `to {SLOT} PRON` | 0.93 | 622 | 236 |
| 2 | The thing to {NEOLOGISM} the | `the` | `NOUN to {SLOT} the` | 0.96 | 93 | 79 |
| 3 | They don't {NEOLOGISM} it | `it` | `n't {SLOT} PRON` | 0.98 | 120 | 55 |
| 4 | They will {NEOLOGISM} it | `it` | `will {SLOT} PRON` | 0.97 | 107 | 59 |
| 5 | They can {NEOLOGISM} it | `it` | `PRON can {SLOT} PRON` | 0.94 | 66 | 46 |
| 6 | I {NEOLOGISM} to do | `to do` | `<s> PRON {SLOT} to VERB` | 0.97 | 96 | 27 |
| 7 | I {NEOLOGISM} it to | `it to` | `PRON {SLOT} PRON to` | 0.94 | 61 | 35 |
| 8 | It was when it {NEOLOGISM} it | `it` | `when PRON {SLOT} PRON` | 1.00 | 37 | 29 |
| 9 | Someone to {NEOLOGISM} the | `the` | `PROPN to {SLOT} the` | 0.95 | 20 | 20 |
| 10 | They don't {NEOLOGISM} the | `the` | `do n't {SLOT} the` | 1.00 | 24 | 13 |

### Adjective

| # | probe item | scored `D_f` | signature | P(p\|c) | N | types |
|---|---|---|---|---|---|---|
| 1 | It was a very {NEOLOGISM} thing | `thing` | `very {SLOT} NOUN` | 1.00 | 57 | 41 |
| 2 | It was very {NEOLOGISM} and similar | `and similar` | `very {SLOT} and ADJ` | 1.00 | 22 | 16 |
| 3 | They have seen the same, {NEOLOGISM} and | `and` | `ADJ , {SLOT} and` | 0.92 | 24 | 21 |
| 4 | It is a {NEOLOGISM} thing to | `thing to` | `is a {SLOT} NOUN to` | 1.00 | 13 | 12 |
| 5 | They would be {NEOLOGISM} to do | `to do` | `would be {SLOT} to VERB` | 0.92 | 12 | 10 |
| 6 | A more {NEOLOGISM} thing | `thing` | `a more {SLOT} NOUN` | 1.00 | 9 | 9 |
| 7 | A {NEOLOGISM} thing to it | `thing to it` | `a {SLOT} NOUN to PRON` | 1.00 | 9 | 8 |
| 8 | The other and {NEOLOGISM} and | `and` | `ADJ and {SLOT} and` | 1.00 | 8 | 8 |
| 9 | It is {NEOLOGISM} same | `same` | `is {SLOT} , ADJ` | 1.00 | 8 | 8 |
| 10 | The {NEOLOGISM} one is that | `one is that` | `<s> the {SLOT} NOUN is that` | 1.00 | 8 | 7 |
