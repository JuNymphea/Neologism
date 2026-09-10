# Corpus-derived syntactic slots for Task 2

Corpus candidate pool from UD English EWT (train split), up to 40 per POS. Identical thresholds for all three categories; the model-side screening picks the final set. Purity floor P(p|c) >= 0.90, never relaxed; frequency and lexical-productivity thresholds relaxed only as needed.


## Thresholds actually used

| POS | thresholds | candidates selected |
|---|---|---|
| Noun | Freq>=5, Types>=4, P(p|c)>=0.90 | 40 |
| Verb | Freq>=5, Types>=4, P(p|c)>=0.90 | 40 |
| Adjective | Freq>=5, Types>=4, P(p|c)>=0.90 | 29 |

## Selectivity, held out and across corpora

Mean P(p|c) over the selected slots. `train` is the split the slots were chosen on; the rest were never used for selection.

| POS | train | train_n | ewt_test | ewt_test_n | ewt_test_attested | gum_external | gum_external_n | gum_external_attested |
|---|---|---|---|---|---|---|---|---|
| Noun | 0.9916 | 2257 | 0.9781 | 228 | 26/40 | 0.9914 | 3477 | 38/40 |
| Verb | 0.9924 | 2092 | 0.9918 | 245 | 33/40 | 0.9849 | 1849 | 39/40 |
| Adjective | 0.9899 | 298 | 0.9667 | 60 | 20/29 | 0.9164 | 275 | 27/29 |

**Frames whose selectivity did not hold up:**

| POS | signature | corpus | train | held-out | n |
|---|---|---|---|---|---|
| Noun | `and the {SLOT} is` | gum_external | 0.9375 | 0.75 | 12 |
| Noun | `NUM {SLOT} after PRON` | gum_external | 1.0 | 0.8333 | 6 |
| Verb | `NOUN to {SLOT} to` | ewt_test | 0.9231 | 0.8 | 5 |
| Verb | `NOUN to {SLOT} with` | gum_external | 0.9375 | 0.7143 | 7 |
| Verb | `PRON {SLOT} to VERB ,` | gum_external | 1.0 | 0.6429 | 14 |
| Adjective | `would be {SLOT} to VERB` | gum_external | 0.9231 | 0.6667 | 9 |
| Adjective | `a more {SLOT} NOUN` | gum_external | 1.0 | 0.8462 | 13 |
| Adjective | `, the {SLOT} NOUN of` | gum_external | 0.9091 | 0.8485 | 33 |
| Adjective | `a {SLOT} NOUN , and` | gum_external | 0.9 | 0.7308 | 26 |
| Adjective | `some {SLOT} NOUN NOUN` | gum_external | 1.0 | 0.7778 | 9 |

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
| 11 | One - {NEOLOGISM} of | `of` | `NUM - {SLOT} of` | 1.00 | 130 | 8 |
| 12 | They have seen the {NEOLOGISM} it was | `it was` | `the {SLOT} PRON was` | 1.00 | 18 | 16 |
| 13 | The different {NEOLOGISM} and | `and` | `the ADJ {SLOT} , and` | 1.00 | 15 | 14 |
| 14 | The different {NEOLOGISM} with someone | `with someone` | `ADJ {SLOT} with PROPN` | 1.00 | 14 | 13 |
| 15 | The different {NEOLOGISM} is that | `is that` | `ADJ {SLOT} is that` | 1.00 | 15 | 11 |
| 16 | The different {NEOLOGISM} and one thing | `and one thing` | `ADJ {SLOT} and NOUN NOUN` | 0.95 | 19 | 16 |
| 17 | They did it and the {NEOLOGISM} is | `is` | `and the {SLOT} is` | 0.94 | 15 | 11 |
| 18 | It was as the {NEOLOGISM} happened | `happened` | `as the {SLOT} VERB` | 1.00 | 11 | 11 |
| 19 | It was one {NEOLOGISM} before | `before` | `NUM {SLOT} before` | 1.00 | 14 | 7 |
| 20 | A different {NEOLOGISM} and | `and` | `a ADJ {SLOT} , and` | 1.00 | 9 | 9 |
| 21 | The different {NEOLOGISM} is the thing | `is the thing` | `ADJ {SLOT} is the NOUN` | 1.00 | 9 | 9 |
| 22 | They have seen the different {NEOLOGISM} it is | `it is` | `ADJ {SLOT} , PRON is` | 0.91 | 10 | 10 |
| 23 | The {NEOLOGISM} s | `s` | `the {SLOT} s` | 1.00 | 9 | 8 |
| 24 | They have seen the different {NEOLOGISM} where | `where` | `ADJ {SLOT} where` | 0.91 | 10 | 9 |
| 25 | The different {NEOLOGISM} will be | `will be` | `ADJ {SLOT} will be` | 0.90 | 9 | 9 |
| 26 | This {NEOLOGISM} has | `has` | `this {SLOT} has` | 1.00 | 8 | 7 |
| 27 | The {NEOLOGISM} for | `for` | `<s> the {SLOT} for` | 1.00 | 7 | 7 |
| 28 | The one {NEOLOGISM} and other | `and other` | `NOUN {SLOT} , and ADJ` | 1.00 | 7 | 7 |
| 29 | The one's {NEOLOGISM} and | `and` | `NOUN 's {SLOT} and` | 1.00 | 7 | 7 |
| 30 | It is the {NEOLOGISM} to | `to` | `is the {SLOT} to` | 1.00 | 8 | 6 |
| 31 | It was, this {NEOLOGISM} happened | `happened` | `, this {SLOT} VERB` | 1.00 | 7 | 6 |
| 32 | They have seen the {NEOLOGISM} it would do | `it would do` | `the {SLOT} PRON would VERB` | 1.00 | 7 | 6 |
| 33 | One {NEOLOGISM} of other | `of other` | `NUM {SLOT} of ADJ` | 1.00 | 7 | 6 |
| 34 | They have seen a {NEOLOGISM} when it | `when it` | `a {SLOT} when PRON` | 1.00 | 7 | 6 |
| 35 | They did this {NEOLOGISM} happened | `happened` | `VERB this {SLOT} VERB` | 1.00 | 6 | 6 |
| 36 | The different {NEOLOGISM} behind | `behind` | `ADJ {SLOT} behind` | 1.00 | 6 | 6 |
| 37 | All the {NEOLOGISM} happened | `happened` | `all the {SLOT} VERB` | 1.00 | 6 | 6 |
| 38 | It was one {NEOLOGISM} after it | `after it` | `NUM {SLOT} after PRON` | 1.00 | 7 | 5 |
| 39 | They tried to the {NEOLOGISM} it did | `it did` | `to the {SLOT} PRON VERB` | 1.00 | 6 | 5 |
| 40 | One {NEOLOGISM} of | `of` | `<s> NUM {SLOT} of` | 1.00 | 5 | 5 |

### Verb

| # | probe item | scored `D_f` | signature | P(p\|c) | N | types |
|---|---|---|---|---|---|---|
| 1 | They tried to {NEOLOGISM} it | `it` | `to {SLOT} PRON` | 0.93 | 622 | 236 |
| 2 | They tried to {NEOLOGISM} the | `the` | `to {SLOT} the` | 0.93 | 345 | 209 |
| 3 | They don't {NEOLOGISM} it | `it` | `n't {SLOT} PRON` | 0.98 | 120 | 55 |
| 4 | They will {NEOLOGISM} it | `it` | `will {SLOT} PRON` | 0.97 | 107 | 59 |
| 5 | They can {NEOLOGISM} it | `it` | `can {SLOT} PRON` | 0.94 | 78 | 55 |
| 6 | I {NEOLOGISM} to do | `to do` | `<s> PRON {SLOT} to VERB` | 0.97 | 96 | 27 |
| 7 | I {NEOLOGISM} that | `that` | `<s> PRON {SLOT} that` | 1.00 | 65 | 30 |
| 8 | I {NEOLOGISM} it to | `it to` | `PRON {SLOT} PRON to` | 0.94 | 61 | 35 |
| 9 | It was when it {NEOLOGISM} it | `it` | `when PRON {SLOT} PRON` | 1.00 | 37 | 29 |
| 10 | They don't {NEOLOGISM} the | `the` | `n't {SLOT} the` | 0.93 | 40 | 24 |
| 11 | They'll {NEOLOGISM} it | `it` | `PRON 'll {SLOT} PRON` | 1.00 | 37 | 20 |
| 12 | They would {NEOLOGISM} it | `it` | `PRON would {SLOT} PRON` | 0.95 | 36 | 21 |
| 13 | I {NEOLOGISM} that it | `that it` | `PRON {SLOT} that PRON` | 0.90 | 78 | 31 |
| 14 | They tried to {NEOLOGISM} this | `this` | `VERB to {SLOT} this` | 1.00 | 18 | 15 |
| 15 | They did it and {NEOLOGISM} it to | `it to` | `and {SLOT} PRON to` | 1.00 | 18 | 13 |
| 16 | If it {NEOLOGISM} any | `any` | `if PRON {SLOT} any` | 1.00 | 45 | 6 |
| 17 | They can {NEOLOGISM} to | `to` | `can {SLOT} to` | 1.00 | 17 | 12 |
| 18 | They have seen the thing to {NEOLOGISM} to | `to` | `NOUN to {SLOT} to` | 0.92 | 25 | 17 |
| 19 | They don't {NEOLOGISM} any | `any` | `n't {SLOT} any` | 1.00 | 17 | 11 |
| 20 | I just {NEOLOGISM} it | `it` | `PRON just {SLOT} PRON` | 0.94 | 17 | 12 |
| 21 | They tried to {NEOLOGISM} this thing | `this thing` | `to {SLOT} this NOUN .` | 1.00 | 13 | 12 |
| 22 | It does not {NEOLOGISM} it | `it` | `does not {SLOT} PRON` | 1.00 | 14 | 11 |
| 23 | When it {NEOLOGISM} the | `the` | `when PRON {SLOT} the` | 0.94 | 15 | 12 |
| 24 | They tried to {NEOLOGISM} that | `that` | `VERB to {SLOT} that` | 0.91 | 21 | 16 |
| 25 | Because it {NEOLOGISM} it | `it` | `because PRON {SLOT} PRON` | 0.93 | 13 | 13 |
| 26 | The thing to {NEOLOGISM} with | `with` | `NOUN to {SLOT} with` | 0.94 | 15 | 11 |
| 27 | It also {NEOLOGISM} it | `it` | `PRON also {SLOT} PRON` | 1.00 | 12 | 11 |
| 28 | They do not {NEOLOGISM} the | `the` | `do not {SLOT} the` | 1.00 | 11 | 7 |
| 29 | Then {NEOLOGISM} it to | `it to` | `ADV {SLOT} PRON to` | 1.00 | 9 | 8 |
| 30 | They also {NEOLOGISM} that | `that` | `also {SLOT} that` | 1.00 | 9 | 8 |
| 31 | They do not {NEOLOGISM} it | `it` | `do not {SLOT} PRON` | 1.00 | 9 | 8 |
| 32 | They can {NEOLOGISM} for | `for` | `can {SLOT} for` | 1.00 | 9 | 7 |
| 33 | They did not {NEOLOGISM} it | `it` | `did not {SLOT} PRON` | 1.00 | 9 | 7 |
| 34 | They {NEOLOGISM} it with | `it with` | `PRON {SLOT} PRON with` | 1.00 | 8 | 7 |
| 35 | They did it and {NEOLOGISM} the one thing | `the one thing` | `and {SLOT} the NOUN NOUN` | 1.00 | 8 | 7 |
| 36 | The thing to {NEOLOGISM} this | `this` | `NOUN to {SLOT} this` | 1.00 | 7 | 7 |
| 37 | They'll {NEOLOGISM} again | `again` | `'ll {SLOT} ADV` | 1.00 | 8 | 6 |
| 38 | They {NEOLOGISM} to do | `to do` | `PRON {SLOT} to VERB ,` | 1.00 | 9 | 5 |
| 39 | They tried to {NEOLOGISM} those | `those` | `to {SLOT} those` | 1.00 | 7 | 6 |
| 40 | The thing to {NEOLOGISM} some | `some` | `NOUN to {SLOT} some` | 1.00 | 7 | 6 |

### Adjective

| # | probe item | scored `D_f` | signature | P(p\|c) | N | types |
|---|---|---|---|---|---|---|
| 1 | It was a very {NEOLOGISM} thing | `thing` | `very {SLOT} NOUN` | 1.00 | 57 | 41 |
| 2 | It was very {NEOLOGISM} and similar | `and similar` | `very {SLOT} and ADJ` | 1.00 | 22 | 16 |
| 3 | They have seen the same, {NEOLOGISM} and | `and` | `ADJ , {SLOT} and` | 0.92 | 24 | 21 |
| 4 | It is a {NEOLOGISM} thing to | `thing to` | `is a {SLOT} NOUN to` | 1.00 | 13 | 12 |
| 5 | They would be {NEOLOGISM} to do | `to do` | `would be {SLOT} to VERB` | 0.92 | 12 | 10 |
| 6 | They were very {NEOLOGISM} with | `with` | `very {SLOT} with` | 1.00 | 13 | 7 |
| 7 | A more {NEOLOGISM} thing | `thing` | `a more {SLOT} NOUN` | 1.00 | 9 | 9 |
| 8 | A {NEOLOGISM} thing to it | `thing to it` | `a {SLOT} NOUN to PRON` | 1.00 | 9 | 8 |
| 9 | It was very {NEOLOGISM} to do | `to do` | `very {SLOT} to VERB` | 0.91 | 10 | 9 |
| 10 | The other and {NEOLOGISM} and | `and` | `ADJ and {SLOT} and` | 1.00 | 8 | 8 |
| 11 | It is {NEOLOGISM} same | `same` | `is {SLOT} , ADJ` | 1.00 | 8 | 8 |
| 12 | They have seen the {NEOLOGISM} thing is that | `thing is that` | `the {SLOT} NOUN is that` | 1.00 | 8 | 7 |
| 13 | They are {NEOLOGISM} and similar | `and similar` | `are {SLOT} and ADJ` | 1.00 | 7 | 7 |
| 14 | It was so {NEOLOGISM} that | `that` | `so {SLOT} that` | 1.00 | 7 | 7 |
| 15 | It was too {NEOLOGISM} to | `to` | `too {SLOT} to` | 1.00 | 7 | 6 |
| 16 | It was a very {NEOLOGISM} same thing | `same thing` | `very {SLOT} ADJ NOUN` | 1.00 | 7 | 6 |
| 17 | It was, the {NEOLOGISM} one of | `one of` | `, the {SLOT} NOUN of` | 0.91 | 11 | 9 |
| 18 | They are very {NEOLOGISM} and | `and` | `are very {SLOT} and` | 1.00 | 6 | 5 |
| 19 | A {NEOLOGISM} thing and | `thing and` | `a {SLOT} NOUN , and` | 0.90 | 10 | 8 |
| 20 | It was, {NEOLOGISM} and other | `and other` | `, {SLOT} and ADJ .` | 1.00 | 5 | 5 |
| 21 | It is very {NEOLOGISM} and | `and` | `is very {SLOT} and` | 1.00 | 5 | 5 |
| 22 | It was a more {NEOLOGISM} thing | `thing` | `more {SLOT} NOUN . </s>` | 1.00 | 5 | 5 |
| 23 | They know how {NEOLOGISM} it is | `it is` | `how {SLOT} PRON is` | 1.00 | 5 | 5 |
| 24 | They were {NEOLOGISM} and similar | `and similar` | `were {SLOT} and ADJ` | 1.00 | 5 | 5 |
| 25 | The {NEOLOGISM} thing to | `thing to` | `<s> the {SLOT} NOUN to` | 1.00 | 5 | 4 |
| 26 | They have seen a very {NEOLOGISM} similar | `similar` | `a very {SLOT} ADJ` | 1.00 | 5 | 4 |
| 27 | Some {NEOLOGISM} one thing | `one thing` | `some {SLOT} NOUN NOUN` | 1.00 | 5 | 4 |
| 28 | It was very {NEOLOGISM} and | `and` | `was very {SLOT} and` | 1.00 | 5 | 4 |
| 29 | They be a {NEOLOGISM} thing for | `thing for` | `be a {SLOT} NOUN for` | 1.00 | 5 | 4 |
