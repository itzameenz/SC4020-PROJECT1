# The Whole Project in Plain English

A no-jargon walkthrough you can read top-to-bottom and then explain to your group.
Every calculation has a tiny worked example.

---

## 0. The one-sentence goal

> A traveller has booked a few cities in a trip, in order. **Guess the next city
> they'll book.**

Everything in this project is about doing that guess well, and comparing different
ways of guessing.

Think of it like autocomplete: you've typed "Paris → Rome → ..." and we predict
the next word — except the "words" are cities and the "sentence" is a trip.

---

## 1. What the data looks like

Each **trip** is a list of cities a person booked, in order. Example:

```
Trip 123:  Paris → Rome → Venice → (???)
```

The dataset gives us three files:
- **train_set.csv** — full trips we learn from (we can see all the cities).
- **test_set.csv** — trips where the **last city is hidden** (blanked out). This is
  what we must guess.
- **ground_truth.csv** — the answer key: the real last city for each test trip.

So we *learn* on full trips, then we're *tested* by hiding the last city and asking
"what was it?", and we check our guess against the answer key. That's called
**leave-last-out** (leave the last one out and predict it).

---

## 2. Preprocessing = cleaning the data so every model gets the same tidy version

Raw data is messy. Preprocessing fixes it. In simple terms, we:

1. **Put each trip in date order.** Rows come shuffled; we sort by check-in date so
   the sequence is correct.
   - `Rome, Paris, Venice` (random) → `Paris, Rome, Venice` (by date).

2. **Remove accidental repeats.** If someone booked the same city twice in a row,
   that's just a longer stay, not a new destination. We squash it.
   - `Paris, Rome, Rome, Venice` → `Paris, Rome, Venice`.

3. **Throw away trips that are too short.** A 1-city trip has nothing to predict. We
   keep trips with at least 4 cities (to match the test set).

4. **Turn city names into numbers.** Computers work with numbers, not words. So every
   city gets an ID number.
   - `Paris = 2, Rome = 3, Venice = 4`. (We save `0` and `1` for "empty" and
     "unknown city" — that's why real cities start at 2.)

5. **Add a few extra clues** each model *may* use: the month of travel, how many
   nights they stayed, how far into the trip they are, and a "persona" (their country
   + device, e.g. "Germany + mobile").

6. **Make a practice set (validation).** We hide the last city of some *training*
   trips to use as a practice exam — so we can tune the models **without peeking at
   the real test**. (Peeking at the test would be cheating.)

After this, every trip is just a clean list of numbers, e.g. `[2, 3, 4]`, plus its
extra clues. **All five models use this exact same tidy data** — that's what makes
the comparison fair.

---

## 3. The models = five different ways to guess the next city

They go from dumb to smart. Each one adds one new ability.

### Model 0 — Popularity (the dumb floor)
Ignores the trip completely. Just guesses the **most popular cities overall**, the
same guess for everyone.
- If Paris is the most-booked city, it guesses Paris for *every* trip.
- It's meant to be beaten — it tells us "you must do better than this."

### Model 1 — Markov (your teammate's)
Looks only at the **last city** and asks: *"In training, after this city, what city
usually came next?"*
- Worked example: in training, after **Rome** we saw **Venice** 50 times, **Milan**
  30 times, **Naples** 10 times. So given a trip ending in Rome, it ranks:
  `Venice, Milan, Naples, ...`
- Smart about the last step, but blind to anything earlier and to context.

### Model 2 — FPMC (YOURS) = "who you are" + "where you just were"
FPMC adds **personalization** to Markov. It gives every city and every traveller a
short **list of numbers** (called an "embedding") that captures their "flavour". It
scores a candidate next-city with two dot products added together.

**What's a dot product?** Multiply matching numbers in two lists and add them up. It
measures "how much two lists agree". Example:
```
list A = [1, 0]
list B = [0.9, 0.1]
dot = (1×0.9) + (0×0.1) = 0.9        # high = they match
```

**FPMC's score** for "given traveller u, last city = Rome, is Venice next?":
```
score = (traveller's list  ·  Venice's list)      # personalization: does this person like Venice?
      + (Rome's list       ·  Venice's list)      # transition: does Venice usually follow Rome?
```
Tiny numbers example:
```
traveller · Venice = 0.9
Rome      · Venice = 0.62
score(Venice) = 0.9 + 0.62 = 1.52
```
We compute this score for **every** city and rank them from highest to lowest. The
highest-scoring cities are our top guesses.

**How does it learn those lists of numbers?** By practice, using a rule called
**BPR**: take a real example (after Rome the person really went to Venice) and a
random wrong city (say Tokyo). Nudge the numbers so that
`score(Venice) > score(Tokyo)`. Repeat millions of times. Slowly the numbers arrange
themselves so real next-cities score higher than random ones.

**Your ablation (an experiment):** we tried the "who you are" part two ways — using
the exact person (`user_id`) vs a **persona** (their country + device). Since most
people appear only once, the exact-person version is useless at test time (we've
never seen them). The persona ("Germany + mobile") repeats across many people, so it
generalises better. **Result: persona won** — a nice finding.

### Model 3 — GRU4Rec (your teammate's)
A neural network that reads the whole trip one city at a time and keeps a "memory"
that it updates at each step. Can remember things from earlier in the trip, not just
the last city.

### Model 4 — SASRec (YOURS) = "pay attention to the steps that matter"
SASRec reads the whole trip and, for each guess, **decides which earlier cities
matter most** — that mechanism is called **attention**.
- Analogy: to guess the next city after `Paris → Rome → Venice`, it might "pay 70%
  attention to Venice, 20% to Rome, 10% to Paris", then blend them to make its guess.
- It learns those attention weights automatically from data.
- It also gives each city a list of numbers (embedding), and at the end it scores
  every candidate city with a dot product — same ranking idea as FPMC.
- Trained the same way (BPR: real next-city should score higher than a random city).

**Your ablation:** we tried adding extra clues (device, country, booking channel) to
SASRec, in two ways (adding them vs gluing them on). **Result: the extra clues
actually hurt** — for this data they're noise, so plain SASRec was best. That's a
real, reportable finding too.

---

## 4. Scoring = how we grade a guess

Every model outputs a **ranked list of cities** (best guess first), e.g.:
```
Model's guesses:  1.Milan  2.Venice  3.Rome  4.Naples  ...
The real answer:  Venice
```
The real answer **Venice is at position 2** in the list. From that one number
(position = 2) we compute three grades:

### HR@k — "Hit Rate" — *did we get it in the top k?* (yes/no)
- **HR@1** = is it #1? Venice is #2, so **0** (no).
- **HR@5** = is it in the top 5? Yes → **1**.
- **HR@10** = top 10? Yes → **1**.

### NDCG@k — *how HIGH did it rank?* (higher = better)
Same as HR, but rewards being nearer the top. Formula for one answer at position `r`:
```
NDCG = 1 / log2(r + 1)          (0 if it's not in the top k)
```
- Venice at position 2 → `1 / log2(3) = 1 / 1.585 = 0.63`.
- If it had been #1 → `1 / log2(2) = 1.0` (perfect). Lower position → smaller score.

### MRR — "Mean Reciprocal Rank" — *1 divided by the position*
```
MRR contribution = 1 / r
```
- Venice at position 2 → `1 / 2 = 0.5`. (Position 1 → 1.0, position 4 → 0.25.)

### Then we **average** each grade over all 70,662 test trips.
So "NDCG@10 = 0.35" means: on average, across every test trip, the true city landed
fairly near the top of the list.

**Why three grades?** They answer slightly different questions:
- HR = *did we catch it at all* (in the top k)?
- NDCG = *how high up* was it?
- MRR = *on average, how good was its position*?
Together they give an honest picture instead of one cherry-picked number.

**We rank against ALL ~40,000 cities** (not a shortlist), which is the honest, hard
way. And a `@k` like `@10` just means "look at the top 10 guesses".

---

## 5. Why we score everyone the same way

Different models guess differently inside, but they **all output the same thing: a
ranked list of cities**. One grading function (`evaluate.py`) grades them all
identically. It's like a race: runners train differently, but they all run the same
track and one stopwatch times everyone. If each used their own stopwatch, you
couldn't compare times.

**Sanity check we did:** the dumb Popularity model should score exactly what the
official Booking.com repo says (~0.053 on their metric). Ours scored **0.0527** —
basically identical — which proves our grading code is correct and our other numbers
can be trusted.

---

## 6. What the results say (in plain words)

| Model | NDCG@10 (higher = better) | In words |
|---|---|---|
| Popularity | 0.05 | the dumb floor |
| FPMC-persona | 0.35 | ~7× better than the floor |
| FPMC-user | 0.345 | slightly worse (exact-person doesn't generalise) |
| SASRec (plain) | 0.35 | tied best; also best at HR@10 |
| SASRec + extra clues | 0.29–0.33 | worse — the clues were noise |

**The story to tell your friends:**
1. Knowing the trip so far is *hugely* better than guessing popular cities (~7×).
2. For "who you are", a broad **persona beats the exact person**, because most people
   are seen only once.
3. The fancy transformer (SASRec) is about **tied** with the simpler FPMC here,
   because these trips are short and the *last city* already tells you most of the
   answer. Fancy isn't always better — a good finding.
4. Extra context clues **hurt** SASRec here — more features isn't always better.
5. Everyone is good at popular cities and bad at rare ones (the "cold-tail" problem).

---

## 7. 30-second version (for explaining out loud)

> "We predict the next city in a trip. First we clean the data into tidy numbered
> sequences that every model shares. Then five models guess, from dumb (just
> popular cities) to smart (a transformer that pays attention to the trip). Each
> model outputs a ranked list of cities; we grade all of them the same way with
> three scores — did we get it (HR), how high (NDCG), and average position (MRR) —
> averaged over 70k test trips. Our two models, FPMC and SASRec, both beat the dumb
> baseline about 7×, they're roughly tied, and we found that a broad 'persona' beats
> the exact user and that extra context features didn't help."

That's the whole project. 🎯
