# Chapter 32 - How Much Data, and How Often to Retrain

Your model is live. It's extracting memories from real conversations, and most of the time it's good. But "the world" keeps moving: people start talking about new topics, your product picks up a new feature, a class of conversations your model fumbles keeps showing up in the logs. You have a trickle of fresh, corrected examples accumulating in a folder. The question this chapter answers is the deceptively simple one every team hits a few weeks after launch: *I have some new data. Is it enough to retrain yet? And how often should I be doing this at all?*

This is a budgeting chapter, not a training chapter. You already know *how* to train (Parts 3-4) and *how* to tell whether a run worked (Ch17 - Watching Training, Ch18 - Did It Actually Work). What you don't yet have is a feel for the *quantities and rhythm* of running a model as a living system: how big each new batch should be, how much old data to keep mixing in so the model doesn't forget what it knew, and what should trigger a retrain at all. We'll give those as defensible ranges — never a single magic number — with the reasoning behind each, plus a runnable tool for the piece that's easy to get wrong: building a replay-mixed training set.

---

## What you'll learn

- Why **tokens**, not row counts, are the truer measure of "how much data" — and the rough token budget for a narrow LoRA fine-tune
- How dataset size scales for a ~4B model on a narrow task, and *why* you hit diminishing returns sooner than you'd expect
- How many **new** rows justify a retrain in the ongoing loop (and why tiny increments aren't worth the trouble)
- **Replay mixing**: how much old/general data to fold in with new data so the model doesn't forget — with a runnable mixer
- **Cadence**: event-driven vs scheduled retraining, and a concrete decision rule, always gated by eval
- How many epochs and what learning rate to use for an *incremental* round (they differ from your first run)
- A worked multi-round budget so you can see the whole loop add up

---

## Concepts you need first

### This chapter is about the *ongoing loop*, not your first dataset

A clear boundary, because it's easy to blur: **Ch13 - Creating Your Training Data with Synthetic Generation** is about building your *first* dataset from scratch — seeding topics, calling a teacher model, filtering, and sizing that initial batch (Ch13's rule of thumb: start at ~500 rows, scale toward a few thousand). That's the cold start.

**This chapter starts the day after your model ships.** From here on, data arrives in *increments* — small batches of new or corrected examples — and the central tension is no longer "how do I get data" but "how do I add a little without breaking what already works." Everything below assumes you already have a trained model and a baseline dataset behind it. When we talk about "how much," we mean *how much new data per round* and *how much old data to replay alongside it* — not how to size the initial corpus. For that, see Ch13.

### Catastrophic forgetting — the one-paragraph version

Imagine you've trained a new hire for months on your company's procedures, and they're great. Then you send them on a two-week intensive course about one narrow new procedure — and they come back having *forgotten* half of what they used to do well, because all they practiced for two weeks was the new thing. Neural networks do exactly this. Train a model hard on only the newest batch of data and it will drift toward that batch and quietly lose competence on everything else. The name for this is **catastrophic forgetting**, and the cheap, reliable defense — mixing some old data back in — is called **replay**. We use replay heavily in this chapter; the full mechanism and the diagnostics for detecting forgetting get their own treatment in **Ch33 - Catastrophic Forgetting Over Many Rounds**.

### A "round," a "canary," and an "increment"

Three words used throughout:

- **Round** — one complete retrain-and-evaluate cycle: gather new data, mix in replay, train, evaluate, decide whether to ship.
- **Increment** — the batch of *new* rows you've collected since the last round.
- **Canary eval set** — a small, *frozen* set of held-out examples you never train on and never change, run after every round to catch regressions. (You built held-out splits in Ch14; the canary is that idea applied to continual learning. Think of it as a smoke alarm: it doesn't tell you the model is good, it tells you if the model just got *worse*.)

---

## Rows are a proxy; tokens are the truth

When people ask "how much data do I need," they almost always mean *rows* — number of (conversation → memories) examples. Rows are easy to count, so we use them as the working unit. But rows are only a stand-in for the thing the model actually consumes during training: **tokens**.

Here's the intuition. Training doesn't see "examples"; it sees a stream of tokens and learns to predict each next one. A row with a 14-message conversation and eight extracted memories carries several times more tokens — and several times more *learning signal* — than a three-message exchange with one memory. Two datasets with the same row count can differ 3-4x in tokens. Size purely by rows and you're measuring with a ruler whose length changes depending on what you put on it.

For our pinned memory-extraction example, a single training row in TRL's conversational format looks like this:

```python
# One training row — the pinned schema and system prompt from the charter.
{
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},          # the pinned system prompt
        {"role": "user",   "content": "A: ...\nB: ..."},        # the conversation
        {"role": "assistant", "content":                        # the gold memories, as a JSON array
            '[{"text": "Sarah prefers dark roast coffee in the morning", '
            '"type": "preference", "entities": ["Sarah"]}]'},
    ]
}
```

You can measure tokens directly with the same tokenizer you train with — never guess:

```python
# count_tokens.py
# Measure the REAL token budget of a JSONL dataset, using the model's own tokenizer.
# This is the number that actually matters for "how much data," not the row count.
import json
from transformers import AutoTokenizer

# Use the tokenizer of whatever base model you're fine-tuning (see Ch10/Ch16).
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

def count_dataset_tokens(path: str) -> dict:
    rows = 0
    total_tokens = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # apply_chat_template renders the messages exactly as training will see them,
            # so the token count reflects reality (system + user + assistant).
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            total_tokens += len(tokenizer(text)["input_ids"])
            rows += 1
    return {
        "rows": rows,
        "tokens": total_tokens,
        "avg_tokens_per_row": round(total_tokens / max(rows, 1), 1),
    }

if __name__ == "__main__":
    stats = count_dataset_tokens("data/memories_train.jsonl")
    print(stats)
    # Example output:
    # {'rows': 1500, 'tokens': 1180000, 'avg_tokens_per_row': 786.7}
```

So what's the budget? For a **narrow** structured task like memory extraction, trained with LoRA/QLoRA, **roughly 1-5 million training tokens is ample.** Below ~1M you may not have enough signal for the model to settle into the format and the type distinctions; above ~5M, for a task this constrained, you're mostly paying for compute without buying accuracy. With ~600-900 tokens per memory-extraction row (typical for our conversational rows), 1-5M tokens lands you at roughly **1,500-6,000 rows** — which, reassuringly, is the same neighborhood as the row-based table below. The reason to track tokens anyway: if your conversations get much longer (say you move to full meeting transcripts), the *same* row count could blow past 5M tokens, and you'd want to notice that.

Treat the token figure as a sanity check on your row count, not a separate target. If rows and tokens disagree wildly, trust tokens and find out why your rows are unusually long or short.

---

## How dataset size scales for a ~4B model on a narrow task

Here's the picture you should hold in your head for a ~4B model on a narrow structured-extraction task with LoRA. These are the charter's ranges, and they hold for the *cumulative* dataset — everything the current model was trained on, not just the latest increment:

| Stage | Rows (cumulative) | What you get, and why |
|---|---|---|
| **Proof of life** | 200-500 | The model learns the *format* — valid JSON, the four types, the entities field. Accuracy is shaky but it stops emitting garbage. This is the lower bound from the Ch0 speedrun. |
| **Solid baseline** | 1,000-3,000 | Where most projects live. The model handles common conversation shapes well. Adding data here still clearly helps. |
| **Strong** | 3,000-10,000 | Good coverage of edge cases. Past the lower end of this band, **diversity and quality matter more than raw count** — another 1,000 near-duplicate rows barely move the needle. |
| **Diminishing returns** | >~10,000 | For a task this narrow, extra rows mostly buy noise. Spend the effort on better eval and on targeted data for specific failures instead. |

### Why diminishing returns kick in *early* on a narrow task

The intuition that trips people up: surely more data is always better? Not on a narrow task. Memory extraction has a small "concept space" — four memory types, a fixed schema, a bounded set of conversational patterns. Once the model has seen enough examples to cover that space, additional examples are mostly *restating things it already knows*. Picture teaching someone to sort mail into four bins. After a few hundred letters they've got it; the ten-thousandth letter teaches them nothing new. Contrast that with open-ended tasks like creative writing, where the concept space is effectively unbounded and more data keeps helping for far longer.

This is why **quality and diversity beat raw count** past the baseline. Ten new conversations covering a *failure mode your model currently gets wrong* are worth more than a thousand more of what it already handles. The continual-learning loop is therefore not "shovel in more data" — it's "find what's broken, add a little targeted data for it, retrain carefully." That reframing is the whole point of Part 8.

---

## Per-round increments: how much new data justifies a retrain?

Now the ongoing-loop question. You've got a folder filling up with new corrected examples. When is it worth doing a round?

**Rule of thumb: a few hundred to ~1,000 high-quality new rows per round.** The intuition is about signal versus cost. A retrain has fixed overhead — your time, GPU minutes, the eval pass, the risk of regression. If your increment is 30 rows against a 2,000-row baseline, you're spending all that overhead to nudge ~1.5% of the data. The model will barely change, and you won't be able to tell from eval whether it changed for better or worse — the difference is lost in the noise.

Why not wait for a *huge* increment instead? Because a giant batch of new-distribution data, dropped in all at once, is exactly what triggers forgetting (the new-hire-on-a-two-week-course problem). Smaller, more frequent rounds with replay drift the model gently. There's a sweet spot:

- **< ~100 new rows:** usually not worth a round on its own. Let it accumulate. (Exception below.)
- **~few hundred to ~1,000 new rows:** the sweet spot — enough to measurably shift behavior, small enough to mix safely with replay.
- **Much more than ~1,000 at once:** still fine, but lean harder on replay and treat it almost like a fresh baseline run.

The one exception to "let small batches accumulate": if the new rows are **corrections for a specific, important failure** you're seeing in production (e.g., the model keeps mislabeling `decision` as `fact` for a certain phrasing), even ~50-100 sharply targeted rows can be worth an immediate round. Targeted quality, again, beats volume.

---

## Replay mixing: keep the old so you don't lose it

This is the single most important number in the chapter, and the cheapest insurance you'll ever buy.

The analogy from "Concepts you need first": train only on the new batch and the model forgets its old competence, like a worker who practiced nothing but the new procedure for two weeks. **Replay** is the fix: when you train a round, you don't train on *only* the new rows — you blend in a sample of the model's prior and general data so it keeps rehearsing what it already knew while it learns the new thing.

**The rule of thumb: mix ~10-30% prior/general data into each round's training set, with ~20% as a sensible default.** So a round with 800 new rows would pull in roughly 200 prior rows, for a ~1,000-row training set that's 80% new / 20% replay.

Why does even a *little* replay help so much? The striking, well-replicated finding is that the relationship is sharply non-linear: going from 0% replay to even 1-10% replay dramatically reduces forgetting, while going from 20% to 40% buys comparatively little extra protection. A small, regular dose of "remember the old stuff" is enough to keep the old behavior anchored — you don't need to re-train on the whole history every time. (The *mechanism* for why a small fraction does so much, and how to measure forgetting directly, is Ch33's job; here we just use the result.)

The tradeoff in the 10-30% band:

- **Toward 10%:** more of each round is spent learning the new thing, so the model adapts faster — but with thinner protection against drift. Use the low end when the new data is close to your existing distribution.
- **Toward 30%:** stronger anchoring against forgetting, but the new data has to share the round with more old data, so adaptation is slower. Use the high end when the new data is a noticeably different distribution, or when your canary eval has flagged regressions before.
- **~20% default:** good balance for most rounds. Start here; adjust based on what your canary eval tells you.

### Runnable: build a replay-mixed training set

Here's the tool. Give it your new increment and your pool of prior data, name a replay ratio, and it writes a shuffled, mixed JSONL ready to train on.

```python
# build_replay_mix.py
# Build ONE round's training set: new increment + a sampled fraction of prior data.
#
# Usage:
#   python build_replay_mix.py \
#       --new      data/round_07_new.jsonl \
#       --prior    data/memories_history.jsonl \
#       --out      data/round_07_train.jsonl \
#       --ratio    0.20        # fraction of the FINAL set that should be replay (default 0.20)
#
# "ratio" is the share of replay rows in the final mix. ratio=0.20 -> 80% new / 20% prior.
import argparse
import json
import random

def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def build_replay_mix(new_rows: list[dict],
                     prior_rows: list[dict],
                     ratio: float,
                     seed: int = 13) -> list[dict]:
    """
    Returns a shuffled training set where `ratio` of the rows are sampled from
    prior data (replay) and the rest are the new increment.

    We keep ALL new rows (that's the point of the round) and solve for how many
    replay rows make them `1 - ratio` of the total:
        n_new = (1 - ratio) * total   ->   total = n_new / (1 - ratio)
        n_replay = total - n_new
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0.0, 1.0). Charter guidance: 0.10-0.30.")
    rng = random.Random(seed)

    n_new = len(new_rows)
    # How many replay rows we WANT to hit the target ratio.
    target_total = n_new / (1.0 - ratio) if ratio > 0 else n_new
    n_replay_wanted = round(target_total - n_new)

    # We can't sample more prior rows than we have. If the pool is small,
    # sample with replacement so the ratio still holds (rehearsal tolerates repeats).
    if n_replay_wanted <= len(prior_rows):
        replay = rng.sample(prior_rows, n_replay_wanted)
    else:
        replay = [rng.choice(prior_rows) for _ in range(n_replay_wanted)] if prior_rows else []
        print(f"  note: prior pool ({len(prior_rows)}) smaller than requested "
              f"replay ({n_replay_wanted}); sampling with replacement.")

    mixed = new_rows + replay
    rng.shuffle(mixed)  # never train new-then-old in order; interleave them
    return mixed

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--new",   required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--out",   required=True)
    p.add_argument("--ratio", type=float, default=0.20)
    args = p.parse_args()

    new_rows   = read_jsonl(args.new)
    prior_rows = read_jsonl(args.prior)
    mixed      = build_replay_mix(new_rows, prior_rows, args.ratio)
    write_jsonl(mixed, args.out)

    actual_ratio = (len(mixed) - len(new_rows)) / max(len(mixed), 1)
    print(f"new rows:    {len(new_rows)}")
    print(f"replay rows: {len(mixed) - len(new_rows)}")
    print(f"total:       {len(mixed)}  (replay = {actual_ratio:.0%})")
    print(f"wrote -> {args.out}")
```

Running it on an 800-row increment with a 20% target:

```
$ python build_replay_mix.py --new data/round_07_new.jsonl \
      --prior data/memories_history.jsonl --out data/round_07_train.jsonl --ratio 0.20
new rows:    800
replay rows: 200
total:       1000  (replay = 20%)
wrote -> data/round_07_train.jsonl
```

A few practical notes baked into the code:

- **It keeps every new row** and solves for the number of replay rows, so your increment is never diluted away — replay rides *alongside* the new data.
- **It shuffles.** Never feed the trainer all-new-then-all-old; interleaving keeps each batch a healthy mix so the model rehearses old and new together.
- **The prior pool should be representative** — sample it from your *whole* history (and any general examples you want to preserve), not just the last round. A good habit is to keep one growing `data/memories_history.jsonl` that every shipped round appends to, and draw replay from that.

---

## Cadence: when to actually pull the trigger

You have two honest options for *when* to run a round, and the right answer is usually a blend.

**Scheduled cadence** — retrain on a fixed clock: weekly, biweekly, or monthly. Predictable, fits a team's rhythm, keeps the model from going stale. The vice: the clock doesn't know whether you have anything worth training on, so a scheduled retrain on 40 new rows is wasted overhead.

**Event-driven cadence** — retrain when something *happens*:
1. **Enough new data has accumulated** — you've crossed the increment threshold (a few hundred to ~1,000 high-quality new rows).
2. **A quality drop is detected** — production monitoring or the canary eval shows accuracy slipping, or a specific failure mode keeps recurring in the logs.

Event-driven is more efficient (you only pay when there's a reason) but requires you to actually be watching — without monitoring in place, "events" silently never fire.

**The practical rule most teams should use is a hybrid:** an event-driven trigger with a scheduled backstop. In plain English:

> Retrain when you have **≥ ~few hundred high-quality new rows** *or* the canary eval shows a **regression** — but **never** more often than makes sense given your overhead, and run a round **at least once a month** even if neither fires, just to fold in whatever has trickled in. **And no round ships unless it passes the canary eval.** That last clause is non-negotiable.

Here's that rule as a function you can run against the state of your data folder:

```python
# should_retrain.py
# Decide whether to run a retraining round. Event-driven with a scheduled backstop,
# always gated by eval at SHIP time (this function decides whether to TRAIN; you
# still gate the ship on the canary eval afterwards — see Ch17/Ch18).
from datetime import date, timedelta

def should_retrain(new_row_count: int,
                   new_rows_are_high_quality: bool,
                   canary_regressed: bool,
                   days_since_last_round: int,
                   min_increment: int = 300,        # lower end of "few hundred"
                   targeted_fix_count: int = 0,     # sharply-targeted correction rows
                   max_days: int = 30) -> tuple[bool, str]:
    """Returns (decision, human-readable reason)."""
    # 1) Quality regression is the most urgent trigger.
    if canary_regressed:
        return True, "Canary eval regressed — retrain to recover lost behavior."
    # 2) A small batch of sharply targeted corrections is worth an immediate round.
    if targeted_fix_count >= 50:
        return True, f"{targeted_fix_count} targeted correction rows for a known failure."
    # 3) Enough high-quality new data accumulated.
    if new_rows_are_high_quality and new_row_count >= min_increment:
        return True, f"{new_row_count} new high-quality rows >= {min_increment} threshold."
    # 4) Scheduled backstop so the model never goes stale.
    if days_since_last_round >= max_days and new_row_count > 0:
        return True, f"{days_since_last_round} days since last round (>= {max_days}) — backstop."
    # Otherwise: wait.
    return False, (f"Hold: {new_row_count} new rows, {days_since_last_round} days elapsed. "
                   f"Let data accumulate.")

if __name__ == "__main__":
    print(should_retrain(820, True,  False, 9))   # -> (True, "820 new high-quality rows ...")
    print(should_retrain(40,  True,  False, 9))   # -> (False, "Hold: 40 new rows ...")
    print(should_retrain(40,  True,  False, 33))  # -> (True, "33 days since last round ...")
    print(should_retrain(0,   True,  True,  3))   # -> (True, "Canary eval regressed ...")
```

Whatever fires the round, the gate at the *end* is the same: train, then run the frozen canary eval (and your full eval from Ch18). If the new model isn't at least as good as the shipped one on the canary, **you don't ship it** — you keep the old model, investigate (more replay? lower LR? bad new data?), and try again. Eval is the brake; the cadence is just the gas pedal.

---

## Epochs and learning rate for an incremental round

A round is not a from-scratch run, and treating it like one is the most common way to wreck a working model. Two settings change.

### Epochs per round

The charter's guidance, which carries over directly: **2-3 epochs for small datasets (≤ ~2k rows), 1-2 epochs for larger ones — and stop on the eval-loss plateau regardless.** Since most rounds are small (a few hundred to ~1,000 rows after mixing), you're usually in the 2-3 range. The reason fewer epochs suit larger sets: with more rows the model sees enough variety in a single pass, so extra passes mostly invite memorization. Don't treat the epoch number as a target to hit — it's a ceiling. Wire up `EarlyStoppingCallback` from **Ch17 - Watching Training** and let the eval-loss plateau end the round early; that's the real stop signal.

### Learning rate: go gentler than your first run

The intuition: your first training run started from a base model that knew nothing about your task, so you used a bigger learning rate (around `2e-4` for LoRA, as in Ch16/Ch17) to move it a long way fast. An incremental round starts from a model that's *already good*. A big learning rate now is a sledgehammer — it can clobber the carefully-learned weights and undo prior competence, which is forgetting by another name. So **use a lower learning rate for incremental rounds — commonly about half to a quarter of your initial rate** (e.g., `5e-5` to `1e-4` if you trained at `2e-4`). Smaller steps nudge the model toward the new data without bulldozing what's already there. Pair the lower LR with a short warmup, exactly as in your first run.

```python
# round_training_args.py
# The deltas from your first-run SFTConfig (Ch16/Ch17) for an INCREMENTAL round.
# Everything else (output_dir, eval/save strategy, load_best_model_at_end, the
# pinned system prompt) stays as in Ch17.
from trl import SFTConfig

round_args = SFTConfig(
    output_dir="./memory-extractor-round-07",
    # --- gentler than the first run, to avoid clobbering prior competence ---
    learning_rate=7e-5,         # ~1/3 of the initial 2e-4 (range: 5e-5 - 1e-4)
    num_train_epochs=3,         # ceiling for a small round; EarlyStopping ends it sooner
    warmup_ratio=0.05,
    # --- monitoring carried over from Ch17 so the plateau is the real stop signal ---
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    bf16=True,                  # or fp16=True on older GPUs
)
```

One design choice worth a sentence: most teams retrain each round **from the base model** on the cumulative-plus-replay data rather than continuing to train the previous adapter on top of itself round after round. Continuing on top compounds drift and is harder to reason about; restarting from base with a good replay mix gives you a clean, reproducible model each round. Ch33 weighs this tradeoff in more depth.

---

## A worked budget over several rounds

Let's put it together. Suppose you shipped an initial model trained on a 2,000-row baseline (built per Ch13), and you keep one growing `memories_history.jsonl`. Here's a plausible four-round quarter, using a 20% replay default and the gentle-LR settings above.

| Round | Trigger | New rows | Replay (20%) | Round train set | Tokens (~) | Epochs | Notes |
|---|---|---|---|---|---|---|---|
| Baseline | initial ship | 2,000 | — | 2,000 | ~1.5M | 3 | The Ch13 cold start. |
| 1 | 700 new rows ≥ threshold | 700 | 175 | 875 | ~0.65M | 3 | New product feature added vocabulary. |
| 2 | canary regressed on `decision` type | 120 targeted | 30 | 150 | ~0.11M | 3 | Targeted fix; small but urgent. Bump replay to 25% next time if it recurs. |
| 3 | monthly backstop | 250 | 62 | 312 | ~0.23M | 3 | Light month; backstop fired so trickle gets folded in. |
| 4 | 900 new rows ≥ threshold | 900 | 225 | 1,125 | ~0.85M | 2-3 | Larger round; eval-loss plateaus at epoch 2, EarlyStopping ends it. |

What this illustrates:

- **Cumulative grows steadily; rounds stay small.** Total unique data after the quarter is ~3,970 rows — squarely in the "strong" band — but no single round trained on more than ~1,125 rows. You never re-trained on the whole history; replay carried the old competence forward.
- **Per-round token budgets sit well under the 5M ceiling.** Each round is cheap — minutes of GPU time and a few dollars, comparable to the speedrun's ~$5-30 all-in figure from Ch0. The continual loop is the inexpensive part; the cold start (Ch13's teacher-API generation) was the bigger spend.
- **Replay rode along every round** — even round 2's tiny 120-row targeted fix carried 30 replay rows, cheap insurance against the fix nudging the model off its other behaviors.
- **Eval gated every ship.** Round 2 happened *because* the canary caught a regression, and every round ended by re-running the canary before shipping. The cadence decided when to train; eval decided what to keep.

---

## A pinned-schema reminder

Every row, every round, uses the same memory objects you've trained on since Ch12, in TRL's conversational format, with the charter's system prompt reused verbatim at training *and* inference time. Increment files and the replay pool are all JSONL of that one shape:

```python
# Each memory object: {"text": <standalone sentence>,
#                      "type": <preference|fact|decision|relationship>,
#                      "entities": [<named people/places/things>]}
# The assistant turn is a JSON array of such objects; an empty array [] is valid.
{"messages": [
    {"role": "system", "content": SYSTEM_PROMPT},          # pinned, verbatim
    {"role": "user", "content": "<the conversation>"},
    {"role": "assistant",
     "content": '[{"text": "...", "type": "preference", "entities": ["..."]}]'},
]}
```

If an increment ever uses a different schema or a tweaked system prompt, that's not a continual-learning round — it's a new task, and you re-baseline.

---

## Recap

- **Tokens, not rows, are the truer measure** of how much data; for narrow LoRA, **~1-5M training tokens** is ample. Use row counts as the working unit but sanity-check against tokens.
- For a ~4B model on a narrow task: **200-500** proves life, **1k-3k** is a solid baseline, **3k-10k** is strong, **>~10k** hits diminishing returns. The concept space is small, so **quality and diversity beat raw count** well before 10k.
- **Per round, a few hundred to ~1,000 high-quality new rows** is the sweet spot; tiny increments aren't worth the overhead, except for sharply targeted fixes.
- **Replay 10-30% prior/general data (default ~20%)** into every round. Even 1-10% sharply cuts forgetting; the mechanism is Ch33's. The runnable mixer keeps every new row and solves for the replay count.
- **Cadence is event-driven (enough data, or a detected quality drop) with a scheduled backstop**, and **every round is gated by the frozen canary eval** before it ships.
- **Incremental rounds use 2-3 epochs (1-2 for larger sets), stop on the eval-loss plateau, and a lower learning rate (~½ to ¼ of the first run)** to avoid clobbering prior competence.

## Next

**Ch33 - Catastrophic Forgetting Over Many Rounds** — the mechanism behind catastrophic forgetting, how to *measure* it with a frozen canary set, and why a small replay fraction does so much of the work.
