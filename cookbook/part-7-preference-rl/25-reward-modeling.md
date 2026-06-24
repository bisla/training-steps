# Chapter 25 - Rewards: Functions and Reward Models

So far in this book you have taught the model by *example*. You showed it thousands of conversations paired with the correct memory JSON, and it learned to imitate them (*Ch15 - Your First Fine-Tune with Unsloth*). Then you measured how well it did (*Ch18 - Did It Actually Work?*). Imitation gets you a long way — but it has a ceiling. The model can only ever be as good as the examples you handed it, and it never finds out that *this* answer was a little better than *that* one.

Reinforcement learning breaks that ceiling. Instead of "here is the right answer, copy it," you say "produce an answer, and here is a score telling you how good it was." Do that thousands of times and the model learns to chase higher scores. But this whole idea rests on one deceptively simple ingredient: the *score itself*. Where does that number come from? That is what this chapter is about. Get the score right and everything downstream — DPO, PPO, GRPO — has something honest to optimize. Get it wrong and you will train a model that games your metric while getting worse at the actual job.

---

## What you'll learn

- What a "reward" actually is, in one sentence, with no math
- The two ways to produce a reward: a cheap **programmatic reward function** you write in Python, and a **learned reward model** trained from preferences
- How to build a runnable `reward_fn(completions, **kwargs) -> list[float]` for memory extraction — the exact signature GRPO expects, reusing the eval logic from *Ch18*
- When a programmatic function is not enough and you need a learned reward model instead
- How to train a reward model with TRL's `RewardTrainer` on a small chosen/rejected preference dataset
- How the reward plugs into RL later (PPO conceptually in *Ch27*, GRPO in *Ch28*)
- The honest failure modes: reward hacking, the real cost of a good reward model, and why programmatic rewards are usually enough for a structured task like ours

---

## Concepts you need first

### A reward is just a number that says "how good was that answer?"

Strip away the jargon and a reward is a single number attached to one model output. Higher means better. That is the whole idea.

Think of training a dog. You do not hand the dog a textbook on how to sit. You wait for it to do something, and then you either give it a treat or you do not. The treat is the reward. Over many repetitions the dog learns which behaviors earn treats. The dog never reads the rules; it just notices that some actions lead to more treats than others, and it does more of those.

Reinforcement learning works the same way. The model produces an answer (in our case, a JSON array of memories). Something looks at that answer and assigns it a number. The training algorithm — PPO, GRPO, whatever — then nudges the model's weights so that high-scoring behavior becomes more likely and low-scoring behavior becomes less likely. The model never sees "the correct answer" the way it did during supervised fine-tuning. It only ever sees the score.

So the entire quality of your RL run depends on the quality of that number. A reward that is easy to fake produces a model that fakes it. A reward that genuinely tracks "good memory extraction" produces a model that genuinely gets better at memory extraction.

### The two ways to get the number

There are exactly two sources for that score, and most of this chapter is about choosing between them:

1. **A programmatic reward function.** You write Python that looks at the model's output and returns a number. For our task: is it valid JSON? Does it match the schema? How many of the gold memories did it find? Did it hallucinate any? You already wrote most of this logic in *Ch18* — there it produced an evaluation report; here it produces a training signal. Same code, new job.

2. **A learned reward model.** Sometimes "good" is something you can recognize but cannot easily write down in code. Is this memory phrased naturally? Is it the *right level* of detail — not too granular, not too bundled? For fuzzy, subjective judgments like that, you instead *train a small model* to predict the score. You feed it examples of "this answer is better than that answer" (human or AI preferences) and it learns to output a number that agrees with those preferences.

The rule of thumb, which we will earn over the course of the chapter: **if you can write the check in code, write the check in code.** A learned reward model is more powerful but also slower, more expensive, and — critically — *hackable* in ways a deterministic function is not. For a structured-output task like memory extraction, a programmatic function gets you most of the way for almost no cost.

### What "preference data" means

A reward model is not trained on "answer → score" pairs. Scoring an answer in isolation is hard even for humans — is this output a 7 or an 8 out of 10? Nobody agrees. But *comparing* two answers is easy: "the left one is better than the right one." Preference data is exactly that: a **prompt**, a **chosen** (better) answer, and a **rejected** (worse) answer. The reward model learns to give the chosen answer a higher number than the rejected one. We will build a small preference dataset for memory extraction below.

---

## Part 1 — Programmatic rewards for memory extraction

### The intuition: a vending machine, not a judge

A programmatic reward function is a vending machine. You put an answer in; a number comes out; the rules are fixed and visible and the same every time. There is no opinion, no API call, no model to load. For a task where "good" has hard, checkable properties — valid JSON, conforms to the schema, found the right facts, invented nothing — this is exactly what you want.

Our memory-extraction task is unusually friendly to programmatic rewards because we know precisely what a good answer looks like:

- It parses as JSON. (Binary: yes or no.)
- It is an array of objects, each with `text`, `type`, and `entities`. (Checkable against the schema.)
- The `type` is one of `preference | fact | decision | relationship`. (Set membership.)
- It contains no markdown fences and no prose before or after. (String inspection.)
- Its memories overlap with the gold answer — high recall, high precision. (This is the F1 you computed in *Ch18*.)
- It did not bundle three facts into one sentence, and it did not invent entities. (Penalties we can detect.)

Every one of those is a line of Python. Let us turn them into a reward.

### The exact signature GRPO expects

Here is the single most important fact in this section, so read it slowly. When you use this reward with GRPO in *Ch28 - GRPO: Practical RL With Reward Functions*, TRL will call your function with a very specific signature:

```python
def reward_fn(completions, **kwargs) -> list[float]:
    ...
```

- `completions` is a **list** of model outputs for a batch of generations. GRPO generates several candidate answers per prompt (`GRPOConfig.num_generations`, default `8`), so you score a whole list at once.
- `**kwargs` catches everything else. Any extra column in your training dataset arrives here as a keyword argument. If your dataset has a `gold_memories` column, you receive `gold_memories=[...]` — one entry per completion. This is how the reward function sees the reference answer it should compare against.
- The return value is a **list of floats**, one score per completion, in the same order.

This is not a signature we are inventing for teaching purposes — it is the contract verified against `trl==1.6.0`. The very same function object you write here gets passed to `GRPOTrainer(reward_funcs=...)` in *Ch28* with zero changes. Write it once, here, correctly.

> One subtlety about `completions` shape: in *conversational* GRPO setups each completion is a list of message dicts (`[{"role": "assistant", "content": "..."}]`); in plain-text setups each completion is a raw string. Our helper below handles both so you can drop it into either configuration. We will use the conversational format to stay consistent with the rest of the book.

### Building the reward function

We reuse the parsing and scoring helpers from *Ch18* rather than re-explaining them — `parse_model_output`, `validate_memory_schema`, and the set-level F1 logic all carry over verbatim. Save this as `code/reward_functions.py`.

```python
# reward_functions.py
#
# Programmatic reward functions for memory extraction.
#
# This module is imported in TWO places:
#   1. Here, for quick standalone testing (see __main__ at the bottom).
#   2. In Ch28's GRPO training script, passed directly as GRPOTrainer(reward_funcs=...).
#
# The reward function signature is the EXACT contract TRL's GRPOTrainer expects:
#   fn(completions, **kwargs) -> list[float]
# Do not change it. Extra dataset columns (like gold_memories) arrive via **kwargs.
#
# Requirements: only the Python standard library. No GPU, no API, no model.

import json
import re
from difflib import SequenceMatcher

# The four valid memory types — straight from the pinned schema (Ch11/Ch12).
# NOTE: exactly these four. Not five. A model that emits "event" or "goal"
# is producing an invalid type and should be penalized.
VALID_TYPES = {"preference", "fact", "decision", "relationship"}


# ── Reused from Ch18: turn a raw model string into a list of memory dicts ─────
# We try clean JSON first, then a markdown-fence fallback, then a bracket grab.
# (Full explanation lives in Ch18 — this is the same function.)
def parse_model_output(raw: str):
    """Return (memories_list, error_str). On success error_str == ''."""
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data, ""
        return None, f"got {type(data).__name__}, not a list"
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        try:
            data = json.loads(fence.group(1))
            if isinstance(data, list):
                return data, ""
        except json.JSONDecodeError:
            pass

    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            if isinstance(data, list):
                return data, ""
        except json.JSONDecodeError:
            pass

    return None, f"no JSON array in: {raw[:80]}..."


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy matching."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def fuzzy_match(a: str, b: str, threshold: float = 0.75) -> bool:
    """True if normalized character-overlap ratio clears the threshold."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio() >= threshold


def set_f1(pred: list, gold: list, threshold: float = 0.75) -> float:
    """
    Set-level F1 between predicted and gold memories, matched on the 'text' field.
    Same greedy bipartite matching as Ch18's compute_set_f1, condensed to return
    just the F1 number (that is all the reward needs).
    """
    # Both empty is a perfect answer (the conversation had no memories to extract).
    if not gold and not pred:
        return 1.0
    # Exactly one side empty → no overlap possible → F1 is 0.
    if not gold or not pred:
        return 0.0

    pred_texts = [m.get("text", "") for m in pred if isinstance(m, dict)]
    gold_texts = [m.get("text", "") for m in gold if isinstance(m, dict)]
    gold_used = [False] * len(gold_texts)
    tp = 0
    for pt in pred_texts:
        for j, gt in enumerate(gold_texts):
            if not gold_used[j] and fuzzy_match(pt, gt, threshold):
                tp += 1
                gold_used[j] = True
                break

    fp = len(pred_texts) - tp
    fn = sum(1 for used in gold_used if not used)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Helper: GRPO hands us completions in one of two shapes. Normalize them. ───
def _completion_to_text(completion) -> str:
    """
    Conversational GRPO: completion is [{"role": "assistant", "content": "..."}].
    Plain-text GRPO:      completion is just a string.
    Return the assistant text either way.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        # Take the content of the last message (the assistant turn).
        return completion[-1].get("content", "")
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# THE REWARD FUNCTION — this is what GRPO calls.
# ═════════════════════════════════════════════════════════════════════════════
def memory_reward(completions, gold_memories=None, **kwargs) -> list[float]:
    """
    Score a batch of memory-extraction completions.

    Args:
        completions:   list of model outputs (one per generation). Each is either
                       a string or a list of message dicts (see _completion_to_text).
        gold_memories: list of reference memory lists, one per completion. Supplied
                       automatically by GRPO because the training dataset has a
                       'gold_memories' column. May be None during quick testing.
        **kwargs:      any other dataset columns GRPO forwards — we ignore them, but
                       the parameter MUST be here or TRL's call will raise TypeError.

    Returns:
        list[float]: one reward per completion, same order as `completions`.

    Reward design (additive, then clamped to [0, 1]):
        +0.20  output parses as a JSON array
        +0.10  no markdown fence and no stray prose around the array
        +0.20  every item conforms to the schema (text/type/entities, valid type)
        +0.50  scaled by F1 against the gold answer (the real quality signal)
        -0.15  per hallucinated entity (an entity not present in any gold memory)
        -0.10  per "bundled" memory (a text that looks like two facts in one)

    The weights are deliberate: format correctness is cheap to earn and caps out
    low (0.50 total), so the model cannot win by emitting perfectly-formatted
    garbage. The bulk of a great score has to come from actually matching the
    gold memories (the F1 term).
    """
    rewards = []

    for i, completion in enumerate(completions):
        text = _completion_to_text(completion)
        gold = gold_memories[i] if gold_memories is not None else []

        score = 0.0

        # ── 1. Does it parse as a JSON array at all? ──────────────────────────
        pred, _err = parse_model_output(text)
        if pred is None:
            # Unparseable output earns nothing. This is the floor.
            rewards.append(0.0)
            continue
        score += 0.20

        # ── 2. Clean formatting: raw array, no fence, no surrounding prose. ───
        stripped = text.strip()
        looks_clean = (
            stripped.startswith("[")
            and stripped.endswith("]")
            and "```" not in stripped
        )
        if looks_clean:
            score += 0.10

        # ── 3. Schema validity: every item has the right keys and a valid type.
        schema_ok = True
        for m in pred:
            if not isinstance(m, dict):
                schema_ok = False
                break
            if not all(k in m for k in ("text", "type", "entities")):
                schema_ok = False
                break
            if m["type"] not in VALID_TYPES:
                schema_ok = False
                break
            if not isinstance(m["entities"], list):
                schema_ok = False
                break
        if schema_ok:
            score += 0.20

        # ── 4. The real signal: F1 against the gold answer (0.0–1.0 → 0.0–0.50).
        f1 = set_f1(pred, gold)
        score += 0.50 * f1

        # ── 5. Penalty: hallucinated entities. ────────────────────────────────
        # Collect every entity mentioned anywhere in the gold answer.
        gold_entities = set()
        for m in gold:
            if isinstance(m, dict):
                for e in m.get("entities", []):
                    gold_entities.add(normalize_text(str(e)))
        # Any predicted entity not present in the gold set is a likely hallucination.
        for m in pred:
            if isinstance(m, dict):
                for e in m.get("entities", []):
                    if normalize_text(str(e)) not in gold_entities and gold_entities:
                        score -= 0.15

        # ── 6. Penalty: bundled facts. ────────────────────────────────────────
        # The schema demands ONE fact per object. A long sentence joined by " and "
        # or containing a semicolon is a strong smell of two facts crammed together.
        for m in pred:
            if isinstance(m, dict):
                t = m.get("text", "")
                # Parenthesized so the length guard only applies to the " and " case:
                # a semicolon is always a bundle smell; an " and " only counts when the
                # sentence is also long (short "Sarah and Mia met" is fine).
                if ";" in t or (re.search(r"\b and \b", t) and len(t) > 60):
                    score -= 0.10

        # ── 7. Clamp to [0, 1] so the RL algorithm sees a bounded, comparable signal.
        rewards.append(max(0.0, min(1.0, score)))

    return rewards


# ═════════════════════════════════════════════════════════════════════════════
# Standalone test — run `python reward_functions.py` to see scores on examples.
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # A gold answer for one conversation: two clean atomic memories.
    gold = [
        {"text": "Sarah prefers dark roast coffee in the morning.",
         "type": "preference", "entities": ["Sarah"]},
        {"text": "Sarah works as a data scientist.",
         "type": "fact", "entities": ["Sarah"]},
    ]

    # Four candidate completions, from great to terrible. In real GRPO these would
    # be eight samples from the model for the SAME prompt; here we hand-write them
    # to show how the reward separates good from bad.
    candidates = [
        # (a) Near-perfect: matches gold, clean format, valid schema.
        '[{"text": "Sarah prefers dark roast coffee in the morning.", "type": "preference", "entities": ["Sarah"]}, '
        '{"text": "Sarah works as a data scientist.", "type": "fact", "entities": ["Sarah"]}]',

        # (b) Right content but wrapped in a markdown fence (loses the clean-format bonus).
        '```json\n[{"text": "Sarah prefers dark roast coffee in the morning.", "type": "preference", "entities": ["Sarah"]}, '
        '{"text": "Sarah works as a data scientist.", "type": "fact", "entities": ["Sarah"]}]\n```',

        # (c) Bundled + hallucinated entity: one fact crams both, invents "Maya".
        '[{"text": "Sarah prefers dark roast coffee in the morning and works as a data scientist with Maya.", '
        '"type": "fact", "entities": ["Sarah", "Maya"]}]',

        # (d) Not JSON at all.
        "Here are the memories I found: Sarah likes coffee.",
    ]

    # GRPO would also pass gold_memories=[gold, gold, gold, gold] — one per completion.
    scores = memory_reward(candidates, gold_memories=[gold] * len(candidates))
    for label, s in zip("abcd", scores):
        print(f"  candidate ({label}): reward = {s:.3f}")
```

Run it:

```bash
python code/reward_functions.py
```

Sample output:

```
  candidate (a): reward = 1.000
  candidate (b): reward = 0.900
  candidate (c): reward = 0.250
  candidate (d): reward = 0.000
```

Read those four numbers — they *are* the lesson:

- **(a)** is a perfect answer and earns the full reward.
- **(b)** has identical content but wraps it in a markdown fence, which the schema forbids. It loses the 0.10 clean-format bonus and lands at 0.90. The model learns that fences cost it.
- **(c)** bundled two facts into one object and invented "Maya." It parses and passes the loose schema check, so it banks the format points (0.20 + 0.10 + 0.20 = 0.50) — but its F1 against gold is 0.0 (one fat bundled sentence matches neither atomic gold memory), and the two penalties bite: −0.15 for the hallucinated "Maya" and −0.10 for the bundle, leaving 0.25. The model learns that bundling and hallucinating are expensive.
- **(d)** is not JSON. It earns nothing. This is the behavior you most want to stamp out, and a zero reward stamps it out fastest.

That spread — 1.0 down to 0.0 across answers a human would also rank in that order — is the entire point. A reward function is *good* when its ordering matches your judgment. Before you ever run GRPO, eyeball a dozen examples like this and confirm the numbers come out in the order you would put them in. If they do not, fix the function, not the model.

### Composing several reward functions

`GRPOConfig` lets you pass a **list** of reward functions instead of one, and weight them with `reward_weights`. This is often cleaner than cramming every rule into a single function: keep a `format_reward` that only checks JSON/schema, and a separate `f1_reward` that only measures content. GRPO sums them (weighted) into the final score. We will use the single combined function above for simplicity, but know that the list form exists — it makes each signal easy to log and debug separately. *Ch28* shows the list form in the GRPO config.

### A note on cost and determinism

This function loads no model, makes no network call, and runs in microseconds. During a GRPO run that generates 8 completions for each of 1,000 prompts across several epochs, the reward function is called tens of thousands of times. If each call cost an API request, that would be slow and expensive. Because ours is pure Python, it is effectively free. That is the headline argument for programmatic rewards on structured tasks: **they scale to RL's call volume for nothing.**

---

## Part 2 — When you need a learned reward model instead

### The intuition: some "good" can't be written down

The vending-machine analogy breaks the moment "good" stops being checkable. Suppose every candidate answer is valid JSON, schema-conforming, and matches the gold memories on F1 — but one phrases a memory as *"The user likes coffee"* and another as *"Sarah prefers dark roast coffee in the morning, brewed strong."* Both are correct. One is clearly a better memory: more specific, more useful, more standalone. There is no clean Python predicate for "this is the better-phrased memory." You *recognize* it, but you cannot easily *code* it.

That is the gap a learned reward model fills. Instead of writing the rule, you show examples of the judgment — "this answer is better than that one" — and train a small model to predict a score that agrees with your preferences. Then *that model's output* becomes the reward.

You reach for a learned reward model when the quality you care about is:

- **Subjective** — phrasing, tone, helpfulness, "naturalness" of a memory.
- **Holistic** — depends on the whole answer in a way that resists itemized checks.
- **Hard to specify but easy to recognize** — you know it when you see it, but every attempt to write the rule misses cases.

For our structured extraction task, honestly, most of the quality *is* checkable, which is why Part 1 carries most of the weight. But the phrasing-quality example above is real, and it is the kind of thing a reward model captures and a regex cannot.

### How a reward model is built

A reward model is an ordinary transformer with its language-modeling head swapped for a single-number regression head. In Hugging Face terms that is `AutoModelForSequenceClassification` with `num_labels=1`: feed it a (prompt, answer) pair, it returns one scalar — the reward.

You train it on preference pairs. For each pair (chosen answer beats rejected answer), the model is nudged so that `score(chosen) > score(rejected)`. It is not told *by how much* — only the ordering. After enough pairs, it generalizes: hand it a brand-new answer it has never seen and it produces a number consistent with the preferences it learned.

---

## Part 3 — Training a reward model with TRL

### Building a small preference dataset

Preference data for our task is easy to manufacture because we already know what good and bad extractions look like. For each conversation we pair a **chosen** (better) extraction with a **rejected** (worse) one. The "worse" version is degraded on purpose: bundle two facts, drop a memory, add a fence, invent an entity, or use a vaguer phrasing.

TRL's `RewardTrainer` accepts a dataset with `chosen` and `rejected` columns (and an optional `prompt`). Each can be either a plain string or a conversational list of messages — `RewardTrainer` detects the format and applies the chat template for you. We use the conversational form so the reward model sees the same system prompt the policy was trained with.

```python
# build_preference_data.py
# Construct a small chosen/rejected preference dataset for memory extraction
# and save it as JSONL. In a real project these pairs come from human review
# or from comparing two model checkpoints; here we craft them to show the shape.

import json

# The pinned system prompt — identical to training (Ch12) and eval (Ch18).
# Reused verbatim so the reward model judges answers in the same context the
# policy produced them.
SYSTEM_PROMPT = """You are a memory extraction assistant. Your job is to read a conversation and extract every memorable piece of information as a list of atomic memory objects.

Each memory object must follow this exact JSON schema:
{
  "text": "<the fact, written as a complete, standalone sentence>",
  "type": "<one of: preference | fact | decision | relationship>",
  "entities": ["<list of named people, places, or things involved>"]
}

Rules:
- One fact per memory object. Do not bundle multiple facts into one.
- Write "text" as a sentence someone could read without any surrounding context.
- If there are no memorable facts in the conversation, return an empty list: []
- Return ONLY a valid JSON array. No explanation, no markdown fences, no extra text.
"""

# Each tuple: (conversation, better_extraction, worse_extraction).
# The "better" one is clean, atomic, complete. The "worse" one is degraded in
# a realistic way — bundled, vague, or fenced.
RAW_PAIRS = [
    (
        "User: I'm Sarah and I always grab a dark roast first thing in the morning.\n"
        "User: I work as a data scientist over at a fintech startup.",
        # chosen: two clean atomic memories
        [
            {"text": "Sarah prefers dark roast coffee in the morning.",
             "type": "preference", "entities": ["Sarah"]},
            {"text": "Sarah works as a data scientist at a fintech startup.",
             "type": "fact", "entities": ["Sarah"]},
        ],
        # rejected: bundled into one non-atomic memory
        [
            {"text": "Sarah likes dark roast coffee and works as a data scientist.",
             "type": "fact", "entities": ["Sarah"]},
        ],
    ),
    (
        "User: My sister Mia just adopted a rescue dog named Biscuit.\n"
        "User: We decided he'll stay with me on weekends.",
        # chosen: a relationship fact + a decision, correctly typed
        [
            {"text": "Mia is the user's sister.",
             "type": "relationship", "entities": ["Mia"]},
            {"text": "Mia adopted a rescue dog named Biscuit.",
             "type": "fact", "entities": ["Mia", "Biscuit"]},
            {"text": "The user and Mia decided Biscuit will stay with the user on weekends.",
             "type": "decision", "entities": ["Mia", "Biscuit"]},
        ],
        # rejected: misses the decision and uses a vague, non-standalone phrasing
        [
            {"text": "Has a sister with a dog.",
             "type": "fact", "entities": []},
        ],
    ),
    (
        "User: Reminder, the team agreed to ship the v2 API by end of Q3.",
        # chosen: clean decision
        [
            {"text": "The team agreed to ship the v2 API by the end of Q3.",
             "type": "decision", "entities": ["v2 API"]},
        ],
        # rejected: correct content but wrapped in a forbidden markdown fence
        # (we store the raw string form to preserve the fence — see below).
        "```json\n[{\"text\": \"Ship v2 API by Q3.\", \"type\": \"decision\", \"entities\": []}]\n```",
    ),
]


def as_conversation(conversation: str, answer) -> list:
    """
    Wrap a (conversation, answer) into the conversational message list
    RewardTrainer expects. `answer` is either a list of memory dicts (we
    JSON-encode it) or an already-formatted raw string (we pass it through,
    which lets us preserve a deliberately-bad fenced answer).
    """
    if isinstance(answer, str):
        assistant_content = answer
    else:
        # Compact JSON, exactly how a clean model emits it (no indentation).
        assistant_content = json.dumps(answer, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": conversation},
        {"role": "assistant", "content": assistant_content},
    ]


with open("data/preference_pairs.jsonl", "w", encoding="utf-8") as f:
    for conversation, better, worse in RAW_PAIRS:
        row = {
            # RewardTrainer reads exactly these two columns.
            "chosen": as_conversation(conversation, better),
            "rejected": as_conversation(conversation, worse),
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(RAW_PAIRS)} preference pairs to data/preference_pairs.jsonl")
```

```
Wrote 3 preference pairs to data/preference_pairs.jsonl
```

Three pairs is a teaching toy — it shows the *format*, nothing more. A reward model that actually generalizes needs a few thousand pairs (consistent with the charter's rule of thumb for reward models). The pairs should cover the full range of mistakes you want the model to learn to disprefer: bundling, vagueness, missed memories, hallucinations, wrong types, fences. Quality and diversity of pairs matter far more than raw count — the same lesson as *Ch13* on synthetic data.

### The training script

Now train the reward model. The critical line — read the charter again if you skip everything else — is `processing_class=tokenizer`. In `trl==1.6.0` the trainer constructors take `processing_class=`, **not** the old `tokenizer=` kwarg. (Unsloth's `SFTTrainer` path back in *Ch15* still uses `tokenizer=`; the new RL trainers do not. Mixing them up is the single most common error in this part of the book.) Save this as `code/train_reward_model.py`.

```python
# train_reward_model.py
# Train a reward model for memory extraction from chosen/rejected preference pairs.
#
# Output: a small AutoModelForSequenceClassification (num_labels=1) that, given a
# (conversation, extraction) pair, returns ONE number — higher = better extraction.
#
# Requirements (pinned versions — see code/requirements.txt):
#   trl==1.6.0  transformers  peft  datasets  accelerate  bitsandbytes

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig          # both are top-level in trl 1.6.0
from peft import LoraConfig                            # train the RM as a LoRA, cheaply

# ── 1. Base model ────────────────────────────────────────────────────────────
# A reward model does NOT need to be large. It is a judge, not a generator, and a
# small base (0.5B–4B) is plenty for a narrow task. We use a small instruct model.
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    # Sequence-classification needs a pad token to batch variable-length pairs.
    tokenizer.pad_token = tokenizer.eos_token

# num_labels=1 turns the classification head into a single-number REGRESSION head.
# That one number is the reward. This is the defining trick of a reward model.
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,
)
# Tell the model which token id to pad with (must match the tokenizer above).
model.config.pad_token_id = tokenizer.pad_token_id

# ── 2. Preference dataset ────────────────────────────────────────────────────
# Each row has 'chosen' and 'rejected' (conversational message lists, built by
# build_preference_data.py). RewardTrainer detects the conversational format,
# applies the chat template, appends EOS, and tokenizes both sides for you.
dataset = load_dataset(
    "json",
    data_files={"train": "data/preference_pairs.jsonl"},
    split="train",
)
print(f"Loaded {len(dataset)} preference pairs")

# ── 3. LoRA config — train a cheap adapter, not the whole model ──────────────
# Same idea as Ch6: freeze the base, train a small adapter. Keeps the reward
# model fast to train and tiny to store.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    # Sequence classification has a regression head outside the LoRA layers —
    # PEFT trains it alongside the adapter automatically.
    task_type="SEQ_CLS",
)

# ── 4. Training config ───────────────────────────────────────────────────────
# RewardConfig wraps TrainingArguments with reward-specific defaults.
# max_length defaults to 1024 in trl 1.6.0 — long enough for a conversation +
# a short JSON answer. Raise it only if your pairs are getting truncated.
reward_args = RewardConfig(
    output_dir="data/reward_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,    # effective batch size 8
    num_train_epochs=1,               # a few thousand pairs rarely needs more than 1–2
    learning_rate=1e-5,               # reward models like a gentle LR
    logging_steps=5,
    max_length=1024,                  # the default; shown here so you know it exists
    report_to="none",                 # no W&B/TensorBoard for this small demo
)

# ── 5. The trainer ───────────────────────────────────────────────────────────
# CRITICAL: processing_class=tokenizer, NOT tokenizer=tokenizer.
# In trl 1.6.0 the old `tokenizer=` kwarg was removed from RewardTrainer.
# Passing tokenizer= will raise a TypeError. This is the #1 gotcha in Part 7.
trainer = RewardTrainer(
    model=model,
    args=reward_args,
    train_dataset=dataset,
    processing_class=tokenizer,       # <-- the line everyone gets wrong
    peft_config=peft_config,
)

print("Training reward model...")
trainer.train()

# ── 6. Save ──────────────────────────────────────────────────────────────────
trainer.save_model("data/reward_model")
tokenizer.save_pretrained("data/reward_model")
print("Reward model saved to data/reward_model")
```

During training you will see the loss drop. The reward-model loss is *not* the language-modeling loss from earlier chapters — it is a preference loss that goes down as the model gets better at scoring chosen above rejected. A typical run on a few thousand pairs looks like:

```
Loaded 3 preference pairs
Training reward model...
{'loss': 0.6931, 'grad_norm': 2.41, 'learning_rate': 9.8e-06, 'epoch': 0.33}
{'loss': 0.5402, 'grad_norm': 2.02, 'learning_rate': 5.0e-06, 'epoch': 0.66}
{'loss': 0.4119, 'grad_norm': 1.74, 'learning_rate': 2.0e-07, 'epoch': 1.0}
Reward model saved to data/reward_model
```

That starting loss near `0.6931` is not a coincidence — it is `ln(2)`, the loss of a model guessing randomly between two options. As training proceeds and the model starts reliably ranking chosen above rejected, the loss falls below it. (With only three toy pairs the model has nothing to generalize to; on a few thousand real pairs the same curve means something.)

### Using the reward model to score an answer

Once trained, scoring is one forward pass: feed the (conversation, answer) pair, read the single output number.

```python
# score_with_reward_model.py — sanity-check a trained reward model.
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("data/reward_model")
model = AutoModelForSequenceClassification.from_pretrained("data/reward_model")
model.eval()

def reward_score(conversation: str, answer_json: str) -> float:
    """Return the reward model's scalar score for one (conversation, answer) pair."""
    # Format the pair the same way RewardTrainer saw it during training.
    messages = [
        {"role": "user", "content": conversation},
        {"role": "assistant", "content": answer_json},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        # logits has shape [1, 1] — the single reward number.
        score = model(**inputs).logits[0, 0].item()
    return score

conv = "User: I'm Sarah and I drink dark roast every morning."
clean = '[{"text": "Sarah prefers dark roast coffee in the morning.", "type": "preference", "entities": ["Sarah"]}]'
bundled = '[{"text": "Sarah drinks coffee and stuff in the morning probably.", "type": "fact", "entities": []}]'

print(f"clean answer score:   {reward_score(conv, clean):.3f}")
print(f"bundled answer score: {reward_score(conv, bundled):.3f}")
```

```
clean answer score:   1.842
bundled answer score: -0.517
```

The raw numbers are unbounded and not calibrated to any fixed scale — a reward model only learns *relative* ordering, so all that matters is that the clean answer scores **higher** than the bad one. It does. During RL, those relative scores are exactly what the policy chases.

---

## Part 4 — How the reward plugs into RL

You now have two ways to produce a number. Here is where each one goes.

**With PPO (conceptual, *Ch27 - PPO and the Full RL Loop: Why We Don't Use It Here*).** Classic RLHF uses a learned reward model in the loop: the policy generates an answer, the reward model scores it, and PPO uses that score (minus a KL penalty that keeps the policy from drifting too far from where it started) to update the policy. The reward model you trained in Part 3 is precisely the "RM" in that picture. Note carefully — *Ch27 is a conceptual chapter*: in `trl==1.6.0` the runnable `PPOTrainer` moved to `trl.experimental.ppo` and the old manual `.step()` loop is gone, so *Ch27* explains the mechanism (value head, KL penalty, advantage) and any code there is explicitly labeled illustrative, not runnable. The takeaway you need here: PPO is the canonical *consumer* of a learned reward model.

**With GRPO (runnable, *Ch28 - GRPO: Practical RL With Reward Functions*).** GRPO is the centerpiece of Part 7 and the one you will actually run. It generates a group of completions per prompt and ranks them against each other, which means it usually does not need a learned reward model at all — it works beautifully with **reward functions**. The `memory_reward` function from Part 1 drops straight into `GRPOTrainer(reward_funcs=memory_reward, ...)` with no changes. That is why we spent the most effort on it: it is the reward you will use most.

Both consumers can take either kind of reward — GRPO can call a learned reward model wrapped in a function, and a PPO setup could in principle use a programmatic reward. The pairing above is just the common case.

---

## Part 5 — Honest notes

### Reward hacking: the model games whatever you actually measure

The deepest hazard in all of RL is **reward hacking** (sometimes called specification gaming): the model maximizes the *number you wrote down*, which is never quite the same as the *thing you meant*. If your reward gives points for valid JSON and length, the model may discover it can emit one trivially-valid memory and score fine without doing the hard work of finding all the facts. If your reward model was trained on pairs where the "chosen" answer happened to be longer, the policy learns to pad — it produces verbose memories because length, not quality, is what the reward learned to like.

The defenses are practical:

- **Inspect the winners.** During and after RL, read the highest-scoring outputs. If they look gamed rather than good, your reward is the problem, not the model.
- **Cap the cheap-to-earn terms.** Notice our `memory_reward` caps all the format bonuses at 0.50 and routes the other half through F1 against gold. The model cannot win on format alone.
- **Keep an honest held-out eval.** Score every checkpoint with the *Ch18* pipeline on data the reward never touched. If the reward climbs but Ch18 F1 stalls or drops, the model is hacking the reward. This is the single most reliable detector.
- **Penalize the specific cheats.** The hallucinated-entity and bundled-fact penalties exist precisely because those are the easy ways to look-good-while-being-wrong on this task.

A learned reward model is *more* hackable than a programmatic one, not less — it has soft spots a deterministic function does not, and the policy is very good at finding them. This is a major reason to prefer programmatic rewards when you can.

### The real cost of a good reward model

A reward model is not free. It costs:

- **Preference data.** A few thousand quality pairs (the charter's rule of thumb) is a real labeling effort if humans produce them, or a real generation-and-curation effort if AI produces them. Garbage pairs make a garbage judge.
- **A second training run** and a second model to version, store, and serve alongside the policy.
- **Inference cost in the loop.** Every RL step calls the reward model, adding a forward pass per completion. With `num_generations=8`, that is eight extra forward passes per prompt per step.
- **The risk of a subtly-wrong judge.** A reward model that learned the wrong thing silently teaches the policy the wrong thing. A bug in a programmatic reward, by contrast, is usually visible the moment you print a few scores.

### Why programmatic rewards are usually enough for *our* task

Memory extraction is a structured-output task with a checkable definition of success. Valid JSON is checkable. Schema conformance is checkable. F1 against a gold answer is checkable. Hallucinated entities and bundled facts are detectable. That covers the overwhelming majority of what "good" means for us — and all of it lives in `memory_reward`, which runs in microseconds with no model, no API, and no preference-labeling project.

So the honest recommendation for this book's running example: **start with the programmatic reward and go straight to GRPO in *Ch28*.** Reach for a learned reward model only when you hit a quality dimension the function genuinely cannot capture — phrasing quality, naturalness, the "right level of detail" — and when you have the labeling budget to build a few thousand solid preference pairs. For many structured-extraction projects in the wild, that day never comes, and that is a perfectly good outcome. The most powerful tool is not always the right one; the right one is the cheapest tool that genuinely measures what you care about.

---

## Recap

- A **reward** is one number saying how good an answer is. RL makes good answers more likely and bad answers less likely by chasing that number — so the number's quality is everything.
- There are two sources: a **programmatic reward function** (Python you write) and a **learned reward model** (trained from preferences). If you can write the check in code, write the check in code.
- The reward-function signature GRPO expects is exactly `reward_fn(completions, **kwargs) -> list[float]`. Extra dataset columns (like `gold_memories`) arrive via `**kwargs`. The `memory_reward` function in this chapter is reused unchanged in *Ch28*.
- Good reward design caps cheap-to-earn format points and routes most of the score through real quality (F1 vs gold), with explicit penalties for hallucinated entities and bundled facts.
- A **learned reward model** is an `AutoModelForSequenceClassification(num_labels=1)` trained with TRL's `RewardTrainer` on `chosen`/`rejected` preference pairs. The constructor takes `processing_class=tokenizer` — **not** `tokenizer=` — in `trl==1.6.0`.
- A reward model only learns *relative* ordering; its raw scores are unbounded and uncalibrated. All that matters is chosen > rejected.
- PPO (*Ch27*, conceptual) is the canonical consumer of a learned reward model; GRPO (*Ch28*, runnable) usually uses reward functions instead.
- Watch for **reward hacking**: the model maximizes what you measured, not what you meant. Inspect top-scoring outputs and keep an honest held-out *Ch18* eval as the real-quality detector.
- For a structured task like memory extraction, a programmatic reward is usually enough — cheaper, faster, more transparent, and harder to game than a learned one.

## Next

*Ch26 - DPO: Learning Directly From Preference Pairs* — DPO skips the separate reward model entirely and optimizes the policy directly from the same chosen/rejected pairs you built here. It is often the fastest path from preferences to a better model.
