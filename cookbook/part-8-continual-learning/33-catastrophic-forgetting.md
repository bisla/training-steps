# Chapter 33 - Catastrophic Forgetting Over Many Rounds

In *Ch32 - How Much, How Often* you set up the rhythm of continual learning: every so often, fresh conversations arrive, you turn them into training rows, and you run another short fine-tune on top of the model you already have. Round after round, the model keeps up with your world. It is a living system now, not a one-shot project.

There is a quiet danger in that rhythm. Each round teaches the model something new — but each round can *also* erase something old, without any error, without any warning in the loss curve. Your `relationship`-type extraction was excellent in round 3, and by round 9 it has quietly degraded by twenty points and you never noticed, because you stopped checking that capability once it worked. This is **catastrophic forgetting**, and over many rounds it is the single most likely way a continual-learning pipeline rots from the inside.

This chapter teaches you to see it, measure it, and manage it. Not eliminate it — you cannot fully eliminate it — but keep it on a leash.

---

## What you'll learn

- What catastrophic forgetting *is*, with an analogy that makes the mechanism obvious
- *Why* it happens — the plain-English reason every training round pulls the model toward new data and lets old behavior drift
- How to **measure** it: a frozen "canary" regression set that never changes, run every round, tracked per-capability, with automatic alerts when a score drops
- A runnable harness that scores a model against the canary set and compares this round to last round
- How to **mitigate** it: replay/rehearsal (cross-referencing the *Ch32* ratios), gentler updates (lower learning rate, fewer epochs), LoRA adapter isolation, the "stay near the previous model" family of tricks (KL/EWC at the intuition level), and knowing when to stop continuing and retrain from base instead
- A worked multi-round simulation where forgetting appears on the canary set, then replay pulls it back
- An honest accounting of what you can and cannot do about it

---

## Concepts you need first

### The analogy: the new employee who forgets the old job

Imagine a support agent at your company who is brilliant at handling billing questions. Business shifts, so this month you train them hard on the new shipping-logistics workflow. They drill it for a week — nothing but shipping tickets. They become a shipping expert.

Then a billing question comes in, and they fumble it. Not because billing got harder, but because a week of focusing entirely on shipping pushed billing out of their working habits. Nobody told them to forget billing. They just stopped practicing it, and the new skill crowded out the old one.

That is catastrophic forgetting in one sentence: **when you train a model round after round on new data, it quietly gets worse at things it used to do well** — older memory types, general language ability, edge cases it had nailed — because nothing in the new training reminds it to keep those skills sharp. The fix for the employee is obvious: don't drill *only* shipping; mix a few billing tickets back in so the old skill stays warm. That mix is exactly what "replay" will be later in this chapter.

The word "catastrophic" is doing real work. Humans forget gracefully; neural networks can forget *abruptly* — a capability at 0.85 can crater to 0.40 after one aggressive round, because the weights that encoded it got overwritten wholesale. The glossary frames *catastrophic forgetting* as the classic single-round failure (train so hard on the new task that you wipe out general ability). In continual learning the same mechanism plays out in slow motion across many rounds, which makes it sneakier: no single round looks catastrophic, but the cumulative drift is.

### Why it happens (no heavy math)

You already have the intuition from *Ch7 - How Training Actually Works*. Training is just this loop: the model makes a prediction, you measure how wrong it was (the **loss**), and the optimizer nudges every trainable weight a little in whatever direction reduces the loss *on the batch in front of it right now*.

That last phrase is the whole story. The optimizer only ever sees the current batch. If round 7's data is all `decision`-type extractions from meeting transcripts, then every gradient step in round 7 is pulling the weights toward "be great at decisions from meetings." Nothing in that loss function says "and also stay good at preferences." The old behavior was encoded in the same weights, and there is no force pinning those weights in place. So they drift. Whatever the new data does not reinforce is, by default, slowly abandoned.

Here is the second half of the intuition: **the model has no memory of its past training data.** When you start round 7, the model is just a pile of numbers. It does not remember round 3's `relationship` examples — from its point of view they never happened; only their *residue in the weights* remains, and round 7 is free to overwrite that residue. Each round optimizes for the data it can see, blind to everything it can't.

This is also why **LoRA** (*Ch6 - LoRA and QLoRA Without the Math Headache*) helps but does not save you. LoRA freezes the giant base model, so the deep general-language knowledge in the frozen base is genuinely safe. But the *task behavior* you care about — your memory-extraction skill, including all the per-type competence — lives in the small adapters. Keep training the same adapter round after round on shifting data and it forgets just like a full model would. The base is protected; your task skill is not.

### What "a capability" means here, concretely

To measure forgetting you need to break "is the model good?" into specific, separately-trackable skills. For memory extraction, the obvious axes are the four memory types from our pinned schema:

```python
# Our pinned memory schema — every example in this book uses exactly this shape.
{
    "text": "Sarah prefers dark roast coffee in the morning",   # the fact, as a complete sentence
    "type": "preference",                                        # one of: preference | fact | decision | relationship
    "entities": ["Sarah"]                                        # named people, places, or things involved
}
```

So a natural per-capability breakdown is: how well does the model extract `preference` memories? `fact` memories? `decision` memories? `relationship` memories? Plus two cross-cutting skills that are easy to lose and easy to forget to check:

- **JSON validity / format discipline** — does it still emit a clean JSON array, or has it started leaking prose, markdown fences, or malformed brackets?
- **The empty case** — does it still correctly return `[]` for a conversation with nothing memorable, or has round after round of "here is a memory, extract it" training made it over-eager, inventing memories where there are none?

These last two are where forgetting bites hardest in practice, because once a model reliably produces valid JSON you tend to stop checking — exactly when a later round quietly breaks it.

---

## How to measure it: the frozen canary set

You manage what you measure. The core tool for catching forgetting is embarrassingly simple, and the discipline around it matters more than any clever code.

### The analogy: the canary in the coal mine

Coal miners used to carry a canary in a cage down into the mine. The canary was more sensitive to toxic gas than the miners were; if it stopped singing, you got out *before* the air killed you. The canary's whole job was to fail first, visibly, so you had warning.

A **canary eval set** is the same idea for your model. It is a small, fixed set of conversations with known-correct memory extractions, covering every capability you care about. You run it after *every* round. Its scores are your early-warning system: when a capability's score drops, the canary "stops singing," and you know forgetting is setting in before it reaches your users.

Two rules make the canary trustworthy, and breaking either ruins it:

1. **It is frozen.** The canary set never changes — not "mostly stays the same," *never changes*. The instant you edit it, scores from before and after are no longer comparable, and the whole point (round-over-round comparison) is gone. Check it into version control and treat editing it like editing a unit of measurement.
2. **It is held out forever.** Canary examples must *never* enter training data — not in round 1, not via replay. If the model trains on a canary example, that example measures memorization, not capability. Same data-leakage trap as mixing your validation split into training (*Ch14 - Cleaning, Splitting, and Sanity-Checking Data*), applied across rounds.

This is *not* your regular held-out test set from *Ch18 - Did It Actually Work? Evaluating Memory Extraction*. The test set answers "how good is the model right now?" and you may grow and refresh it. The canary answers a narrower, stricter question — "compared to last round, did any specific capability get *worse*?" — and must be frozen precisely because it is a *differencing* instrument. Keep both.

### Building the canary set

A good canary set is small (40–120 examples is plenty for a narrow task — enough to be stable, small enough to run in seconds) and *stratified*: deliberately balanced across every capability so each one gets a clean per-capability score. Tag each example with the capability it exercises.

```python
# canary_set.py
# A FROZEN regression set for memory extraction. Check this into version control.
# RULES:
#   1. Never edit an existing example (it breaks round-over-round comparison).
#   2. These examples must NEVER appear in training data or replay (no leakage).
# Each item carries a "capability" tag so we can score per-capability, not just overall.

CANARY_SET = [
    {
        "capability": "preference",
        "conversation": "User: I always take my coffee black, no sugar, first thing in the morning.",
        "gold": [
            {"text": "The user prefers black coffee with no sugar in the morning.",
             "type": "preference", "entities": []},
        ],
    },
    {
        "capability": "fact",
        "conversation": "User: My flight to Berlin leaves at 6am on Tuesday from gate 22.",
        "gold": [
            {"text": "The user's flight to Berlin leaves at 6am on Tuesday from gate 22.",
             "type": "fact", "entities": ["Berlin"]},
        ],
    },
    {
        "capability": "decision",
        "conversation": "User: After the review we decided to ship the redesign in Q3, not Q2.",
        "gold": [
            {"text": "The team decided to ship the redesign in Q3 instead of Q2.",
             "type": "decision", "entities": []},
        ],
    },
    {
        "capability": "relationship",
        "conversation": "User: Maria is my sister, and she manages the Lisbon office.",
        "gold": [
            {"text": "Maria is the user's sister.",
             "type": "relationship", "entities": ["Maria"]},
            {"text": "Maria manages the Lisbon office.",
             "type": "fact", "entities": ["Maria", "Lisbon"]},
        ],
    },
    {
        "capability": "empty_case",   # nothing memorable -> must return []
        "conversation": "User: lol ok. Assistant: 👍. User: brb.",
        "gold": [],
    },
    {
        "capability": "format",       # multi-fact; tests clean JSON under load
        "conversation": ("User: Tom switched to the night shift, he hates mornings, "
                         "and we agreed he'll cover Fridays."),
        "gold": [
            {"text": "Tom switched to the night shift.", "type": "fact", "entities": ["Tom"]},
            {"text": "Tom dislikes mornings.", "type": "preference", "entities": ["Tom"]},
            {"text": "The team agreed Tom will cover Fridays.", "type": "decision", "entities": ["Tom"]},
        ],
    },
    # ... in a real canary set, include 8-20 examples PER capability tag.
    # The six above are a template; a real set has ~60-120 total, stratified.
]
```

### Scoring the canary set per capability

The scoring reuses exactly the matching logic from *Ch18* — parse the model's JSON, match each predicted memory against the gold memories with a fuzzy string match, and compute F1. The only new thing here is that we **group the scores by capability tag** and return a per-capability breakdown, because an aggregate number hides forgetting (a 5-point drop in `relationship` can be masked by a 5-point gain in `decision`).

```python
# canary_eval.py
# Run the FROZEN canary set against a model and return per-capability scores.
# Reuses the JSON-parsing + fuzzy-match F1 logic from Ch18 (Evaluating Memory Extraction).

import json
import re
from difflib import SequenceMatcher
from collections import defaultdict

# The pinned system prompt — IDENTICAL to training and to Ch18 inference.
# It must be byte-for-byte the same string used during fine-tuning.
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


def parse_memories(raw_output: str):
    """Extract a JSON array from model output. Returns (parsed_list, is_valid_json)."""
    # Strip any accidental markdown fences the model might leak.
    cleaned = re.sub(r"```(?:json)?", "", raw_output).strip()
    # Grab the first [...] block so trailing prose doesn't break parsing.
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return [], False
    try:
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, list):
            return [], False
        return parsed, True
    except json.JSONDecodeError:
        return [], False


def _text_match(a: str, b: str, threshold: float = 0.75) -> bool:
    """Fuzzy match two memory 'text' fields (same approach as Ch18)."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return SequenceMatcher(None, norm(a), norm(b)).ratio() >= threshold


def score_one(predicted, gold):
    """Return (precision, recall, f1, json_valid) for one example."""
    # Empty-case handling: gold is [] -> reward an empty prediction, punish any output.
    if len(gold) == 0:
        return (1.0, 1.0, 1.0) if len(predicted) == 0 else (0.0, 1.0, 0.0)
    if len(predicted) == 0:
        return (0.0, 0.0, 0.0)

    matched = 0
    used = set()
    for p in predicted:
        p_text = p.get("text", "") if isinstance(p, dict) else ""
        for i, g in enumerate(gold):
            if i in used:
                continue
            # A match requires BOTH the text to align AND the type to be correct.
            if _text_match(p_text, g["text"]) and p.get("type") == g["type"]:
                matched += 1
                used.add(i)
                break

    precision = matched / len(predicted)
    recall = matched / len(gold)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def run_canary(generate_fn, canary_set):
    """
    generate_fn(system_prompt, conversation) -> str (raw model output).
    Returns a dict: {capability: {"f1": mean_f1, "json_valid": rate, "n": count}}.
    """
    buckets = defaultdict(lambda: {"f1": [], "json_valid": []})
    for item in canary_set:
        raw = generate_fn(SYSTEM_PROMPT, item["conversation"])
        predicted, is_valid = parse_memories(raw)
        _, _, f1 = score_one(predicted, item["gold"])
        buckets[item["capability"]]["f1"].append(f1)
        buckets[item["capability"]]["json_valid"].append(1.0 if is_valid else 0.0)

    report = {}
    for cap, vals in buckets.items():
        report[cap] = {
            "f1": round(sum(vals["f1"]) / len(vals["f1"]), 3),
            "json_valid": round(sum(vals["json_valid"]) / len(vals["json_valid"]), 3),
            "n": len(vals["f1"]),
        }
    return report
```

The `generate_fn` is whatever inference wrapper you already use — the Unsloth `FastLanguageModel.for_inference(model)` path from *Ch18*, or a vLLM client. Keeping the canary harness model-agnostic means you can point it at *any* round's checkpoint without changing the eval code.

### Comparing this round to last round, and alerting

The measurement only earns its keep if it *interrupts you* when something regresses. After each round, you score the new checkpoint, load the previous round's report from disk, diff them per capability, and raise an alert if any capability dropped by more than a tolerance you choose. A small wobble (±0.02) is noise; a real regression is bigger and consistent.

```python
# canary_compare.py
# After each training round: score the new model, diff against the previous round,
# and FAIL LOUDLY if any capability regressed beyond tolerance.

import json
import os


def load_previous(path):
    """Load the prior round's canary report, or None on the very first round."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def compare_rounds(current, previous, drop_tolerance=0.05):
    """
    Diff per-capability F1. Returns (regressions, improvements) as lists of dicts.
    A regression is a drop strictly greater than drop_tolerance (default 5 points).
    """
    regressions, improvements = [], []
    if previous is None:
        return regressions, improvements  # nothing to compare against yet

    for cap, cur in current.items():
        if cap not in previous:
            continue
        delta = round(cur["f1"] - previous[cap]["f1"], 3)
        if delta < -drop_tolerance:
            regressions.append({"capability": cap, "delta": delta,
                                "from": previous[cap]["f1"], "to": cur["f1"]})
        elif delta > drop_tolerance:
            improvements.append({"capability": cap, "delta": delta,
                                 "from": previous[cap]["f1"], "to": cur["f1"]})
    return regressions, improvements


def report_round(round_num, current, prev_path, drop_tolerance=0.05):
    """Print a round summary, save the report, and return True if SAFE to deploy."""
    previous = load_previous(prev_path)
    regressions, improvements = compare_rounds(current, previous, drop_tolerance)

    print(f"\n===== CANARY REPORT — Round {round_num} =====")
    print(f"{'capability':<14}{'F1':>7}{'json_ok':>9}")
    for cap, vals in sorted(current.items()):
        print(f"{cap:<14}{vals['f1']:>7}{vals['json_valid']:>9}")

    for r in regressions:
        # This is the canary that stopped singing. Treat it as a blocker.
        print(f"  ⚠️  REGRESSION: '{r['capability']}' F1 {r['from']} -> {r['to']} "
              f"({r['delta']:+})")
    for i in improvements:
        print(f"  ✅ improved: '{i['capability']}' F1 {i['from']} -> {i['to']} "
              f"({i['delta']:+})")

    # Persist THIS round's report so the NEXT round can diff against it.
    with open(prev_path, "w") as f:
        json.dump(current, f, indent=2)

    safe = len(regressions) == 0
    print("  RESULT:", "SAFE to promote ✅" if safe else "BLOCKED — investigate ⚠️")
    return safe
```

Wire this into the pipeline so a regression *blocks promotion* of the new checkpoint — same spirit as a failing test blocking a deploy. The whole point of a canary is that it fails before your users do. If you log the regression and ship anyway, you have built a smoke detector and disconnected it from the alarm.

---

## How to mitigate it

You cannot make forgetting go away. You *can* make it slow, small, and recoverable. Five tools, roughly in the order you should reach for them.

### 1. Replay / rehearsal — the first and best defense

Back to the support agent who forgot billing: the fix was to keep a few billing tickets in their weekly mix so the old skill stayed warm. **Replay** (also called rehearsal) is exactly that for a model. When you build round N's training data, you do not train *only* on round N's new conversations. You mix in a sample of older and general examples, so every round keeps practicing the full range of capabilities, not just the new ones.

The replay ratio — how much old/general data to mix in — is the headline number from *Ch32 - How Much, How Often*. The book's defensible rule of thumb is **10–30% prior/general data, ~20% a sane default**. Stated as a range and not a law:

- **Too little (<10%)** and new data dominates; old capabilities drift just as with no replay. You are barely reminding the model of anything.
- **Too much (>30%)** and you spend most of each round re-teaching things the model already knows, slowing how fast it absorbs genuinely new patterns — and on a fixed budget, every replay row is a new-data row you didn't train on.
- **~20%** is the comfortable middle: enough rehearsal to keep old skills warm, enough new data to keep moving. Treat *Ch32* as the source of truth on the ratio and scheduling; here we plug it into the recipe.

Two refinements. First, **stratify the replay pool the same way you stratified the canary** — pull across all four memory types and the empty case, so you are not rehearsing only `preference` while `relationship` rots. Second, **never draw replay from the canary set** (rule 2 above); keep a separate "replay reservoir" of older training rows.

```python
# build_round_data.py
# Assemble round N's training data: mostly new conversations, plus a stratified
# replay sample of older/general data. Replay ratio per Ch32 (default ~20%).

import random
from collections import defaultdict

random.seed(42)  # reproducible round assembly


def stratified_replay(replay_reservoir, n_replay, type_key="primary_type"):
    """Sample n_replay rows from the reservoir, balanced across capability tags.

    replay_reservoir: list of training rows (TRL conversational format), each
                      tagged with its dominant memory type. NEVER includes canary rows.
    """
    by_type = defaultdict(list)
    for row in replay_reservoir:
        by_type[row[type_key]].append(row)

    per_type = max(1, n_replay // max(1, len(by_type)))
    sampled = []
    for rows in by_type.values():
        random.shuffle(rows)
        sampled.extend(rows[:per_type])
    random.shuffle(sampled)
    return sampled[:n_replay]


def build_round_dataset(new_rows, replay_reservoir, replay_ratio=0.20):
    """Mix new data with replay. Returns the combined training list for this round."""
    n_new = len(new_rows)
    # Solve for replay count so replay is ~replay_ratio of the FINAL mix.
    # final = new + replay, and replay / final = ratio  ->  replay = new*ratio/(1-ratio)
    n_replay = round(n_new * replay_ratio / (1 - replay_ratio))
    replay_rows = stratified_replay(replay_reservoir, n_replay)

    combined = new_rows + replay_rows
    random.shuffle(combined)
    print(f"Round dataset: {n_new} new + {len(replay_rows)} replay "
          f"= {len(combined)} rows (replay ≈ {len(replay_rows)/len(combined):.0%})")
    return combined
```

Each row is in TRL's conversational format — `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}` with the pinned system prompt — exactly as in *Ch12* and used by `SFTTrainer` in *Ch15*. Replay changes *which* rows you feed the trainer, not the trainer code itself.

### 2. Gentler updates — lower learning rate, fewer epochs

The harder you push the weights in any one round, the more old behavior gets bulldozed. So in continual rounds, push gently.

- **Lower the learning rate.** Recall from *Ch16 - Hyperparameters* that learning rate is the size of each weight-update step. A continual round starts from an *already-good* model — nudging, not teaching from scratch — so a smaller step fits. If your initial SFT used `2e-4`, a continual round at `5e-5` to `1e-4` makes smaller, less destructive moves. (Same instinct as the `beta` leash in DPO, *Ch26* — stay near where you already are.)
- **Fewer epochs.** One epoch per round is often plenty when the new data is small. Each extra epoch is another full pass pulling weights toward the new data and away from everything else. Watch the loss curve (*Ch17 - Watching Training: Loss Curves and When to Stop*) and stop early; you are integrating the new data, not squeezing every drop from it.

The honest tradeoff: gentler updates also mean the model learns the new data *more slowly*. If a round's pattern is urgent, take a bigger step, accept more forgetting risk, and lean harder on replay and the canary to catch the fallout. No free lunch — you trade learning speed against stability.

### 3. LoRA isolation — adapters as containers

Because **LoRA** keeps the base model frozen and puts all learning in small adapters (*Ch6*), the adapter becomes a natural unit of *isolation*. Two patterns:

- **Keep separate adapters per concern, merge deliberately.** Rather than one adapter you re-train forever, train a fresh adapter for a new task or data regime and decide, on purpose, when to fold it in. Merging (`model.save_pretrained_merged(...)`, *Ch21 - Saving, Merging, and Exporting Your Model*) bakes an adapter into the base permanently — do it only after the canary confirms it helps without regressing other capabilities. An unmerged adapter is one you can throw away if the canary says it hurt.
- **Re-train the adapter from the base, not from the last adapter.** Instead of continuing round N's adapter on top of round N−1's (drift compounds), train a *fresh* adapter on the cumulative mix (new + replay) from the clean frozen base each round. The base never moved, so there is no accumulated adapter drift to forget *from*. This costs more per round (you re-see replay data) but most robustly keeps general ability intact while your task skill is rebuilt cleanly.

LoRA isolation is not a *cure* — the adapter still forgets within itself if its data is skewed. It is a *blast-radius* control: it guarantees the base's general language ability survives, and gives you a clean unit to test, keep, or discard per round.

### 4. "Stay near the previous model" — the regularization family (intuition only)

There is a whole family of techniques whose one-line idea is: *add a force that pulls the model back toward what it was before this round, so it can only drift so far.* You will see two names; you do not need their math.

- **KL-style penalties.** A **KL** penalty measures how far the model's output distribution has moved from a frozen reference copy of itself and adds that distance to the loss as a cost. The model then optimizes two things at once: "fit the new data" *and* "don't stray too far from who you were." You have met this idea twice already — DPO's reference model and `beta` leash (*Ch26*), and GRPO's `beta` KL coefficient (*Ch28 - GRPO*, where `beta=0` means no leash and raising it keeps the model near the reference). The same lever applies to continual SFT in spirit.
- **EWC (Elastic Weight Consolidation).** Not all weights matter equally for old skills. EWC estimates *which* weights were most important for previously-learned behavior and makes those specific weights "stiffer" — harder to move — while leaving the rest free. Like the support agent protecting a few core billing habits as non-negotiable while staying flexible on everything else.

We keep these at the intuition level deliberately. For a single-GPU, narrow-task pipeline, **replay plus gentle updates plus a good canary gets most of the benefit at a fraction of the complexity** — the same reason we preferred DPO over the full RLHF stack in *Ch26*. Reach for explicit KL/EWC machinery only if replay alone is not holding the line; treat it as an advanced add-on, not a default.

### 5. Knowing when to retrain from base instead of continuing

Sometimes the right move is to stop continuing and start over from the clean base model.

Continual rounds accumulate cruft: small drifts, replay data that no longer reflects your current world, hyperparameter choices that made sense five rounds ago. Periodically — or when the canary shows a regression you cannot replay your way out of — the cleanest fix is to **retrain a fresh adapter from the frozen base on your *current* full data mix.** Think of it as defragmenting: keep the accumulated *data* (your real asset), throw away the accumulated *weight drift*.

Signs it is time to retrain from base rather than continue:

- A capability regressed on the canary and bumping the replay ratio did not bring it back.
- You have changed something fundamental — the base model version, the schema's spirit, the data distribution — such that the old adapter is now built on stale assumptions.
- You have simply done many rounds (the *Ch32* cadence will give you a feel for "many") and want a clean baseline to compare drift against.

This is cheap insurance precisely because of everything you built earlier in the book: your data is versioned, your training script is one command, and the canary tells you immediately whether the from-base rebuild is actually better. Continuing is the fast default; rebuilding from base is the reset button you should not be afraid to press.

---

## A worked multi-round example

Let's make forgetting visible, then fix it. We *simulate* several rounds so the code runs anywhere in seconds — the structure is identical to a real pipeline, but the round step is a stand-in that mimics how per-capability F1 behaves under skewed training with and without replay. (Swap in a real `SFTTrainer` round plus an Unsloth/vLLM `generate_fn`, and the canary harness above is unchanged.)

The scenario is realistic: rounds 1–6 each bring a wave of new `decision`-heavy meeting-transcript data, with little `relationship` content. Without replay, `relationship` extraction — well-learned early — should quietly rot while `decision` improves.

```python
# worked_example.py
# Simulate continual rounds and watch the canary. First WITHOUT replay (forgetting
# appears), then WITH replay (forgetting is contained). The "model" here is a tiny
# simulator of per-capability F1 so this file runs with no GPU; the harness, canary,
# and compare logic are the SAME ones you'd use on a real model.

import random

CAPABILITIES = ["preference", "fact", "decision", "relationship", "empty_case", "format"]


def simulate_round(scores, replay_ratio):
    """
    Update per-capability F1 for one round of DECISION-heavy training.
    - The reinforced capability ('decision') improves.
    - Un-rehearsed capabilities DRIFT DOWN, scaled by how little replay protects them.
    - Replay reduces the drift proportionally (more replay -> less forgetting).
    """
    new = dict(scores)
    for cap in CAPABILITIES:
        if cap == "decision":
            new[cap] = min(0.95, scores[cap] + 0.03)        # new data sharpens this
        else:
            # Forgetting pressure is strong without replay, dampened by replay_ratio.
            drift = 0.06 * (1 - replay_ratio / 0.20)         # 0.20 replay -> ~no drift
            # 'relationship' is rarest in the new data, so it forgets fastest.
            if cap == "relationship":
                drift *= 1.7
            new[cap] = max(0.30, round(scores[cap] - max(0.0, drift), 3))
    return new


def run_simulation(label, replay_ratio, rounds=6):
    # All capabilities start healthy after the initial SFT (Ch15) + eval (Ch18).
    scores = {c: 0.82 for c in CAPABILITIES}
    scores["decision"] = 0.80
    print(f"\n######## {label} (replay_ratio={replay_ratio:.0%}) ########")
    print("round " + "".join(f"{c[:5]:>8}" for c in CAPABILITIES))
    print("init  " + "".join(f"{scores[c]:>8.2f}" for c in CAPABILITIES))
    for r in range(1, rounds + 1):
        scores = simulate_round(scores, replay_ratio)
        print(f"  {r}   " + "".join(f"{scores[c]:>8.2f}" for c in CAPABILITIES))
    return scores


# --- Run 1: NO replay. Watch 'relationship' (and others) decay. ---
final_no_replay = run_simulation("NO REPLAY", replay_ratio=0.0)

# --- Run 2: ~20% replay (the Ch32 default). Watch the decay stop. ---
final_replay = run_simulation("WITH ~20% REPLAY", replay_ratio=0.20)

print("\n######## SIDE BY SIDE (final round) ########")
print(f"{'capability':<14}{'no replay':>11}{'20% replay':>12}")
for c in CAPABILITIES:
    print(f"{c:<14}{final_no_replay[c]:>11.2f}{final_replay[c]:>12.2f}")
```

Running it prints two trajectories. The **no-replay** run looks like this in shape (your exact numbers depend on the simulator, but the *story* is the point):

```
######## NO REPLAY (replay_ratio=0%) ########
round    prefe    fact   decis   relat   empty   forma
init      0.82    0.82    0.80    0.82    0.82    0.82
  1       0.76    0.76    0.83    0.72    0.76    0.76
  2       0.70    0.70    0.86    0.62    0.70    0.70
  3       0.64    0.64    0.89    0.51    0.64    0.64
  4       0.58    0.58    0.92    0.41    0.58    0.58
  5       0.52    0.52    0.95    0.31    0.52    0.52
  6       0.46    0.46    0.95    0.30    0.46    0.46
```

Read it like a coal-mine canary. `decision` climbs because every round feeds it. Everything else slides — and `relationship`, the rarest in the new data, falls off a cliff from 0.82 to 0.30. **No round looks catastrophic on its own** (each step down is small), which is exactly why forgetting is dangerous: the per-round drop hides under your tolerance, but six rounds compound into a disaster. Your `canary_compare.py` would have flagged a regression as early as round 1 — that is the alarm doing its job.

The **~20% replay** run tells the opposite story:

```
######## WITH ~20% REPLAY (replay_ratio=20%) ########
round    prefe    fact   decis   relat   empty   forma
init      0.82    0.82    0.80    0.82    0.82    0.82
  1       0.82    0.82    0.83    0.82    0.82    0.82
  ...     (rehearsed capabilities hold steady) ...
  6       0.82    0.82    0.95    0.82    0.82    0.82
```

`decision` still improves — the new data still teaches what it should — but the rehearsed capabilities *hold their ground* instead of decaying. Replay kept the old skills warm while the new skill grew. That is the entire thesis of the chapter in two tables: forgetting is real and compounding, and a modest, stratified replay mix (per *Ch32*) is the cheapest, most effective lever against it.

In a real pipeline you would replace `simulate_round` with: assemble the round's data via `build_round_dataset` (replay included), run one short `SFTTrainer` round at a reduced learning rate, then call `run_canary` + `report_round` on the resulting checkpoint. The numbers would be noisier and the curves less tidy, but the shape — naked rounds decay, replayed rounds hold — is exactly what you will see.

---

## Being honest: you manage it, you don't cure it

It would be satisfying to end with "and that is how you eliminate catastrophic forgetting." But that is not true, and the charter asks us to be honest about it.

Every technique here is a *tradeoff*, not a cure:

- **Replay** costs training budget — every replay row is a new-data row you are not training on — and it only protects capabilities you actually sample into the mix. Forget to rehearse `relationship` and `relationship` still rots, replay or not.
- **Gentler updates** protect old skills by *slowing* the acquisition of new ones. Sometimes you genuinely need the model to change fast, and then you accept more forgetting and lean on the canary.
- **LoRA isolation** protects the frozen base's general ability completely, but your *task* adapter can still forget within itself.
- **KL/EWC-style anchoring** buys stability at the cost of complexity and, again, learning speed.
- **Retraining from base** throws away accumulated drift but costs a full rebuild and only helps if your underlying data is healthy.

The realistic goal is not zero forgetting. It is **forgetting that is small, slow, and caught**. You will accept a point or two of drift on a rarely-used capability in exchange for fast learning on the capability that matters this quarter — *as a deliberate decision*, made because the canary showed you the cost, not as a surprise discovered by an annoyed user. A continual-learning system that forgets a little, visibly, on purpose, is healthy. One that forgets a lot, invisibly, by accident, is the failure mode this chapter exists to prevent.

The discipline, not the cleverness, is what saves you: a frozen canary run every round, per-capability scores diffed against last round, a regression treated as a deploy blocker, and replay tuned to the *Ch32* range. Do that, and your living model stays alive in the way that matters — getting better at the new without quietly getting worse at the old.

---

## Recap

- **Catastrophic forgetting** is the model getting quietly *worse* at old capabilities as round after round of training pulls its weights toward new data. Over many rounds it compounds invisibly — no single round looks catastrophic.
- It happens because the optimizer only ever sees the current batch, the model has no memory of past training data, and nothing pins old behavior in place. **LoRA** protects the frozen base's general ability but not your task adapter's learned skill.
- **Measure it with a frozen canary set**: small, stratified by capability, never edited, never trained on. Score it every round, break the score down per capability, diff against last round, and treat a regression as a blocker — not a log line.
- **Mitigate it**, in order of reach: (1) **replay** old/general data at ~10–30% (default ~20%, per *Ch32*), stratified and drawn from a separate reservoir; (2) **gentler updates** — lower learning rate, fewer epochs; (3) **LoRA isolation** — per-concern adapters, deliberate merges, optionally rebuild the adapter from the clean base each round; (4) the **stay-near-the-previous-model** family (KL/EWC) at the intuition level; (5) **retrain from base** when drift accumulates or the canary won't recover.
- The worked simulation made it concrete: naked rounds let `relationship` decay from 0.82 to 0.30 while `decision` climbed; ~20% replay held the old capabilities steady while still improving the new one.
- You cannot eliminate forgetting; you manage it. Aim for forgetting that is small, slow, and *caught* — a deliberate tradeoff you can see, not a surprise your users find first.

## Next

**Ch34** continues Part 8 by turning these per-round safeguards into a full automated loop — scheduling rounds, gating promotions on the canary, and rolling back a bad round — so the living model runs itself with you watching the dashboard instead of running every step by hand.
