# Chapter 30 - The Continual Learning Loop as a System

Everything before this point in the book taught you to do one thing well: take a base model, feed it good data, train it, evaluate it, and ship the result. You ran that pipeline once — maybe a few times while you were iterating — and you ended up with a fine-tuned memory extractor that turns raw conversations into clean `[{text, type, entities}]` JSON.

That single pass is a **photograph**. It is a sharp capture of one moment: the data you had, the failure modes you knew about, the conversations your users were sending the week you collected your training set. A photograph is genuinely useful — you can frame it and ship it to production. But the world keeps moving after the shutter clicks, and a photograph does not.

A continual learning system is not a photograph. It is a **living process** — closer to a person who keeps showing up to work, notices what is changing, learns from this week's mistakes, and is a little better next month than they were this month. The model is no longer the deliverable; the *loop that keeps producing better models* is the deliverable. *Ch23 - Continual Learning and Scaling Up* sketched this idea and gave you a first cron job. This chapter — the opening of Part 8 — turns that sketch into a real **architecture**: named components, clear interfaces, and an honest account of what is hard about running it.

---

## What you'll learn

- Why a one-shot fine-tune is a *photograph* and a continual system is a *living process* — and why that reframe changes how you build
- The full loop as named components with defined interfaces: **collect → select → label → train → eval-gate → deploy → monitor → repeat**
- For each component: what goes *in*, what comes *out*, and the simplest thing that works
- Runnable Python skeletons — dataclasses, function signatures, and a small run registry — that you could actually drop into a repo, not toy pseudo-code
- Exactly where the Part 7 preference and RL methods (DPO, GRPO) slot into the **train** step, and where Chapters 31–34 deep-dive each component
- The cadence question at a high level — event-driven vs scheduled retraining — with the full treatment deferred to *Ch32 - How Much Data, How Often to Retrain*
- The honest truth: this is mostly software engineering and operational discipline, not ML magic

---

## Concepts you need first

### A pipeline is a line; a loop is a circle that feeds itself

In *Ch13 - Creating Your Training Data with Synthetic Generation* through *Ch22 - Serving Your Model and Using It in an App*, you built a **pipeline**: data goes in one end, a deployed model comes out the other. It is a line — a beginning and an end, and when you reach the end, you are done.

A **loop** wires the output end back to the input end. The model you deployed produces traffic; that traffic becomes raw material for the next round of data; that data trains the next model; that model deploys and produces more traffic. The crucial difference is the *feedback edge* — the arrow from "deployed and running" back to "collecting data." That one arrow turns a line into a circle, and a circle is what lets the system improve without you starting over each time.

**One-line definition:** A continual learning loop is a pipeline whose output (a running model and its production traffic) is automatically routed back to become the input (training data) for the next training round, gated by evaluation so a worse model never replaces a better one.

### Why "automatically" is the hard word in that sentence

You already know every individual step. You generated data (Ch13), trained (Ch15), evaluated (Ch18), deployed (Ch22). The hard part of continual learning is almost never the machine learning. It is the *plumbing and the discipline*: capturing traffic without leaking user data, deciding which logged conversations are worth labeling, refusing to deploy a regression even when you are impatient, knowing how to roll back at 2 a.m. when the new model is worse, and keeping a paper trail of which model came from which data.

We will be blunt throughout: **a continual learning system is mostly ordinary software engineering wearing an ML hat.** If you have shipped a backend service, you already have most of the skills. What is new is the eval-gate discipline and the forgetting hazard — and even those are more "be careful and measure" than "invent new math."

### The component vocabulary

The rest of the chapter uses these names. Skim them now; we define each properly in its own section.

- **collect** — capture production inputs, outputs, and feedback into durable logs.
- **select / curate** — choose which logged events are worth turning into training data (full treatment: *Ch31 - Selecting and Curating Data*).
- **label / generate** — produce gold answers for the selected inputs, via a teacher model or humans.
- **train** — run SFT or a preference/RL method (DPO, GRPO from Part 7) on the new + replayed data.
- **eval-gate** — score the candidate model on a held metric and a frozen canary set; pass or fail (metrics from *Ch18 - Did It Actually Work?*).
- **deploy** — roll the passing model out carefully (canary/shadow, then promote or roll back — *Ch34 - Production Ops*).
- **monitor** — watch the live model and decide when the loop should run again (cadence: *Ch32 - How Much Data, How Often to Retrain*).

---

## The loop, drawn once

Here is the whole architecture on one page. Read it top to bottom, then follow the arrow back up.

```
            ┌──────────────────────────────────────────────────────┐
            │                                                        │
            ▼                                                        │
   ┌─────────────────┐                                              │
   │   1. COLLECT    │  production logs + user/system feedback      │
   │  in:  live req/resp        out: append-only event log          │
   └────────┬────────┘                                              │
            ▼                                                        │
   ┌─────────────────┐                                              │
   │  2. SELECT /    │  pick the events worth learning from  (Ch31) │
   │     CURATE      │  in:  raw events           out: candidates   │
   └────────┬────────┘                                              │
            ▼                                                        │
   ┌─────────────────┐                                              │
   │  3. LABEL /     │  make gold answers (teacher LLM or humans)   │
   │     GENERATE    │  in:  candidates           out: labeled rows │
   └────────┬────────┘                                              │
            ▼                                                        │
   ┌─────────────────┐                                              │
   │   4. TRAIN      │  SFT, or DPO / GRPO from Part 7              │
   │                 │  in:  new rows + replay    out: candidate    │
   └────────┬────────┘                          model adapter       │
            ▼                                                        │
   ┌─────────────────┐                                              │
   │  5. EVAL-GATE   │  Ch18 metrics + frozen canary set            │
   │                 │  in:  candidate model      out: PASS / FAIL  │
   └────────┬────────┘                                              │
       PASS │   └── FAIL ──► discard candidate, alert, do not ship  │
            ▼                                                        │
   ┌─────────────────┐                                              │
   │  6. DEPLOY      │  canary / shadow → promote or rollback (Ch34)│
   │                 │  in:  passing model        out: live model   │
   └────────┬────────┘                                              │
            ▼                                                        │
   ┌─────────────────┐                                              │
   │  7. MONITOR     │  watch quality; decide when to run again     │
   │                 │  (event-driven vs scheduled — Ch32)          │
   └────────┬────────┘                                              │
            └──────────────── repeat ─────────────────────────────►─┘
```

Notice three things before we build it.

First, the **eval-gate is the only place an arrow can leave the loop early**. Every other step hands off to the next; only the gate can say "stop, this candidate is not good enough, throw it away." That asymmetry is the whole safety story. A loop that always deploys whatever it trained will eventually deploy a regression and not notice.

Second, the components are **independent stages with simple, file-shaped interfaces**. Each stage reads files and writes files — collect writes an event log, select reads it and writes candidates, label reads candidates and writes labeled rows. That is deliberate: file-shaped interfaces mean any stage can be rerun, inspected, or replaced without touching the others, and a failure leaves a debuggable artifact behind.

Third, **the loop has a memory of itself**. Every run should record *which data produced which model and what the gate said*. We build a tiny registry for that at the end, because without it you cannot answer the question you will eventually be asked: "the model got worse last Tuesday — what changed?"

---

## A shared vocabulary in code

Before the components, let's pin down the data that flows between them as plain Python dataclasses. These are the "wire format" of the loop — boring on purpose, because boring interfaces are what let the interesting parts stay decoupled. Everything here is standard library; no framework required.

```python
# loop_types.py
# The shared data shapes that flow between loop stages.
# Deliberately minimal: a dataclass per artifact, JSON-serializable.

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time


# The pinned memory schema from the charter. A single memory object is
# {"text": ..., "type": ..., "entities": [...]}; the model emits a JSON
# array of zero or more of these. We do NOT add fields to this schema.
Memory = dict  # e.g. {"text": "...", "type": "preference", "entities": ["Sarah"]}


@dataclass
class CollectedEvent:
    """One production inference, captured by the serving layer (stage 1)."""
    id: str                       # uuid, stable across stages
    ts: float                     # unix time of the request
    conversation: str             # the raw input the model received
    raw_output: str               # the model's text output, pre-parse
    memories: Optional[list]      # parsed JSON array, or None if parse failed
    parse_ok: bool                # did raw_output parse as valid JSON?
    feedback: Optional[str] = None  # optional user/system signal: "good" | "bad" | None
    model_version: str = "unknown"  # which model produced this (set by the server)


@dataclass
class Candidate:
    """An event chosen for the next training round (output of stage 2)."""
    id: str
    conversation: str
    reason: str                   # why it was picked: "parse_fail" | "negative_feedback" | "sampled"


@dataclass
class LabeledRow:
    """A conversation paired with a gold answer (output of stage 3)."""
    conversation: str
    memories: list                # the gold JSON array, [{text, type, entities}, ...]
    source: str                   # "teacher" | "human" | "original_synthetic"


@dataclass
class GateResult:
    """The eval-gate's verdict on a candidate model (output of stage 5)."""
    passed: bool
    primary_metric: float         # e.g. F1 on the held test set
    canary_metric: float          # F1 on the frozen anti-forgetting set
    detail: dict = field(default_factory=dict)


def write_jsonl(path: str, rows: list) -> None:
    """Serialize a list of dataclass instances (or dicts) to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            obj = asdict(r) if hasattr(r, "__dataclass_fields__") else r
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file back into a list of plain dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
```

That is the entire shared library. Each stage below imports from `loop_types` and does exactly one transformation: dicts in, dicts out. Now let's walk the loop.

---

## Stage 1 — Collect

**In:** every live request/response your serving layer handles, plus any feedback signal you can get. **Out:** an append-only event log on disk (one JSONL file per day).

**Intuition first.** Collection is the security camera of your system. It is not glamorous and you mostly ignore the footage — until something interesting happens, at which point you are deeply grateful it was recording. The discipline is to record everything relevant and cheaply, so that when you later decide what is worth learning from (Stage 2), you have the raw material. You cannot curate footage you never captured.

The simplest thing that works is to log each inference to a daily JSONL file from inside the serving wrapper you built in *Ch22* — restated here against our `CollectedEvent` dataclass so it produces the wire format the rest of the loop expects.

```python
# collect.py — drop into the serving layer from Ch22.
import uuid, time
from pathlib import Path
from loop_types import CollectedEvent, asdict
import json

LOG_DIR = Path("data/events")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_event(conversation: str, raw_output: str, parsed: list | None,
              model_version: str, feedback: str | None = None) -> None:
    """Append one inference event to today's log. Called per request."""
    event = CollectedEvent(
        id=str(uuid.uuid4()),
        ts=time.time(),
        conversation=conversation,
        raw_output=raw_output,
        memories=parsed,
        parse_ok=parsed is not None,   # None means the model emitted invalid JSON
        feedback=feedback,             # e.g. a thumbs-down from the product UI
        model_version=model_version,   # so we can attribute behavior to a model later
    )
    path = LOG_DIR / f"events_{time.strftime('%Y-%m-%d')}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
```

**Two things that are easy to get wrong, stated plainly:**

1. **Privacy is not optional.** If you log real user conversations, you are now holding user data, with everything that implies legally and ethically. The minimal responsible posture is to anonymize before writing (strip or placeholder personal names) or collect explicit consent. We do not build a full anonymizer here, but treat it as a blocker before production, not a nice-to-have.
2. **`feedback` is the highest-value field and the easiest to skip.** A thumbs-down in your product UI, a downstream system that detected a contradiction, an edit a user made to a stored memory — any of these is a far stronger learning signal than a random sampled row. Wire even a crude `"good"`/`"bad"` signal through to this field; Stage 2 will thank you. *Ch31* goes deep on harvesting feedback as a curation signal.

---

## Stage 2 — Select / curate

**In:** the raw event log. **Out:** a `candidates.jsonl` file of conversations worth labeling. **Deep dive:** *Ch31 - Selecting and Curating Data.*

**Intuition first.** Not all footage is worth reviewing. If you trained on every conversation your system ever saw, you would mostly re-teach it things it already knows while drowning the few instructive examples in noise. Curation is the editor who clips out only the moments that matter: parse failures (the model visibly struggled), thumbs-downs (a human said it was wrong), and a *small* random sample of ordinary successes (so the dataset does not become only failures and skew the model's sense of normal). This is the single highest-leverage stage for quality, which is why it gets its own chapter.

The simplest thing that works is a priority order with a budget: take all the negative-feedback and parse-failure events first, then fill the remaining budget with a random sample of clean successes.

```python
# select.py
import random
from loop_types import read_jsonl, write_jsonl, Candidate

def select_candidates(events: list[dict], budget: int = 200,
                      sample_success_frac: float = 0.3) -> list[Candidate]:
    """
    Priority: explicit negative feedback > parse failures > random successes.
    Returns at most `budget` candidates. This is a starting heuristic; Ch31
    replaces it with smarter curation (clustering, novelty, difficulty).
    """
    neg     = [e for e in events if e.get("feedback") == "bad"]
    fails   = [e for e in events if not e["parse_ok"] and e.get("feedback") != "bad"]
    success = [e for e in events if e["parse_ok"] and e.get("feedback") != "bad"]

    picked: list[dict] = []
    for bucket, reason in [(neg, "negative_feedback"), (fails, "parse_fail")]:
        take = bucket[: max(0, budget - len(picked))]
        picked += [{"e": e, "reason": reason} for e in take]

    # Fill remaining budget with a random sample of ordinary successes so the
    # training set reflects normal traffic, not only failure cases.
    remaining = budget - len(picked)
    if remaining > 0 and success:
        n = min(remaining, int(budget * sample_success_frac), len(success))
        picked += [{"e": e, "reason": "sampled"} for e in random.sample(success, n)]

    random.shuffle(picked)
    return [Candidate(id=p["e"]["id"], conversation=p["e"]["conversation"],
                      reason=p["reason"]) for p in picked]


if __name__ == "__main__":
    events = read_jsonl("data/events/today.jsonl")   # or glob a date range
    cands = select_candidates(events, budget=200)
    write_jsonl("data/candidates.jsonl", cands)
    print(f"Selected {len(cands)} candidates from {len(events)} events.")
```

The budget is deliberately conservative. Continual learning fails far more often from too much mediocre data than from too little good data. When in doubt, label fewer, better rows. (See the charter's data table: a few hundred clean rows per round is plenty for a narrow LoRA task.)

---

## Stage 3 — Label / generate

**In:** `candidates.jsonl` (conversations with no gold answer yet). **Out:** `labeled.jsonl` of `{conversation, memories, source}` rows in the pinned schema.

**Intuition first.** A curated conversation is a question with no answer key. Labeling writes the answer key. Two ways, not mutually exclusive: a stronger **teacher model** produces the gold answer (fast, cheap, scales — the synthetic-generation move from *Ch13*, now pointed at real traffic), or **humans** do (slow, expensive, but the ground truth when the teacher is unreliable on your hardest cases). The right default is teacher-with-a-judge for volume, humans for the cases the judge flags as uncertain or for periodic audits.

Here is the teacher path, reusing your Ch13 helpers. The labeler must emit the pinned schema exactly — `{text, type, entities}`, no extra fields — because Stage 4 will feed these straight into training and any drift here becomes drift in the model.

```python
# label.py
import json
from loop_types import read_jsonl, write_jsonl, LabeledRow

# Reuse the exact teacher call + quality judge you built in Ch13.
# (Rename the import if you saved them under different module names.)
from generate import call_teacher          # str(prompt) -> str(raw model text)
from judge import judge_row                # (conversation, memories) -> bool

# The charter's system prompt IS the labeling instruction — reuse it verbatim
# so the gold answers match exactly what the student model is asked to produce.
from prompts import SYSTEM_PROMPT


def label_one(conversation: str) -> list | None:
    """Ask the teacher for gold memories; return the parsed array or None."""
    # Same contract the student model is trained on: system prompt + the
    # conversation as the user turn, expecting a JSON array back.
    raw = call_teacher(SYSTEM_PROMPT, conversation)
    try:
        memories = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    # Enforce the pinned schema. Reject anything that isn't a list of objects
    # with exactly the three allowed keys.
    if not isinstance(memories, list):
        return None
    allowed = {"text", "type", "entities"}
    for m in memories:
        if not isinstance(m, dict) or set(m.keys()) != allowed:
            return None
    return memories


def label_candidates(in_path: str, out_path: str) -> int:
    rows, accepted = read_jsonl(in_path), []
    for c in rows:
        memories = label_one(c["conversation"])
        if memories is None:
            continue                                   # bad/teacher-malformed
        if not judge_row(c["conversation"], memories): # Ch13 quality judge
            continue                                   # judge rejected it
        accepted.append(LabeledRow(conversation=c["conversation"],
                                   memories=memories, source="teacher"))
    write_jsonl(out_path, accepted)
    print(f"Labeled {len(accepted)}/{len(rows)} candidates (judge-accepted).")
    return len(accepted)


if __name__ == "__main__":
    label_candidates("data/candidates.jsonl", "data/labeled.jsonl")
```

The empty-list case is real and important: `[]` is a valid label. A conversation with no memorable facts should produce `memories: []`, and training on those rows is what teaches the model *restraint* — when not to invent memories. Do not silently drop empty-label rows.

---

## Stage 4 — Train

**In:** the new `labeled.jsonl` plus a replay sample of prior data. **Out:** a candidate model adapter on disk.

**Intuition first.** This stage reuses everything from Parts 4 and 7. There is no new training technique here — there is a *choice* of technique, and a single discipline (replay) that keeps the model from forgetting.

**The replay discipline.** The core danger of training round after round is **catastrophic forgetting**: weight updates that make the model better on this week's data quietly overwrite what made it good last month — the musician who practices only the new piece forgets the old repertoire. The cheap, effective mitigation is **replay**: never train on new data alone; always mix in a sample of prior/general data so old skills keep getting rehearsed. The charter's rule of thumb is ~10–30% prior data (default ~20%), plus a frozen canary set (Stage 5) as your forgetting alarm.

```python
# build_training_set.py
import random
from loop_types import read_jsonl, write_jsonl

def build_round_dataset(new_path: str, prior_path: str, out_path: str,
                        replay_frac: float = 0.2, max_new: int = 500) -> int:
    """
    Mix this round's new rows with a replay sample of prior rows.
    replay_frac ~0.2 means the final set is ~20% prior data — the charter's
    default for keeping old skills alive without drowning out the new signal.
    """
    new   = read_jsonl(new_path)
    prior = read_jsonl(prior_path)

    if len(new) > max_new:                 # don't let one round dominate
        new = random.sample(new, max_new)

    # Size the replay sample so prior data is ~replay_frac of the total.
    # If new is N rows and we want prior = f*(N+prior), then prior = f/(1-f)*N.
    n_replay = min(len(prior), int(replay_frac / (1 - replay_frac) * len(new)))
    replay = random.sample(prior, n_replay) if prior else []

    merged = new + replay
    random.shuffle(merged)
    write_jsonl(out_path, merged)
    print(f"Round dataset: {len(new)} new + {len(replay)} replay = {len(merged)}")
    return len(merged)
```

**Where Part 7 slots in.** The "train" step is a pluggable method; your choice depends on what improvement this round needs:

- **SFT** (*Ch15*) is the default — use it when new data adds *coverage* (new topics, slang, conversation shapes the model hasn't seen). Build TRL conversational rows (`{"messages": [system, user, assistant]}`) from your `LabeledRow`s and run `SFTTrainer` as in Ch15.
- **DPO** (*Ch26 - DPO*) is for *quality* rounds where you have, or can build, preference pairs `(prompt, chosen, rejected)` — the model produces valid-but-not-best answers and you want to teach the *direction* from worse to better. Your loop is a natural pair factory: a thumbs-down event paired with the teacher's corrected label is exactly a `(rejected, chosen)` pair. Run `DPOTrainer(model=..., ref_model=None, args=DPOConfig(...), train_dataset=..., processing_class=tokenizer, peft_config=...)`.
- **GRPO** (*Ch28 - GRPO*) is for rounds where quality is a *checkable reward* you want the model to explore toward — valid JSON, entities that appear in the source, no hallucinated facts. Run `GRPOTrainer(model=..., reward_funcs=<your reward fn>, args=GRPOConfig(...), train_dataset=..., processing_class=tokenizer)`. *Ch29 - Choosing a Method* is the decision guide; this loop just calls whichever you picked.

None of this changes the interface: whatever method you run, a training dataset goes in and a candidate adapter directory comes out. We launch it as a subprocess so the orchestrator stays method-agnostic:

```python
# train.py — method-agnostic launcher for one round.
import subprocess, time
from pathlib import Path

def run_training(dataset_path: str, method: str = "sft") -> str:
    """
    Launch the appropriate Part 3-7 training script as a subprocess and return
    the output model dir. Each script (train_sft.py / train_dpo.py /
    train_grpo.py) is the runnable code from its own chapter — unchanged.
    """
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"models/memory-extractor-{run_id}"
    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)

    script = {"sft": "train_sft.py",      # Ch15
              "dpo": "train_dpo.py",      # Ch26
              "grpo": "train_grpo.py"}[method]   # Ch28

    # Restart from the BASE model each round (not the previous adapter): replaying
    # the merged dataset from a clean base is more stable than stacking adapters,
    # whose round-to-round errors compound. (Ch23 makes this case in full.)
    subprocess.run(["python", script, "--dataset", dataset_path,
                    "--output-dir", out_dir], check=True)
    return out_dir
```

---

## Stage 5 — Eval-gate

**In:** the candidate model directory. **Out:** a `GateResult` — `passed: True/False` — and nobody downstream sees the model unless `passed` is `True`. **Metrics:** *Ch18 - Evaluating Memory Extraction.*

**Intuition first.** The gate is the bouncer at the door. A model does not get into production by showing up; it gets in by passing the test. This is the single most important component for safety — the only place the loop can refuse to ship. A loop without a real gate is an automated way to deploy regressions.

Two measurements answer two different questions:

1. **The primary metric** on a current held-out test set answers *"is this model good at today's traffic?"* — the F1 from *Ch18* (overlap between predicted and gold memories).
2. **The canary metric** on a *frozen* set answers *"did this model forget what it used to know?"* The canary is hand-labeled once, early, and never changed. It is your forgetting alarm: a candidate can score beautifully on today's test set while quietly collapsing on the canary — catastrophic forgetting caught red-handed. That is why replay (Stage 4) and the canary (here) are a matched pair.

```python
# gate.py
import json, subprocess
from loop_types import GateResult

PRIMARY_MIN = 0.75   # F1 floor on the current test set (set from your Ch18 baseline)
CANARY_MIN  = 0.80   # F1 floor on the frozen anti-forgetting set
CANARY_MAX_DROP = 0.03  # also fail if canary fell >3 pts vs the live model

def _eval(model_dir: str, test_set: str) -> float:
    """Call the Ch18 evaluation script; return its F1. The script writes
    {"f1": ..., "precision": ..., "recall": ...} to the --output path."""
    subprocess.run(["python", "evaluate.py", "--model-dir", model_dir,
                    "--test-set", test_set, "--output", "eval.json"], check=True)
    with open("eval.json") as f:
        return json.load(f).get("f1", 0.0)

def gate(model_dir: str, live_canary_f1: float | None = None) -> GateResult:
    primary = _eval(model_dir, "data/memories_test.jsonl")    # current traffic
    canary  = _eval(model_dir, "data/canary_frozen.jsonl")    # never changes

    passed = primary >= PRIMARY_MIN and canary >= CANARY_MIN
    # Regression guard: even above the floor, refuse a big canary drop vs live.
    if live_canary_f1 is not None and (live_canary_f1 - canary) > CANARY_MAX_DROP:
        passed = False

    result = GateResult(passed=passed, primary_metric=primary, canary_metric=canary,
                        detail={"primary_min": PRIMARY_MIN, "canary_min": CANARY_MIN})
    print(f"GATE {'PASS' if passed else 'FAIL'} | "
          f"primary F1={primary:.3f} canary F1={canary:.3f}")
    return result
```

One honest caveat from *Ch23*: your *current* test set goes stale (**evaluation drift**). A model can look great on a test set sampled from six-month-old traffic and be quietly worse on today's. The fix is operational, not algorithmic: every 4–8 weeks, hand-label 50–100 recent rows into the current test set and retire the oldest — while the *canary stays frozen forever*. The two sets have different jobs and lifecycles; confusing them defeats both. *Ch33 - Catastrophic Forgetting Over Many Rounds* is dedicated to the frozen canary and forgetting alarm.

---

## Stage 6 — Deploy

**In:** a model that passed the gate. **Out:** that model serving live traffic, with a way back. **Deep dive:** *Ch34 - Production Ops.*

**Intuition first.** Passing the gate means the model is good on your *test data*, not yet that it is good on *live traffic* — no test set is the world. So you do not flip the whole switch at once. You let the new model serve a small, observed slice first — a **canary** (a few percent of real traffic) or a **shadow** (the new model runs on the same requests as the old one, but only the old one's answers are returned, so you compare without risk). If the slice looks healthy, **promote** to 100%. If it looks worse, **roll back** — instantly revert to the previous model.

The simplest thing that works, from *Ch23*: serve from a `models/current` symlink, keep a `models/previous` symlink, and make promotion and rollback two-line operations. The gotcha bears repeating because it bites everyone once: **updating the symlink does not reload a running server.** You must restart (or signal) the serving process, or it will keep serving the old weights for days after a "successful" deploy.

```python
# deploy.py
import os, subprocess
from pathlib import Path

def promote(model_dir: str) -> None:
    """Point `current` at the new model, keep `previous` for instant rollback."""
    cur = Path("models/current")
    if cur.exists():
        prev = Path("models/previous")
        if prev.is_symlink() or prev.exists():
            prev.unlink()
        os.symlink(os.readlink(cur), prev)   # remember what we're replacing
        cur.unlink()
    os.symlink(os.path.abspath(model_dir), cur)
    _reload_server()                          # <-- the step everyone forgets

def rollback() -> None:
    """Swap current back to previous and reload. The 2 a.m. button."""
    cur, prev = Path("models/current"), Path("models/previous")
    target = os.readlink(prev)
    cur.unlink(); os.symlink(target, cur)
    _reload_server()

def _reload_server() -> None:
    # Restart the serving process so it loads the new weights. Match to your setup;
    # WITHOUT this, the symlink changes but the live model does NOT.
    subprocess.run(["systemctl", "restart", "vllm"], check=False)
```

Canary/shadow routing, gradual traffic ramps, and automated rollback triggers are *Ch34*'s job. For your first loop, "promote to a symlink, keep one rollback step, restart the server" is a completely respectable starting point.

---

## Stage 7 — Monitor, and the cadence question

**In:** the live model and its fresh traffic. **Out:** the decision *when to run the loop again* — which closes the circle back to Stage 1. **Deep dive:** *Ch32 - How Much Data, How Often to Retrain.*

**Intuition first.** Once a model is live, it immediately starts generating the very logs Stage 1 collects. Monitoring watches those logs for trouble — a rising parse-failure rate, a spike in thumbs-downs, a drift in incoming conversations — and decides whether it is time for another round. This is what makes the loop a loop rather than a one-time pipeline with extra steps.

The big question monitoring answers is **cadence**, with two high-level philosophies (Ch32 does the full treatment with thresholds and code):

- **Scheduled retraining.** Run on a fixed clock — every Sunday at 2 a.m. The win is predictability: you know when training happens and when to be on call. The cost is retraining even when nothing changed (wasteful) and waiting up to a week to react when something breaks (slow). Scheduled is the right *first* cadence: simple, boring, and you learn your system's rhythm before automating cleverness on top.
- **Event-driven retraining.** Trigger when a *signal* crosses a threshold — parse-failure rate doubles, negative feedback spikes, enough fresh candidates accumulate. The win is responsiveness; the cost is complexity and the risk of thrashing on noise. Most mature systems end up *hybrid*: a scheduled floor ("at least monthly") plus event-driven triggers for emergencies.

A minimal monitor that supports both is just a function over recent events:

```python
# monitor.py
from loop_types import read_jsonl

def should_retrain(recent_events: list[dict], min_candidates: int = 50,
                   parse_fail_alarm: float = 0.10) -> tuple[bool, str]:
    """Decide whether to kick off a round. Returns (trigger?, reason).
    This is the seed of Ch32's cadence logic — start simple, here."""
    n = len(recent_events)
    if n == 0:
        return False, "no traffic"
    fail_rate = sum(1 for e in recent_events if not e["parse_ok"]) / n
    n_negative = sum(1 for e in recent_events if e.get("feedback") == "bad")

    if fail_rate >= parse_fail_alarm:               # event-driven: emergency
        return True, f"parse-fail rate {fail_rate:.0%} >= {parse_fail_alarm:.0%}"
    if (sum(1 for e in recent_events if not e["parse_ok"]) + n_negative) >= min_candidates:
        return True, f"accumulated >= {min_candidates} learnable events"
    return False, "nothing changed enough to retrain"
```

The honest guardrail, carried from *Ch23*: a round on too little data is worse than no round. If a trigger fires but the curated candidate count is thin (say under ~20–50 rows), skip the round. A quiet week is allowed to be quiet.

---

## The loop's memory: a tiny run registry

A system you cannot *interrogate* is a system you cannot trust. When someone asks "the model regressed — what changed?", you need to answer: *which data produced which model, what did the gate say, and is it the one currently live?* That is a job for a small registry. SQLite is plenty — no service, no migrations, just a file you can query.

```python
# registry.py — the loop's logbook. One row per training round.
import sqlite3, json, time
from loop_types import GateResult

DB = "loop_registry.db"

def _conn():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, ts REAL, method TEXT, dataset TEXT,
        model_dir TEXT, n_new INTEGER, n_total INTEGER,
        primary_f1 REAL, canary_f1 REAL, passed INTEGER,
        deployed INTEGER DEFAULT 0)""")
    return c

def record_run(run_id: str, method: str, dataset: str, model_dir: str,
               n_new: int, n_total: int, gate: GateResult) -> None:
    """Write one round's full provenance: data in, model out, gate verdict."""
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,0)",
                  (run_id, time.time(), method, dataset, model_dir, n_new, n_total,
                   gate.primary_metric, gate.canary_metric, int(gate.passed)))

def mark_deployed(run_id: str) -> None:
    """Flag which model is actually live (exactly one should be)."""
    with _conn() as c:
        c.execute("UPDATE runs SET deployed = 0")           # clear old flag
        c.execute("UPDATE runs SET deployed = 1 WHERE run_id = ?", (run_id,))

def history(limit: int = 10) -> list[tuple]:
    with _conn() as c:
        return list(c.execute(
            "SELECT run_id, method, primary_f1, canary_f1, passed, deployed "
            "FROM runs ORDER BY ts DESC LIMIT ?", (limit,)))
```

This registry is also where the **canary trend** lives. Plotting `canary_f1` across runs over months is your clearest forgetting detector: a slow downward drift in a frozen metric means the loop is eroding old skills even as it improves on new traffic. That single column, watched over time, is worth more than any clever algorithm.

---

## Wiring it together: one round, orchestrated

Here is the orchestrator that runs one full pass of the loop. Notice it is almost entirely *control flow* — call a stage, check a result, branch — which is the point. The intelligence is in the stages; the orchestrator just enforces the order and the gate.

```python
# run_round.py — one pass through the loop. Schedule this (cron) or trigger it
# from monitor.should_retrain(). This is the whole system, assembled.
import time, glob
from loop_types import read_jsonl
import select, label, build_training_set, train, gate, deploy, registry, monitor

def run_round(method: str = "sft") -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")

    # 1. COLLECT already happened live; just read the recent event logs.
    events = [r for p in glob.glob("data/events/*.jsonl") for r in read_jsonl(p)]

    # 7-as-precondition: should we even run? (cadence — Ch32)
    go, reason = monitor.should_retrain(events)
    if not go:
        print(f"[{run_id}] skipping round: {reason}")
        return

    # 2. SELECT / CURATE  (Ch31)
    cands = select.select_candidates(events, budget=200)
    from loop_types import write_jsonl
    write_jsonl("data/candidates.jsonl", cands)

    # 3. LABEL / GENERATE
    n_new = label.label_candidates("data/candidates.jsonl", "data/labeled.jsonl")
    if n_new < 20:                                    # the thin-data guard
        print(f"[{run_id}] only {n_new} labeled rows — skipping (too thin).")
        return

    # 4. TRAIN  (SFT / DPO / GRPO — Part 7)
    n_total = build_training_set.build_round_dataset(
        "data/labeled.jsonl", "data/prior_train.jsonl", "data/round_train.jsonl")
    model_dir = train.run_training("data/round_train.jsonl", method=method)

    # 5. EVAL-GATE  (Ch18 metrics + frozen canary)
    result = gate.gate(model_dir)
    registry.record_run(run_id, method, "data/round_train.jsonl",
                        model_dir, n_new, n_total, result)
    if not result.passed:
        print(f"[{run_id}] GATE FAILED — discarding candidate, NOT deploying.")
        return                                        # the only safe early exit

    # 6. DEPLOY  (canary/shadow → promote — Ch34)
    deploy.promote(model_dir)
    registry.mark_deployed(run_id)
    print(f"[{run_id}] deployed {model_dir} "
          f"(primary F1={result.primary_metric:.3f}).")

if __name__ == "__main__":
    run_round(method="sft")
```

Schedule that with the same cron one-liner from *Ch23* (`0 2 * * 0 /path/to/run_round.py`), and you have a living system: it collects, curates, labels, trains, gates, and — only if the gate passes — deploys, recording every step. That is the photograph turned into a process.

---

## The honest part: this is engineering, not magic

Worth saying directly, because "continual learning" can sound like the model is doing something autonomous and clever. It is not. Look back at the seven stages and notice how little is machine learning:

- **Collect** is logging.
- **Select** is filtering with a budget.
- **Label** is an API call plus a validity check.
- **Train** is the same SFT/DPO/GRPO you already learned, wrapped in a subprocess call.
- **Eval-gate** is running a test and comparing to a threshold.
- **Deploy** is a symlink and a server restart.
- **Monitor** is counting events and checking a number against a limit.

The ML is one box out of seven, and even that box is code you already wrote. The other six are plumbing, discipline, and provenance. This is not a knock on the approach — it is the *reason it works*. A system built from boring, inspectable, file-shaped stages is one you can debug at 2 a.m.; one that hides everything behind a "self-improving" black box is not. The genuinely-unsolved research problems — catastrophic forgetting, evaluation drift, compounding label noise — are real (we faced them in *Ch23* and revisit them per-component in Chapters 31–34), but your defenses are mundane and effective: replay a slice of old data, keep a frozen canary, gate before you ship, and write down what you did.

A system that retrains on real traffic, refuses to deploy regressions, and keeps a frozen canary as a forgetting alarm will get measurably better over months — more than most production ML systems ever achieve, reached with software-engineering discipline, not ML wizardry.

---

## Common mistakes

**Treating "deploy whatever we trained" as the loop.** The eval-gate is not optional ceremony; it is the only thing standing between you and an automated regression-deployment machine. If you skip it to move faster, you are not running a continual learning system — you are running a random walk through model quality.

**Confusing the canary set with the test set.** The canary is *frozen forever* and answers "did we forget?". The test set is *refreshed* and answers "are we good now?". They have opposite lifecycles. Refreshing the canary defeats its purpose (you lose your forgetting alarm); freezing the test set guarantees evaluation drift. Keep them separate, on purpose.

**Coupling the stages.** The moment "select" reaches into the trainer's internals or "deploy" assumes a particular labeler, you lose the ability to rerun, inspect, or replace a stage. Keep the interfaces file-shaped (dicts in, dicts out) even when it feels verbose. The decoupling is what makes the system debuggable.

**No provenance.** If you cannot answer "which data produced the live model and what did the gate say," you cannot debug a regression. The registry is twenty lines of SQLite. Write it before you need it, not after the model mysteriously got worse.

**Reacting to noise.** An event-driven trigger that fires on every small fluctuation will thrash — retraining constantly on data that hasn't really changed. Pair triggers with thin-data guards and a scheduled floor; let quiet weeks be quiet. (*Ch32* makes this precise.)

---

## Recap

- A one-shot fine-tune is a **photograph**; a continual system is a **living process** that routes its own output back as next round's input. The feedback edge is what turns a line into a loop.
- The architecture is seven named, file-interfaced stages: **collect → select → label → train → eval-gate → deploy → monitor → repeat.** Each is "dicts in, dicts out," so any stage can be rerun, inspected, or swapped.
- **Train** is where Part 7 plugs in: SFT (*Ch15*) for coverage, DPO (*Ch26*) for quality pairs, GRPO (*Ch28*) for checkable-reward exploration — same interface, different method. **Replay** (~10–30% prior data) is the cheap defense against forgetting.
- The **eval-gate** is the only place a candidate can be rejected, and the single most important safety component: Ch18 metrics on a refreshed test set *plus* a F1 floor on a **frozen canary** that is your forgetting alarm.
- **Cadence** (Stage 7) is scheduled vs event-driven, usually hybrid; full detail in *Ch32*. Chapters 31–34 deep-dive curation, cadence, catastrophic forgetting, and production ops respectively.
- This is **mostly software engineering and discipline** — six of seven stages are plumbing, and the one ML box is code you already wrote. That boringness is the feature.

## Next

*Ch31 - Selecting and Curating Data* — the highest-leverage stage. We replace this chapter's simple priority-budget heuristic with real curation: harvesting feedback signals, clustering for novelty, scoring difficulty, and deciding which fraction of the flood of production traffic is actually worth labeling and learning from.
