# Chapter 34 - Production Ops: Monitoring, Versioning, Gating, and Rollback

You have a fine-tuned memory-extraction model serving real traffic. Conversations flow in, JSON memories flow out, and somewhere a user is getting a better assistant because of it. That is a genuine accomplishment — most people who start a fine-tuning project never get a model into production at all.

But "in production" is not a finish line. It is the start of a new job. The model that was excellent on last month's eval set will quietly drift as your users start talking about new topics. A retraining run that looked great in a notebook can ship a regression that breaks JSON parsing for ten percent of requests — and you will not notice until support tickets pile up. Three months from now, someone (possibly you) will ask "which exact model is live right now, what data trained it, and can we roll it back?" and the honest answer will be a shrug.

This chapter is about turning your one good model into a *living system* that improves over many rounds without ever shipping a regression you cannot instantly undo. We will build the unglamorous machinery that separates a demo from a product: logging, dataset and adapter versioning, a model registry, an eval gate, canary deploys, and one-command rollback. None of it is hard. All of it is the difference between "we fine-tuned a model once" and "we run a continually-improving model in production."

---

## What you'll learn

- What to log in production — inputs, outputs, JSON-valid rate, latency, drift signals, and user feedback — and a simple metrics logger you can drop in today
- How to set alert thresholds tied back to the eval metrics from *Ch18 - Did It Actually Work? Evaluating Memory Extraction*
- How to make every training run reproducible by versioning the dataset (content hash + manifest), the config, and the resulting adapter
- How to build a lightweight model **registry** that tracks which adapter is live, its base model, its data version, and its eval scores
- How to write an **eval gate** that promotes a new adapter only if it beats the current one *and* passes a no-forgetting regression check (cross-ref *Ch33 - Catastrophic Forgetting Over Many Rounds*)
- How to run **canary** and **shadow** deploys to validate a new adapter on live traffic before promoting it
- How to **roll back** to the last-good adapter instantly via the registry

---

## Concepts you need first

### Observability: you cannot fix what you cannot see

Observability is just the practice of making a running system explain itself. When the model misbehaves at 2 a.m., you do not want to be reconstructing what happened from memory. You want a log line that says: this conversation came in, this JSON came out, it took 840 milliseconds, it parsed cleanly, and the user kept the memory. With that, debugging is reading. Without it, debugging is guessing.

In *Ch18 - Did It Actually Work?* you measured your model offline, on a held-out test set, before shipping. Production observability is the *same metrics, computed continuously on live traffic*. The held-out test set tells you the model was good last Tuesday. The production logs tell you whether it is good right now.

### Drift: the world moves, your model does not

A fine-tuned model is a snapshot. It learned the distribution of conversations you trained it on, and it is frozen there. But your users keep talking about new things — a product launch, a slang term, a new kind of meeting. The gap between "what the model was trained on" and "what it is now seeing" is called **drift**, and it shows up as a slow decay in quality that no single request makes obvious. Monitoring exists largely to catch drift early, while it is a nudge and not a crisis.

### Reproducibility: a result you cannot rebuild is a rumor

If you cannot regenerate a model from recorded ingredients — the exact data, the exact config, the exact base model — then you do not really know what you have. Reproducibility is what lets you say "v7 is worse than v6, let me diff exactly what changed" instead of "v7 feels worse, but who knows." The trick is boringly mechanical: hash everything that goes into a run, write it down in a manifest, and never reuse a name.

### A registry: the single source of truth for "what is live"

A registry is a small database — it can be a JSON file — that records every model version you have produced and which one is currently serving traffic. It is the difference between a drawer full of unlabeled USB sticks and a librarian who can tell you exactly where everything is and which copy is checked out. Once you have a registry, "what is live?", "promote this one", and "roll back" all become one-line operations.

### Gating, canary, and rollback: ship scared, ship safe

These three are the safety rails of continual learning:

- A **gate** is an automated quality check a new model must pass before it is allowed anywhere near production. No human judgment, no vibes — it beats the current model on the numbers or it does not ship.
- A **canary** is a small slice of live traffic (say 5%) routed to the new model so you can watch it on *real* requests before trusting it with all of them. (Named after the canary in a coal mine — it gets exposed to danger first so the rest are safe.)
- A **rollback** is the undo button: instantly reverting to the last model you know was good.

We anchor all of this on the serving setup from *Ch22 - Serving Your Model and Using It in an App* and the adapter/merge formats from *Ch21 - Saving, Merging, and Exporting Your Model*.

---

## Part 1 — Monitoring and observability

### The intuition: a flight recorder for every request

Think of a commercial aircraft's black box. It does not try to predict crashes. It simply records everything — altitude, speed, control inputs — so that *if* something goes wrong, the story is already written down. Your production logger is the same idea. For every extraction request, you record a compact row of facts. Most rows you will never look at. But the day JSON parsing drops to 80%, those rows are the difference between a five-minute fix and a five-hour mystery.

The art is in *what* to record. Log too little and the black box is empty when you need it. Log everything verbatim and you have a privacy liability and a storage bill. For memory extraction, here is the field set that earns its keep:

- **Inputs** — the conversation text (or a hash of it, if it is sensitive; more on that below).
- **Outputs** — the raw string the model produced, before parsing. You want the *raw* output, because the most important failures are exactly the ones that did not parse.
- **JSON-valid flag** — did the output parse as a JSON array and pass the schema check from *Ch11 - Defining the Task*? This is your single most important production signal.
- **Memory count** — how many memory objects came out. A sudden shift (everything returning `[]`, or everything returning twenty memories) is a loud drift signal.
- **Latency** — milliseconds from request to response. Users feel this, and it is your early warning for an overloaded server.
- **Drift signals** — cheap proxies for "is this input unlike training data?": input length, out-of-vocabulary rate, or just the memory-count distribution over time.
- **Feedback** — explicit (the user deleted or edited the memory) or implicit (the memory was retrieved and used later). This is the gold you will mine for the next training round.

### A simple metrics logger

Here is a logger you can wrap around the `extract_memories()` function from *Ch22*. It writes one JSON line per request (the "JSON Lines" format — one independent JSON object per line, trivially appendable and trivially parseable later).

```python
# metrics_logger.py
# A drop-in observability wrapper around the extract_memories() call from Ch22.
# Writes one JSON-Lines record per request. No external services required.

import json
import time
import hashlib
import datetime as dt
from pathlib import Path

# Reuse the exact schema validator from Ch11 / Ch22 so "valid" means the same
# thing in production as it did in evaluation.
from end_to_end import validate_memory_output  # from Ch22's end_to_end.py

VALID_TYPES = {"preference", "fact", "decision", "relationship"}


def _hash_text(text: str) -> str:
    """Stable short hash of an input — lets us group/count identical inputs
    and reference a conversation in logs WITHOUT storing its raw content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class MetricsLogger:
    """
    Wraps a single extraction call and records a structured row about it.

    log_inputs=False is the privacy-safe default: we store a hash of the
    conversation, not the text itself. Flip it to True only in environments
    where logging raw user text is allowed (e.g. your own dev traffic).
    """

    def __init__(self, log_path: str = "logs/extractions.jsonl",
                 model_version: str = "unknown", log_inputs: bool = False):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version
        self.log_inputs = log_inputs

    def record(self, conversation: str, raw_output: str,
               latency_ms: float, user_id: str | None = None) -> dict:
        """Build and append one log row. Returns the row so callers can inspect it."""

        # The single most important production signal: did it parse + validate?
        json_valid = True
        memory_count = 0
        parse_error = None
        try:
            memories = validate_memory_output(raw_output)  # raises on bad schema
            memory_count = len(memories)
        except ValueError as e:
            json_valid = False
            parse_error = str(e)[:200]  # truncate — we just need a fingerprint

        row = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_version": self.model_version,
            "user_id": user_id,
            "input_hash": _hash_text(conversation),
            "input_chars": len(conversation),        # cheap drift proxy
            "json_valid": json_valid,                # the headline metric
            "memory_count": memory_count,            # distribution = drift signal
            "latency_ms": round(latency_ms, 1),
            "parse_error": parse_error,              # null when it parsed fine
        }
        if self.log_inputs:
            row["conversation"] = conversation
            row["raw_output"] = raw_output

        # One line, append-only. Cheap, durable, grep-able, replayable.
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


# ── Wrapping the Ch22 client ─────────────────────────────────────────────────

def extract_and_log(conversation: str, client, logger: MetricsLogger,
                    user_id: str | None = None):
    """
    Calls the served model (Ch22) and logs the result.
    Returns (memories_or_None, log_row). memories is None when parsing failed.
    """
    from openai import OpenAI  # noqa: F401  (client was built with make_client)
    from memory_prompt import SYSTEM_PROMPT

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=logger.model_version if "/" not in logger.model_version
              else "memory-extractor",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": conversation},
        ],
        temperature=0.1,   # low temp for JSON — see Ch22
        max_tokens=1024,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    raw = resp.choices[0].message.content.strip()

    row = logger.record(conversation, raw, latency_ms, user_id=user_id)

    memories = None
    if row["json_valid"]:
        memories = validate_memory_output(raw)
    return memories, row


if __name__ == "__main__":
    from client_http import make_client  # from Ch22
    client = make_client("http://localhost:8000/v1")
    logger = MetricsLogger(model_version="memory-extractor-v3", log_inputs=False)

    mems, row = extract_and_log(
        "User: I switched from Notion to Obsidian. Also I'm vegetarian.",
        client, logger, user_id="user_42",
    )
    print(json.dumps(row, indent=2))
```

A single recorded row looks like this:

```json
{
  "ts": "2026-06-24T18:30:11.402913+00:00",
  "model_version": "memory-extractor-v3",
  "user_id": "user_42",
  "input_hash": "9f2a1c0b7e4d8a55",
  "input_chars": 61,
  "json_valid": true,
  "memory_count": 2,
  "latency_ms": 712.4,
  "parse_error": null
}
```

Notice what we did *not* do: we did not stand up a time-series database or wire in a metrics SaaS. A JSON-Lines file is enough to start, and you can graduate to Prometheus, Datadog, or a logging pipeline later without changing the field set. Start with the file. Most teams over-engineer this and never actually look at the dashboard.

### Rolling up the logs into the Ch18 metrics

The logger records raw rows. To watch health, you summarize a recent window into the *same numbers you cared about in Ch18* — so "good in eval" and "good in production" mean the same thing.

```python
# metrics_rollup.py
# Summarize a window of the production log into the Ch18 metrics + drift signals.

import json
from pathlib import Path
from statistics import median


def summarize(log_path: str = "logs/extractions.jsonl", last_n: int = 1000) -> dict:
    """Read the most recent `last_n` rows and roll them up."""
    rows = []
    with Path(log_path).open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[-last_n:]
    if not rows:
        return {"n": 0}

    n = len(rows)
    valid = sum(1 for r in rows if r["json_valid"])
    latencies = sorted(r["latency_ms"] for r in rows)
    counts = [r["memory_count"] for r in rows]

    def pct(p):  # simple percentile
        return latencies[min(int(p / 100 * n), n - 1)]

    return {
        "n": n,
        # The headline production metric — the live analogue of Ch18 parse-rate.
        "json_valid_rate": round(valid / n, 4),
        "empty_rate": round(sum(1 for c in counts if c == 0) / n, 4),
        "avg_memories": round(sum(counts) / n, 2),
        "p50_latency_ms": round(median(latencies), 1),
        "p95_latency_ms": round(pct(95), 1),
        "p99_latency_ms": round(pct(99), 1),
    }


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
    # Example:
    # {
    #   "n": 1000, "json_valid_rate": 0.981, "empty_rate": 0.142,
    #   "avg_memories": 2.31, "p50_latency_ms": 690.0,
    #   "p95_latency_ms": 1180.0, "p99_latency_ms": 2040.0
    # }
```

### Alert thresholds, tied back to Ch18

An alert is a threshold plus a "tell someone." The thresholds should be derived from your offline numbers, not invented. In *Ch18*, the chapter made the point that **parse-rate is more important than F1** — a model with 98% parse-rate and 0.75 F1 beats one with 60% parse-rate and 0.80 F1, because unparseable outputs are silent production failures. So your loudest alarm is on the JSON-valid rate.

```python
# alerts.py
# Compare a live rollup against thresholds anchored on the Ch18 eval numbers.

from metrics_rollup import summarize

# Anchor thresholds on what the model achieved on the Ch18 held-out set,
# then allow a small live tolerance. If eval json-valid was ~0.99, alert
# when live drops below 0.95 (a real degradation, not statistical noise).
THRESHOLDS = {
    "json_valid_rate_min": 0.95,   # headline: format breakdown == silent failure
    "p95_latency_ms_max": 2500.0,  # user-facing latency ceiling
    "empty_rate_max": 0.45,        # if ~half of inputs suddenly extract nothing,
                                   #   either traffic shifted (drift) or the model broke
}


def check_alerts(log_path: str = "logs/extractions.jsonl") -> list[str]:
    s = summarize(log_path)
    if s["n"] < 50:
        return []  # too few samples to judge — avoid noisy false alarms

    alerts = []
    if s["json_valid_rate"] < THRESHOLDS["json_valid_rate_min"]:
        alerts.append(
            f"JSON-valid rate {s['json_valid_rate']:.3f} below "
            f"{THRESHOLDS['json_valid_rate_min']} — possible format regression "
            f"or drift. Check raw outputs in the log."
        )
    if s["p95_latency_ms"] > THRESHOLDS["p95_latency_ms_max"]:
        alerts.append(
            f"p95 latency {s['p95_latency_ms']:.0f}ms over "
            f"{THRESHOLDS['p95_latency_ms_max']:.0f}ms — server overloaded?"
        )
    if s["empty_rate"] > THRESHOLDS["empty_rate_max"]:
        alerts.append(
            f"Empty-output rate {s['empty_rate']:.3f} over "
            f"{THRESHOLDS['empty_rate_max']} — input distribution may have drifted "
            f"(Ch33: collect these for the next training round)."
        )
    return alerts


if __name__ == "__main__":
    for a in check_alerts() or ["All metrics within thresholds."]:
        print(a)
```

Two honest caveats. First, these thresholds are *starting points*, not laws — watch your own traffic for a week and set them where real problems live, not where noise does. Second, a fired alert is a question, not a verdict: a spike in empty-output rate might mean the model broke, or it might mean a flood of small-talk conversations that genuinely have nothing to extract. The log rows tell you which. That distinction — degradation versus drift — is exactly what the next training round needs to resolve.

---

## Part 2 — Dataset versioning: making every run reproducible

### The intuition: a recipe card, not a memory

Imagine a chef whose signature dish keeps coming out slightly different and who cannot say why — a pinch more salt here, a different supplier there, nothing written down. That is an unversioned ML pipeline. The fix is a recipe card: the exact ingredients (data), the exact method (config), and a label on the finished dish (the adapter) that points back to both. With the card, any cook can reproduce the dish, and you can diff two cards to see what changed.

Concretely, three things must be pinned for a run to be reproducible:

1. **The dataset** — captured by a *content hash* (so any edit changes the hash) plus a small *manifest* describing it.
2. **The config** — hyperparameters, base model, replay ratio (from *Ch33*), seed.
3. **The resulting adapter** — named so it points unambiguously back at the data and config that produced it.

### A versioning helper

```python
# dataset_version.py
# Content-hash a training dataset and write a manifest. The hash is the version:
# identical data -> identical hash -> reproducible run.

import json
import hashlib
import datetime as dt
from pathlib import Path


def hash_dataset(jsonl_path: str) -> str:
    """
    Content hash of a JSONL training file. We hash the bytes of each line in
    file order, so reordering rows changes the hash too (order can affect a run).
    Returns a 12-char hex digest used as the dataset version id.
    """
    h = hashlib.sha256()
    with Path(jsonl_path).open("rb") as f:
        for line in f:
            h.update(line)
    return h.hexdigest()[:12]


def write_manifest(jsonl_path: str, manifest_dir: str = "data/manifests",
                   notes: str = "") -> dict:
    """
    Create a manifest describing this exact dataset snapshot.
    The manifest filename embeds the content hash so it is self-identifying.
    """
    digest = hash_dataset(jsonl_path)

    # Count rows and the replay/general mix (Ch33) so the manifest records
    # the actual composition, not just a hash.
    n_rows, n_empty = 0, 0
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_rows += 1
            # An "empty" training row teaches the model to return [] (Ch18).
            assistant = row["messages"][-1]["content"].strip()
            if assistant in ("[]", "[ ]"):
                n_empty += 1

    manifest = {
        "dataset_version": digest,
        "source_file": str(jsonl_path),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_rows": n_rows,
        "n_empty_rows": n_empty,            # negatives matter for precision (Ch18)
        "empty_fraction": round(n_empty / n_rows, 3) if n_rows else 0.0,
        "schema": "[{text, type, entities}]",  # the pinned book schema
        "notes": notes,
    }

    out_dir = Path(manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dataset-{digest}.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest {out_path}")
    return manifest


if __name__ == "__main__":
    m = write_manifest(
        "data/train_round4.jsonl",
        notes="Round 4: added 600 rows of production-corrected examples + 20% replay",
    )
    print(json.dumps(m, indent=2))
    # Example:
    # {
    #   "dataset_version": "a1b2c3d4e5f6",
    #   "source_file": "data/train_round4.jsonl",
    #   "n_rows": 2840, "n_empty_rows": 312, "empty_fraction": 0.11,
    #   "schema": "[{text, type, entities}]",
    #   "notes": "Round 4: ... + 20% replay"
    # }
```

A note on scale. For datasets up to a few hundred MB, hashing the whole file every run is fine — it takes a second or two. If your training files grow into the gigabytes, hash a manifest of per-file hashes instead of re-reading everything, or adopt a purpose-built tool like DVC (Data Version Control) or `git-lfs`. The principle is unchanged: the version is a function of the content, so identical content always yields an identical id.

### A naming scheme that points backward

Adopt one naming convention and never deviate. The point is that a name should let you reconstruct the lineage without opening anything:

```
memory-extractor-v{round}-{base_short}-ds{dataset_version}
                 │         │            └─ dataset content hash (from above)
                 │         └─ base model short name, e.g. qwen3-1.7b
                 └─ training round number (1, 2, 3, …)

# Example:
memory-extractor-v4-qwen3-1.7b-dsa1b2c3d4e5f6
```

Given that string, you know it is the 4th round, fine-tuned on Qwen3-1.7B, from the dataset whose manifest is `dataset-a1b2c3d4e5f6.json`. No spreadsheet required.

---

## Part 3 — Adapter versioning and a model registry

### The intuition: the librarian, not the drawer

You will accumulate adapters fast — *Ch21* showed how a LoRA adapter is only 40–80 MB, so there is no reason to ever throw one away. But a folder of adapters is a drawer of unlabeled USB sticks. The registry is the librarian: it knows every version, what each was trained from, how each scored, and crucially *which one is currently checked out to production*.

For each adapter we track exactly the fields you would need to either trust it or replace it:

- **adapter version** (the naming-scheme string above)
- **base model** — its parent (from *Ch10*); an adapter is meaningless without it
- **dataset version** — the content hash that produced it
- **eval scores** — F1, parse-rate, judge score from *Ch18*, plus the no-forgetting canary score from *Ch33*
- **status** — `candidate`, `live`, or `archived`
- **created timestamp**

### A registry on a JSON file

A JSON file is genuinely enough for a single-model project — it is human-readable, diff-able, and commit-able. (If you later run many models or need concurrent writers, the same interface drops onto SQLite without changing callers; I will note where.)

```python
# registry.py
# A minimal model registry backed by a JSON file. Tracks every adapter version,
# its lineage, its eval scores, and which one is currently live.

import json
import datetime as dt
from pathlib import Path

REGISTRY_PATH = Path("registry/models.json")


def _load() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"live_version": None, "models": {}}


def _save(reg: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def register(version: str, base_model: str, dataset_version: str,
             adapter_path: str, eval_scores: dict, notes: str = "") -> None:
    """
    Add a newly-trained adapter as a CANDIDATE (not yet live).
    eval_scores should carry the Ch18 metrics + the Ch33 forgetting check, e.g.
        {"f1": 0.83, "json_valid_rate": 0.99, "judge": 0.88, "canary_f1": 0.81}
    """
    reg = _load()
    if version in reg["models"]:
        raise ValueError(f"Version '{version}' already registered — names are immutable.")
    reg["models"][version] = {
        "version": version,
        "base_model": base_model,           # the parent (Ch10) — adapter needs it
        "dataset_version": dataset_version,  # links to the data manifest (Part 2)
        "adapter_path": adapter_path,        # where the 40-80MB adapter lives (Ch21)
        "eval_scores": eval_scores,
        "status": "candidate",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notes": notes,
    }
    _save(reg)
    print(f"Registered candidate '{version}'.")


def get_live() -> dict | None:
    """Return the record of the currently-live adapter, or None if none set."""
    reg = _load()
    v = reg["live_version"]
    return reg["models"].get(v) if v else None


def promote(version: str) -> None:
    """Mark a version live. The previously-live version is archived (kept!)."""
    reg = _load()
    if version not in reg["models"]:
        raise ValueError(f"Unknown version '{version}'.")
    prev = reg["live_version"]
    if prev and prev in reg["models"]:
        reg["models"][prev]["status"] = "archived"
    reg["models"][version]["status"] = "live"
    reg["live_version"] = version
    _save(reg)
    print(f"Promoted '{version}' to live (was '{prev}').")


def history() -> list[dict]:
    """All versions, newest first — your audit trail."""
    reg = _load()
    return sorted(reg["models"].values(), key=lambda m: m["created"], reverse=True)


if __name__ == "__main__":
    register(
        version="memory-extractor-v4-qwen3-1.7b-dsa1b2c3d4e5f6",
        base_model="unsloth/Qwen3-1.7B",
        dataset_version="a1b2c3d4e5f6",
        adapter_path="outputs/memory-extractor-v4-adapter",
        eval_scores={"f1": 0.83, "json_valid_rate": 0.99,
                     "judge": 0.88, "canary_f1": 0.81},
        notes="Round 4 candidate",
    )
    live = get_live()
    print("Currently live:", live["version"] if live else "(none yet)")
```

> **When to move to SQLite.** Keep the *exact same functions* (`register`, `get_live`, `promote`, `history`) but back them with a single `models` table and a one-row `state` table for `live_version`. SQLite gives you atomic writes and safe concurrent reads — worth it once more than one process touches the registry, or once you have hundreds of versions. Until then, the JSON file is simpler and you can read it with your eyes.

---

## Part 4 — Eval gating: a candidate must earn promotion

### The intuition: a bouncer with a checklist

A new adapter does not get into production because it is new, or because the loss curve looked nice, or because you are excited about it. It gets in if and only if it is *measurably better* than what is already live — and does not break anything the old one handled. The gate is a bouncer with a fixed checklist: beat the incumbent on the held-out set, hold up on the live-traffic canary set, and pass the no-forgetting regression. Fail any one and you stay outside.

That last check matters because of a failure mode you met in *Ch33 - Catastrophic Forgetting Over Many Rounds*: when you fine-tune on round 4's new data, the model can *forget* skills it had in round 3. The new data might be all `decision`-type memories, and the model quietly gets worse at `preference`. Average F1 can even go *up* while a specific capability collapses. So the gate checks a **frozen canary eval set** — a fixed, never-changing set of examples spanning every memory type and edge case — and requires no significant regression on it. (This frozen canary eval set is the offline cousin of the live canary deploy in Part 5. Same word, two roles: a fixed battery of *examples* here; a slice of live *traffic* there.)

### A gate function

```python
# gate.py
# Decide whether a candidate adapter may be promoted over the current live one.
# Returns a verdict ("promote" / "reject") plus human-readable reasons.

from dataclasses import dataclass
from registry import get_live


@dataclass
class GateThresholds:
    min_f1_improvement: float = 0.01     # must beat live F1 by at least this much
    min_json_valid_rate: float = 0.97    # hard floor (Ch18: format == everything)
    max_canary_f1_drop: float = 0.02     # live-traffic canary must not drop more
    max_forgetting_drop: float = 0.03    # frozen no-forgetting set (Ch33)


def evaluate_gate(candidate_scores: dict,
                  forgetting_baseline: dict,
                  candidate_forgetting: dict,
                  thresholds: GateThresholds = GateThresholds()) -> dict:
    """
    candidate_scores:     Ch18 metrics for the candidate, incl. 'f1',
                          'json_valid_rate', and live 'canary_f1'.
    forgetting_baseline:  per-type F1 of the CURRENT live model on the frozen
                          no-forgetting set (Ch33), e.g.
                          {"preference": 0.85, "fact": 0.88, ...}
    candidate_forgetting: same per-type F1, measured for the candidate.

    Returns {"verdict": "promote"|"reject", "reasons": [...], "checks": {...}}.
    """
    reasons, checks = [], {}
    live = get_live()

    # ── Check 1: hard floor on JSON validity ────────────────────────────────
    jv = candidate_scores["json_valid_rate"]
    ok_json = jv >= thresholds.min_json_valid_rate
    checks["json_valid"] = ok_json
    if not ok_json:
        reasons.append(f"JSON-valid {jv:.3f} below floor "
                       f"{thresholds.min_json_valid_rate} — format regression.")

    # ── Check 2: beats the incumbent on held-out F1 ─────────────────────────
    if live is None:
        ok_f1 = True  # first model ever — nothing to beat
        reasons.append("No live model yet; F1-improvement check skipped.")
    else:
        delta = candidate_scores["f1"] - live["eval_scores"]["f1"]
        ok_f1 = delta >= thresholds.min_f1_improvement
        checks["f1_improvement"] = round(delta, 4)
        if not ok_f1:
            reasons.append(f"Held-out F1 gain {delta:+.4f} below required "
                           f"+{thresholds.min_f1_improvement}.")
        else:
            reasons.append(f"Held-out F1 improved {delta:+.4f}.")

    # ── Check 3: live canary set did not regress ────────────────────────────
    ok_canary = True
    if live is not None and "canary_f1" in candidate_scores \
            and "canary_f1" in live["eval_scores"]:
        c_drop = live["eval_scores"]["canary_f1"] - candidate_scores["canary_f1"]
        ok_canary = c_drop <= thresholds.max_canary_f1_drop
        checks["canary_f1_drop"] = round(c_drop, 4)
        if not ok_canary:
            reasons.append(f"Canary F1 dropped {c_drop:.4f} (> "
                           f"{thresholds.max_canary_f1_drop}).")

    # ── Check 4: no catastrophic forgetting, per memory type (Ch33) ─────────
    ok_forget = True
    worst = []
    for mem_type, base_f1 in forgetting_baseline.items():
        drop = base_f1 - candidate_forgetting.get(mem_type, 0.0)
        if drop > thresholds.max_forgetting_drop:
            ok_forget = False
            worst.append(f"{mem_type} -{drop:.3f}")
    checks["no_forgetting"] = ok_forget
    if not ok_forget:
        reasons.append("Catastrophic forgetting on: " + ", ".join(worst) +
                       " (Ch33: raise the replay ratio and retrain).")

    verdict = "promote" if (ok_json and ok_f1 and ok_canary and ok_forget) else "reject"
    return {"verdict": verdict, "reasons": reasons, "checks": checks}


if __name__ == "__main__":
    result = evaluate_gate(
        candidate_scores={"f1": 0.83, "json_valid_rate": 0.99, "canary_f1": 0.81},
        forgetting_baseline={"preference": 0.85, "fact": 0.88,
                             "decision": 0.82, "relationship": 0.79},
        candidate_forgetting={"preference": 0.84, "fact": 0.89,
                              "decision": 0.83, "relationship": 0.71},  # regressed!
    )
    import json
    print(json.dumps(result, indent=2))
    # verdict: "reject" — relationship F1 fell 0.08, far past the 0.03 limit.
```

The gate is deliberately conservative. It would rather reject a genuine improvement than ship a hidden regression, because a rejected candidate costs you one more training round, while a shipped regression costs you user trust and a 2 a.m. rollback. When the gate rejects on forgetting, the fix is the one from *Ch33*: bump the replay ratio (mix in more prior-round and general data — the charter's rule of thumb is ~10–30%, default ~20%) and retrain.

---

## Part 5 — Canary and shadow deploys

### The intuition: taste before you serve the whole table

Your gate passed. The candidate is better on every offline number. Do you flip it to 100% of traffic right now? No — because offline eval, however careful, is a sample, and production always has surprises your eval set never imagined. You taste the dish before serving the whole table. Two ways to taste:

- **Shadow deploy:** run the candidate on real traffic *but throw its answers away* — the live model still serves users; the candidate just runs alongside so you can compare. Zero user risk. Perfect for the first look.
- **Canary deploy:** route a small slice of real traffic (say 5%) to the candidate *for real*, watch its live metrics, and ramp up — 5% → 25% → 100% — only as it proves itself. Some user risk, but bounded, and it catches problems shadowing cannot (e.g. how the candidate behaves when its output actually flows downstream).

Both build directly on the *Ch22* serving setup. The cleanest pattern is to run two vLLM servers — live on port 8000, candidate on port 8001 — each loaded with the merged 16-bit model for its adapter (*Ch21*: vLLM needs the merged model, not the raw adapter). A thin router in front decides who handles each request.

### A traffic router with canary and shadow modes

```python
# router.py
# Routes extraction requests between the live and candidate models.
# Builds on Ch22's make_client() / extract_memories() and the MetricsLogger.

import random
import threading
from client_http import make_client, extract_memories   # Ch22
from metrics_logger import MetricsLogger


class CanaryRouter:
    """
    Sends most traffic to the live model and a small fraction to the candidate.
    In shadow mode, the candidate also runs on EVERY request but its result is
    discarded (logged only) — the user always gets the live answer.
    """

    def __init__(self, live_url: str, candidate_url: str | None = None,
                 live_version: str = "live", candidate_version: str = "candidate",
                 canary_fraction: float = 0.0, shadow: bool = False):
        self.live = make_client(live_url)
        self.candidate = make_client(candidate_url) if candidate_url else None
        self.canary_fraction = canary_fraction   # e.g. 0.05 = 5% real canary
        self.shadow = shadow
        self.live_log = MetricsLogger("logs/live.jsonl", model_version=live_version)
        self.cand_log = MetricsLogger("logs/candidate.jsonl",
                                      model_version=candidate_version)

    def _shadow_call(self, text, user_id):
        """Run the candidate in the background; log it; ignore its return value."""
        try:
            import time
            t0 = time.perf_counter()
            mems = extract_memories(text, self.candidate)
            self.cand_log.record(text, _to_raw(mems),
                                 (time.perf_counter() - t0) * 1000.0, user_id)
        except Exception as e:
            print(f"[shadow] candidate errored (user impact: none): {e}")

    def handle(self, text: str, user_id: str | None = None) -> list[dict]:
        """Return memories for one request, routing per the configured policy."""
        # Shadow: fire-and-forget the candidate, but always serve from live.
        if self.shadow and self.candidate is not None:
            threading.Thread(target=self._shadow_call,
                             args=(text, user_id), daemon=True).start()

        # Canary: a fraction of real traffic is SERVED by the candidate.
        use_candidate = (self.candidate is not None and not self.shadow
                         and random.random() < self.canary_fraction)

        client = self.candidate if use_candidate else self.live
        log = self.cand_log if use_candidate else self.live_log

        import time
        t0 = time.perf_counter()
        memories = extract_memories(text, client)
        log.record(text, _to_raw(memories),
                   (time.perf_counter() - t0) * 1000.0, user_id)
        return memories


def _to_raw(memories) -> str:
    """Re-serialize parsed memories so the logger's validator sees clean JSON."""
    import json
    return json.dumps(memories, ensure_ascii=False)


if __name__ == "__main__":
    # Stage 1: shadow — candidate runs on all traffic, users see only live.
    router = CanaryRouter(
        live_url="http://localhost:8000/v1",
        candidate_url="http://localhost:8001/v1",
        live_version="memory-extractor-v3",
        candidate_version="memory-extractor-v4",
        shadow=True,
    )
    router.handle("User: I prefer tea over coffee. My manager is Priya.", "user_7")
    # Now compare logs/live.jsonl vs logs/candidate.jsonl with metrics_rollup.
```

The promotion ramp is then a sequence of small, reversible steps, each watched with the same `metrics_rollup.summarize()` from Part 1:

1. **Shadow** for a day. Compare `logs/candidate.jsonl` against `logs/live.jsonl`. The candidate should match or beat live on JSON-valid rate and latency, on the same real inputs.
2. **5% canary** (`shadow=False, canary_fraction=0.05`). Watch the candidate log for an hour or a day depending on your traffic volume.
3. **Ramp** to 25%, then 100%, pausing at each step.
4. **Promote in the registry** (`promote("memory-extractor-v4-…")`) once 100% has held cleanly. Now the candidate *is* live, and the old live version is archived but kept.

Compare like with like: a canary on overnight traffic and a baseline measured at midday peak is not a fair fight. Match the time windows. If at any step the candidate's live JSON-valid rate dips below the live model's, stop and roll back — which is the next section, and it is the easiest part of the whole pipeline.

---

## Part 6 — Rollback: the undo button

### The intuition: keep the old key on the ring

Every safe-deployment story ends the same way: *and we could undo it instantly.* You kept every adapter (they are tiny — *Ch21*), the registry remembers which version was live before this one, and the merged models for recent versions are still on disk or on the Hub. So rollback is not a rebuild. It is handing the server a different — already-built — model and updating one pointer.

```python
# rollback.py
# Revert production to the last-good adapter. Two parts:
#   1. flip the registry pointer back  2. point serving at the restored model.

from registry import _load, _save, get_live, history


def rollback_to_previous() -> dict:
    """
    Revert to the version that was live immediately before the current one.
    Returns the record we rolled back to.
    """
    reg = _load()
    current = reg["live_version"]

    # The last archived version, newest first, that isn't the current one.
    prior = next((m for m in history()
                  if m["version"] != current and m["status"] == "archived"), None)
    if prior is None:
        raise RuntimeError("No prior version to roll back to.")

    if current and current in reg["models"]:
        # Quarantine the bad version so the gate won't re-promote it by accident.
        reg["models"][current]["status"] = "rolled_back"
    reg["models"][prior["version"]]["status"] = "live"
    reg["live_version"] = prior["version"]
    _save(reg)
    print(f"Rolled back: '{current}' -> '{prior['version']}'.")
    return reg["models"][prior["version"]]


def rollback_to(version: str) -> dict:
    """Roll back to a specific named version (for targeted reverts)."""
    reg = _load()
    if version not in reg["models"]:
        raise ValueError(f"Unknown version '{version}'.")
    cur = reg["live_version"]
    if cur and cur in reg["models"]:
        reg["models"][cur]["status"] = "rolled_back"
    reg["models"][version]["status"] = "live"
    reg["live_version"] = version
    _save(reg)
    print(f"Rolled back to '{version}'.")
    return reg["models"][version]


if __name__ == "__main__":
    restored = rollback_to_previous()
    print("Now live:", restored["version"])
    print("Restart/redirect serving to:", restored["adapter_path"])
```

The registry flip is instant. The *serving* side then has to actually load the restored model, and how fast that is depends on your setup from *Ch22*:

- **Two warm servers (recommended for production):** keep the previous version's vLLM server running on its own port even after promotion. Rollback is then just pointing the router's `self.live` at that already-warm port — sub-second, zero reload. This is why the canary router keeps both clients around.
- **Single server:** restart vLLM pointed at the restored merged model directory. That is the 20–60 second weight-load from *Ch22* — acceptable for most apps, but plan for the brief blip.

Because adapters are 40–80 MB and you never delete them, you can roll back not just one step but to *any* prior version with `rollback_to("memory-extractor-v2-…")`. The registry is your time machine; the kept adapters are the snapshots it travels to.

A short runbook is worth taping to the wall: **(1)** an alert fires (Part 1), **(2)** you glance at the candidate/live rollups to confirm it is a real regression and not drift, **(3)** `rollback_to_previous()`, **(4)** point serving at the restored model, **(5)** the bad version is quarantined as `rolled_back` so the gate cannot silently re-promote it, **(6)** you debug the regression offline, calmly, with production no longer on fire.

---

## The whole book, in one breath

Step back and look at what you have built — not in this chapter, but across the entire book.

In *Ch0 - The Afternoon Speedrun* you rented a GPU, generated a few hundred synthetic examples, ran one training script, and had a working memory extractor by dinner for under $30. It felt almost too easy, and a little like magic.

Parts 0–2 took the magic apart. You learned what a model actually is, why *LoRA* attaches cheap "sticky notes" instead of reprinting the whole textbook, and how tokens, context windows, and chat templates fit together — so the speedrun stopped being magic and became *mechanism*.

Parts 3–6 were the craft. You defined the task and pinned the schema — `[{text, type, entities}]` — and its system prompt (*Ch11–12*). You generated and cleaned real training data (*Ch13–14*), ran a proper fine-tune and learned which knobs matter (*Ch15–17*), and then — the part most tutorials skip — you *measured* it honestly with parse-rate, precision/recall/F1, and an LLM judge (*Ch18*), debugged the failures (*Ch19*), iterated (*Ch20*), and shipped it: exported (*Ch21*) and served (*Ch22*).

Part 7 went beyond imitation. Supervised fine-tuning teaches a model to copy good examples; preference and RL methods teach it to *prefer* better answers over worse ones. You built a reward signal, ran DPO (*Ch26*), understood honestly why hand-rolled PPO is impractical for this reader (*Ch27*) and reached for GRPO instead (*Ch28*), and learned to pick the right method for the job (*Ch29*).

And Part 8 — this final stretch — turned a model into a *system*. You learned to guard against catastrophic forgetting with a replay buffer (*Ch33*), and in this chapter you wrapped the whole thing in production ops: it watches itself (monitoring), it can rebuild any version from recorded ingredients (dataset + adapter versioning + registry), it refuses to ship a regression (the gate), it tries new versions on a sliver of real traffic before trusting them (canary and shadow), and it can undo any mistake in seconds (rollback).

That is the full arc: **working → measured → better → preferred → continually improving and safely shippable.** A loop, not a line. Production traffic generates logs; logs surface corrections and drift; corrections and drift become the next dataset version; that trains the next adapter; the gate vets it; the canary proves it; the registry promotes it — and the loop turns again, a little better each round.

You did not buy this system. You did not call an API and hope. You *built* it, layer by layer, on a single consumer GPU or a rented cloud instance, starting from "I write Python but have never trained a neural net." The memory extractor was only ever the example. The real thing you walk away with is the technique — and the confidence to point it at whatever domain is yours: the meeting transcripts, the clinical notes, the support tickets, the contracts, the personal recall feature you have been imagining.

Go build the next one. You already know how.

---

## Recap

- **Monitoring** is your offline Ch18 metrics computed continuously on live traffic. Log, per request: input (or its hash), raw output, JSON-valid flag, memory count, latency, and feedback. A JSON-Lines file is enough to start; roll it up into json-valid rate and latency percentiles, and alert when they cross thresholds anchored on your eval numbers — parse-rate first, because a format breakdown is a silent failure.
- **Dataset versioning** makes runs reproducible: content-hash the data, write a manifest recording row counts and the replay mix, and use a naming scheme that points an adapter back at its data and base model.
- A **registry** (a JSON file, or SQLite at scale) is the single source of truth for every adapter version — its base model, dataset version, eval scores, and which one is live. It makes promote and rollback one-liners.
- The **eval gate** promotes a candidate only if it beats the live model on held-out F1, holds up on the live canary set, clears a hard JSON-valid floor, and shows no catastrophic forgetting on the frozen per-type set (Ch33). It is intentionally conservative.
- **Shadow** deploys run the candidate on real traffic with its output discarded (zero user risk); **canary** deploys serve a small, ramping fraction of real traffic (bounded risk). Both build on the Ch22 two-server serving setup; watch them with the same rollup.
- **Rollback** is a registry pointer flip plus pointing serving at an already-built model. Keep the previous version warm for sub-second reverts; never delete adapters (they are 40–80 MB), so you can roll back to *any* prior version.
- This is the capstone: the speedrun became a measured, preferred, continually-improving, observable, safely-shippable system — a loop where production feedback feeds the next training round. You built it.
```
