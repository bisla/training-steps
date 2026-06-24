# Chapter 31 - Selecting and Curating Data That Actually Helps

In the speedrun and in Parts 3 and 4 you had the opposite of a luxury problem: you had to *manufacture* training data out of thin air, generating synthetic conversations because you didn't have enough real ones. By the time you are running a continual loop (the round-after-round system you set up in *Ch30 - The Continual Learning Loop*), the problem has flipped completely. Every day your deployed model sees real conversations. Every day you can capture the ones it handled, the ones it fumbled, the ones a human corrected. Within a few weeks you are drowning. You have tens of thousands of candidate rows and room in your next training run for maybe two thousand.

This is the moment most people get wrong. The instinct is to throw all of it at the trainer — surely more data is better? It is not. A continual loop that trains on everything it collects gets *worse*, not better: it drowns the rare, hard, instructive examples under a flood of near-identical easy ones, it amplifies whatever noise leaked in, and it slowly forgets the things it used to know. Curation is the single highest-leverage skill in continual learning. This chapter is about choosing the two thousand rows that teach the most, and throwing the other twenty-eight thousand away without guilt.

This is the advanced counterpart to *Ch14 - Cleaning, Splitting, and Sanity-Checking Data*. Ch14 was about making sure your data is not *broken*. This chapter assumes the data is already valid and asks a harder question: of all the valid data, which rows are actually worth training on?

---

## What you'll learn

- Why, in a continual loop, curation beats volume — and how redundant or low-quality data actively *hurts* the model rather than just wasting compute
- How to remove exact duplicates (hashing) and near-duplicates (embedding cosine similarity and MinHash) from a pool of candidate memory-extraction rows
- How to score rows for quality using cheap heuristics (schema validity, fact density, length sanity) and, when you need more nuance, a model-based scorer (an LLM judge or the reward model from *Ch25 - Training a Reward Model*)
- How to do **hard-example mining**: run your *current* model on the candidates and keep the ones it gets wrong, because those are the ones with something left to teach
- How to sample for **coverage** so you don't over-represent easy, common cases — balancing across the four memory types and across conversation styles
- How to assemble all of this into one curation pipeline that turns a raw candidate pool into a lean, high-value training set, plugged straight into Ch30's "select / curate" step
- How aggressive to be, and the honest tradeoffs of each knob

---

## Concepts you need first

### Why "more data" stops helping (and starts hurting)

Back in the charter's rule-of-thumb table, you saw that for a narrow structured-extraction task on a ~4B model, quality and diversity start to matter more than raw count somewhere around 3,000–10,000 rows, and past ~10,000 you hit diminishing returns. In a continual loop you blow past those numbers fast. The naive move — append everything you collected this round and retrain — fails for three concrete reasons:

1. **Redundancy dilutes the gradient.** Training nudges the model a little for every example it sees. If 4,000 of your 5,000 rows are minor variations of "user states a coffee preference," the model spends 80% of its learning budget getting *even better* at the one thing it was already great at, and almost no budget on the rare decision-extraction case it keeps fumbling. The signal you care about is drowned out.

2. **Noise gets amplified.** Real captured data is messier than your clean synthetic set. A small fraction will be mislabeled, partially correct, or genuinely ambiguous. At small volumes a few bad rows wash out. But if your collection process systematically lets through a *type* of bad row (say, conversations where the user's message got truncated), training faithfully teaches the model that broken pattern.

3. **Easy data teaches nothing.** This is the deepest point and we'll spend a whole section on it. A row the model *already gets right* contributes almost no useful gradient — the model is already confident and correct, so there is little error to learn from. You paid to collect it, you'll pay GPU time to train on it, and it moves the model approximately nowhere.

The mental model: think of curation like a coach building a practice plan for an athlete who is already pretty good. A bad coach makes them run the drills they've already mastered, because those drills look impressive and feel productive. A good coach spends the limited practice hours on the *specific weaknesses* — the backhand that keeps failing — and only revisits the strengths occasionally to keep them sharp. Your training run is a fixed number of practice hours. Curation decides what the model practices.

### The four levers

Curation, for our purposes, is four filters applied in sequence, each removing a different kind of low-value row:

- **Dedup** removes rows that are *redundant* — you already have one like it.
- **Quality scoring** removes rows that are *bad* — malformed, thin, or wrong.
- **Hard-example mining** removes rows that are *too easy* — the model already nails them.
- **Coverage / importance sampling** removes rows that are *over-represented* — you have plenty of that flavor already.

Order matters, and we'll justify the order when we assemble the pipeline. For now, the intuition: cheap-and-broad filters first (dedup, heuristics), expensive-and-precise filters last (model scoring on the survivors).

### The data we're curating

Everything in this chapter operates on a pool of *candidate* rows in the same conversational format you've used since *Ch12 - Data Format*. Each row is a training example whose assistant turn is a JSON array of memory objects following the pinned schema:

```python
# One memory object — the schema we have used since Ch12. Do not drift from this.
{
    "text": "Sarah prefers dark roast coffee in the morning",   # the fact, a complete sentence
    "type": "preference",                                        # one of: preference | fact | decision | relationship
    "entities": ["Sarah"]                                        # named people, places, or things
}
```

In a continual loop these candidates come from a few places (you wired up the collection in Ch30): conversations your deployed model handled in production, examples a human reviewer corrected, and a slice of freshly generated synthetic data to cover gaps. They arrive already validated by the Ch14 pipeline — valid JSON, required fields present, no truncation. Our job starts after that.

We'll build a small, realistic candidate pool in code so every example in this chapter actually runs.

```python
# build_candidates.py
# PURPOSE: assemble a small, realistic pool of candidate rows to curate.
# In your real loop, this comes from Ch30's collection step; here we make one
# in-memory so the rest of the chapter is runnable end to end.

import json
from datasets import Dataset

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

def make_row(user_text, memories):
    """Build one conversational training row in the format the trainer expects."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            # The assistant content is the JSON array, serialized as a string.
            {"role": "assistant", "content": json.dumps(memories, ensure_ascii=False)},
        ],
        # We also keep the parsed memories alongside for convenient scoring.
        "memories": memories,
    }

# A deliberately lopsided pool: lots of easy coffee-preference rows (the common
# case in our imaginary product), a few near-duplicates, and a thin scattering
# of the harder decision / relationship rows that we actually want more of.
raw_rows = []

# 1) A pile of near-identical "preference" rows (the over-represented easy case).
coffee_variants = [
    "Maya: I always grab a dark roast before my 9am standup.",
    "Maya: Dark roast in the morning, every single day for me.",
    "Maya: Can't start the day without a dark roast coffee.",
    "Maya: Morning ritual is a big mug of dark roast.",
    "Maya: I drink dark roast coffee first thing every morning.",
]
for v in coffee_variants:
    raw_rows.append(make_row(
        f"Extract memories from this conversation:\n\n{v}",
        [{"text": "Maya prefers dark roast coffee in the morning",
          "type": "preference", "entities": ["Maya"]}],
    ))

# 2) An EXACT duplicate of one of the above (a retry that got written twice).
raw_rows.append(make_row(
    "Extract memories from this conversation:\n\nMaya: I drink dark roast coffee first thing every morning.",
    [{"text": "Maya prefers dark roast coffee in the morning",
      "type": "preference", "entities": ["Maya"]}],
))

# 3) A few harder, rarer rows — multi-fact, decisions, relationships.
raw_rows.append(make_row(
    "Extract memories from this conversation:\n\n"
    "Priya: After the incident review we're moving the deploy window to Tuesdays.\n"
    "Tom: Agreed. And Dana will own the on-call rotation starting next sprint.",
    [{"text": "The team decided to move the deploy window to Tuesdays.",
      "type": "decision", "entities": ["the team"]},
     {"text": "Dana will own the on-call rotation starting next sprint.",
      "type": "decision", "entities": ["Dana"]}],
))
raw_rows.append(make_row(
    "Extract memories from this conversation:\n\n"
    "Sam: My sister Lena just started at the same hospital as Dr. Okafor.\n"
    "Sam: She's doing her residency in cardiology.",
    [{"text": "Lena is Sam's sister.",
      "type": "relationship", "entities": ["Lena", "Sam"]},
     {"text": "Lena is doing a residency in cardiology.",
      "type": "fact", "entities": ["Lena"]}],
))
raw_rows.append(make_row(
    "Extract memories from this conversation:\n\n"
    "Jordan: We picked Postgres over Mongo for the billing service.\n"
    "Jordan: Mostly because Ravi's team already runs it in prod.",
    [{"text": "The billing service will use Postgres instead of MongoDB.",
      "type": "decision", "entities": ["billing service", "Postgres", "MongoDB"]},
     {"text": "Ravi's team already runs Postgres in production.",
      "type": "fact", "entities": ["Ravi"]}],
))

# 4) A thin / low-quality row: a long conversation that yielded almost nothing.
raw_rows.append(make_row(
    "Extract memories from this conversation:\n\n"
    "Alex: hey\nBmsg: hey\nAlex: you around later?\nBmsg: maybe, will ping you\n"
    "Alex: cool cool\nBmsg: 👍\nAlex: ok talk soon\nBmsg: yep",
    [],   # genuinely nothing memorable — an empty extraction is correct here
))

candidates = Dataset.from_list(raw_rows)
print(f"Candidate pool: {len(candidates)} rows")
# Candidate pool: 11 rows
```

Eleven rows is a toy pool, but it has every pathology a real pool has: exact duplicates, near-duplicates, an over-represented easy type, rare hard types, and a thin row. Each technique below will show its effect on this pool, and the final pipeline runs over the whole thing.

A small helper we'll reuse everywhere — pulling the user text and the parsed memories out of a row:

```python
# helpers.py — small accessors used across the curation steps.

def user_text(row):
    """Return the user-turn content (the conversation we extract from)."""
    return next(m["content"] for m in row["messages"] if m["role"] == "user")

def assistant_memories(row):
    """Return the parsed list of memory objects from the assistant turn."""
    # We stored the parsed list on the row for convenience; in a pool that
    # only has the serialized string, json.loads the assistant content instead.
    if "memories" in row:
        return row["memories"]
    import json
    content = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    return json.loads(content)
```

---

## Step 1 — Deduplication

### Intuition

Imagine flashcards for studying. If you accidentally photocopy one card forty times, your deck is now mostly that one fact. You'll spend most of your study time on it and starve everything else. Worse, you'll *feel* productive because the deck is thick. Deduplication is throwing out the photocopies so the deck reflects what you actually need to learn.

There are two flavors of duplicate, and they need different tools:

- **Exact duplicates** — byte-for-byte identical inputs. These happen constantly in a real loop: a generator retried a failed call and wrote both, or the same production conversation got logged twice. Catching these is cheap: hash the input, drop repeats.
- **Near-duplicates** — different wording, same content. "I drink dark roast every morning" vs "Dark roast in the morning, every day for me." A hash sees these as completely different (one changed character flips the whole hash). Catching them needs a notion of *similarity*, not equality.

For near-duplicates we'll show two approaches. **Embedding cosine similarity** is the most accurate and reuses the exact model you already met in *Ch18 - Evaluating Memory Extraction* (`all-MiniLM-L6-v2`); it's the right default for a few thousand rows. **MinHash** is the approach you reach for when the pool is huge (hundreds of thousands of rows), because comparing every pair of embeddings becomes too slow — MinHash approximates similarity cheaply. We'll cover the embedding version as the default and MinHash as the scale-up option.

### Exact dedup with hashing

```python
# dedup_exact.py
import hashlib

def normalize(text):
    """Collapse whitespace so trivially-different formatting hashes the same.
    (Same trick as Ch14's dedup, but here it's the first of several steps.)"""
    return " ".join(text.split())

def exact_dedup(dataset):
    """Keep the first occurrence of each unique (normalized) user input."""
    seen = set()
    keep = []
    for i, row in enumerate(dataset):
        # Hash the normalized user text. md5 is fine here — we are not doing
        # cryptography, just bucketing identical strings cheaply.
        key = hashlib.md5(normalize(user_text(row)).encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return dataset.select(keep)

deduped = exact_dedup(candidates)
print(f"After exact dedup: {len(deduped)} rows  ({len(candidates) - len(deduped)} removed)")
# After exact dedup: 10 rows  (1 removed)
```

The exact duplicate (the second copy of "I drink dark roast coffee first thing every morning") is gone. The five *near*-duplicate coffee rows survive — they're worded differently, so their hashes differ. That's what near-dedup is for.

### Near-dedup with embedding cosine similarity

The idea: convert each input into a vector (an embedding) where similar meanings land near each other in space, then measure the cosine of the angle between vectors. Cosine similarity runs from −1 (opposite) to 1 (identical direction). Two paraphrases of the same sentence score around 0.85–0.95; two unrelated sentences score near 0.

```python
# dedup_near.py
import numpy as np
from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2: ~22 MB, fast on CPU, the same model Ch18 used for its
# embedding-similarity match. Loading it here keeps the whole pipeline on one
# small dependency rather than two.
_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def near_dedup(dataset, threshold=0.92):
    """Greedy near-dedup: encode every input, then walk the rows keeping a row
    only if it is NOT too similar to any row already kept.

    threshold is the cosine similarity above which two rows count as 'the same'.
    Higher threshold = more cautious (only drops very close pairs)."""
    texts = [user_text(row) for row in dataset]
    # normalize_embeddings=True makes vectors unit-length, so a plain dot
    # product equals cosine similarity — no separate normalization needed.
    embs = _embedder.encode(texts, normalize_embeddings=True,
                            show_progress_bar=False)

    keep = []                       # indices we decide to keep
    kept_embs = []                  # their embeddings, for comparison
    for i, emb in enumerate(embs):
        if kept_embs:
            # Cosine sim against everything kept so far = dot products.
            sims = np.dot(np.array(kept_embs), emb)
            if sims.max() >= threshold:
                continue            # too close to something we already kept
        keep.append(i)
        kept_embs.append(emb)
    return dataset.select(keep)

near = near_dedup(deduped, threshold=0.92)
print(f"After near dedup:  {len(near)} rows  ({len(deduped) - len(near)} removed)")
# After near dedup:  6 rows  (4 removed)

for row in near:
    print(" -", user_text(row).split("\n\n", 1)[-1][:60])
# - Maya: I always grab a dark roast before my 9am standup.
# - Priya: After the incident review we're moving the deploy window ...
# - Sam: My sister Lena just started at the same hospital as Dr. Okafor.
# - Jordan: We picked Postgres over Mongo for the billing service.
# - Alex: hey
# - Bmsg: hey ...
```

Four of the five coffee paraphrases collapsed into one. The pool is now six rows: one coffee preference, three hard multi-fact rows, and the thin row. Notice the threshold is a real knob: at `0.92` we kept genuine paraphrases out; at `0.80` we'd start dropping rows that share a *topic* but carry different facts, which is dangerous. We'll come back to how aggressive to be in the tradeoffs section. The sane default is **high (0.90–0.95)**: only kill rows that are nearly the same input.

### MinHash — the scale-up option

Embedding dedup compares every survivor against every prior survivor. At 200,000 rows that's billions of comparisons and a slow afternoon. **MinHash with Locality-Sensitive Hashing (LSH)** approximates the same answer cheaply: it turns each document into a small signature such that documents with similar word-sets get similar signatures, and an LSH index lets you query "what's near this?" without scanning everything. The similarity it measures is **Jaccard similarity** — overlap of the sets of word-shingles — which is a lexical cousin of cosine similarity: great for catching reworded-but-overlapping text, blind to deep paraphrase.

```python
# dedup_minhash.py — reach for this when the pool is too big for pairwise embeddings.
# pip install datasketch
from datasketch import MinHash, MinHashLSH

def shingles(text, k=5):
    """Break text into overlapping k-word windows (the 'set' we compare.)"""
    words = normalize(text).lower().split()
    return {" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}

def minhash_dedup(dataset, threshold=0.7, num_perm=128):
    """Drop rows whose shingle-set Jaccard similarity to an earlier kept row
    exceeds `threshold`. num_perm trades accuracy for memory/speed."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = []
    for i, row in enumerate(dataset):
        m = MinHash(num_perm=num_perm)
        for sh in shingles(user_text(row)):
            m.update(sh.encode("utf-8"))
        # query() returns keys of already-inserted rows that look similar.
        if lsh.query(m):
            continue                       # near-duplicate of something kept
        lsh.insert(str(i), m)
        keep.append(i)
    return dataset.select(keep)

# On a small pool the embedding version is more accurate; use MinHash at scale.
mh = minhash_dedup(candidates, threshold=0.7)
print(f"MinHash dedup:     {len(mh)} rows")
# MinHash dedup:     7 rows
```

MinHash is lexical, so it catches the heavily-overlapping coffee rows but is more conservative than the embedding model on the ones that share few literal words. That's the tradeoff: MinHash is cheap and scales to millions of rows; embeddings are more accurate on paraphrase but cost more per comparison. **Rule of thumb:** embeddings under ~50k rows, MinHash (or a hybrid — MinHash to shrink the pool, embeddings to refine) above that.

---

## Step 2 — Quality scoring

### Intuition

Dedup answers "do I already have one like this?" Quality scoring answers a different question: "is this row any good in the first place?" A row can be perfectly unique and still be junk — a conversation that's all "hey / yep / 👍" yielding nothing, or an extraction that technically parses but bundles three facts into one bloated `text` field.

Think of it like a copy editor reading submissions. Some get rejected on sight for obvious problems — wrong format, way too short, way too long. That's the *heuristic* pass: fast, mechanical, no judgment required. The pieces that survive get a real read for substance — is this actually well-written and on-topic? That deeper read is expensive, so you only do it on what survived the cheap pass. That second read is the *model-based* score.

### Heuristic scoring

Heuristics are cheap, deterministic rules. They won't catch subtle wrongness, but they catch the obvious 80% for free and run in milliseconds over millions of rows. Three that matter for memory extraction:

- **Schema validity** — does every memory object have the three required fields and a `type` from the pinned set? (Ch14 already filtered hard failures, but in a continual loop with human edits, re-checking is cheap insurance.)
- **Fact density** — how many memories per unit of conversation? A 400-word conversation yielding zero memories *might* be correct (some chats are empty) but is often a sign of a missed extraction or a degenerate chat. A one-line conversation yielding eight memories is suspicious the other way.
- **Length sanity** — each `text` should read like one standalone sentence. A 200-character `text` is probably several facts bundled together (a schema violation in spirit even if not in letter).

```python
# quality_heuristic.py
VALID_TYPES = {"preference", "fact", "decision", "relationship"}  # pinned schema
REQUIRED_FIELDS = {"text", "type", "entities"}

def heuristic_quality(row):
    """Return a score in [0, 1]. Higher = looks healthier. Cheap and deterministic.
    These weights are a sensible starting point, not laws — tune them on your data."""
    mems = assistant_memories(row)
    convo = user_text(row)
    n_words = max(1, len(convo.split()))

    score = 1.0

    # --- Schema validity: heavily penalize anything malformed. ---
    for m in mems:
        if REQUIRED_FIELDS - set(m.keys()):
            score -= 0.5
        if m.get("type") not in VALID_TYPES:
            score -= 0.5
        if not isinstance(m.get("entities"), list):
            score -= 0.3
        # Length sanity: a 'text' over ~160 chars is probably bundled facts.
        if isinstance(m.get("text"), str) and len(m["text"]) > 160:
            score -= 0.2

    # --- Fact density: penalize the extremes, not the middle. ---
    # memories per 100 words of conversation.
    density = len(mems) / (n_words / 100.0)
    if len(mems) == 0 and n_words > 40:
        # A long conversation that yielded nothing: usually a missed extraction
        # or a degenerate chat. Not always wrong, so a soft penalty.
        score -= 0.4
    elif density > 12:
        # Implausibly many memories for how little was said.
        score -= 0.3

    return max(0.0, min(1.0, score))

for row in near:
    snippet = user_text(row).split("\n\n", 1)[-1][:40].replace("\n", " ")
    print(f"{heuristic_quality(row):.2f}  {snippet}")
# 1.00  Maya: I always grab a dark roast before my 9am sta
# 1.00  Priya: After the incident review we're moving the
# 1.00  Sam: My sister Lena just started at the same hospi
# 1.00  Jordan: We picked Postgres over Mongo for the bill
# 0.60  Alex: hey Bmsg: hey Alex: you around later? Bmsg:
```

The thin "hey / yep" row drops to 0.60 — flagged, as it should be. Note the *soft* penalty: an empty extraction on a long chat is suspicious but sometimes genuinely correct, so we dock points rather than deleting outright. Heuristics set the floor; they should rarely be the only thing standing between a row and the trash.

### Model-based scoring

Heuristics are blind to meaning. They can't tell that `"text": "Maya likes coffee"` is a weaker extraction than `"text": "Maya prefers dark roast coffee in the morning"` — both are valid, short, single sentences. For that judgment you need a model. You have two options, and you've met both already:

1. **An LLM judge** (the technique from *Ch18 - Evaluating Memory Extraction*): prompt a capable model with a tight rubric and ask for a single number. Flexible, no training, costs an API call per row.
2. **The reward model from *Ch25 - Training a Reward Model***: you already trained a small model that takes a (conversation, extraction) pair and emits a scalar "how good is this." It's essentially free to run locally and was trained on *your* notion of quality. If you have it, prefer it.

Here is the LLM-judge version, since it stands alone without the Ch25 artifact. The rubric is deliberately narrow — one number, no prose — exactly as Ch18 argued.

```python
# quality_model.py — LLM-as-judge scorer for extraction quality.
# pip install anthropic
import json, re
from anthropic import Anthropic

# Per the charter's running example, the book uses Claude as the teacher/judge.
# Pin the model id you actually have access to.
client = Anthropic()
JUDGE_MODEL = "claude-sonnet-4-5"   # any capable model; pin what you have

JUDGE_RUBRIC = """You are grading a memory-extraction output. Given a CONVERSATION and the EXTRACTED MEMORIES (a JSON array), rate the extraction's quality from 1 to 5:

5 = every memorable fact captured, each as one clean standalone sentence, correct type, no hallucinations
4 = mostly correct, minor wording or a single missed/extra fact
3 = captures the gist but bundles facts, vague wording, or one wrong type
2 = significant misses, hallucinations, or several malformed objects
1 = mostly wrong or unusable

Respond with ONLY the single digit. No explanation."""

def llm_quality(row):
    """Return a normalized quality score in [0, 1] from an LLM judge."""
    payload = (
        f"CONVERSATION:\n{user_text(row)}\n\n"
        f"EXTRACTED MEMORIES:\n{json.dumps(assistant_memories(row), ensure_ascii=False)}"
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=5,                       # we only want one digit
        system=JUDGE_RUBRIC,
        messages=[{"role": "user", "content": payload}],
    )
    text = resp.content[0].text.strip()
    m = re.search(r"[1-5]", text)           # be robust to stray whitespace
    raw = int(m.group()) if m else 3        # default to neutral on a weird reply
    return (raw - 1) / 4.0                   # map 1..5 -> 0.0..1.0

# If you trained the Ch25 reward model, swap the body above for a local call:
#
#   from transformers import AutoTokenizer, AutoModelForSequenceClassification
#   import torch
#   rm_tok = AutoTokenizer.from_pretrained("path/to/your/reward-model")
#   rm     = AutoModelForSequenceClassification.from_pretrained(
#                "path/to/your/reward-model", num_labels=1)
#   def reward_quality(row):
#       text = rm_tok.apply_chat_template(row["messages"], tokenize=False)
#       inputs = rm_tok(text, return_tensors="pt", truncation=True, max_length=1024)
#       with torch.no_grad():
#           score = rm(**inputs).logits[0, 0].item()   # the scalar reward
#       return score   # rank by this; no need to normalize for selection
```

Whichever scorer you use, the output is a number per row. Combine the cheap and expensive signals into one column — but only spend the expensive call on rows that already cleared the heuristic floor:

```python
# Attach a combined quality score. Heuristics are free, so run them on all rows;
# only call the (expensive) model scorer on rows that passed a cheap gate.
def score_quality(dataset, model_scorer=None, heuristic_floor=0.5):
    scored = []
    for row in dataset:
        h = heuristic_quality(row)
        if model_scorer is not None and h >= heuristic_floor:
            q = 0.4 * h + 0.6 * model_scorer(row)   # trust the model more
        else:
            q = h                                    # heuristic-only
        scored.append(q)
    return dataset.add_column("quality", scored)

# For a runnable demo without API calls, score with heuristics only:
near = score_quality(near, model_scorer=None)
print([round(q, 2) for q in near["quality"]])
# [1.0, 1.0, 1.0, 1.0, 0.6]
```

---

## Step 3 — Hard-example mining

### Intuition

This is the technique that separates a curated continual loop from a glorified data hoarder, so it's worth slowing down.

Picture a student reviewing for a final. They have a stack of 500 practice problems and time for 50. Which 50? Not the ones they answer instantly and correctly — re-solving those teaches nothing; the knowledge is already there. The valuable 50 are the ones they currently get *wrong* or get right only by luck. Each of those carries information the student doesn't yet have. The same is true, almost literally, for your model. Training updates weights in proportion to *error*. A row the model already extracts perfectly produces a tiny loss and a tiny gradient — you train on it and the weights barely move. A row the model botches produces a large loss and a large, *informative* gradient. Per row of GPU time, hard examples teach far more.

So the move is: run your **current** model (the one from last round, the one you're about to improve) over the candidate pool, measure how badly it does on each row, and keep the rows it struggles with. "How badly" can be measured two ways:

- **High loss** — the model's own surprise at the correct answer. Directly measures "how much would this row move the weights." This is the purest signal but requires a forward pass through the training model.
- **Low eval F1** — run the model in inference mode, compare its extraction against the reference using the precision/recall/F1 machinery from *Ch18*, and keep rows where F1 is low. Slower (it generates), but it measures the thing you actually care about: getting the right memories.

We'll show the F1 version because it's the most interpretable and reuses Ch18's metric directly. (For the loss version, the recipe is: tokenize each row with the chat template, run a forward pass with `labels` set, and read the per-row loss — the same number the trainer minimizes. Rank descending.)

```python
# hard_mining.py — keep the rows the CURRENT model gets wrong.
# Reuses the model-loading pattern from Ch15/Ch18 and the F1 metric from Ch18.
import json, re
from difflib import SequenceMatcher

# --- (1) Score the current model on a candidate. Returns its extraction. ---
# In a real run you'd load your last-round model with Unsloth, exactly as in
# Ch18, and batch-generate. We sketch a single-row generate() to stay focused.
#
#   from unsloth import FastLanguageModel
#   model, tokenizer = FastLanguageModel.from_pretrained(
#       "outputs/round-N/merged", max_seq_length=1024, load_in_4bit=True)
#   FastLanguageModel.for_inference(model)
#
def model_extract(model, tokenizer, row):
    """Run the current model on a row's conversation; parse its JSON output."""
    prompt_msgs = [m for m in row["messages"] if m["role"] != "assistant"]
    inputs = tokenizer.apply_chat_template(
        prompt_msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(inputs, max_new_tokens=512, do_sample=False)
    text = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []   # a parse failure is itself a 'hard' signal — F1 will be 0

# --- (2) The F1 between predicted and reference memory sets (from Ch18). ---
def _match(a, b, threshold=0.75):
    """Fuzzy text match — same difflib approach Ch18 used as its zero-dep option."""
    ra = " ".join(a.lower().split())
    rb = " ".join(b.lower().split())
    return SequenceMatcher(None, ra, rb).ratio() >= threshold

def memory_f1(pred, gold):
    """Set-level F1 over memory 'text' fields, greedy fuzzy matching."""
    if not pred and not gold:
        return 1.0                         # correctly extracted nothing
    if not pred or not gold:
        return 0.0
    gold_texts = [g["text"] for g in gold]
    used = set()
    tp = 0
    for p in pred:
        for j, gt in enumerate(gold_texts):
            if j not in used and _match(p.get("text", ""), gt):
                used.add(j); tp += 1; break
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# --- (3) Mine: keep the rows where the current model scores low F1. ---
def hard_mine(dataset, model, tokenizer, keep_below=0.7):
    """Attach the current model's F1 per row; keep rows it struggles with."""
    f1s = []
    for row in dataset:
        pred = model_extract(model, tokenizer, row)
        f1s.append(memory_f1(pred, assistant_memories(row)))
    dataset = dataset.add_column("model_f1", f1s)
    # Keep the hard ones — low F1 means the model has something to learn here.
    hard = dataset.filter(lambda r: r["model_f1"] < keep_below)
    return dataset, hard
```

The shape of a real run, with numbers you might actually see on our six-row pool after a couple of continual rounds:

```python
# Illustrative output once you wire in a real loaded model:
#   row                                            model_f1
#   Maya / dark roast preference                      1.00   <- model nails it; SKIP
#   Priya / deploy window + on-call decisions         0.50   <- multi-fact decision; KEEP
#   Sam / Lena sister + cardiology residency          0.40   <- relationship + fact; KEEP
#   Jordan / Postgres vs Mongo decision               0.33   <- the model invents a fact; KEEP
#   Alex / empty chat                                 1.00   <- correctly extracts []; SKIP
#
# Kept (model_f1 < 0.7): the three hard multi-fact rows.
# Dropped: the easy preference and the (correct) empty row.
```

This is exactly the inversion that makes curation powerful. The coffee-preference row — the *most common* thing in your pool — is the *least* valuable to train on, because the model already aces it. The rare decision and relationship rows, which you have few of and which the model fumbles, are gold. Hard-example mining surfaces them automatically.

**One critical caveat — don't only keep the hard rows.** A pool that is 100% the model's failures is a distorted picture of reality; train on that alone and you can destabilize the things the model already does well (a cousin of the catastrophic-forgetting problem from Ch30). Treat hard-mining as a *prioritization* signal, not an absolute filter: heavily favor hard rows, but keep a minority of easy ones so the model is reminded of the full distribution. We bake this in next, with sampling.

---

## Step 4 — Importance sampling for coverage

### Intuition

Even after dedup, quality filtering, and hard-mining, your surviving pool reflects whatever your *product* happens to see most. If 70% of real conversations are casual chats producing preference memories, then 70% of your pool — even the deduped, hard-mined pool — leans preference. Train on that distribution and the model gets sharper on preferences and stays mediocre on the rare-but-important `decision` and `relationship` types.

A balanced diet, not a balanced plate of whatever the cafeteria served most. You don't want to mirror the input distribution; you want to *over-sample the under-represented but important categories* so the model gets enough practice on each. This is importance sampling: deliberately reweighting the pool toward coverage rather than frequency.

The two axes worth balancing for our task:

- **Memory type** — the four pinned types: `preference | fact | decision | relationship`. Decisions and relationships are usually rarer and harder; give them a floor.
- **Conversation style** — a crude but useful proxy: short single-turn snippets vs longer multi-turn threads, casual vs work-context. You can label these with a cheap heuristic (turn count, word count) or a quick classifier.

```python
# coverage_sample.py — sample toward balanced coverage instead of mirroring frequency.
import random
from collections import defaultdict, Counter

def primary_type(row):
    """Label a row by the memory type it most contains (empty -> 'none')."""
    mems = assistant_memories(row)
    if not mems:
        return "none"
    return Counter(m.get("type") for m in mems).most_common(1)[0][0]

def coverage_sample(dataset, target_total, type_floor=0.15, seed=42):
    """Select `target_total` rows, guaranteeing each memory type gets at least
    `type_floor` fraction of the budget if enough rows of that type exist.

    Within a type we KEEP THE HARDEST FIRST (lowest model_f1 if present),
    which is how hard-mining and coverage combine: balance across types,
    prioritize difficulty within a type."""
    rng = random.Random(seed)

    # Bucket rows by primary type.
    buckets = defaultdict(list)
    for i, row in enumerate(dataset):
        buckets[primary_type(row)].append(i)

    # Order each bucket hardest-first when we have a difficulty signal.
    has_f1 = "model_f1" in dataset.column_names
    for t, idxs in buckets.items():
        if has_f1:
            idxs.sort(key=lambda i: dataset[i]["model_f1"])   # low F1 == hard, first
        else:
            rng.shuffle(idxs)

    # Give each NON-empty type a floor of the budget; split the rest by what's
    # available. ('none' rows — correct empty extractions — are useful in small
    # doses so the model learns to say [], but we don't floor them.)
    real_types = [t for t in buckets if t != "none"]
    floor_n = int(target_total * type_floor)

    selected = []
    for t in real_types:
        take = min(floor_n, len(buckets[t]))
        selected.extend(buckets[t][:take])

    # Fill the remaining budget from whatever is left, hardest-first overall.
    remaining = target_total - len(selected)
    leftovers = [i for t in buckets for i in buckets[t] if i not in set(selected)]
    if has_f1:
        leftovers.sort(key=lambda i: dataset[i]["model_f1"])
    else:
        rng.shuffle(leftovers)
    selected.extend(leftovers[:max(0, remaining)])

    return dataset.select(selected[:target_total])

balanced = coverage_sample(near, target_total=5)
print("Selected types:", Counter(primary_type(r) for r in balanced))
# Selected types: Counter({'decision': 2, 'preference': 1, 'relationship': 1, 'none': 1})
```

On the real pool the effect is the point: the single surviving preference row no longer crowds anything out, and the rare `decision` / `relationship` rows are guaranteed seats at the table. In a continual loop where decisions are 5% of traffic but a third of what you *need* the model to be good at, this floor is what keeps the model improving on them round over round.

A related lever you already know from the charter: **replay**. Coverage sampling balances *this round's* new data; replay mixes in ~10–30% prior-round and general data so the model doesn't drift. They're complementary — coverage shapes the new slice, replay protects the old skills. Ch30 owns the replay mixing; here we just make sure the new slice is itself well-balanced before it gets mixed.

---

## Step 5 — The full curation pipeline

Now assemble the four levers in order. The ordering principle is **cheap-and-broad before expensive-and-precise**, so each stage hands a smaller pool to the next:

1. **Exact dedup** (hashing) — pennies, removes obvious repeats, shrinks everything downstream.
2. **Near dedup** (embeddings or MinHash) — moderate cost, removes paraphrase redundancy.
3. **Quality filter** (heuristics, then model on survivors) — drops junk before you waste model-inference on it.
4. **Hard mining** (current model F1) — the most expensive step (it generates), so it runs on the smallest pool.
5. **Coverage sampling** — free, shapes the final selection and folds in the hard-mining signal.

```python
# curate.py — raw candidate pool  ->  lean, high-value training set.
# This is the "select / curate" component Ch30's loop calls each round.

def curate(
    candidates,
    target_total=2000,        # how many rows the next training run can absorb
    near_threshold=0.92,      # near-dedup cosine cutoff (high = cautious)
    quality_min=0.5,          # drop rows below this combined quality score
    model_scorer=None,        # plug in llm_quality or reward_quality; None = heuristics only
    current_model=None,       # your last-round model, for hard mining
    tokenizer=None,
    hard_keep_below=0.7,      # rows with model_f1 below this are 'hard'
    type_floor=0.15,          # min fraction of the budget per memory type
):
    print(f"start:        {len(candidates)} candidates")

    # 1) exact dedup
    ds = exact_dedup(candidates)
    print(f"exact dedup:  {len(ds)}")

    # 2) near dedup
    ds = near_dedup(ds, threshold=near_threshold)
    print(f"near dedup:   {len(ds)}")

    # 3) quality score + filter
    ds = score_quality(ds, model_scorer=model_scorer)
    ds = ds.filter(lambda r: r["quality"] >= quality_min)
    print(f"quality >= {quality_min}: {len(ds)}")

    # 4) hard mining (only if a current model is supplied — it needs a model to run)
    if current_model is not None and tokenizer is not None:
        ds, hard = hard_mine(ds, current_model, tokenizer, keep_below=hard_keep_below)
        # We don't drop the easy rows outright; we keep the difficulty signal
        # (model_f1) and let coverage_sample prioritize hard rows within each type.
        print(f"hard-mined (kept difficulty signal): {len(ds)}")

    # 5) coverage sampling down to the budget
    if len(ds) > target_total:
        ds = coverage_sample(ds, target_total=target_total, type_floor=type_floor)
    print(f"final:        {len(ds)} rows")
    return ds

# Runnable demo end-to-end on the toy pool (heuristics only, no model needed):
final = curate(
    candidates,
    target_total=5,
    quality_min=0.5,
    model_scorer=None,     # set to llm_quality once you want model-based scoring
    current_model=None,    # set to your loaded Ch15 model to enable hard mining
)
print("\nFinal training set:")
for row in final:
    print(f"  [{primary_type(row)}] {user_text(row).split(chr(10)+chr(10),1)[-1][:45]}")
# start:        11 candidates
# exact dedup:  10
# near dedup:   6
# quality >= 0.5: 6
# final:        5 rows
#
# Final training set:
#   [decision] Priya: After the incident review we're moving th
#   [relationship] Sam: My sister Lena just started at the same hosp
#   [decision] Jordan: We picked Postgres over Mongo for the b
#   [preference] Maya: I always grab a dark roast before my 9am s
#   [none] Alex: hey Bmsg: hey Alex: you around later? Bm
```

Eleven noisy candidates became five lean, balanced, high-value rows: the redundant coffee photocopies collapsed to one, the thin chat survived only as a single "learn to output `[]`" example, and the rare hard rows dominate. Hand `final` back to Ch30's loop, which mixes in replay data and ships it to the `SFTTrainer` (the same trainer from *Ch15 - Your First Fine-Tune with Unsloth*). That's a full curation cycle.

---

## How aggressive to be — the honest tradeoffs

Every knob above trades one risk against another. There is no universally correct setting; there is a setting that's right for *how much data you have and how much you trust it*. The honest guidance:

**Dedup threshold.** Set it high (cosine 0.90–0.95, MinHash Jaccard ~0.7). The failure mode of aggressive dedup is subtle and expensive: two rows about the same *topic* often carry *different facts*, and a low threshold throws away the second fact thinking it's a duplicate. Over-keeping a few near-dupes costs you a little training budget; over-dropping costs you coverage you can't get back. When unsure, keep more.

**Quality floor.** Heuristics should rarely be the sole gate — they're blunt and will occasionally punish a correct-but-unusual row (the genuinely-empty chat). Use them to flag, then let a model score the survivors. Set the floor low enough that only clear junk dies on heuristics alone (≈0.5 on our scale), and lean on the model score for the close calls. If you have no model scorer, be *more* lenient with heuristics, not less — a blunt instrument should cut conservatively.

**How hard is too hard?** Hard-example mining has a trap at the extreme: the rows your model gets *most* wrong are sometimes wrong because the *label* is wrong, not because the model is bad. A row with a mislabeled or genuinely ambiguous reference will always score F1 ≈ 0 and will always be "mined" — and training on it teaches the model the mistake. So: prioritize hard rows, but spot-check the hardest few by hand before each round, and never let hard rows be 100% of the set. A blend of roughly 60–80% hard / 20–40% easy keeps the model improving on weaknesses without forgetting strengths. (This is the same forgetting risk Ch30 manages with replay; here it shows up as "don't train only on failures.")

**How much to curate down to.** This is the volume-vs-quality dial from the charter's table. If your curated pool lands in the 1,000–3,000 range, you're in the sweet spot for a continual round on a ~4B model — most projects live there. Curating *below* ~500 rows per round risks too little signal to move the model; curating to *more* than ~3,000 when most of the marginal rows are easy is just burning GPU time for diminishing returns. When you have 30,000 candidates and room for 2,000, the 2,000 you pick matter far more than finding room for 3,000.

**The meta-tradeoff: curation effort vs. just training.** Every filter here costs engineering and compute. For your *first* continual round, you can get away with exact dedup + heuristic quality + a coverage floor and skip the expensive model-based scoring and hard-mining — it's better than training on raw data and costs almost nothing. Add hard-mining and model scoring once you're collecting enough that the easy/redundant rows genuinely dominate (typically a few rounds in, when your pool is several times bigger than your training budget). Don't build the full pipeline before you have the data volume that justifies it.

---

## Common mistakes

**1. Treating volume as progress.** "We collected 50,000 rows this month" is not good news on its own. The number that matters is how many *distinct, hard, well-labeled* rows you have. A loop optimized for collection volume quietly degrades.

**2. Deduping too aggressively and losing coverage.** A cosine threshold of 0.80 feels safe until you realize it merged "we picked Postgres" and "we picked Redis for caching" because they share sentence structure. Keep thresholds high; coverage is harder to recover than budget.

**3. Hard-mining on a stale model.** Mine with your *current* (last-round) model, not the original base model. Rows that were hard three rounds ago may be trivial now; mining against an old model keeps feeding the system problems it already solved.

**4. Forgetting the easy and the empty.** Curate toward hard and rare, but keep a minority of easy rows and some correct-empty (`[]`) rows. A model trained only on hard multi-fact extractions forgets how to say "nothing memorable here," and starts hallucinating memories into casual chatter.

**5. Letting bad labels become "hard examples."** The mining step cannot tell "model is wrong" from "label is wrong." Spot-check the very hardest rows before each round; one mislabeled row mined repeatedly across rounds becomes a persistent bug.

**6. Running the expensive filters first.** Model scoring and hard-mining both run a model per row. Doing them before dedup means paying to score forty copies of the same coffee preference. Cheap-and-broad first, always.

---

## Recap

- In a continual loop you have far more candidate data than you should train on. Curation, not volume, is the lever — redundant and low-quality rows actively hurt by drowning the signal, amplifying noise, and wasting your fixed training budget on things the model already knows.
- **Dedup** in two passes: hash for exact duplicates, then embedding cosine similarity (default, accurate) or MinHash (for very large pools) for near-duplicates. Keep thresholds high.
- **Quality-score** cheaply with heuristics (schema validity, fact density, length sanity), then more precisely with a model — an LLM judge (Ch18 technique) or the reward model from Ch25 — but only on rows that cleared the heuristic floor.
- **Hard-example mining** runs your *current* model over the pool and surfaces the rows it gets wrong (low F1, the Ch18 metric, or high loss). Those carry the most learning signal per GPU-second. Prioritize them, but never train on failures alone.
- **Coverage sampling** balances across the four pinned memory types and conversation styles so rare-but-important categories (decisions, relationships) aren't crowded out by the common easy case.
- The **pipeline** chains them cheap-to-expensive: exact dedup → near dedup → quality filter → hard mining → coverage sampling → a lean training set, handed straight to Ch30's loop for replay-mixing and training.
- Be **cautious** with dedup and quality thresholds, **prioritize but don't isolate** hard examples, and only build the expensive stages once your data volume justifies them.

## Next

**Ch32 — How Much Data, How Often to Retrain**: you've selected the best new data; now decide how much of it justifies a round, how often to retrain, and how much prior data to replay alongside it — the quantities and cadence that keep a continual loop improving instead of drifting.
