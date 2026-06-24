# Ch3 - Prompting vs RAG vs Fine-Tuning vs Full Training

You have a problem. You want an LLM to do something specific — in our case, read a chunk of conversation and spit out a clean JSON list of memories. You open the docs and find four different ways people solve this kind of problem: prompting, RAG, fine-tuning, and full training. Nobody tells you which one to actually use, or why. This chapter fixes that.

By the end you'll know exactly where each approach lives on the tradeoff map, and why a small QLoRA fine-tune is the right answer for memory extraction specifically.

---

## What you'll learn

- What each of the four approaches actually does (one-paragraph level, no ML background assumed)
- A simple decision flowchart you can apply to any LLM task
- Concrete cost, effort, and maintenance tradeoffs for each approach
- Why memory extraction sits in the fine-tuning sweet spot
- The one thing fine-tuning genuinely cannot do (and what to use instead)

---

## Concepts you need first

### LLM "inference" vs "training"

**Analogy.** A human can answer questions about a book they've already read. That's *inference* — applying existing knowledge. Re-reading a new book so they can answer questions about it later — that's *training* (or at least studying). These are fundamentally different activities with different costs.

**One-line definition.** *Inference* is running a model to get an output. *Training* is adjusting the model's internal numbers to change its future behavior.

**Why it matters here.** Prompting and RAG happen at inference time — you feed the model more words. Fine-tuning and full training happen before inference — you change the model itself. This distinction drives every tradeoff below.

### Parameters

**Analogy.** A model is like a massive mixing board with billions of knobs. Each knob is a *parameter* — a number. The values of all those knobs together determine how the model responds to any input. Training means turning knobs. Inference means playing through the board as-is.

**One-line definition.** Parameters are the numerical weights stored inside a model. A "7B model" has roughly 7 billion of them.

**Why it matters here.** Different approaches touch different numbers of knobs — from zero (prompting) to all of them (full training). More knobs touched = more compute, more risk, more potential upside.

---

## The four approaches, plainly

### 1. Prompting

You write careful instructions in the prompt. You include examples. You describe the output format precisely. The model's weights never change — you're just being very explicit about what you want each time you call it.

**What it costs:** Your time writing the prompt. A few cents per API call. Zero setup.

**What it buys:** Fast iteration. Works today. No infrastructure.

**Where it breaks down:** Every call re-reads your instructions. The model may drift — follow your format 90% of the time, hallucinate the other 10%. Harder tasks with subtle output requirements drift more. And because the instructions go in the context window every single time, you pay for them in tokens on every request.

### 2. RAG (Retrieval-Augmented Generation)

RAG stands for Retrieval-Augmented Generation. Before calling the model, you search a database for relevant documents and stuff them into the prompt alongside the user's question. The model never changes — you're just giving it better context each call.

**What it costs:** A vector database, an embedding model, a retrieval pipeline. More moving parts than raw prompting.

**What it buys:** The model can "know" things that weren't in its training data. You update the database, not the model. Fresh facts stay fresh.

**Where it breaks down:** RAG is about *knowledge*, not *skill*. If the model doesn't already know *how* to do your task, retrieving more documents won't teach it. RAG also doesn't make the model better at following a rigid output format.

### 3. Fine-tuning (LoRA / QLoRA)

You take a pretrained model and continue training it on your own dataset — but only for a specific task, and only for a short while. With LoRA/QLoRA (covered in depth in Ch6 - LoRA and QLoRA Without the Math Headache), you don't retrain all 7 billion parameters. You train a small adapter — a thin layer of new weights — and leave the base model frozen underneath.

**What it costs:** A GPU for a few hours (roughly $5–$20 for a 7B model on a cloud instance). A dataset of a few hundred to a few thousand examples. Some setup time.

**What it buys:** The model learns the *skill* itself. It becomes reliably better at your specific task. Outputs are consistent without burning tokens on long instructions every call. Inference gets cheaper because the system prompt shrinks dramatically.

**Where it breaks down:** Fine-tuning teaches *behavior*, not *facts*. If you want the model to know about events after its training cutoff, or proprietary documents it's never seen, that's RAG's job. Fine-tuning on facts tends to produce confident hallucinations, not reliable recall.

### 4. Full training (pretraining from scratch)

You train a model from random weights on a massive corpus — hundreds of billions of tokens, thousands of GPUs, weeks of compute. This is what Anthropic, Google, and Meta do to produce the base models everyone else starts from.

**What it costs:** Millions of dollars. Hundreds of GPU-months. A specialized research team.

**What it buys:** A model that genuinely knows a domain from the ground up — not just a skill on top of someone else's knowledge.

**Where it breaks down:** Almost certainly not what you need. Unless you're a well-funded lab with a very specific domain (e.g., genomics sequences that look nothing like natural language), this column doesn't apply to you.

---

## The decision flowchart

Work through these questions in order. The first "yes" is your answer.

```
Is the model already good at this task with a careful prompt?
  └─ Yes → Use prompting. Ship it.
  └─ No ↓

Does the model fail because it lacks up-to-date or proprietary facts?
  └─ Yes → Use RAG (possibly with prompting on top).
  └─ No ↓

Does the model fail because it doesn't reliably produce the right *structure* or *style*,
or because it doesn't consistently apply a specific *skill*?
  └─ Yes → Fine-tune with LoRA/QLoRA.
  └─ No ↓

Is the domain so alien (non-natural-language sequences, ultra-niche jargon corpus)
that even a top frontier model is near-random?
  └─ Yes → Full pretraining or continued pretraining (rare; get expert help).
  └─ No → Re-examine the problem. You likely have a data or prompt issue, not a model issue.
```

Print this out. Tape it above your monitor. You'll use it more than you expect.

---

## Where memory extraction lands

Let's run our task through the flowchart.

**Step 1 — Does prompting work?**

Sort of. A frontier model like Claude or GPT-4 can extract memories from a conversation if you write a detailed prompt with a JSON schema and a few examples. You'll get decent results ~80–85% of the time on clean input.

But "sort of" is the problem. Memory extraction is a *structured-output skill*:

- The JSON schema must be exact — wrong field names break downstream code
- The memory text must be atomic (one fact per memory, not a paragraph summary)
- The `type` classification must be consistent (`preference` vs `fact` vs `relationship`) — not approximate
- Subtle implied facts ("she mentioned not liking meetings before 9am" → a `preference` memory) must be caught

A prompt alone drifts. On edge cases — ambiguous phrasing, multi-party conversations, code-mixed text — the output degrades unpredictably. You end up with a long system prompt (hundreds of tokens, billed on every call) *and* inconsistent behavior.

**Step 2 — Is the failure about missing facts?**

No. The model isn't failing because it doesn't *know* things. It's failing because it doesn't reliably *apply* the skill in exactly the way we need it applied.

**Step 3 — Is it a structure/skill problem?**

Yes. That's exactly what this is. The model needs to learn:

- Our specific JSON schema, not a generic one
- Our definition of "atomic memory" (not summaries, not paraphrases, not duplicates)
- Our taxonomy of memory types
- When to extract vs. when to skip something that isn't worth remembering

This is a skill. Skills are learned. A small fine-tune on 500–2000 examples will encode this behavior reliably into the adapter weights, so the model applies it consistently at inference without a 500-token preamble.

**The verdict: QLoRA fine-tune.**

A QLoRA fine-tune on a 7B model (Qwen3-8B or Gemma 3 7B) is the sweet spot because:

- 7B models are small enough to train on a single consumer GPU (≈24 GB VRAM) or a cheap cloud instance
- QLoRA compresses the base model during training, cutting memory use roughly in half again
- The resulting model can run inference cheaply — even on CPU if needed, though slowly
- 500–2000 high-quality training examples is achievable (Ch13 - Creating Your Training Data with Synthetic Generation covers how to generate them)

---

## What the tradeoffs look like side by side

| | Prompting | RAG | QLoRA fine-tune | Full training |
|---|---|---|---|---|
| **Setup time** | Hours | Days–weeks | 1–2 days | Months |
| **GPU needed** | No | No | Yes (training) | Many, for weeks |
| **Approx. cost** | $0 upfront | $50–$500/mo infra | $5–$30 one-time | $1M+ |
| **Consistent output format** | Moderate | Moderate | High | High |
| **Handles fresh facts** | No | Yes | No | Depends |
| **Inference token cost** | High (long prompt) | High | Low (short prompt) | Low |
| **Maintenance** | Rewrite prompt as needs evolve | Keep DB fresh | Retrain on new data | Retrain entire model |
| **Good for skills** | Sometimes | No | Yes | Yes |
| **Good for knowledge** | Sometimes | Yes | No | Yes |

The numbers in the cost row are rough ballparks for a mid-2020s cloud setup — treat them as order-of-magnitude guidance, not quotes.

---

## The one honest limit of fine-tuning

This deserves its own section because it's the most common misconception.

**Fine-tuning does not reliably inject new facts.**

If you fine-tune on a dataset full of statements like "The CEO of Acme Corp is Jane Smith," the model will *sometimes* recall that fact — but it will also sometimes confuse it with similar facts, hallucinate a different name, or confidently state the opposite. The model hasn't added a row to a database. It's nudged the statistics of a billion-parameter function. That's not the same thing.

The rule of thumb: **facts go in RAG; skills go in fine-tuning.**

For memory extraction, this isn't a problem — we're not teaching the model facts about the world. We're teaching it a *procedure*: how to read text and output a specific JSON structure. That's exactly what fine-tuning is good at.

Where it would become a problem: if you wanted your model to know every product in your catalog, every customer's history, every internal policy document. Those belong in a retrieval system that stays current, not baked into weights that you retrain infrequently.

---

## A minimal prompting baseline (so you can compare)

Before you fine-tune anything, you should always have a prompting baseline. Here's one for memory extraction. Run this first — it's your benchmark to beat.

```python
# baseline_prompt.py
# A simple prompting baseline for memory extraction.
# Run this BEFORE fine-tuning so you have something to compare against.
# Requires: pip install anthropic  (or swap in openai — schema is the same)

import json
import anthropic  # pip install anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

# This is our target JSON schema. Every approach in this book outputs this shape.
MEMORY_SCHEMA = """
[
  {
    "text": "A single, self-contained fact or preference, written as a statement.",
    "type": "fact | preference | decision | relationship",
    "entities": ["list", "of", "named", "things", "mentioned"]
  }
]
"""

# The system prompt carries all the instructions. Notice how long it has to be
# to get consistent output — this is what fine-tuning lets us replace.
SYSTEM_PROMPT = f"""You are a memory extraction engine. 
Given a conversation or note, extract every memorable fact, preference, 
decision, or relationship mentioned. 

Output ONLY a valid JSON array. No explanation, no markdown fences.
Each item must follow this schema exactly:
{MEMORY_SCHEMA}

Rules:
- Each memory must be ONE atomic fact. Split compound facts into separate items.
- Write each "text" as a complete standalone sentence (it will be read without context).
- "type" must be exactly one of: fact, preference, decision, relationship
- "entities" lists people, places, products, or organizations mentioned in that memory.
- Skip trivial filler ("they said hello"). Keep only things worth remembering.
- If nothing is worth remembering, return an empty array: []
"""

def extract_memories(conversation_text: str) -> list[dict]:
    """
    Send a conversation to the model and get back a list of memory dicts.
    Returns an empty list if parsing fails — never raises.
    """
    response = client.messages.create(
        model="claude-opus-4-5",     # frontier model; good baseline but expensive
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": conversation_text}
        ]
    )
    
    raw = response.content[0].text.strip()
    
    try:
        memories = json.loads(raw)
        return memories
    except json.JSONDecodeError:
        # The model didn't return clean JSON — a real failure mode we'll measure
        print(f"[WARN] JSON parse failed. Raw output:\n{raw[:300]}")
        return []


# --- Try it on a sample conversation ---
sample = """
Alice: I finally switched to oat milk. Can't do dairy anymore.
Bob: Oh nice. Are you going vegan?
Alice: No, just lactose intolerant. Anyway, I've decided to go with Figma for 
      the new design system — we looked at Sketch but the team prefers the 
      web-based workflow.
Bob: Makes sense. I'll set up the shared library this week.
Alice: Perfect. Also, can we move our Monday syncs to 9:30? 8am is brutal.
"""

memories = extract_memories(sample)

print(f"Extracted {len(memories)} memories:\n")
for i, m in enumerate(memories, 1):
    # Pretty-print each memory so it's easy to scan
    print(f"  [{i}] ({m.get('type', '?')}) {m.get('text', '')}")
    print(f"       entities: {m.get('entities', [])}")
    print()
```

When you run this, you'll probably get reasonable results. A frontier model on a clean example like this one does fine. The cracks show when you run it on:

- 20-message threads with interruptions and topic switches
- Conversations with ambiguous pronoun references
- Inputs where the user wants a very specific `type` taxonomy consistently applied
- High volume (every call re-pays the full prompt cost in tokens)

That's what the fine-tuned model fixes. Ch18 - Did It Actually Work? Evaluating Memory Extraction will show you how to measure the gap quantitatively.

---

## Can you combine approaches?

Yes, and you should consider it. A common production pattern:

1. **Fine-tune for the skill** — memory extraction with your exact schema
2. **RAG for context** — if the extraction needs to deduplicate against existing memories ("we already have this fact"), retrieve them and pass them in

The fine-tuned model handles the skill reliably. The retrieval layer handles freshness and deduplication. Each tool does what it's good at.

For this book, we'll build the fine-tuning piece. Adding RAG on top is a natural next step once the core model works.

---

## Common mistakes

**Mistake 1: Fine-tuning when prompting hasn't been tried properly.**

Spending a day training a model when you could have gotten there with a better prompt. Before you fine-tune, spend at least two hours iterating on the prompt. Add examples (few-shot prompting). Be specific about failure cases. If a good prompt gets you to 90%+ on your test set, you may not need to fine-tune at all.

*Fix:* Build the prompting baseline first. Measure it. Fine-tune only when prompting has a real ceiling.

**Mistake 2: Using fine-tuning to inject facts.**

Training a model on "our product costs $49/month" and expecting it to recall that reliably. It won't. The model may confidently say $49 sometimes and $99 others.

*Fix:* Facts go in RAG or the system prompt. Fine-tuning is for skills and behaviors.

**Mistake 3: Skipping the baseline comparison.**

Fine-tuning, evaluating, and declaring success — without comparing to what prompting alone would have produced. You don't know if you actually improved anything.

*Fix:* Always run the prompting baseline first and save its outputs. Ch18 - Did It Actually Work? Evaluating Memory Extraction covers this systematically.

**Mistake 4: Conflating fine-tuning cost with inference cost.**

"Fine-tuning is expensive" — yes, the training run costs money. But once the model is trained, each inference call is cheap because the system prompt shrinks from 400 tokens to ~20. At high call volume, fine-tuning pays for itself quickly.

*Fix:* Think in total cost of ownership. Estimate your call volume and multiply by token cost for both approaches.

**Mistake 5: Reaching for full training when fine-tuning would work.**

Unless your domain requires building a model from scratch (rare), LoRA/QLoRA fine-tuning of an existing base model is almost always the right call. Full pretraining is not a "better" version of fine-tuning; it's a different, enormously more expensive tool for a different problem.

*Fix:* Trust the flowchart. The "full training" branch is for teams with research budgets, not individual practitioners.

---

## Recap

- **Prompting** requires no setup but costs tokens per call and drifts on structured tasks
- **RAG** solves the *knowledge* problem (fresh facts, private documents) but not the *skill* problem
- **Fine-tuning** (LoRA/QLoRA) encodes a skill into the model — consistent, cheap at inference, wrong tool for facts
- **Full training** is for well-funded labs, not application developers
- Memory extraction is a structured-output skill — consistent schema, consistent taxonomy — which is exactly what fine-tuning solves
- Fine-tuning does not reliably inject facts; combine with RAG when freshness matters
- Always build the prompting baseline first; measure both before calling fine-tuning a win
- The decision flowchart: can prompting do it? → missing facts? → missing skill? → fine-tune
- QLoRA on a 7B model is the practical sweet spot: single GPU, hours of training, ~$5–$30

## Next

Ch4 - Transformers and LLMs in 20 Minutes: before we touch the training code, you need a working mental model of what's actually inside these models — just enough to understand what LoRA is modifying and why it works.
