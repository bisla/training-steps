# Chapter 29 - Choosing Your Method: SFT vs DPO vs KTO/ORPO vs GRPO vs PPO

You've reached the end of Part 7. You now know — at least in outline — five or six different ways to push a model past plain imitation: DPO, KTO, ORPO, GRPO, PPO, and the reward models that feed some of them. Each chapter sold you on its own technique. That's the job of a chapter. The job of *this* chapter is the opposite: to stop you from reaching for the fanciest tool when the simplest one is sitting right there, already finished, in Part 4.

Here's the uncomfortable truth that organizes everything below. **Most teams who think they need RL actually need a better SFT dataset.** The preference and RL methods in this part are real, they work, and they have a genuine sweet spot — but that sweet spot is narrow, and it sits *on top of* a solid supervised fine-tune, never instead of one. This chapter is the decision guide that tells you which rung of the ladder you're actually on, so you spend your GPU hours on the method that will move your eval numbers and not on the one with the coolest acronym.

---

## What you'll learn

- A decision **flow** you can run in your head: when SFT is enough, and the single signal that tells you it's time to add preference or RL
- A side-by-side **comparison table** of SFT, DPO, KTO, ORPO, GRPO, and PPO (plus a note on RLOO): what data each needs, how many extra models, how much compute, how stable, and when to reach for it
- The concrete **recommended path** for the memory-extraction example — what to do first, second, and (almost) never
- **How much data** each method wants, as ranges with the reasoning behind them
- The honest **failure modes** — including the most important one: when *not* to bother with RL at all
- How the method you pick becomes one **step in a continual-learning loop** (Part 8)

---

## Concepts you need first

### "Imitation" vs "preference" — the textbook and the editor

**Analogy.** Picture two ways of teaching someone to write. The first: hand them a stack of excellent essays and say "write like these." They read, they imitate, they absorb the format and the moves. The second: they hand *you* a draft, and you say "this paragraph is better than that one — do more of the first kind." The first is imitation. The second is preference. Both teach, but they teach differently. Imitation can only show what *good* looks like. Preference can also push *away* from what bad looks like, even when you can't write the perfect example yourself.

**One-line definition.** *Supervised fine-tuning (SFT)* trains the model to reproduce demonstrations — exact target outputs. *Preference and RL methods* train the model to prefer better outputs over worse ones, using comparisons or scores rather than a single gold answer.

**Why it matters here.** This is the entire decision. If you can *write the right answer*, SFT is the tool — it's cheaper, more stable, and you already know how to run it (Ch15 - Your First Fine-Tune with Unsloth). If you can *recognize* a better answer but struggle to demonstrate it consistently, that's the gap preference methods fill. Knowing which side of that line you're on is 90% of choosing a method.

### "Reference model," "reward model," "extra models"

**Analogy.** Some of these methods need a chaperone. When you nudge a model toward preferred answers, there's a risk it wanders off and forgets how to speak English entirely (this really happens — it's called reward hacking or mode collapse). A *reference model* is a frozen copy of where you started, used as a leash: "you can change, but don't drift too far from this." A *reward model* is a separate trained judge that scores outputs so the policy has something to optimize against. These are *extra models* you have to hold in memory or train separately — and each one costs VRAM and complexity.

**One-line definition.** A *reference model* is a frozen snapshot used to penalize drift; a *reward model* is a learned scorer that rates outputs. Methods differ sharply in how many of these they require.

**Why it matters here.** "Number of extra models" is the single best proxy for how painful a method is to run on one GPU. SFT needs zero. DPO needs one (sometimes zero — see below). Full PPO needs *three* extra models live at once. That ladder of complexity is the spine of the comparison table.

### "On-policy" vs "off-policy"

**Analogy.** Off-policy learning is studying from a fixed answer key someone else wrote. On-policy learning is taking the test yourself, getting graded, and studying *your own* mistakes. On-policy is more powerful — the model learns from the exact errors it actually makes — but more expensive, because you have to generate fresh answers every step.

**One-line definition.** *Off-policy* methods (DPO, KTO, ORPO) learn from a pre-collected dataset of comparisons. *On-policy* methods (GRPO, PPO, RLOO) generate fresh samples from the current model during training and score them on the fly.

**Why it matters here.** On-policy methods need a reward *signal you can compute on demand* (a function or a reward model), plus generation during training — which is why they're slower and why GRPO leans on vLLM (Ch9 - The Toolbox) for fast sampling. Off-policy methods just need a dataset sitting on disk. This distinction decides whether you need a programmatic reward at all.

---

## The decision flow (intuition first)

Before any table, internalize the shape of the decision. It's a ladder, and you climb it one rung at a time, only when forced.

```
Start here. Always.
┌─────────────────────────────────────────────────────────────┐
│ RUNG 1 — SFT (supervised fine-tuning)                         │
│ You need format and skill first. Period.                      │
│ Train on demonstrations until the model reliably produces     │
│ the right JSON shape and catches the obvious memories.        │
└─────────────────────────────────────────────────────────────┘
        │
        │  Run your eval (Ch18). Does the model already hit
        │  your eval bar?
        │
        ├─ YES → STOP. You are done. Ship it. Do not do RL.
        │
        └─ NO ↓  But WHY is it failing?
        │
        ├─ It produces malformed JSON / wrong fields / misses
        │  obvious facts.
        │     → This is a DEMONSTRATION gap. Go back to RUNG 1:
        │       more / cleaner / more diverse SFT data. NOT RL.
        │
        └─ The format is fine and the skill is mostly there, but
           the model keeps making a QUALITY choice you can
           describe but can't easily demonstrate at scale
           (over-extracts trivia, splits facts badly, picks the
           wrong `type`, bundles two facts into one).
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ RUNG 2 — PREFERENCE (DPO / KTO / ORPO)                        │
│ You can RECOGNIZE better vs worse even when you can't always  │
│ write the perfect answer. Do you have, or can you cheaply     │
│ build, comparison data?                                       │
│   • pairs (this output > that output)        → DPO            │
│   • single thumbs-up / thumbs-down signals   → KTO            │
│   • want to skip the separate SFT step       → ORPO           │
└─────────────────────────────────────────────────────────────┘
        │
        │  Still short, AND you can write a PROGRAMMATIC reward
        │  (a function that scores an output: valid JSON? schema
        │  match? F1 against a key?) ...
        ▼
┌─────────────────────────────────────────────────────────────┐
│ RUNG 3 — ONLINE RL (GRPO; RLOO as a lighter cousin)           │
│ The model generates its own attempts, your reward function    │
│ grades them, and it learns from its OWN mistakes. Reach here  │
│ when "better" is checkable by code, not just by a static set  │
│ of pairs.                                                     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ RUNG 4 — PPO                                                  │
│ For this reader: basically never. Three extra models, fragile │
│ tuning, and in TRL 1.6.0 it's moved to experimental. GRPO     │
│ gets you the same on-policy benefit at a fraction of the      │
│ pain. See Ch27 for the full conceptual treatment of why.      │
└─────────────────────────────────────────────────────────────┘
```

The two rules that matter most are at the top, so read them twice:

1. **You always start at Rung 1.** There is no path that skips SFT. Preference and RL methods *sharpen* a model that already has the skill; they cannot install the skill from nothing. A model that can't produce valid JSON has a demonstration problem, and no amount of preference optimization fixes a demonstration problem efficiently.

2. **You only climb when SFT plateaus on quality you can describe but not demonstrate.** That phrase is the whole trigger. If you can write the perfect output, add it to your SFT set — that's cheaper and more stable than any method below it. You climb the ladder precisely when you've hit a quality ceiling that *more demonstrations won't raise*, because the problem is one of judgment, not of examples.

---

## The comparison table

Here is the whole of Part 7 on one page. Read it as "what does this cost me, and what do I get." The "extra models" column counts models you must hold or train *beyond* the policy you're optimizing.

| Method | What data you need | Extra models | Compute / VRAM | Stability | Reference model? | When to reach for it |
|---|---|---|---|---|---|---|
| **SFT** | Demonstrations — prompt + the exact correct output | 0 | Lowest. One model, one forward/backward pass. QLoRA fits on a single 24 GB GPU. | Very high — boring and predictable | No | Always first. Teaches format and skill. Your default for memory extraction. |
| **DPO** | Preference **pairs** — same prompt, a `chosen` and a `rejected` completion | 0–1 (frozen reference; can be skipped on a PEFT adapter) | Moderate — two forward passes (policy + reference) per step | High — the gentlest preference method | Yes (often implicit: with LoRA, the frozen base *is* the reference, so `ref_model=None`) | You have or can cheaply build pairs and want to sharpen quality after SFT. The default preference method. |
| **KTO** | **Unpaired** thumbs-up / thumbs-down labels — each output tagged good or bad, no pairing needed | 0–1 (reference, like DPO) | Moderate — similar to DPO | High | Yes (same implicit-reference trick) | You have binary feedback (a 👍/👎 button in your app) but *can't* form clean pairs. Cheaper labels than DPO. |
| **ORPO** | Preference **pairs** (like DPO) — but folded into one stage | 0 | Moderate — single combined SFT+preference objective, no separate reference pass | High-ish — newer, less battle-tested | **No** — that's its selling point | You want to combine SFT and preference into one run and skip maintaining a reference model. Convenience over control. |
| **GRPO** | **Prompts + a reward function** (or reward model). No gold outputs, no pairs — just a way to *score* an attempt | 0 reward models if your reward is a plain Python function; 1 if you train a reward model | Higher — generates `num_generations` samples per prompt every step; wants vLLM for speed | Moderate — needs a sane reward and KL control | Optional (KL penalty to a reference; `beta=0.0` by default = *no* KL leash) | "Better" is checkable by code — valid JSON, schema match, F1 against a key. The RL centerpiece of this book. |
| **PPO** | Prompts + a **reward model** (and a value head) | **3** — reference, reward, and value models live alongside the policy | Highest — four models in play, fragile hyperparameters | Low — the classic "RL is finicky" experience | Yes (KL penalty to reference) | Almost never for this reader. Conceptual only in TRL 1.6.0. See Ch27. |
| *RLOO* (one line) | *Prompts + reward (like GRPO)* | *0–1 (reward), no value model* | *Lower than PPO — drops the value model* | *Moderate* | *Optional KL* | *A lighter online-RL alternative to PPO; if GRPO doesn't fit your problem, look here before PPO.* |

A few things worth saying out loud about that table:

- **The "extra models" column tracks the pain.** Notice the cliff at PPO: going from GRPO's "zero or one extra model" to PPO's "three" is exactly why PPO is impractical on a single consumer GPU and why this book hands it off conceptually.
- **The reference-model column has a happy surprise for LoRA users.** When you train with a LoRA/QLoRA adapter (which you are — Ch6), the frozen base model *is* a perfect reference model. DPO and KTO exploit this: you pass `ref_model=None` and the trainer computes reference log-probs by disabling the adapter. Zero extra VRAM for the reference. ORPO removes the reference entirely.
- **GRPO's KL leash is OFF by default.** In TRL 1.6.0, `GRPOConfig.beta` defaults to `0.0`, meaning no KL penalty toward the reference. That's a deliberate, modern default, but it means you own the responsibility of raising `beta` if your model starts drifting into gibberish. This is the kind of footgun that makes online RL "moderate" stability rather than "high."

---

## The recommended path for memory extraction

Enough generality. Here's what you, specifically, should do for the running example — reading a conversation and emitting a JSON array of memory objects in the pinned schema:

```python
# The pinned schema every method in this book targets — do not drift from it.
{
    "text": "Sarah prefers dark roast coffee in the morning",   # the fact, as a complete sentence
    "type": "preference",                                        # one of: preference | fact | decision | relationship
    "entities": ["Sarah"]                                        # named people, places, or things involved
}
# Output is a JSON array of zero or more such objects ([] is valid).
```

### Step 1 — SFT first, always

Do exactly what Parts 3 and 4 built. Generate or collect demonstrations (Ch13 - Creating Your Training Data with Synthetic Generation), format them as TRL conversational rows with the pinned system prompt verbatim, and run `SFTTrainer` on a QLoRA adapter (Ch15). This is non-negotiable, because the very first thing the model must learn is the *shape* of the answer — valid JSON, the four allowed `type` values, one fact per object. No preference method can teach that shape efficiently; they assume it's already there.

Then **evaluate** (Ch18 - Did It Actually Work?). If your SFT model already clears your eval bar — valid JSON rate near 100%, good precision/recall on the facts that matter — **stop here**. You are done. Everything below is for the case where SFT got you *most* of the way and then plateaued.

### Step 2 — If you have or can cheaply build preference pairs → DPO

Suppose your SFT model is reliable on format but keeps making a *judgment* error you can describe: it over-extracts ("Bob said hi" should not become a memory), or it bundles two facts into one object, or it labels a `decision` as a `fact`. These are quality calls. You can *recognize* the better output even when writing a perfect demonstration for every case is tedious.

That's DPO's home turf. You build `prompt` / `chosen` / `rejected` triples — often by taking two outputs (one from your SFT model, one from a stronger teacher, or two samples and a quick human/heuristic vote) and labeling which is better. The construction matters more than the algorithm.

```python
# DPO on top of the SFT adapter — TRL 1.6.0, pinned APIs.
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()  # apply Unsloth's DPO patch BEFORE constructing the trainer

from trl import DPOTrainer, DPOConfig

# `model` is your SFT-tuned LoRA model; `tokenizer` came from from_pretrained.
# Dataset columns must be exactly: prompt / chosen / rejected.
trainer = DPOTrainer(
    model=model,
    ref_model=None,                 # None → the frozen base under the LoRA adapter IS the reference.
    args=DPOConfig(
        beta=0.1,                   # default; lower = follow preferences harder, higher = stay near reference
        loss_type="sigmoid",        # default DPO loss
    ),
    train_dataset=pref_dataset,     # ~1,000 pairs is a sensible start (see "how much data" below)
    processing_class=tokenizer,     # NOTE: processing_class, NOT tokenizer= — the #1 TRL gotcha
    peft_config=lora_config,
)
trainer.train()
```

If your feedback comes as standalone thumbs-up/thumbs-down rather than pairs (say, an in-app button), reach for **KTO** instead — same idea, unpaired labels, top-level import `from trl import KTOTrainer, KTOConfig`. If you'd rather fold preference into the SFT run and skip maintaining any reference, **ORPO** is the option — but note its import lives in experimental: `from trl.experimental.orpo import ORPOTrainer, ORPOConfig`. For most readers, plain DPO after a separate SFT pass is the clearest, most stable choice.

### Step 3 — If you can write a good programmatic reward → GRPO

Here's the thing about memory extraction that makes it almost suspiciously well-suited to online RL: **a lot of "quality" is checkable by code.** Is the output valid JSON? Do all objects use one of the four allowed `type` values? Does the set of extracted facts match a known key with decent F1? Those are functions, not opinions. And when you can *write the grader as a function*, you no longer need to pre-build a static pile of pairs — you can let the model generate attempts and score them live. That's GRPO.

```python
# GRPO with a programmatic reward — TRL 1.6.0 + Unsloth, pinned APIs.
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)  # verify this patch name against unsloth==2026.6.9;
                                        # if absent, a plain Unsloth-loaded PEFT model + TRL GRPOTrainer works.

from trl import GRPOTrainer, GRPOConfig
import json

def reward_valid_json(completions, **kwargs):
    """+1 for parseable JSON array whose objects use only allowed types; else 0.
    GRPO reward signature: fn(completions, **kwargs) -> list[float].
    Extra dataset columns (e.g. a gold key) arrive as keyword args."""
    allowed = {"preference", "fact", "decision", "relationship"}
    scores = []
    for c in completions:
        try:
            objs = json.loads(c)
            ok = isinstance(objs, list) and all(
                isinstance(o, dict)
                and set(o.keys()) == {"text", "type", "entities"}   # exact pinned schema keys
                and o["type"] in allowed
                for o in objs
            )
            scores.append(1.0 if ok else 0.0)
        except (json.JSONDecodeError, TypeError):
            scores.append(0.0)  # malformed output earns nothing
    return scores

trainer = GRPOTrainer(
    model=model,                       # your SFT adapter, loaded with fast_inference=True
    reward_funcs=reward_valid_json,    # a callable, or a list of callables weighted by reward_weights
    args=GRPOConfig(
        num_generations=8,             # default; samples per prompt to compare against each other
        max_completion_length=256,     # default
        beta=0.0,                      # default: NO KL penalty. Raise it if the model drifts.
        use_vllm=True,                 # set True for fast generation (needs vllm — Ch9)
    ),
    train_dataset=prompt_dataset,      # just PROMPTS — no gold outputs needed
    processing_class=tokenizer,        # again: processing_class, not tokenizer=
)
trainer.train()
```

In practice you'd combine several reward functions (valid-JSON, schema-exactness, F1 against a key) in a list and let `GRPOConfig.reward_weights` balance them. The deep treatment is in the GRPO chapter (Ch28); this guide's job is only to tell you *when* you've arrived at GRPO's doorstep — which is: when your notion of "better" is something a function can check, and you'd rather grade the model's own attempts than hand-build pairs.

### Step 4 — PPO: basically never, and why

You will be tempted by PPO because it's the name everyone knows. Resist. For this reader and this task, PPO is the wrong tool, and not by a little:

- **Three extra models live at once** — a frozen reference, a separate reward model, *and* a value model (the value head) — on top of your policy. That's four models competing for VRAM on a GPU that was already tight with one.
- **It's notoriously fragile.** PPO's hyperparameters (KL coefficient, value loss weight, clip range, advantage normalization) interact in ways that take real RL experience to tune. The failure mode isn't "slightly worse results," it's "the run silently collapses."
- **In TRL 1.6.0 it's been relocated to experimental.** `PPOTrainer`, `PPOConfig`, and `AutoModelForCausalLMWithValueHead` are no longer top-level imports — they live in `trl.experimental.ppo` and emit a `TRLExperimentalWarning`. The old hand-rolled `trainer.step(query, response, rewards)` loop is gone; the current trainer is `.train()`-based. This is the library telling you, structurally, that PPO is not the paved path.
- **GRPO gives you the same on-policy benefit for far less.** GRPO drops the value model entirely (it compares a group of samples against each other instead of estimating a value baseline), which is most of where PPO's pain and VRAM go.

So PPO stays where Ch27 puts it: a *conceptual* chapter that explains the full loop — value head, KL penalty, advantage estimation — so you understand the machinery GRPO descends from, then explicitly hands you off to GRPO. Any PPO code you see in this book is labeled illustrative and not runnable, on purpose. If you genuinely need on-policy RL and GRPO somehow doesn't fit, look at **RLOO** (`from trl import RLOOTrainer`) before PPO — it keeps the online-RL benefit while dropping the value model, landing well short of PPO's complexity.

---

## How much data each method wants

Numbers are ranges, not laws — present them to yourself with the *reason* attached, because the reason tells you which way to adjust.

- **SFT — 500 to a few thousand demonstrations.** Proof-of-life learns the format at 200–500 rows; a solid baseline for a narrow structured task like ours lives around 1,000–3,000; you reach "strong" by 3,000–10,000, after which diversity and quality matter far more than raw count. *Reason:* format is cheap to learn, judgment is expensive; once the shape is locked in, each additional near-duplicate row teaches almost nothing. Epochs: 2–3 for ≤2k rows, 1–2 for larger; stop on eval-loss plateau (Ch17).

- **DPO — 500 to 5,000 pairs, start around 1,000.** *Reason:* each pair carries less information than a full demonstration (it only says "this beats that," not "here is the right answer"), so you need a fair few — but you're sharpening an already-competent model, not teaching from scratch, so you don't need tens of thousands. Quality of the *contrast* (is the `rejected` actually worse in the way you care about?) matters more than count.

- **KTO — similar order to DPO**, but you're labeling individual outputs good/bad rather than pairing them, which is usually cheaper to collect at the same scale. *Reason:* unpaired labels are easier to gather (one button click), so KTO trades a slightly weaker per-example signal for much cheaper labeling.

- **ORPO — pair count comparable to DPO**, since it consumes the same `chosen`/`rejected` structure, just in a single combined stage. *Reason:* it's doing SFT and preference at once, so budget for both being learned from the same data.

- **Reward model (if GRPO/PPO uses a learned scorer) — a few thousand preference pairs.** *Reason:* you're training a small classifier to predict "better/worse"; that's a real model that needs enough labeled comparisons to generalize.

- **GRPO — 500 to 2,000 *prompts*, with `num_generations` of 4–8.** *Reason:* the effective training signal is prompts × generations (each prompt yields a small group of scored attempts to compare), so 1,000 prompts at 8 generations is already 8,000 graded samples per epoch. And crucially, you need *prompts*, not gold outputs — the reward function supplies the "right answer signal," which is exactly why GRPO is attractive when labeled outputs are scarce but a grader is easy to write.

A rough token sanity check for the SFT stage: ~1–5M training tokens is ample for a narrow LoRA fine-tune. If you're far above that, suspect redundancy in your data, not insufficiency.

---

## Honest failure modes — and when NOT to bother with RL at all

This is the section the acronyms don't want you to read.

**If SFT already hits your eval bar, stop.** This is the most common and most expensive mistake in the whole part: doing RL because it's interesting, not because the numbers demand it. Preference and RL methods add models, compute, instability, and a labeling burden. If your SFT model already clears your precision/recall and valid-JSON targets on the eval set (Ch18), every one of those costs buys you nothing. Ship the SFT model and move to deployment (Part 6).

**Don't use RL to fix a demonstration gap.** If the model emits malformed JSON, invents field names, or misses obvious facts, that's not a preference problem — it's a "the model never properly learned the skill" problem. The fix is more, cleaner, more diverse SFT data, possibly a better base model (Ch10). Throwing DPO or GRPO at a model that can't reliably produce the format is like hiring an editor for a writer who hasn't learned the alphabet. Diagnose with the debugging playbook (Ch19) before you climb the ladder.

**Reward hacking is real, especially in GRPO.** When you optimize against a function, the model will find the cheapest way to maximize it — which is often not what you meant. A reward that only checks "valid JSON" can be maximized by emitting `[]` every time (always valid, never wrong!). The defense is to make the reward capture what you actually care about (combine validity *and* F1 against a key so empty output scores zero on recall), and to watch sample outputs during training, not just the reward curve. The remember-that-`beta=0.0` footgun lives here too: with no KL leash, a model chasing a sloppy reward can drift into degenerate text fast.

**Preference data can teach the wrong lesson.** DPO/KTO/ORPO are only as good as the contrast in your data. If your `rejected` examples are worse for an *incidental* reason (they happen to be shorter, say) rather than the reason you care about, the model learns the incidental thing — "longer is better" — and you've made it worse. Audit a sample of your pairs by hand before training.

**Online RL is slow and VRAM-hungry.** GRPO generates `num_generations` completions per prompt every step. Without vLLM (Ch9) for fast generation, this can be painfully slow on a single GPU. Budget accordingly, and don't reach for it when a static DPO dataset would have answered the question.

The throughline: **every rung up the ladder is a cost you should be forced into by a number on your eval set, never a cost you volunteer for because the technique is exciting.**

---

## Putting it together — your method is one step in a loop

Here's the reframe that connects this chapter to Part 8 (Continuous Learning). You've been reading this as a one-time decision: pick a method, train once, done. It isn't. The method you choose is a *step in an ongoing loop*, and that changes how you should think about the choice.

A living memory-extraction system looks like this, round after round: serve the model → log real traffic and failures → harvest the hard cases → retrain → evaluate against a frozen canary set to catch forgetting → serve again. SFT is the backbone of every round. But the *signal* you collect from production is often preference-shaped, not demonstration-shaped — users click 👍/👎, or a downstream check flags an output as wrong. That feedback is exactly the fuel for the methods in this part.

So the realistic shape is layered, not either/or:

1. **Round 0 — SFT** on synthetic and curated demonstrations. Gets you a working model.
2. **Production feedback accrues** — thumbs, corrections, automated checks on live outputs.
3. **Next round — SFT on new demonstrations + a preference pass** (DPO from collected pairs, or KTO from the thumbs) to sharpen the specific judgment errors production revealed. Or, if your checks are programmatic, a GRPO pass against them.
4. **Re-evaluate on the frozen canary set** (Ch18) and watch for forgetting; mix in ~10–30% prior/general data on retrains (default ~20%) so the model doesn't lose old skills while gaining new ones.
5. **Serve. Repeat.**

Part 8 builds this loop in full — the replay ratios, the canary sets, the data-flywheel mechanics. The point to carry out of *this* chapter is that "choosing a method" isn't a fork in the road you take once. It's a tool you pick up at the right moment in each turn of the wheel: SFT to install and maintain the skill, preference methods to fold in the judgment signal your users hand you for free, GRPO when your quality bar is something code can check. Pick the rung the current round's eval gap actually puts you on — no higher.

---

## Recap

- **Start at SFT, always.** Preference and RL sharpen a skill; they cannot install one. A model that can't produce valid JSON has a demonstration problem, not a preference problem.
- **Climb the ladder only when SFT plateaus on quality you can describe but not demonstrate.** If you can write the right answer, add it to SFT — it's cheaper and more stable.
- **The "extra models" column is the pain meter:** SFT 0, DPO/KTO 0–1 (the frozen LoRA base doubles as the reference), ORPO 0, GRPO 0–1, PPO *three*.
- **For memory extraction:** SFT first → DPO if you have/can build pairs → GRPO if your reward is a function (valid JSON, schema, F1) → PPO basically never (Ch27 explains why; in TRL 1.6.0 it's experimental and needs four models).
- **API locations to get right:** `KTOTrainer`/`KTOConfig` are top-level; `ORPOTrainer`/`ORPOConfig` live in `trl.experimental.orpo`; `PPOTrainer` lives in `trl.experimental.ppo`. Every RL trainer takes `processing_class=tokenizer`, not `tokenizer=`. GRPO's `beta` defaults to `0.0` (no KL leash).
- **Data, as ranges:** SFT 500–few-thousand demos; DPO 500–5,000 pairs (start ~1,000); GRPO 500–2,000 prompts × 4–8 generations; reward model a few thousand pairs.
- **The biggest failure mode is doing RL you didn't need.** If SFT clears your eval bar, stop. Watch for reward hacking (empty `[]` games a JSON-only reward) and bad preference contrasts.
- **Your method is one step in a loop.** Part 8 turns this single decision into an ongoing flywheel: SFT to maintain the skill, preference methods to absorb production feedback, GRPO when "better" is checkable by code.

## Next

Part 8 - Continuous Learning: take the method you just chose and run it as a living system — collecting feedback, retraining in rounds, guarding against forgetting with replay and a frozen canary eval set.
