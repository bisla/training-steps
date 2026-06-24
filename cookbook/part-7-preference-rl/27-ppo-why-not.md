# Chapter 27 - PPO and the Full RL Loop: Why We Don't Use It Here

For years, when people said "reinforcement learning from human feedback" — RLHF, the thing
that supposedly turned a raw language model into ChatGPT — they meant one specific, heavy,
finicky machine. That machine is called **PPO**, Proximal Policy Optimization. It is the
algorithm behind the original InstructGPT, and for a long stretch it *was* how you did RL on
language models. If you read a blog post from a few years ago about "aligning" a model, it was
almost certainly describing a PPO loop.

So we owe it to you to explain it. Not because you are going to run it — you are not, and this
chapter is going to spend most of its length convincing you that's the right call — but because
the *next* chapter (Ch28 - GRPO) makes a hundred times more sense once you've seen the machine
it replaces. GRPO is "PPO with the most expensive, most fragile part ripped out and thrown
away." You can't appreciate that surgery until you've seen the patient.

This is the one conceptual chapter in the book. Everywhere else, you copy a code block and run
it. Here, the code is a *sketch* — a labeled diagram in Python — because the honest truth is
that hand-rolling a PPO loop is no longer something the ecosystem wants you to do, and the
library we pin (`trl==1.6.0`) has quietly moved PPO into an "experimental" corner to say so.
We'll show you exactly what that looks like so you recognize it in the wild.

---

## What you'll learn

- The classic RLHF/PPO loop in plain English — every moving part and what it's *for*: the
  policy, the reference model and its KL "leash," the reward model, the value head (critic),
  and the advantage signal
- Why PPO is impractical for a solo developer with one GPU: three or four models in memory at
  once, a reward model you have to train first, notorious instability, real cost, and a strong
  tendency to "reward-hack"
- The TRL reality in `trl==1.6.0`: `PPOTrainer`, `PPOConfig`, and
  `AutoModelForCausalLMWithValueHead` are no longer top-level imports — they live in
  `trl.experimental.ppo`, the old manual `trainer.step(...)` loop is gone, and the ecosystem
  has shifted toward GRPO and RLOO
- How Andrej Karpathy frames the whole thing — that RL's real gift is letting a model *discover*
  strategies instead of merely imitating, while full PPO-style RLHF is heavy machinery and
  "barely RL" in practice
- Why all of this sets up GRPO (Ch28) as the practical win: keep the reward signal, drop the
  value model and most of the pain

---

## Concepts you need first

### Where we are in the arc

In Ch24 we made the case for *preference* learning: SFT (Parts 3–4) teaches your model to
imitate good answers, but it can't teach it to *prefer* a better answer over a merely-acceptable
one. In Ch25 you built a **reward model** — a small model that reads a memory-extraction output
and scores how good it is. In Ch26 you used **DPO**, which learns directly from pairs of
"this answer is better than that one" without ever spinning up a reward model at training time.

This chapter sits between the reward model (Ch25) and GRPO (Ch28). It answers a question a sharp
reader will already be asking: *"We built a reward model in Ch25. Isn't the classic way to use a
reward model to run PPO? Why aren't we doing that?"* Good instinct. Here's the full answer.

### "Policy" is just a fancy word for "your model"

RL has its own vocabulary, borrowed from robotics and game-playing, and it can make simple ideas
sound intimidating. The single most important translation:

**The "policy" is your model.** When an RL paper says "we update the policy," it means "we nudge
the model's weights." A policy is a thing that, given a situation, decides what to do. For a
language model, the situation is the prompt-so-far and the "action" is the next token. That's it.
Whenever you see "policy" in this chapter, mentally substitute "the model we're training."

### A reward is a score, not a loss

In SFT, training is driven by a **loss**: the gap between what the model said and the one
"correct" answer in your dataset. RL doesn't have a single correct answer. Instead it has a
**reward** — a number that says "that output was good (high) or bad (low)" *after* the model
produced a whole answer its own way. There's no gold answer to copy; there's only a judge handing
out scores. The model's job is to produce answers that earn higher scores over time.

That's the whole shift from Part 4 to Part 7: from "copy this exact answer" to "do whatever you
like, but earn a high score." For our running example, the "score" is how well a generated batch
of memory objects matches what a good extraction looks like.

### The running example, unchanged

Everything below is framed around the same task as the rest of the book: read a conversation,
emit a JSON array of memory objects. The pinned schema is

```python
# A single memory object (verbatim from the charter — do not drift):
{
    "text": "Sarah prefers dark roast coffee in the morning",  # the fact, as a full sentence
    "type": "preference",                                       # preference | fact | decision | relationship
    "entities": ["Sarah"]                                       # named people, places, or things
}
# The model emits a JSON array of zero or more of these. [] is a valid answer.
```

and the model is trained and served with the exact `SYSTEM_PROMPT` we pinned back in Ch12 and
have reused in every chapter since. None of that changes here. What changes is *how* we'd push the
model to produce better arrays — and why we won't push it this particular way.

---

## The classic RLHF loop, told as a story

Forget code for a moment. Picture training a junior analyst whose job is to read meeting
transcripts and write down the important facts — exactly our memory-extraction task, but with a
person.

**You give them a transcript.** They write a list of facts. (The analyst is the *policy*. The
transcript is the prompt. The list they write is the *action* — really a sequence of actions, one
word at a time.)

**A reviewer scores their list.** Not "here's the right answer," just a number: 8 out of 10. Maybe
the facts were mostly right but one was bundled (two facts crammed into one object), so points off.
The reviewer never writes the list themselves; they only judge. (The reviewer is the **reward
model** from Ch25.)

Now, two problems show up immediately, and the cleverness of PPO is entirely in how it handles
them.

### Problem one: the analyst could go feral chasing the score

If your only instruction is "maximize the reviewer's score," a sufficiently motivated analyst will
find ways to game the reviewer. Maybe the reviewer gives high marks to long lists, so the analyst
starts padding every list with plausible-sounding junk. Maybe it likes the word "decision," so
every fact mysteriously becomes a decision. The analyst is *technically* scoring well while
producing garbage. This is **reward hacking**, and it is not a hypothetical — it is the single
most common way RLHF runs go wrong.

PPO's defense is a **leash**. Before training starts, you make a frozen photocopy of the analyst —
their skills exactly as they came out of SFT. This copy never learns anything; it just sits there.
During training, every time the live analyst writes something, you compare it to what the frozen
copy *would have* written. If the live analyst is drifting far from its sensible original self —
producing weird, unnatural text just to please the reviewer — you penalize that drift.

That frozen photocopy is the **reference model**. The penalty for drifting away from it is the
**KL penalty** (KL stands for Kullback–Leibler divergence, a measure of how far two probability
distributions have wandered apart — but you can hold the whole idea as "how different is the new
behavior from the original behavior"). The KL penalty is the leash that keeps the model from
sprinting off into gibberish-land in pursuit of reward. Tighten the leash and the model barely
changes; loosen it and the model is free to drift — and free to reward-hack.

So now we're already holding **two** models: the live policy and the frozen reference.

### Problem two: a raw score is hard to learn from

Say the analyst earns an 8. Is that good? You genuinely can't tell without context. If this was an
easy, fact-dense transcript where any competent analyst scores a 9, then an 8 is a *disappointment*
— it should push the analyst's behavior *down*. If it was a sparse, ambiguous transcript where most
attempts score a 4, then an 8 is *excellent* — reinforce whatever they just did.

What actually teaches the analyst is not the raw score but **"better or worse than expected."** And
to know what to expect, you need someone who, *before seeing the answer*, predicts the likely
score for this transcript. "This one looks easy — I'd expect about a 9." Then when the real score
comes in, you compute the surprise: actual minus expected.

That predictor is the **value head**, also called the **critic**. In practice it's a small extra
output bolted onto the model that, for any prompt (and partial answer), predicts the expected
future reward. And the surprise — *how much better or worse the outcome was than the critic
predicted* — is the **advantage**. Positive advantage: "do more of that." Negative advantage: "do
less of that." The advantage, not the raw reward, is what actually drives the weight updates.

> **One sentence on GAE, no derivation.** The specific recipe PPO uses to turn a sequence of
> per-token value predictions and rewards into one clean advantage number per step is called
> *Generalized Advantage Estimation* (GAE); intuitively it's a smoothing trick that blends "the
> reward we actually saw" with "the reward the critic expected," trading off noise against bias.
> You will never compute it by hand, and we will not write the equation.

The critic is a model too — usually the same size as your policy, with that extra value head. So
now we're holding the policy, the reference, the reward model, and the critic. **Three to four
full models in GPU memory at once.** Hold that thought; it's the crux of why we won't run this.

### Putting the story together

Here is the full loop in five beats, still no code:

1. **Generate.** The policy reads a batch of transcripts and writes memory-object lists, its own
   way (sampling, not greedy — it needs variety to learn from).
2. **Score.** The reward model (Ch25) scores each list. The reference model says how surprised it
   is by the policy's word choices — that's the KL term.
3. **Estimate.** The critic predicts the expected reward for each; subtract to get the advantage
   (the "better/worse than expected" signal), with the KL penalty folded in so drift is punished.
4. **Update.** Nudge the policy's weights to make high-advantage behavior more likely and
   low-advantage behavior less likely — but only a *little* per step (that's the "Proximal" in PPO:
   small, clipped, careful steps so one weird batch can't blow up the model). Also nudge the
   critic to predict better next time.
5. **Repeat** for thousands of batches, babysitting the whole thing the entire way.

That's RLHF-by-PPO. Every alignment blog post you half-remember is some retelling of those five
beats.

---

## Why this is the wrong machine for you

The loop is elegant. It also asks for almost everything a solo developer with one GPU doesn't
have. Five concrete reasons.

### 1. You're juggling three to four models at once

Re-read step 3. At the moment of an update you may need the **policy** (training, so it needs
gradients and optimizer state — the expensive kind of resident), the **reference** (frozen, but
still occupying memory), the **reward model** (Ch25's scorer, resident), and the **critic** (the
value model, also training). For a 4B base model, the policy alone in a QLoRA setup is the kind of
thing you fought to fit on a consumer card back in Ch8. Now imagine three more model-shaped things
sharing that card. People run PPO on multi-GPU clusters for a reason. On the single L4 or rented
A100 this book targets, it's a memory-management nightmare before you've learned anything.

### 2. You have to train a good reward model *first*

PPO's entire learning signal comes from the reward model. If that scorer is mediocre, PPO will
faithfully optimize your model toward mediocre — or worse, toward the scorer's blind spots. So
before you can even start PPO, you owe yourself a *good* reward model: a few thousand preference
pairs (Ch25's rule of thumb), trained and validated, with its own failure modes understood. That's
a whole sub-project standing between you and your first training step. (Contrast DPO in Ch26, which
skips the separate reward model entirely, and GRPO in Ch28, which can run off a plain Python
function as its reward.)

### 3. It is notoriously unstable and finicky to tune

PPO has a reputation, and it earned it. The KL coefficient (how tight the leash is), the learning
rates for policy *and* critic, the clipping range, the number of optimization passes per batch,
the reward normalization — these all interact, and a bad combination doesn't fail loudly. It fails
*quietly*: the model collapses into repeating one high-scoring phrase, or the KL term explodes and
the model freezes, or rewards drift up while real quality drifts down. Diagnosing that is a
specialist skill. As a reader who, a few chapters ago, was meeting the word "gradient" for the
first time, this is not where you want to spend your one GPU's weekend.

### 4. It's expensive

Generation, scoring with a separate model, value estimation, and small clipped updates — repeated
over thousands of batches — is a lot of compute per unit of learning. Recall the book's speedrun
budget: a working fine-tune for under $30 (Ch0). A serious PPO run blows past that on compute
alone, and most of that spend goes into the machinery (critic, reference, scoring passes) rather
than into the part you care about.

### 5. It's the most reward-hacking-prone method we cover

Every reward-based method can be gamed, but PPO's long, aggressive optimization against a fixed
reward model is especially good at finding the cracks. Give it ten thousand batches to maximize a
scorer and it *will* discover that scorer's weaknesses — the padded lists, the favored words, the
formatting quirk that nudges the score up. The KL leash helps, but tuning the leash is yet another
fragile knob (see reason 3). You spend real effort defending against your own training algorithm.

None of this means PPO is bad. On a well-resourced team with ML engineers, a strong reward model,
and a cluster, PPO is a legitimate, powerful tool. It's just the wrong tool for *this* book's
reader, hardware, and task. We optimize for "run it this weekend and see it work," and PPO fails
that test on every axis.

---

## The TRL reality: PPO has been moved to the back room

Here's the part that surprises people coming from older tutorials. If you learned RLHF a couple of
years ago, you remember a tidy `from trl import PPOTrainer` and a hand-written loop where *you*
called `trainer.step(query_tensors, response_tensors, rewards)` once per batch, feeding in your
own rewards. That world is gone in the version this book pins.

In **`trl==1.6.0`**, three things are true and worth stating plainly:

- **`PPOTrainer`, `PPOConfig`, and `AutoModelForCausalLMWithValueHead` are no longer top-level
  imports.** They have been relocated to `trl.experimental.ppo`. Importing them prints a
  `TRLExperimentalWarning` — the library's way of telling you, in writing, "this is not the
  supported happy path."
- **The old manual `trainer.step(...)` loop is gone.** The modern `PPOTrainer` is `.train()`-based,
  like every other trainer in TRL — you hand it the policy, the reference model, the reward model,
  and the value model up front, and call `.train()`. You no longer hand-feed rewards batch by batch.
- **The ecosystem has shifted toward GRPO and RLOO.** Those are the trainers that get top-level
  imports, active development, and the documentation real estate. PPO is kept around for
  reproducing older work, not for new projects. RLOO (`from trl import RLOOTrainer`) is a lighter
  online-RL alternative worth knowing the name of; GRPO (Ch28) is the one we'll actually run.

The signal from the maintainers couldn't be clearer: a method gets demoted to `experimental` and
slapped with a warning when the community has moved on. So we move on too.

Below is a sketch of what the conceptual loop looks like. **Read it as a labeled diagram, not as
something to run.**

```python
# ----------------------------------------------------------------------------
# Illustrative — conceptual, not runnable.
# For the runnable path, see Chapter 28 (GRPO).
#
# This sketch exists ONLY to show you the moving parts of a PPO/RLHF loop and
# how trl==1.6.0 frames them. Do not copy this into a script and expect it to
# train a model. The import below intentionally prints a TRLExperimentalWarning.
# ----------------------------------------------------------------------------

# PPO no longer lives at the top level of trl. It is in the experimental module,
# and importing it warns you that this is not the supported path:
from trl.experimental.ppo import PPOTrainer, PPOConfig          # -> TRLExperimentalWarning
from trl.experimental.ppo import AutoModelForCausalLMWithValueHead

# The four model-shaped things the loop needs simultaneously (see "Why this is
# the wrong machine for you", reason #1). Each line below is a full model in
# memory:
#
#   policy     = your SFT'd memory-extraction model, with a value head bolted on
#                (this is the "policy" AND the "critic" — the value head is the critic)
#   ref_model  = a FROZEN copy of the policy — the leash (KL penalty)
#   reward_model = the Ch25 scorer that judges each generated memory list
#
# Conceptually:
policy        = AutoModelForCausalLMWithValueHead.from_pretrained("your-sft-memory-model")
ref_model     = AutoModelForCausalLMWithValueHead.from_pretrained("your-sft-memory-model")  # frozen
reward_model  = load_your_reward_model_from_ch25()   # the judge, num_labels=1

# Modern PPOTrainer is .train()-based: you wire the pieces together up front,
# then call .train(). There is NO hand-written trainer.step(query, response,
# rewards) loop anymore — that legacy API is gone in trl 1.6.0.
trainer = PPOTrainer(
    args=PPOConfig(...),            # KL coefficient, clip range, lrs, ... the fragile knobs
    model=policy,                   # the model being trained (policy + value head)
    ref_model=ref_model,           # the leash
    reward_model=reward_model,     # the Ch25 judge
    # ...plus the tokenizer/processing_class and dataset of memory-extraction prompts...
)
trainer.train()                    # generate -> score -> estimate advantage -> small clipped update -> repeat

# What .train() is doing under the hood, in the five beats from the story above:
#   1. policy generates memory-object lists for a batch of transcripts
#   2. reward_model scores each list; ref_model supplies the KL (drift) penalty
#   3. the value head predicts expected reward; advantage = actual - expected
#   4. nudge policy weights toward high-advantage behavior (small, clipped steps);
#      nudge the value head to predict better
#   5. repeat for thousands of batches, babysitting throughout
#
# Notice the cost: steps 1-3 touch three-to-four models every single batch.
# That is the machinery GRPO (Ch28) is about to delete.
```

If you ever *do* see that `TRLExperimentalWarning` print in your own terminal, it's not a bug —
it's the library confirming everything in this section.

---

## Karpathy's framing: RL is the good part, PPO is the heavy part

It helps to hear this from someone who has built these systems at scale. Andrej Karpathy — a
co-founder of OpenAI and former head of AI at Tesla — has talked and written extensively about why
RL matters for language models, and his framing maps almost perfectly onto our argument. The
following is a **paraphrase of his publicly stated views**, not a direct quotation; treat it as
"the spirit of what he's said," and go read his talks and posts for the exact words.

The valuable idea, in his telling, is this: supervised fine-tuning can only teach a model to
**imitate** the answers it was shown. It can never teach the model to find a *better* strategy than
the demonstrations — because there are no demonstrations of the better strategy to copy. RL is what
lets a model **discover** approaches on its own, by trying things and keeping what scores well.
That discovery — going beyond imitation — is the genuine prize, and it's exactly why Part 7 of this
book exists. For our task, it's the difference between a model that mimics the example extractions
in your dataset and one that learns, through trial and reward, to produce *cleaner* atomic facts
than any single example showed it.

But — and this is the part that sets up Ch28 — Karpathy has also been candid that the full
PPO-style RLHF apparatus is, in his words, heavy and in practice "barely RL." The reward model is a
learned, imperfect stand-in for a human judge, the loop spends enormous effort just keeping itself
stable, and a great deal of the machinery is bookkeeping rather than the actual learning you care
about. The *insight* of RL is precious; the *PPO implementation* of it is a lot of expensive
scaffolding around that insight.

That tension — keep the insight, drop the scaffolding — is the entire pitch for the next chapter.

---

## The bridge to Chapter 28: keep the signal, drop the value model

Look back at the story. Which parts are the *point*, and which are *overhead*?

- The reward signal — "this memory list scored well, that one didn't" — is **the point.** It's how
  the model goes beyond imitation. We keep it.
- The reference model and KL leash — keeping the model from drifting into gibberish — is a
  **reasonable safety idea** worth keeping in some lighter form. (GRPO keeps it available but, per
  our pinned config, off by default — more on that in Ch28.)
- The **critic / value model** — the entire apparatus for predicting expected reward so you can
  compute advantage — is the **most expensive and most fragile** piece. It's a whole second
  training model, and it's the part most responsible for PPO's instability.

The breakthrough behind **GRPO** (Group Relative Policy Optimization) — the centerpiece of Ch28 —
is a wonderfully simple swap. Instead of training a separate critic to *predict* the expected
reward, GRPO just generates **a group of answers for the same prompt** and uses *the group's own
average score* as the baseline. "Better or worse than expected" becomes "better or worse than the
other answers I just gave to this exact prompt." No value head. No critic to train, tune, or fit in
memory. The advantage falls out of the group for free.

For our task, that means: for one transcript, generate (say) eight different memory-object lists,
score all eight with a reward signal, and reinforce the ones that beat the group average while
discouraging the ones below it. You keep the thing that makes RL valuable — learning from a score,
discovering better extractions than your examples showed — and you delete the model that caused
most of the pain.

That's why this chapter has no runnable code and the next one does. PPO is the machine worth
understanding and *not* building. GRPO is the machine worth building. Turn the page (Ch28 - GRPO)
and let's actually run it.

---

## Recap

- The classic RLHF loop is **PPO**, and it has five named parts: the **policy** (your model), the
  frozen **reference model** whose **KL penalty** is a leash against drift, the **reward model**
  (Ch25) that scores outputs, the **value head / critic** that predicts expected reward, and the
  **advantage** ("better or worse than expected") that actually drives learning.
- PPO is the wrong tool for a solo developer on one GPU: **three to four models resident at once**,
  a **reward model you must train first**, **notorious instability**, **real cost**, and a strong
  **reward-hacking** tendency.
- In **`trl==1.6.0`**, PPO has been **moved to `trl.experimental.ppo`** (importing it prints a
  `TRLExperimentalWarning`), the old **manual `trainer.step(...)` loop is gone** in favor of a
  `.train()`-based trainer, and the ecosystem has **shifted to GRPO and RLOO**.
- **Karpathy's framing (paraphrased):** RL's real value is letting a model *discover* strategies
  beyond imitation — but full PPO-style RLHF is heavy machinery and "barely RL" in practice.
- **GRPO (Ch28)** keeps the valuable part (learning from a reward signal) and **deletes the value
  model** by using a group of answers as its own baseline — the practical RL win this book actually
  runs.
