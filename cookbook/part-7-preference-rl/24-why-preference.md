# Chapter 24 - Beyond Imitation: Why Preference and RL

Up to this point, the whole book has been built on one quiet assumption: that for every conversation, there is *the* right memory extraction, and our job is to teach the model to copy it. We generated training data where each conversation had one gold answer. We trained the model to reproduce that answer. We evaluated by checking how close the model got to it.

That assumption took us a long way. Our fine-tuned memory extractor reliably emits clean JSON, uses the right field names, and picks up most of the durable facts in a conversation. For a huge class of problems, that is the finish line.

But it has a ceiling. Some things you want from a model are not "copy this exact answer" — they are "of two reasonable answers, prefer the better one." There is no single gold string to imitate. There is a *spectrum* of better and worse, and the only way to teach a spectrum is to show the model comparisons, not copies. That is what Part 7 is about. This chapter is the on-ramp: no heavy training code yet, just the intuition for why a whole new family of techniques exists and what each one is for.

---

## What you'll learn

- Why supervised fine-tuning (SFT) is fundamentally *imitation*, and what that buys you — plus where imitation hits a wall
- The difference between teaching "copy this answer" and teaching "this answer is better than that one"
- What a *preference signal* (a "reward") actually is, in plain English, with concrete chosen-vs-rejected examples on a real conversation from our memory task
- Andrej Karpathy's framing of why reinforcement learning (RL) can *discover* strategies that imitation never could — and his honest caution that the common form of RL used on language models "is just barely RL"
- A map of the rest of Part 7: reward functions and reward models, DPO, why we skip PPO, GRPO as the practical method, and how to choose between them

---

## Concepts you need first

### What SFT actually did (a one-paragraph refresher)

In Part 4 we ran *supervised fine-tuning*. "Supervised" means every training example came with the correct answer attached — a conversation paired with its gold memory JSON. The model's entire job during training was to make its own output look as much like that gold answer as possible, token by token. When it guessed a token that matched the gold, it was nudged to do that more; when it guessed wrong, it was nudged away. Do that across a few thousand examples and the model internalizes the pattern. We called this teaching a behavior. A more precise word for it is **imitation**: the model learns to imitate the demonstrations you gave it. Hold onto that word — it is the hinge this entire chapter turns on.

### "Better" is not the same as "correct"

For some tasks, an answer is simply right or wrong. Is this JSON valid? Yes or no. Does it parse? Yes or no. But for most of what we actually care about, answers come in shades. One memory extraction can be *more complete* than another. One can be *more faithful* to what was actually said. One can avoid *inventing* facts the conversation never contained. None of these are pass/fail — they are more-or-less. The moment your quality bar is a "more-or-less," pure imitation starts to struggle, because imitation only knows how to chase one target.

### A reward, in the most ordinary sense

When this chapter says "reward," do not picture an equation. Picture a coach. After the model produces an answer, something looks at it and says "that one's better" or "that one's worse." That judgment — a thumbs up, a score, a ranking of two answers — is the reward signal. It does not tell the model *what to say*. It tells the model *how good what it just said was*. That single shift, from "here is the answer, copy it" to "here is a verdict on the answer you just produced," is the whole conceptual leap of Part 7. Everything else is plumbing.

---

## Imitation has a ceiling

Here is an analogy worth sitting with.

Imagine teaching someone to cook by handing them one recipe card per dish and saying: "Make it exactly like the card." They get good — impressively good — at reproducing those specific cards. Hand them the card for risotto and they make a fine risotto. This is SFT. It works, and for a lot of cooking it is genuinely all you need.

But notice what they *cannot* learn this way. They cannot learn to taste two versions of the same risotto and tell you which is better. They have never been asked to compare. They have only ever been asked to copy. Show them a dish that is slightly under-seasoned versus one that is perfectly seasoned and, if neither matches a card they memorized, they have no way to express a preference. The recipe-card method has no vocabulary for "better." It only has a vocabulary for "same as the card."

Now push on it harder. What if, for a given dish, there are *many* good versions? A risotto can be excellent in five different ways. If you train only by "match this one card," you implicitly tell the cook that the four other excellent versions are wrong — they don't match the card. That is the deep limitation of imitation: **it can only ever be as good as the single demonstration you gave it, and it treats every other good answer as a mistake.**

This is not a flaw in how we did SFT. It is a flaw in what imitation *is*. To teach "better," you need a fundamentally different signal — one based on comparison, not copying.

---

## Where our memory extractor hits the wall

Let's make this concrete on the running example. Our SFT model is good. Here is what "good" means and where it quietly falls short.

Take this conversation:

```
Maya: Quick heads up — I'm switching my standups to async, mornings are rough for me lately.
Tom: No problem. Is that just this sprint or going forward?
Maya: Going forward. Also my sister Priya is visiting from Seattle next week so I'll be offline Thursday.
Tom: Got it. I'll let the team know about Thursday.
Maya: Thanks. Oh and we locked the vendor — going with Brightline for the redesign.
```

After Part 4, our fine-tuned model produces clean, valid JSON every time. The *format* is a solved problem. But ask it to extract this conversation a few times (generation is not perfectly deterministic) and you'll see outputs that are all well-formed yet meaningfully different in quality.

Here is a **weaker** extraction — perfectly valid JSON, nothing malformed:

```json
[
  {
    "text": "Maya is switching her standups to async.",
    "type": "preference",
    "entities": ["Maya"]
  },
  {
    "text": "Maya has a sister.",
    "type": "relationship",
    "entities": ["Maya"]
  }
]
```

And here is a **stronger** extraction of the same conversation — also valid JSON:

```json
[
  {
    "text": "Maya is switching her standups to async going forward because mornings have been rough for her.",
    "type": "preference",
    "entities": ["Maya"]
  },
  {
    "text": "Maya will be offline on Thursday because her sister Priya is visiting from Seattle.",
    "type": "fact",
    "entities": ["Maya", "Priya", "Seattle"]
  },
  {
    "text": "Priya is Maya's sister.",
    "type": "relationship",
    "entities": ["Priya", "Maya"]
  },
  {
    "text": "The team decided to go with Brightline as the vendor for the redesign.",
    "type": "decision",
    "entities": ["Brightline"]
  }
]
```

Look at the gap. The weaker one is *not wrong* — every memory in it is true, the JSON is clean, the field names are right. An SFT loss would not scream about it. But the stronger one is **more complete** (it caught the vendor decision and the Thursday absence), **more faithful** (it captured that the async switch is permanent and the reason for it), and it correctly named Priya and her relationship instead of vaguely noting "Maya has a sister."

Now here is a third extraction — the one we most want to *push the model away from*:

```json
[
  {
    "text": "Maya prefers async standups.",
    "type": "preference",
    "entities": ["Maya"]
  },
  {
    "text": "Maya's sister Priya recently moved to Seattle from another city.",
    "type": "fact",
    "entities": ["Maya", "Priya", "Seattle"]
  },
  {
    "text": "Maya and Tom agreed to evaluate two more vendors before deciding.",
    "type": "decision",
    "entities": ["Maya", "Tom"]
  }
]
```

This one is the most dangerous, because it is the most *confident-looking* and the JSON is immaculate. But Priya did not "recently move to Seattle" — she is *visiting from* Seattle. And nobody agreed to evaluate two more vendors — they *locked* Brightline. These are **hallucinations**: plausible, well-formatted, and false. In a memory system, a hallucinated fact is worse than a missing one, because it gets stored and trusted and recalled later as if it were real.

So we have three valid-JSON answers and a clear ranking: the complete-and-faithful one is best, the thin-but-true one is in the middle, and the confident-hallucination one is worst. **SFT has no clean way to teach this ranking.** It was trained to reproduce one gold answer per conversation. It cannot be told "prefer the complete one over the thin one, and prefer either of those over the one that invents facts." That instruction is a *comparison*, and SFT does not speak comparison.

This is the wall. And the thing on the other side of it is preference learning.

---

## From "copy this" to "this one is better"

Preference learning flips the training signal. Instead of one gold answer per input, you give the model **pairs**: two answers to the same prompt, with a label saying which one is preferred. The jargon is **chosen** (the better one) and **rejected** (the worse one).

Concretely, for the conversation above, a preference pair looks like this — the same prompt, two completions, a verdict:

```python
# A single preference example. This is the shape of preference data;
# we'll build real datasets like this in Ch25 and Ch26.
{
    "prompt": "<the SYSTEM_PROMPT + the Maya/Tom conversation>",

    # The completion we want the model to lean toward:
    "chosen":  '[{"text": "Maya is switching her standups to async going forward '
               'because mornings have been rough for her.", "type": "preference", '
               '"entities": ["Maya"]}, {"text": "Maya will be offline on Thursday '
               'because her sister Priya is visiting from Seattle.", "type": "fact", '
               '"entities": ["Maya", "Priya", "Seattle"]}, {"text": "Priya is Maya\'s '
               'sister.", "type": "relationship", "entities": ["Priya", "Maya"]}, '
               '{"text": "The team decided to go with Brightline as the vendor for '
               'the redesign.", "type": "decision", "entities": ["Brightline"]}]',

    # The completion we want the model to lean away from
    # (here, the hallucinated one — invented a move to Seattle and a fake decision):
    "rejected": '[{"text": "Maya prefers async standups.", "type": "preference", '
                '"entities": ["Maya"]}, {"text": "Maya\'s sister Priya recently moved '
                'to Seattle from another city.", "type": "fact", "entities": ["Maya", '
                '"Priya", "Seattle"]}, {"text": "Maya and Tom agreed to evaluate two '
                'more vendors before deciding.", "type": "decision", "entities": '
                '["Maya", "Tom"]}]',
}
```

Notice what is *not* here: there is no claim that `chosen` is the one and only correct answer. There may be other extractions just as good or better. All the pair asserts is a *relative* judgment: between these two, the first is better. That is a far weaker, far more honest claim than "this is the gold answer" — and paradoxically, that weakness is its strength. You can express preferences you could never express as a single demonstration. "Prefer faithful over hallucinated." "Prefer complete over thin." "Prefer the one that names entities correctly." Each becomes a pile of pairs.

The training process then does something imitation never could: it learns the *direction* of "better." It does not just memorize the chosen answers — it learns what makes chosen answers tend to beat rejected ones, and it can apply that pull to conversations it has never seen. The recipe-card cook finally gets to taste two risottos and learn what "better" tastes like.

Where do these pairs come from? Several places the next chapters cover: a strong model can judge pairs of your model's outputs, you can construct rejected examples deliberately (take a good extraction and corrupt it — drop a memory, inject a hallucination, vague-up an entity), or humans can rank outputs. The corruption trick is especially nice for our task, because we know exactly what "worse" looks like: we can *manufacture* the thin and hallucinated variants from a good answer.

---

## Karpathy's framing: imitation matches, RL discovers

It helps to hear this from someone who has thought about it for a living. Andrej Karpathy — a founding member of OpenAI and former head of AI at Tesla — has a framing for the difference between imitation and reinforcement learning that is worth borrowing, with two caveats stated up front: I am paraphrasing his *ideas* below, and where I put words in quotation marks I have verified them; everything else is my paraphrase and an editor should treat it as such.

The core idea: **imitation can only match the demonstrations; reinforcement learning can discover things beyond them.** When you train purely by copying expert examples — SFT — the very best you can hope for is to reproduce the experts. You are bounded by your demonstrations. You cannot exceed the people you imitated, because exceeding them would mean doing something they never demonstrated, and imitation has no mechanism for that.

Reinforcement learning is different in kind. Because the signal is "how good was the thing you just tried" rather than "copy this," the model is free to *try things that were never demonstrated* and find out they work better. Karpathy points to AlphaGo's famous "Move 37" — a move in its 2016 match against Lee Sedol so unusual that human experts estimated almost no human player would have chosen it, yet it turned out to be brilliant. His framing (paraphrased): this is the kind of new, surprising, genuinely creative move you can *only* get from large-scale reinforcement learning, never from imitating human experts — because no human expert would have shown it to you. Imitation copies the known; RL can find the unknown. (The AlphaGo system did use human-game imitation as a starting point, then improved far past it through self-play RL — which is, conveniently, almost exactly the imitation-then-preference arc this book follows.)

For our humble memory extractor, there is no "Move 37" waiting to be discovered — let's be honest about scale. But the *shape* of the benefit is real: a model trained to prefer complete, faithful extractions can learn to be selective in ways your hand-written gold answers never explicitly demonstrated. It can discover, across thousands of comparisons, a general pull toward "say what was actually said, all of it, and nothing more" that no single recipe card ever spelled out.

### The honest caution: "RLHF is just barely RL"

Here is the part that keeps us grounded, and it is also Karpathy's. The flavor of RL most commonly applied to language models is *RLHF* — reinforcement learning from human feedback — where you train a separate "reward model" to predict which answers humans prefer, then optimize against that. Karpathy's well-known and verified verdict on this: **"RLHF is just barely RL."**

His reasoning, paraphrased: in real RL like AlphaGo, the reward is *the actual thing you want* — did you win the game of Go? That is a true, unfakeable objective. In RLHF, the reward is a learned model of human preferences — a stand-in, a "vibe check," not the real goal. And a learned stand-in can be *gamed*. Optimize hard enough against a reward model and the policy will discover weird, adversarial outputs that the reward model mistakenly loves — the textbook example being a model that learns to emit nonsense the reward model happens to score highly. The reward model is a proxy, and proxies leak.

Why tell you this on the *on-ramp*, before you've trained anything? Because it sets honest expectations, which this book cares about. Preference and RL methods are powerful and they will make our memory extractor meaningfully better. They are also not magic, they add real complexity, and the reward signal you build is only as good as you make it. A sloppy reward function teaches the model to be sloppy in exactly the ways your reward was sloppy. We will come back to this hazard — "reward hacking" — in the reward-functions chapter, because for our task we get to do something nicer than train a fuzzy human-vibe reward model: we can write *checkable* rewards (is the JSON valid? do the entities actually appear in the conversation?) that are far less gameable than a learned vibe check.

---

## A map of Part 7

With the intuition in place, here is the road ahead. Each chapter builds on the last; this one owed you no code, but every chapter after this one pays it back.

**Ch25 - Reward Functions and Reward Models.** The foundation. Before you can teach a model to prefer better answers, you need something that can *say* which answer is better. There are two flavors, and we use both. A **reward function** is plain Python you write yourself — for our task, things like "give points for valid JSON, give points for entities that actually appear in the source conversation, subtract points for hallucinated facts." It is checkable and hard to game. A **reward model** is a small neural network you *train* on preference pairs (using TRL's `RewardTrainer`) to predict which of two answers is better — useful when "better" is too fuzzy to write as a rule. This chapter is where the chosen/rejected pairs from above become a working signal.

**Ch26 - DPO: Preference Tuning Without the RL Machinery.** The most practical first step into preference learning, and where most readers should start. Direct Preference Optimization (DPO) takes your chosen/rejected pairs and tunes the model to prefer the chosen ones *directly* — no separate reward model, no sampling loop, no RL plumbing. If you can run SFT, you can run DPO; the dataset is just `prompt`/`chosen`/`rejected` instead of one gold answer. It is the gentlest on-ramp from imitation to preference, and for many projects it is the whole journey.

**Ch27 - PPO and Why We Don't Use It.** The classic RL algorithm behind the original RLHF results. We explain how it works — the policy generating answers, a reward scoring them, a value estimate, a penalty for drifting too far from the original model — because the ideas are worth understanding. But we do *not* hand you a runnable PPO loop. It is heavy, finicky, memory-hungry, and in our pinned tooling it has been moved to an experimental corner; the code in that chapter is explicitly labeled illustrative, not runnable. The chapter exists to explain the landscape and then point you somewhere better.

**Ch28 - GRPO: The Practical RL Method.** The centerpiece of Part 7. Group Relative Policy Optimization (GRPO) keeps the discovery-driven spirit of real RL — the model generates several answers to each prompt, they get scored, and the model learns to lean toward the ones that scored higher relative to their peers — but it strips out PPO's most painful machinery (notably the separate value model). It pairs beautifully with the *checkable reward functions* from Ch25, which sidesteps much of the "barely RL" hazard. This is the chapter with the real, runnable RL code for our memory extractor.

**Ch29 - Choosing a Method.** The decision guide. SFT, DPO, or GRPO — and when each is right. The short version, which the chapter will justify properly: master SFT first (you have), reach for DPO when you have clean preference pairs and want a simple, stable improvement, and reach for GRPO when you can express quality as a checkable reward and want the model to actively explore toward it. We will also nod briefly at the cousins — KTO, ORPO, RLOO — so you recognize them in the wild without getting lost in them.

---

## How this changes the running example

It is worth being precise about what we are and are not changing.

We are **not** abandoning SFT. Everything from Parts 3 through 6 stays. The SFT model is the foundation — it is what taught the model the schema, the JSON discipline, and the basic skill of pulling facts from a conversation. Preference and RL are a *second pass on top of a model that already works*, not a replacement for it. You imitate first, then you refine toward "better." (This is exactly the imitation-then-RL arc that worked for AlphaGo, scaled down to our living room.)

We are **not** changing the pinned schema. Every memory object is still `{text, type, entities}`, with `type` being one of `preference | fact | decision | relationship`, and the model still emits a JSON array. The system prompt stays verbatim. Preference learning does not touch the *format* — that was SFT's job and SFT nailed it. Preference learning operates entirely in the realm of *quality*: completeness, faithfulness, restraint from hallucination, correct entity attribution. Same schema, better judgment.

And remember the framing from the very first chapter: memory extraction is *one example* of domain fine-tuning, chosen because it is concrete and runnable. Everything in Part 7 transfers. If your domain is extracting decisions and owners from meeting transcripts, or pulling structured fields from clinical notes, or any reliable structured-output task — the same progression applies. SFT teaches the format and the basic skill. Preference and RL teach the model to prefer the *good* version of that skill over the merely *valid* version. The conversation in this chapter happened to be about Maya and Priya; the lesson is about every domain model you will ever train.

---

## Common mistakes

**Reaching for RL before SFT is solid.** Preference learning refines a model that already works; it does not rescue one that doesn't. If your SFT model still produces malformed JSON or misses obvious facts, that is a *data and SFT* problem (see Ch20 - Iterating). Fix it there first. Preference tuning a broken base just teaches it to prefer slightly-less-broken outputs. Imitation before preference, always.

**Confusing "better" with "correct."** If your task genuinely has one right answer that you can check exactly, you may not need preference learning at all — a checkable rule or more SFT data is simpler. Preference and RL earn their complexity specifically when quality is a *spectrum*: more complete, more faithful, more selective. Use them for the spectrum problems, not the pass/fail ones.

**Treating the reward as the true goal.** This is the "barely RL" trap in practice. Whatever you reward, the model will chase — including the dumb literal interpretation of what you wrote. If your reward gives points for "more memories," the model learns to over-extract and hallucinate to rack up points. The reward is a proxy for what you want; design it suspiciously, and watch for the model gaming it. Ch25 spends real time here.

**Assuming RL means "from scratch" or "huge."** None of these methods retrain the model from zero, and none require a research cluster. DPO and GRPO run as LoRA passes on the same single GPU you used for SFT, on the same scale of data (hundreds to a few thousand examples — see the charter's data table). This is preference *fine-tuning*, in the same practical spirit as everything before it.

---

## Recap

- SFT is **imitation**: it teaches the model to copy one gold answer per input. That is powerful and, for format and basic skill, often sufficient — but it has a hard ceiling, because it treats every good answer other than the demonstration as a mistake.
- Many tasks — including ours — have a **spectrum** of better and worse answers, not a single right one. Our memory extractor produces valid JSON every time, yet some valid extractions are more complete, more faithful, and free of hallucination than others. Imitation cannot teach that ranking.
- **Preference learning** flips the signal from "copy this answer" to "this answer is better than that one," using chosen/rejected pairs. The claim is relative and honest, and it lets you express quality goals (prefer faithful over hallucinated, complete over thin) that no single demonstration could.
- Karpathy's framing (paraphrased, with "RLHF is just barely RL" quoted verbatim): imitation can only *match* the demonstrations, while RL can *discover* strategies beyond them — but the common reward-model form of RL optimizes a fuzzy proxy that can be gamed, so keep expectations honest and rewards hard to hack.
- Part 7's road: **Ch25** builds the reward signal (functions and reward models), **Ch26** does DPO (preference tuning without RL machinery), **Ch27** explains PPO and why we skip it, **Ch28** is GRPO (the practical, runnable RL method), and **Ch29** helps you choose among them.

## Next

**Ch25 - Reward Functions and Reward Models** — before we can teach the model to prefer better answers, we need something that can score them. We'll write checkable Python reward functions for our memory task and train a small reward model on preference pairs, setting up everything DPO and GRPO will need.
