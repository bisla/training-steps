# Ultraplan cloud prompt — Domain Fine-Tuning Handbook expansion

Paste the block below as the `/ultraplan <prompt>` task once the Claude GitHub app is
installed on this repo (https://github.com/apps/claude/installations/new → select
`bisla/training-steps`). It refines the plan in `~/.claude/plans/` against this repo.

---

```
Refine the implementation plan in ~/.claude/plans/ (the "Domain Fine-Tuning Handbook" plan)
against the existing book in cookbook/.

CONTEXT
- cookbook/ has 27 chapters (parts 0–6 + appendices), ~110k words, plus README.md,
  SUMMARY.md, build/book.toml, build/build.sh (html+epub+pdf). It currently teaches ONLY
  one-shot SFT with Unsloth, and is branded "Engram".
- Running example throughout: memory extraction (conversation -> JSON [{text,type,entities}]).
- Reader: Python dev, ZERO ML background. Tone: intuition-first, "for dummies", runnable code.

GOAL OF THIS WORK (already decided — do not re-litigate)
1. De-brand "Engram" -> "The Domain Fine-Tuning Handbook" everywhere (md + book.toml + build.sh).
2. Add Part 7 — Preference & RL: why-preference; reward functions + reward model (RewardTrainer);
   DPO (DPOTrainer); GRPO (GRPOTrainer); FULL runnable PPO loop (reward model + PPOTrainer,
   value head/KL/advantage); SFT-vs-DPO-vs-KTO/ORPO-vs-GRPO-vs-PPO decision guide. ALL runnable.
3. Add Part 8 — Continuous Learning as a system: the loop architecture; data selection/curation
   (dedup, quality scoring, hard-example mining, importance sampling); how-much-data & cadence
   (rows vs tokens, 4B scaling, replay mix ratios); catastrophic forgetting over many rounds;
   production ops (monitoring, dataset/adapter versioning, gating/canary, rollback).
4. Re-thread Ch1 (mission), Ch3 (add post-training + continual axis), Ch23 (bridge to Part 8),
   appendix A glossary (DPO/PPO/GRPO/reward model/KL/advantage/value head/replay buffer/...).
5. Add cookbook/AUTHORING_CHARTER.md: mission + acceptance bar + reusable WRITER and EDITOR
   prompt templates, injected into every chapter-writing agent so the editor enforces:
   teaches RL + continual PRACTICALLY, intuition-before-code, runnable real TRL/Unsloth APIs,
   de-branded, stays on the memory-extraction example.
6. Build via writer->editor->fix agent loop; rebuild SUMMARY/README/build.sh CHAPTERS+CONTENT_DIRS.

WHAT I WANT FROM YOU
- Sharpen chapter boundaries, ordering, and the AUTHORING_CHARTER acceptance bar.
- Verify the TRL/Unsloth APIs named (DPOTrainer, GRPOTrainer, PPOTrainer, RewardTrainer) are
  current and that the PPO-full-loop chapter is realistic; flag any that changed.
- Pin concrete, defensible numbers for "how much data for a ~4B model" and replay ratios.
- Output the final plan as an ordered, executable checklist with per-chapter "must cover" bullets.
Keep all decisions above fixed; refine HOW, not WHETHER.

ADDITIONAL REQUIREMENTS (fold in, do not drop)
7. PROGRESSIVE ON-RAMP is the spine of the book: order it so a reader gets a WORKING fine-tuned
   model FAST and CHEAP first, then layers sophistication. Add an early Quickstart that fine-tunes
   Gemma-3-4B or Qwen3-4B on synthetic data end-to-end for UNDER $20–30 total (state the exact
   cost + time + GPU, e.g. one cloud A100/L4 hour or free Colab). Each later part is an explicit
   "now make it better" step: better data -> evaluation -> DPO -> GRPO/PPO -> continuous learning.
   Every stage names its $ budget and what it buys.
8. SYNTHETIC DATA: make the synthetic-data chapter (currently Ch13) deep and practical — exact
   generation pipeline, teacher-model prompts, diversity/personas, self-verification/judge filtering,
   dedup, and CONCRETE VOLUME guidance (how many rows/tokens for a first cheap run vs a good model
   vs each continual-update cycle; rows-vs-tokens; diminishing returns). Tie its numbers to the
   how-much-data chapter in Part 8 so they agree.
9. HOSTING & SERVING (inference): make the serving chapter (currently Ch22) a practical, cheap-first
   guide — local Ollama (GGUF) for $0, then vLLM OpenAI-compatible server for throughput, then a
   managed/endpoint option; include a tiny client calling the served model with the SAME system
   prompt used in training, plus latency/cost notes.
10. TONE BAR everywhere: super simple to follow, "dumbed down but technically correct", builds
    intuition and context for a zero-ML reader who still ends up understanding WHY. Practical steps
    over theory. The AUTHORING_CHARTER acceptance bar must encode this and the progressive-cost spine.
```
