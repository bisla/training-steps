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
```
