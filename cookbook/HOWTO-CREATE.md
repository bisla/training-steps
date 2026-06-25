
Right — instead of pasting those long prompts every time, package the loop as **project slash commands**. That's the goal: turn each repeated instruction into a one-word `/command` that any fresh session in this repo can run.

Custom commands are just Markdown files in `.claude/commands/` — each file becomes a slash command, and `$ARGUMENTS` lets you pass a chapter path or number.

A useful set for this workflow:

| Command | File | Does |
|---|---|---|
| `/write-chapter <path>` | `.claude/commands/write-chapter.md` | Runs writer→editor→fix on one chapter using the charter + plan |
| `/edit-pass <paths>` | `.claude/commands/edit-pass.md` | Re-runs **only** the EDITOR template on given chapters + applies fixes |
| `/build-phase <n>` | `.claude/commands/build-phase.md` | Runs a whole plan phase, stops for review |

Example — `.claude/commands/edit-pass.md`:
```markdown
Run the EDITOR prompt template from @cookbook/AUTHORING_CHARTER.md against
these chapters: $ARGUMENTS

For each: return PASS/FAIL on all 7 acceptance items with line references,
then apply the fix list. A chapter is done only when all 7 PASS. Verify every
TRL/Unsloth API against code/requirements.txt, not from memory.
```

Then in a fresh session you just type `/edit-pass cookbook/part-7-preference-rl/28-grpo.md` — no pasting. Editing the editor's behavior still happens in `AUTHORING_CHARTER.md` (lines 208–224 / 253–272); the command just *invokes* it, so your rule changes flow through automatically.

The `@cookbook/AUTHORING_CHARTER.md` reference pulls the charter into the command's context each time, so the commands stay thin and the charter stays the single source of truth.

Want me to create these three command files in `.claude/commands/` now (I can add them to the existing branch/commit)?




Here's a practical terminal runbook. Everything keys off two files already in the repo: `cookbook/EXPANSION_PLAN.md` (what to build) and `cookbook/AUTHORING_CHARTER.md` (how to build + the editor rules).

## 0. Start the session

```bash
cd /path/to/training-steps
git checkout cookbook-expansion-rl-continual   # or main, wherever the two files live
claude
```

Quick sanity check inside the session — paste:
> Confirm `cookbook/EXPANSION_PLAN.md` and `cookbook/AUTHORING_CHARTER.md` exist and summarize the build phases and the 7-point acceptance bar back to me.

That makes sure it has loaded both before doing any work.

## 1. Execute the writer → editor → fix loop

The loop is driven entirely by a prompt — there's no script to run. Paste something like:

> Execute `cookbook/EXPANSION_PLAN.md`. For **each** chapter in a phase, run this loop:
> 1. **Writer:** spawn an agent using the WRITER template from `cookbook/AUTHORING_CHARTER.md`, filling in that chapter's title + "must cover" bullets from the plan. It writes the `.md` file.
> 2. **Editor:** spawn an agent using the EDITOR template; it returns PASS/FAIL on all 7 acceptance items + a fix list.
> 3. **Fix:** if any item FAILs, apply the fixes and re-run the editor until all 7 PASS.
> Do them **one phase at a time** and stop after each phase for my review. Start with Phase 0.

Tips:
- Run **phase by phase** (not the whole book at once) so you can review and keep context manageable.
- To parallelize within a phase: *"Write the Part 7 chapters concurrently — one writer agent per chapter — then run the editor on each."*
- To redo a single chapter: *"Re-run the full writer→editor→fix loop on `cookbook/part-7-preference-rl/28-grpo.md`."*

## 2. "Incorporate the expansion plan"

Two senses — be explicit about which you mean:

- **Use it as the build spec** (normal case): just reference it, as above — *"Follow `cookbook/EXPANSION_PLAN.md`."* Claude reads the per-chapter "must cover" bullets and the pinned numbers/APIs from it.
- **Change the plan, then build to the change:** edit the plan first, then build.
  > Open `cookbook/EXPANSION_PLAN.md`, add a chapter "Ch35 — Evaluating RL'd models" with these must-cover bullets: …. Then update `SUMMARY.md`, `README.md`, and `build/build.sh` (CHAPTERS + CONTENT_DIRS) to match, and run the writer→editor→fix loop for the new chapter only.

Always have it re-wire `build.sh` (CHAPTERS + CONTENT_DIRS), `SUMMARY.md`, and `README.md` whenever chapters are added/renamed — that's Phase 4 and it's easy to forget.

## 3. Editing the editor, then re-running

The editor's behavior lives in `cookbook/AUTHORING_CHARTER.md` — **acceptance bar = lines 208–224**, **EDITOR template = lines 253–272**. To change and re-apply:

```
# In the claude session:
```
> Edit `cookbook/AUTHORING_CHARTER.md`: add an 8th acceptance item — "Every non-PPO code block is self-contained and importable under `code/requirements.txt`" — to the acceptance bar, and add a matching hard check to the EDITOR template. Then re-run the editor (with the updated charter) on all of Part 7 and Part 8, and apply any new fixes.

Because the charter is injected verbatim into each editor agent, the moment you edit those lines, **every subsequent editor run uses the new rules** — no other config to touch. Re-running is just: *"Run the EDITOR template against \<chapter paths\> and fix failures."*

## Closing each cycle

```bash
cd cookbook/build && ./build.sh html      # the step the sandbox couldn't do — verify rendering
git add cookbook/ && git commit -m "..."  # claude will do this if you ask; commits only when asked
git push
```

A few guardrails worth setting at session start so the loop runs smoothly:
- Tell it **"commit after each phase passes the editor"** if you want checkpoints.
- Keep `code/requirements.txt` pinned and tell it **"verify APIs against the installed versions, not memory"** — that's the single biggest quality lever for the RL code.
- If you want it fully autonomous across phases, say so explicitly (*"don't stop between phases"*); otherwise the phase-by-phase default keeps you in control.

Want me to write this runbook into the repo (e.g. `cookbook/RUNBOOK.md`) so it's there when you open the fresh session?
