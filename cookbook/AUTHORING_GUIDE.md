# Authoring Guide — How to Build & Maintain Books with Claude Code

This is the operating manual for this repo's book-authoring system. Read this first when you
open a fresh session. It explains the moving parts, how to run them, how to change the rules,
how to make big edits safely, and how to start an entirely new book.

---

## 1. The mental model (4 pieces)

| Piece | File(s) | Role | Analogy |
|---|---|---|---|
| **Charter** | `cookbook/AUTHORING_CHARTER.md` | The *law*: mission, audience, voice, pinned facts/APIs, the 7-point acceptance bar, and the WRITER/EDITOR prompt templates. | The constitution every agent must obey. |
| **Plan** | `cookbook/EXPANSION_PLAN.md` | The *spec*: ordered part/chapter list with per-chapter "must cover" bullets + build wiring. | The blueprint of what to build. |
| **Commands** | `.claude/commands/*.md` | Repeatable *verbs* you type as `/name`. Thin — they just point agents at the charter + plan. | Buttons on the machine. |
| **Agents** | ad-hoc (Agent tool) or `.claude/agents/*.md` | Reusable *roles* (writer, editor, fact-checker) each with its own context. | The crew. |

**One rule to remember:** the charter is the single source of truth. Commands and agents
*reference* it (`@cookbook/AUTHORING_CHARTER.md`), so when you edit the charter, every future
run follows the new rules automatically — nothing else to update.

---

## 2. The core loop: writer → editor → fix

Every chapter (new or rewritten) goes through the same loop. There is **no script** — it's
driven by a prompt/command:

1. **Writer agent** — fed the charter + that chapter's "must cover" bullets + neighbor
   chapters for voice. Writes the full `.md`. Verifies APIs against `code/requirements.txt`.
2. **Editor agent** — scores PASS/FAIL on all **7 acceptance items** (charter §"Acceptance
   bar") with line references, then a fix list. Read-only.
3. **Fix pass** — applies the fix list; re-runs the editor until all 7 PASS.

A chapter is "done" only when all 7 items PASS.

---

## 3. Starting a fresh session (the runbook)

```bash
cd /path/to/this-repo
git checkout <branch-with-the-book>
claude
```

First prompt — load context:
> Confirm `cookbook/EXPANSION_PLAN.md` and `cookbook/AUTHORING_CHARTER.md` exist and summarize
> the phases and the 7-point acceptance bar back to me.

Then drive it with commands (below). Work **phase by phase** and review between phases — it
keeps quality high and context lean.

---

## 4. Commands reference (`.claude/commands/`)

Type these as slash commands in any session opened in this repo. `$ARGUMENTS` = whatever you
type after the command.

| Command | Use | Example |
|---|---|---|
| `/build-phase <n>` | Run a whole plan phase (writer→editor→fix for each chapter), re-wire build files, build HTML, then stop. | `/build-phase 2` |
| `/write-chapter <path>` | Run the full loop on **one** chapter. | `/write-chapter cookbook/part-7-preference-rl/28-grpo.md` |
| `/edit-pass <paths>` | Re-run **only** the editor + apply fixes (use after changing the charter, or to re-verify). | `/edit-pass cookbook/part-7-preference-rl/*.md` |

To add a new command, drop a new `.md` file in `.claude/commands/`. The filename becomes the
command name; reference shared rules with `@cookbook/AUTHORING_CHARTER.md` and read inputs via
`$ARGUMENTS`. Keep commands thin — logic lives in the charter.

---

## 5. Changing the rules (editing the editor)

The editor's behavior lives entirely in the charter:
- **Acceptance bar** (what "good" means) — `AUTHORING_CHARTER.md`, the "Acceptance bar" section.
- **EDITOR template** (how it reviews, incl. hard checks) — the "EDITOR prompt template" section.

To change and re-apply:
> Edit `cookbook/AUTHORING_CHARTER.md`: add an 8th acceptance item — "Every non-PPO code block
> is self-contained and importable under `code/requirements.txt`" — and a matching hard check
> to the EDITOR template. Then `/edit-pass cookbook/part-7-preference-rl/*.md`.

Because the charter is injected into every editor agent, the change takes effect immediately on
the next run. Same idea for the WRITER template if you want to change how chapters are *drafted*.

---

## 6. Major edits, done goal-oriented

Don't free-prompt big changes. Always: **goal → plan → fan-out → editor-gate.**

> Goal: make Part 7 assume zero RL background. First plan the diff across Ch24–29 (what changes
> per chapter) and show me. Then apply chapter-by-chapter, running an editor pass on each.

For sweeping consistency changes (voice, terminology, a renamed concept), fan out one agent per
chapter in parallel, then run a single **continuity** pass to catch cross-chapter drift. Turn
recurring sweeps into commands (e.g. `/api-audit`, `/retune-voice`).

---

## 7. Starting a NEW book

The system is book-agnostic. Reuse the same two-file pattern:

1. **Bootstrap the charter + plan** in one prompt:
   > Interview me about a new book (topic, target reader, voice, scope, any pinned facts/APIs
   > or a running example). Then generate a `CHARTER.md` and a `PLAN.md` for it, modeled on
   > `cookbook/AUTHORING_CHARTER.md` and `cookbook/EXPANSION_PLAN.md`.
2. Put them in the new book's directory (e.g. `mybook/CHARTER.md`, `mybook/PLAN.md`), set up a
   `build/` (copy `cookbook/build/` and adjust `CHAPTERS`/`CONTENT_DIRS`), and a pinned
   `code/requirements.txt` if it has code.
3. Point the commands at the new paths (either edit the `@...` references, or make
   book-scoped command copies), then `/build-phase 0`.

Everything else — the loop, the acceptance bar, the agent roles — works unchanged.

---

## 8. Subagents & agent teams (primer)

- **Subagent** = a separate context with its own prompt/tools. Two flavors:
  - **Ad-hoc** — the lead spawns one on the fly ("write Ch28"). Good for one-offs.
  - **Defined** — a file in `.claude/agents/<name>.md` (name, description, allowed tools,
    model, system prompt). Reusable role. E.g. make `editor.md` whose body *is* the charter's
    EDITOR template; then commands just delegate to it.
- **Team** = a lead orchestrating defined roles in a pattern. This book's team is
  **writer → editor → fix**. Useful roles to define:
  - `writer` — drafts chapters (needs Write).
  - `editor` — scores against the acceptance bar (**read-only**, no Write).
  - `fact-checker` — verifies APIs/quotes against pinned versions (read-only + web).
  - `continuity` — checks cross-references and voice across chapters (read-only).
- **Parallel vs sequential:** independent chapters → spawn writers concurrently (one message,
  multiple agents); dependent steps (editor *after* writer) → sequential.
- **Keep editors read-only:** they report; the lead or a fix-agent edits. Cleaner accountability.
- **Context stays lean:** the lead keeps each subagent's *conclusion*, not its full transcript.

Rule of thumb: **commands = repeatable verbs · defined agents = reusable roles · charter = the
law they all read.**

---

## 9. Build & ship

```bash
cd cookbook/build
./build.sh html      # mdBook → out/html/index.html
./build.sh epub      # Pandoc
./build.sh pdf       # Pandoc + a LaTeX/HTML engine
```
Whenever chapters are added/renamed, the four wiring points must stay in sync:
`build/build.sh` (`CHAPTERS` + `CONTENT_DIRS`), `SUMMARY.md`, `README.md`. The `/build-phase`
command does this for you.

Commit/push only when you ask (default branch is protected by habit — branch first):
```bash
git checkout -b my-edits
git add cookbook/ .claude/ && git commit -m "..."
git push -u origin my-edits
```

---

## 10. File map

```
cookbook/
  AUTHORING_CHARTER.md   # the law (rules, acceptance bar, WRITER/EDITOR templates)
  EXPANSION_PLAN.md      # the spec (chapters + must-cover bullets + wiring)
  AUTHORING_GUIDE.md     # this file
  code/requirements.txt  # pinned library versions; all code is written against these
  build/build.sh         # multi-format build + the CHAPTERS/CONTENT_DIRS wiring
  SUMMARY.md, README.md  # table of contents (keep in sync with build.sh)
  quickstart/, part-*/   # the chapters
.claude/commands/        # /build-phase, /write-chapter, /edit-pass
.claude/agents/          # (optional) defined roles: writer, editor, fact-checker, continuity
```

That's the whole system. Open a session, load the charter + plan, and drive it with commands.
