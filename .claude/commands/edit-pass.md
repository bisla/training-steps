Re-run ONLY the editor on already-written chapters and apply fixes.

Charter: @cookbook/AUTHORING_CHARTER.md

Chapters: $ARGUMENTS

For each chapter, run the EDITOR template from the charter:
- Return a PASS/FAIL table on all 7 acceptance items, each with line references.
- Hard checks: TRL/Unsloth APIs match code/requirements.txt (flag any deprecated/
  hallucinated call, esp. the old PPO .step() loop); zero "Engram" references;
  pinned schema + system prompt unchanged; numbers as ranges; quotes verified.
- Then apply the fix list and re-run until all 7 PASS.
Report each chapter's final PASS table. Do not commit unless I ask.
