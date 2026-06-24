Execute one phase of the plan, then stop for my review.

Charter: @cookbook/AUTHORING_CHARTER.md
Plan:    @cookbook/EXPANSION_PLAN.md

Phase: $ARGUMENTS

1. List the chapters/tasks in this phase from the plan.
2. For each chapter, run the writer→editor→fix loop (see /write-chapter). Write
   independent chapters concurrently (one writer agent each), then editor each.
3. If this phase adds/renames/removes chapters, re-wire build/build.sh (CHAPTERS +
   CONTENT_DIRS), SUMMARY.md, and README.md to match — keep order identical.
4. Run `cookbook/build/build.sh html` and report warnings.
Summarize per-chapter PASS status and stop. Do not start the next phase or commit
unless I ask.
