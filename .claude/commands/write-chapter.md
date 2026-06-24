Write or rewrite one chapter of the book, running the full writer→editor→fix loop.

Charter (rules): @cookbook/AUTHORING_CHARTER.md
Plan (what to cover): @cookbook/EXPANSION_PLAN.md

Target chapter: $ARGUMENTS

Steps:
1. WRITER: spawn an agent using the WRITER template in the charter. Fill in this
   chapter's title and "must cover" bullets from the plan. It reads the listed
   neighbor chapters for voice, then writes the full .md file (~3,500–5,000 words).
   Verify every API/import/signature against code/requirements.txt — never from memory.
2. EDITOR: spawn an agent using the EDITOR template; it returns PASS/FAIL on all 7
   acceptance items with line references, then a fix list.
3. FIX: apply the fix list. Re-run the editor until all 7 items PASS.
Report the final PASS table. Do not commit unless I ask.
