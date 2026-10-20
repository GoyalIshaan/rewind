# Working on Rewind

## Teaching and scope

- Follow the user's requested pace. For explanations, introduce function inputs,
  outputs, and call flow before implementation details.
- Do not change code during discussion-only requests or when the user asks for
  no edits. Do not advance from analysis to rewriting without task authorization.
- Keep expression matching, chain analysis, history detection, replay legality,
  transformation, lowering, and performance claims distinct.

## GitHub workflow

- The user explicitly authorized committing and pushing project changes to this
  repository as ongoing proof of work. After each completed coherent change,
  perform appropriate validation, make a descriptive commit, and push it to
  `origin`. No additional confirmation is needed for ordinary project pushes.
- Do not fabricate historical commits or backdate work. Do not create empty
  commits for discussion-only turns.
- Preserve unrelated user changes. Do not force-push or rewrite published history
  unless the user specifically requests it. Use the `codex/` prefix if creating a
  new working branch; ordinary work may continue on the current branch.
- Keep `rewind_mvp/DESIGN_NOTES.md` local. Never stage, commit, push, or force-add
  that concepts/ideas tracking note. Preserve it on disk.
- Exclude credentials, environments, and generated caches. Report the pushed
  commit and verification results, distinguishing local test results from CI.

## Validation

- Run relevant tests for code changes; for analyses, include cases that should
  match and cases that should be rejected.
- Full suite: `python3 -m unittest discover -s rewind_mvp -v`.
- Debug demos: `python3 -m rewind_mvp.matcher`, `python3 -m rewind_mvp.chains`,
  and `python3 -m rewind_mvp.history`.
- These analyses are read-only until a rewrite is explicitly introduced. A
  structural match is not proof of replay legality or profitability.
