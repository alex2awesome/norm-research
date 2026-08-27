# isomorphism/ — pending gated move (stage 2)

This directory will receive the ENTIRE `methods/codability/experiments/` subtree (the parts-1–2
policy-isomorphism apparatus: 48 py + 31 json manifests + 4 sh + ~40 tests) once the gate opens.

**Gate preconditions (all must hold):**
1. The frozen 990-cell calibration campaign and the sealed-validation endgame are closed and
   reported.
2. `pgrep -f methods.codability.experiments` is empty on BOTH sk2 and sk3.
3. Remote checkouts are ready to `git pull` immediately after the move lands.

Until then, all access goes through `methods/tacit_channels/_apparatus.py`. Do not copy files
here early — the frozen execution manifests hash-pin the current paths, and live GPU jobs invoke
`python -m methods.codability.experiments.*` by module path.

Move checklist: Part III of the approved plan (`~/.claude/plans/recursive-swimming-unicorn.md`).
