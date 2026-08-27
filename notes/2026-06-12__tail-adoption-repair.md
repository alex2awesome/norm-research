# L0 singleton-tail repair via existing judge labels (adopt_v1)

Date: 2026-06-12. Script: `scripts/adopt_tail_clusters.py` (env:
`MIN_SUPPORT`, `OUT_NAME`). Outputs:
`outputs/analyses/structural_metrics/adopt_v1{,_strict}/`.
Locked tau-0.825 artifacts untouched.

## Do we trust the score=2 ("same rule") judgments?

Spot-checked + quantified:

- **Pairwise precision ~60–80%, task-dependent.** Random cross-cluster
  score=2 edges: peer-review ~75–80% genuinely same-rule; creative-writing
  ~60% (e.g. "simplified and focused" ≈ "clear and easy to understand";
  one sampled pair was near-opposite advice). Errors are over-merges into
  *related* rules — never absurd (zero score-0-style mistakes seen).
- **Not transitive.** Judged triangles with A=B=2 and B=C=2:
  P(A=C=2) = 0.78 (humor) / 0.81 (peer-review) / 0.85 (CW) / 0.93 (code).
  P(A=C=0) = 0.00 everywhere. So "same rule" is a tight *similarity*, not an
  equivalence relation → union-find/transitive closure is invalid: it chained
  1,228 CW clusters into one 3,034-form blob. Any repair must be chain-proof.
- **Support tiering:** adoptions backed by ≥2 independent score=2 edges are
  ~85–90% correct on audit; single-edge adoptions track raw edge precision
  (~60–70%).

## Repair operator: one-round star adoption

Tail clusters (size ≤2) only. Target = cluster with most score=2 edges, no
score=0 edge. Tail→head adoptions are stars (single hop); tail→tail only on
mutual-best (no chains; asserted). Two variants:

| variant | gate | adopted/task | singleton% (e.g. CW) | max size (CW) |
|---|---|---|---|---|
| adopt_v1 | ≥1 edge | 405–970 | 74%→76% of fewer clusters; raw count −29% | 42→57 |
| adopt_v1_strict | ≥2 edges | 153–310 | −12% singleton count | 42→50 |

(Union-find for contrast: CW max size 3,034 — rejected.)

Recommended use: **strict** for any headline statistic; permissive for
sensitivity bounds. v2 should re-adjudicate single-edge adoptions with fresh
judge majority (different seeds) on sk3 — prepped by the audit logs
`adoptions_<task>.jsonl` (absorbed/target texts + support).

## Upward propagation (deterministic, no LLM re-runs)

- **L0→R1**: absorbed cluster inherits the *target's* R1 family; removed from
  its old family; emptied families drop (R1 count −1% to −20% by task).
- **R1→R2**: dropped families removed from aspects; emptied aspects drop
  (R2 barely changes: −0 to −31).
- **Cross-family adoptions are logged, not applied**:
  `candidate_r1_merges_<task>.json` (163–771 weighted family-pair edges per
  task) — high-value input for a Fork3-v2 R1 merge pass, since each edge is
  judge-grounded evidence two families share a rule.

## adopt_v2 (fresh-judge re-adjudication, 2026-06-12 evening) — THE RECOMMENDED VARIANT

All 4,672 single-edge adoptions re-judged on sk3 (Llama-3.3-70B BF16, exact v6
prompt, 3 votes/pair at temp 0.6, confirm = majority score-2). Result: **only
45% confirmed** (36% grant-funding to 54% code-review; 84% of vote triples
unanimous) — the original single greedy score=2 labels had a ~55% test-retest
overturn rate on this slice, worse than the eyeball estimate. Rejected samples
audit as correctly rejected (related-not-same); confirmed samples audit clean.

`adopt_v2` = strict tier + confirmed single-edge tier
(`outputs/analyses/structural_metrics/adopt_v2/`, built via
`ALLOWED_PAIRS=... OUT_NAME=adopt_v2 python scripts/adopt_tail_clusters.py`):
4,398 adoptions across 11 tasks; singleton counts drop 7–19% (e.g. CW
2,180→1,800, news 1,204→974); max cluster sizes stay sane (CW 42→55).
Verdicts: sk3 `norm_embed/readjudicate_verdicts.jsonl` (+ local
/tmp/readj_verdicts.jsonl). Pipeline lesson: **never apply a single greedy
judge label as a merge edge; require either ≥2 independent edges or a fresh
majority.**

## Caveats

- Adoption uses the same v6 labels that defined the validation oracle — the
  repaired clustering can no longer be *evaluated* against those same edges
  (in-sample). Fresh judge labels needed for unbiased re-evaluation.
- Verdict coverage is the kNN candidate net: tails with no judged partner
  (60–83%) may still have unjudged duplicates; the singleton counts after
  repair remain upper bounds on true idiosyncrasy.
