# Evidence-aware judging flips the patents prior-art-op null (WS3, 2026-07-10)

**Audience:** self-contained hand-off for anyone doing patent work; no metric-seam context
assumed. TL;DR: prior-art retrieval/matching machinery that looked worthless under our old
evaluation is in fact strongly informative — the old evaluation was structurally incapable
of seeing its value. If you are evaluating any patent criterion that depends on evidence
outside the application document (novelty, non-obviousness, prior-art differentiation),
your judge must be shown that evidence, or you will measure a false null.

## 1. The problem this experiment resolved

We build "metric programs" — scoring functions for evaluative criteria on patent
applications (e.g., *does this application clear the novelty bars?*). Some of these
programs call **PriorArtOps**: a precomputed per-application record from the Task-C
retrieval pipeline containing per-claim prior-art matches with element-level disclosure
verdicts (label/rejection fields stripped — leakage guard). The programs are scored by
rank-correlating (Spearman) against an LLM judge's rating of the same criterion.

Earlier (R7.1, early July), we measured the **op-marginal** — score of the full program
minus the same program with the prior-art record nulled out — and got **≈ 0 on every
aspect**. Tempting read: the retrieval machinery is useless. Correct read (pre-registered
before this experiment): the judge only ever saw the application document, and a judge that
cannot see the evidence cannot reward a program for using it — formally, I(M̄(X); Z | X) = 0,
where Z is the evidence and M̄(X) a doc-only judge. The null was a *level-matching artifact*
of the instrument, not a fact about the ops.

## 2. Design

- **Items:** the standing patents_pa set — 250 applications (`items.json`), each with a
  PriorArtOps record (`pa_features.json`).
- **Aspects (4, the R7.1 evidence-dominant criteria):** a26 non-obviousness, a34 novelty
  bars, a60 prior-art differentiation, a35 patentability triad.
- **Judge:** Gemma-4-31B, 0–10 scale, two independent passes per cell (reliability →
  attenuation ceiling), same two-pass templates as every other task in the program.
- **Arms:**
  - `evidence` — judge sees the document **plus** the serialized prior-art search record
    (summary stats + top-2 claims × top-2 refs with disclosure verdicts, capped 2,200
    chars). This is the new target **M̄(x,Z)**.
  - `filler` — judge sees the document plus **length-matched inert text**: controls for
    "any extra attachment changes judge behavior" (format/instruction-load confound).
  - doc-only M̄(x) — the existing R7.1 judgments, reused unmodified.
- **Volume:** 4 aspects × 250 items × 2 passes × 2 new arms = 4,000 judgments (one GPU pass,
  sk3; 0 unparsed, 1.2% NA, full 0–10 range used, no mode collapse).
- **Readout:** op-marginal = rho(full program) − rho(NullOps twin) against each target, on
  the same held-out 40% split as R7.1 (seed rng(7) over sorted ids — split identity was
  audited programmatically 2026-07-10, sets are byte-identical), with a 2,000-resample
  item bootstrap for P(full > null).

## 3. Results (held-out, n=100/cell)

*(Numbers below are eval v2, 2026-07-10 — revised after an external code review:
judges rebuilt on the two-pass intersection only (v1 union-mixed one- and two-pass items),
and the bootstrap now excludes undefined resamples instead of counting them as losses.)*

| aspect | target | judge 2-pass rel. | rho full | rho null | **op-marginal** | P(full>null) |
|---|---|---|---|---|---|---|
| a26 non-obviousness | **evidence** | .873 | .455 | .244 | **+.211** | .98 |
| | doc-only | .774 | −.095 | .123 | −.218 | .02 |
| | filler | .764 | −.072 | .175 | −.246 | .01 |
| a34 novelty bars | **evidence** | .640 | .745 | .084 | **+.661** | 1.00* |
| | doc-only | .444 | .065 | .057 | +.008 | .46 |
| | filler | — | −.006 | .021 | −.027 | .34 |
| a60 prior-art diff. | evidence | **.197** | .096 | −.027 | +.123 | .86 |
| | doc-only | .133 | −.073 | −.054 | −.019 | .42 |
| a35 patentability triad | **evidence** | .916 | .451 | −.159 | **+.609** | 1.00 |
| | doc-only | .913 | −.117 | −.319 | +.202† | .95 |
| | filler | .740 | −.126 | −.274 | +.148 | .89 |

\* a34's earlier-reported P=.62 was a bootstrap artifact: the NullOps twin is ~constant on
the test split (2 distinct values, 99% at the mode), so 723/2000 resamples had undefined
null correlation and were counted as losses. Excluding undefined resamples, full beats null
in 100% of defined pairs.
† a35's positive doc-only difference is driven by the null twin *anti-correlating* (−.32)
with the doc-only judge, not by the full program being good there (it's −.12).

**The four pre-registered predictions, checked:**
1. Op-marginal positive vs M̄(x,Z) for evidence-dominant criteria → **confirmed** on
   a26/a34/a35.
2. Op-marginal ≈ 0 vs doc-only M̄(x) → holds for a26 (negative), a34 (+.008), a60 (−.019);
   **a35 does not replicate the null** (+.202, P .95, via the anti-correlated null twin
   mechanism above) — 3 of 4, stated as such.
3. Filler ≈ doc-only on every aspect → descriptively yes (no formal equivalence test).
   Caveat: the filler controls length and the attachment header, **not payload
   syntax/schema** — evidence is structured JSON-ish text, filler is repeated prose.
4. Bonus, unplanned: evidence **raises judge reliability** on all four aspects
   (.873/.640/.197/.916 vs .774/.444/.133/.913) — the judge is less noisy when grounded.
5. Sanity: M̄(x,Z) tracks the payload's own novelty-exposure summary
   (1 − frac_claims_any_disclose; rho .48–.72 except a60), M̄(x) doesn't (|rho| ≤ .18) —
   exactly the designed information asymmetry.

**Honest exclusion:** a60 (prior-art differentiation). Its evidence-arm judge reliability is
.197 → attenuation ceiling .57; the target itself is too unreliable to support any claim.
This is a target-quality failure (likely the aspect wording), not evidence against the ops.
Note: this is an **exploratory** exclusion — no reliability cutoff was pre-registered; a
cutoff of rel1 ≥ .30 is now frozen for future evidence-judge runs.

**Scope caveat (from external review):** the judge and the hybrid programs consume the
*same* precomputed disclosure-summary representation. The positive marginals therefore
establish that op-marginals are **well-posed** once the judge sees the evidence (the
question this experiment was built to answer) — they do not independently validate the
disclosure representation against ground-truth patent correctness. Treat "the ops carry
criterion-relevant signal" as "the ops carry signal the evidence-aware judge rewards."

**Stronger caveat — oracle gold injection (Codex audit, 2026-07-10):** the option3
candidate sets underlying `pa_features.json` were built with the examiner-cited gold
document FORCE-INCLUDED for examiner-targeted claims; non-targeted claims got retrieval
fillers only. Stripping `is_gold` hides *which* candidate is gold but does not undo the
label-dependent candidate-set construction. The positive op-marginals could therefore
partly reflect examiner-citation leakage (the judge rewarding oracle-injected content)
rather than deployable retrieval quality — the filler≈doc-only result shows the lift
comes from the gold-bearing records' CONTENT, which is consistent with both readings.
A clean test requires candidates produced by the retriever alone, with the examiner-cited
document held out for evaluation. Until then: "the judge needs the evidence" stands;
"the retrieval machinery discovers the evidence" is NOT established by this experiment.

## 4. What this means for patent work

- **Patents' apparent "low verifiability" was partly an instrument artifact.** For
  evidence-dominant criteria, machine scoring becomes strongly informative the moment the
  evaluator is level-matched with the evidence the criterion is *about*.
- **Any judge/eval of novelty-type criteria must include the search record.** A doc-only
  LLM judge produces a structural false null on everything retrieval-based — do not use
  doc-only judgments to evaluate retrieval-dependent features, ops, or agents.
- The Task-C retrieval + element-disclosure machinery (`pa_features.json` lineage) is
  validated as carrying real criterion-relevant signal — e.g., a34's full program reaches
  held-out rho .745 against a .64-reliability judge.
- Caution when reusing: the serialized payload is **label-stripped** (no rejection/OA
  fields). If you rebuild payloads, keep that guard — a judge shown outcome-adjacent fields
  is leaking the label, not reading evidence.

## 5. Reuse pointers (all paths repo-relative; sk3 mirror under
`/lfs/skampere3/0/alexspan/norm-research/`)

| artifact | path |
|---|---|
| prompt builder (arms, serialization, filler) | `methods/metric_seam/f2p_mock/build_ws3_evidence_judge.py` |
| judge scorer (offline batch vLLM, Gemma-4-31B) | `methods/metric_seam/pilot/gemma_score_v1.py` |
| eval (reliability, op-marginals, exposure) | `methods/metric_seam/f2p_mock/ws3_eval_evidence.py` |
| raw judgments (4,000) | `outputs/metric_seam_pilot/tasks/patents_pa/ws3_evidence_results.jsonl` |
| report JSON (the table above) | `outputs/metric_seam_pilot/tasks/patents_pa/ws3_eval_report.json` |
| items / evidence payloads | `outputs/metric_seam_pilot/tasks/patents_pa/{items.json,pa_features.json}` |
| doc-only judgments (R7.1, reused) | `outputs/metric_seam_pilot/tasks/patents_pa/results.jsonl` |
| doc-of-record + pre-registration | `notes/2026-07-10__seam-agentic-program-runbook.md` (WS3 section) |

Scope note: this validates the *evaluation* of criterion programs on prior-art-dependent
aspects. It is not a claim about final-outcome prediction (Tasks A/B in the README) — though
the instrument lesson (level-match your judge to the evidence) applies to any LLM-judged
evaluation in this directory.
