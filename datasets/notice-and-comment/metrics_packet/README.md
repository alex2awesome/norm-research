# N&C Explicit-Metrics Packet

Self-contained, distributable collection of every **explicit metric** built for the
notice-and-comment (federal rulemaking public comments) VAT campaign (2026-07, v4.2 corpus).
Three metric families: LLM-judged rubrics (A), regex features (V), and hybrid
LLM-extract/code-decide programs (V_deep). Assembled by `build_packet.py` (re-runnable);
campaign context in `../README.md` §9.

**Task**: score the quality/persuasiveness of a public comment submitted on a proposed
federal rule. Metrics are label-free instruments — none were fit to outcome labels.

## Contents

```
a_rubrics/
  rubrics.jsonl                198 LLM-judge rubrics, enriched with provenance + stats
  rubrics_gepa_optimized.jsonl GEPA fidelity-optimized descriptions (see quoting rule!)
  leaf_criteria.jsonl          every rubric unpacked to its original per-source statements
  sources.csv                  every source document the rubrics were mined from
  judge_protocol.md            how to score a comment against a rubric
v_regex/
  v_features.py                27 deterministic regex/count features (standalone, stdlib+re)
v_deep_programs/
  *_h0.py *_h1.py *_h2.py      14 hybrid programs: LLM extracts <=3 fields, code decides
  ops.py                       shared helper API the programs receive
  cfr_parts_index.json.gz      real eCFR title->parts index (for authority_lookup_h2)
performance_summary.csv        univariate outcome-y AUC + coverage for every metric
build_packet.py                rebuilds this packet from repo artifacts
```

## The three families

### 1. A-rubrics (198) — LLM-judged articulable criteria
Mined from ~1,000 public documents about what makes a good regulatory comment —
agency-authored Federal Register preambles/response-to-comment sections, agency
"how to comment" guidance, OMB Circular A-4, ACUS recommendations, SBA Advocacy,
plus NGO/academic/law-firm guides — then clustered into 198 merged rubrics
(88 `general` craft criteria + 110 `specific` regulatory-regime criteria).

Each rubric in `rubrics.jsonl` carries full provenance:
- `agencies` / `n_distinct_agencies` / `provenance_class` — which agencies' documents
  its leaf metrics came from (`multi-agency` = the criterion was articulated
  independently by 2+ agencies; `single-agency` = agency-idiosyncratic;
  `no-agency-doc` = cross-cutting/non-government sources only)
- `source_documents` — every source doc (file, url, kind, agency)
- `univariate_auc_outcome_y`, `na_rate`, `n_applicable` — behavior on the campaign
  sample (7,482 comments, y = majority rule-change outcome)

Key provenance facts: 31 agencies contribute; EPA documents feed 56/198 rubrics.
87 rubrics are multi-agency (convergent craft criteria — specificity, evidence,
alternatives, transparency), 53 single-agency (mostly regime-specific: Title IX/ED,
SEC disclosure items, EPA CAA SIP standards, FTC non-compete, DHS public charge).
Discrimination power is flat across provenance classes; single-agency `specific`
rubrics apply to far fewer comments (median ~534 vs ~3,542) — treat them as narrow
topical instruments and beware docket-detector artifacts.

**Tracing a rubric back to what the sources actually say**: each rubric's `description`
is a merged distillation; `leaf_criteria.jsonl` (same `rubric_id`) unpacks it into all
3,063 underlying leaf criteria — for each: the source document (file, URL, kind, agency)
plus the criterion as originally extracted from that document (`original_description`,
and `original_guidance` where the source gave operational advice). The URL then takes
you to the source document itself. 46 leaves across 11 legacy files have no recorded
URL but retain their original text.

**GEPA quoting rule**: `rubrics_gepa_optimized.jsonl` contains descriptions rewritten
over 4 rounds to maximize construct fidelity to a frozen Sonnet reference
(fidelity .485→.577). Fidelity rose but **predictive AUC dropped** (A on outcome-y
.592→.578; agree-y within-docket .558→.501). The canonical bank for any predictive
use is `rubrics.jsonl` (pre-GEPA); the GEPA variant is included as the cleaner
*articulation* of each construct and as the fidelity-vs-prediction tradeoff artifact.

### 2. V regex features (27) — fully deterministic
`v_regex/v_features.py` — length/structure counts, citation and docket references,
econ/data/alternatives vocabularies, stance keywords, caps ratio, first-person
density. Zero dependencies beyond `re`. `v_features(text) -> dict`, order in `V_NAMES`.

### 3. V_deep hybrid programs (14) — the metric-seam family
Each program optionally declares `LLM_FIELDS` (≤3 one-line extraction instructions,
filled by any capable LLM) and a deterministic `score(text, extracted, ops) -> [0,1]`
(blend 0.65·code + 0.35·llm; defensive fallback 0.5). Programs run with
`extracted=None` too (code-only mode, degraded).

- Wave `_h0` (8, comment-quality surface): citation_validity, redline_ask, cba_rigor,
  evidence_provenance, alternatives_analysis, structure_org (code-only),
  legal_argument, stake_specificity
- Wave `_h1` (4, persuasion): stance_alignment (best single metric on agree-y, .627),
  ask_modesty, deference_tone (code-only), technical_precision
- Wave `_h2` (2, verification tier — LLM extracts, code *verifies correctness*):
  numeric_consistency (arithmetic coherence of quantities),
  authority_lookup (checks the load-bearing cited CFR/USC authority against the real
  eCFR index in `cfr_parts_index.json.gz`)

Campaign headline: the 12-program V_deep bundle reached **.615** AUC on outcome-y —
above both the regex-V baseline (.595) and a docket-disjoint fine-tuned Llama-8B (.602).

## Quickstart

```python
import importlib.util
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

# V regex features
vf = load("v_regex/v_features.py", "vf")
feats = vf.v_features(comment_text)              # dict of 27 floats, order in vf.V_NAMES

# V_deep hybrid programs (ops is an instance of the shared helper class)
ops = load("v_deep_programs/ops.py", "opsmod").Ops()
prog = load("v_deep_programs/authority_lookup_h2.py", "al")
extracted = {"authority_relied_on": "40 CFR Part 60"}  # filled by your LLM per prog.LLM_FIELDS
s = prog.score(comment_text, extracted, ops)     # float in [0,1]; extracted=None -> code-only

# A rubrics: send (rubric name+description, comment) to an LLM judge
# with the protocol in a_rubrics/judge_protocol.md
```

Sanity anchors (packet copies patched to find `cfr_parts_index.json.gz` next to the
program file): `authority_lookup_h2` must score a real authority ("40 CFR Part 60")
above a fabricated one ("40 CFR Part 9999"), and `citation_validity_h0` must score a
cite-rich comment above an all-caps rant.

## Reading `performance_summary.csv`

One row per metric across all families. `univariate_auc_outcome_y` is the pooled
raw-score AUC against the majority rule-change outcome (no model fitting; NA scores
excluded pairwise). Comments cluster by docket and outcome is largely docket-level,
so treat univariate AUCs as descriptive, not as clean effect sizes. Campaign
multivariate numbers (docket-grouped CV) are in `../README.md` §9.

## Caveats for redistribution

- Rubric provenance is by *source-document author* (who wrote the guidance/preamble a
  leaf metric was mined from), not the agency whose comments were scored. Federal
  Register docs were attributed via their `/agencies/` link (40/479 unattributable).
- Rubric texts are distillations of public-domain government documents and public
  guides; `sources.csv` lists every underlying URL for attribution.
- The A-rubrics assume the judge protocol in `a_rubrics/judge_protocol.md`
  (1.0/0.5/0.0/NA single-token scale). Scores are judge-dependent; campaign scores
  used Gemma-4-31B. Include the three anchor comments in any re-scoring run as a
  sanity check.
