# PREREG-24 — register effect on extraction recall (real-host splice, powered)

**Status: FROZEN 2026-07-27, before any construction or judging.**
Supersedes the P21d design, which was attempted twice and voided twice.

## Why this exists

P21d asks: *does the census extractor recover casual-register statements of a norm less
often than formal-register statements of the same norm?* Two prior attempts failed:

| Attempt | Design | Outcome |
|---|---|---|
| 2026-07-24 | Fully generated pages, 309 words, 2 planted rules | Realism gate voided it. **And** recall was 20/20 formal, 20/20 casual — a ceiling with zero power. |
| 2026-07-26 | Splice: real prose + real rules, 516 words, 2 planted rules | Realism gate voided it first (0/20 passed as real). Same 2-per-page density, so it would very likely have hit the same ceiling. |

The binding defect was never realism. It was **power and task difficulty**:

- Real census prose pages run ~1,058 words median with ~36 criteria competing for
  attention; production recall there is **.29** (Leg 3, grain-matched, ceiling-relative).
- Both prior designs planted 2 salient rules into a short, otherwise-clean page. That is
  a far easier task, and it saturated.
- At n=20 construct pairs the paired sign test has **.24 power** for a 20-point gap and
  **.09** for a 10-point gap (exact calculation, this file's power section). Both attempts
  were underpowered by roughly 5×.

This design fixes density, host realism, and n together, and adds a preregistered
**ceiling check** so a saturated run is declared uninformative rather than reported.

## Hypothesis

**H1 (primary, directional).** Production extraction recall is higher for formal-register
plantings than for casual-register plantings of the same construct.

**H0.** No difference: among construct pairs discordant in recall outcome, formal-only and
casual-only outcomes are equally likely.

## Design

- **Unit of analysis:** the construct pair. Each construct is represented by one *formal*
  and one *casual* real rule sentence expressing the same requirement, mined from
  different subreddits in the AIRules corpus (`datasets/prior_norms/airules_frame.jsonl.gz`,
  99,967 subreddits). Both members are verbatim real text; neither is model-written.
- **n = 140 construct pairs** = 280 plantings across 140 host pages, 2 plantings per page
  (one formal, one casual, **from different constructs**, so no page contains both members
  of a pair and no pair is confounded with a host).
- **Hosts:** real parsed census pages from `notebooks/_explore_cache/parsed_text/`,
  restricted to ≥800 words and to the five tasks used in Leg 3 (code-review, humor,
  creative-writing, peer-review, grant-funding) so recall is comparable to the .29
  benchmark. Host pages are **not** stripped of their pre-existing criteria — those
  criteria are the haystack, and removing them is what made the prior designs easy.
- **Planting:** each rule sentence is inserted verbatim at a paragraph boundary in the
  host. Nothing else about the host is altered.
- **Attribution:** a planting counts as recalled iff an anchor-gated judge matches an
  extracted item to that planted rule's content. Matches to the host's pre-existing
  criteria do not count. The planting key (source, verbatim text, host, char offset) is
  written and hashed before extraction.

## Primary test

Paired one-sided exact binomial (sign) test on discordant construct pairs, α = .05,
direction formal > casual. **One run. No re-running, no arm tuning, no re-scoring.**

## Power (exact, computed 2026-07-27 before freeze)

Assuming baseline recall in the .25–.50 band implied by Leg 3:

| n pairs | 20pt gap (.45/.25) | 15pt gap (.50/.35) | 10pt gap (.40/.30) |
|---:|---:|---:|---:|
| 20 *(both prior attempts)* | .24 | .15 | .09 |
| 60 | .69 | .44 | .25 |
| **140 (this design)** | **.96** | **.78** | **.49** |

Declared in advance: this is powered for gaps ≥15 points and **underpowered for a
10-point gap**. A null result will be reported with that limit stated, not as evidence of
no effect.

## Gates — all pre-specified, all reported whether they pass or fail

- **G1 Realism (hard stop, run FIRST).** A blinded judge classifies 140 spliced pages and
  140 untouched control pages as real or synthetic. **Gate: discrimination accuracy ≤ .70.**
  (The 2026-07-26 design scored 1.00.) If G1 fails, nothing downstream runs and the failure
  is the reported result.
- **G2 Register separation (construction-time).** Judged formality (1–7) of the formal
  member must exceed the casual member by **≥1.5**. Pairs failing are dropped *before*
  planting; the drop count is reported. Anchor gate ≥5/6 on the formality wave.
- **G3 Same-requirement confirmation (construction-time).** A judge must confirm both
  members state the same requirement. Failing pairs dropped before planting.
- **G4 Non-overlap (construction-time).** The planted construct must not duplicate a
  criterion already present on its host page (judge-confirmed), or attribution is
  ambiguous. Failing host/construct assignments are re-drawn.
- **G5 Scoring anchor gate.** ≥5/6 camouflaged known-answer items in the matching wave,
  key frozen before scoring.

### Amendments, 2026-07-27 — made during construction, BEFORE any judging wave ran

Both were forced by inspecting *materials*, not outcomes. No recall, register, or
same-requirement judgment had been made when either was written.

- **G6 Quality-stratum restriction (construction-time).** Candidate rules are restricted to
  the quality stratum of AIRules before pairing. Unrestricted mining returns ~16%
  governance against ~6% quality — flair, NSFW tags, reposts, spam, self-promotion — and
  the production pass recovers only $.05$ of governance gold. Planting governance rules
  would park both arms on the **floor**, the mirror image of the 2026-07-24 ceiling and
  equally uninformative. The filter is a heuristic pre-screen on what the judge sees;
  G2/G3 still decide survival. Pool: 284,394 plantable sentences → 17,616 quality-stratum
  → 700 construct-deduplicated candidate pairs.
- **G7 Host cleanliness (construction-time).** Host text must be real parsed prose:
  no binary container header, ≤2% non-ASCII, ≥70% alphabetic/space, ≥12 sentences.
  Forced by the Leg-3 audit below. Applied: 242 of 6,132 candidate docs rejected
  (105 binary containers, 133 non-ASCII, 4 low-alpha), leaving 4,260 eligible.

### What forced G7: a Leg-3 correction

Auditing the host corpus for this design surfaced a defect in the **2026-07-26 Leg 3**
result. Two of its fifteen adjudicated pages (`l3_00`, `l3_02`) were served to the panel
and to the production extractor as **unparsed PDF byte streams**, while their reference
criteria came from properly parsed text. Both scored $100\%$ recall on 2- and 3-unit
denominators — the signature of a spurious ceiling.

| | pages | matched-grain recall | raw micro recall |
|---|---:|---:|---:|
| as first reported | 15 | 61/208 = **.293** | 85/541 = .157 |
| excluding the two byte-stream pages | 13 | 56/203 = **.276** | 78/534 = .146 |

The paper now carries $.28$, the corrected value, with the exclusion disclosed.

**Scope of the underlying defect.** The census itself is unaffected: criteria extracted
from those documents are clean and coherent, so the extraction pipeline read properly
parsed text. The corruption is confined to the `doc_text` field of `contexts_*.jsonl`,
which carries raw bytes for ~6% of rows (1,795 of ~29K across the five tasks). Any
analysis reading `contexts_*.jsonl:doc_text` as page text inherits it. Leg 3 did; it has
been corrected. This is now a standing landmine, recorded in the ledger.

## Preregistered ceiling check — the lesson from 2026-07-24

If planted-item recall is **≥.90 in both arms**, the design has saturated and **the primary
test is uninformative regardless of its p-value.** In that case the result is reported as
"design saturated, H1 untested," exactly as a failed realism gate would be. Symmetrically,
if recall is ≤.05 in both arms the design is at floor and the same applies.

Observed planted-item recall will be reported next to the .29 Leg 3 panel-union benchmark
as evidence about whether the planted task is comparable in difficulty to the real one.

## Secondary, descriptive only (not confirmatory)

- Planted recall by host task.
- Planted recall against host length and host criterion count.
- Whether extraction of *pre-existing* host criteria changes between spliced and control
  versions of the same page — a behavioral check that splicing does not perturb extractor
  behavior, which is the thing external validity actually requires.

## Standing disciplines applied

LLM judges do all measurement (embedding similarity is used **only** to generate candidate
pairs for judging, never as a measurement). Anchor gates in every judging wave, results
reported including failures. Nothing under `extraction_validity_20260724/` or
`extraction_validity_20260726/` is modified. Outputs to
`outputs/lexicon/extraction_validity_20260727/`.
