# Where else can we run the "group size ↔ formalization" test?

**Date:** 2026-07-09
**Context:** The GitHub PR result (org size predicts written rules but not decision
*consistency*; small teams more consistent) used repo→org as the grouping unit. Which other
tasks in the portfolio have the structure to replicate it?

## What the test actually requires

1. A **grouping unit** whose accept/reject-style decisions are made by an identifiable body.
2. A **size covariate** on that unit that varies a lot (members, budget, submission volume).
3. A **within-group binary outcome** (accept/reject, matched/not, grant/deny) with **enough
   samples per group** (≥40, both classes ≥8) to fit a within-group predictability model.
4. **Many groups** (≥30ish) so the size↔predictability correlation has power.

The scarce ingredient is #3+#4 together: many groups each with enough labeled within-group
contrast. Most tasks fail on group *count*, not on having a size axis.

## Ranked candidates

### ★★★ Notice-and-comment (agency = group) — BEST fit, arguably better than GitHub
- **Grouping:** federal agency. 18 agencies with built splits, **1.85M comments total**
  (cms 607K, epa 397K, fda 125K, … fsis 12K).
- **Outcome:** `judgement` = comment's claim was matched/adopted into the agency's
  Response-to-Comments (accept/reject analogue). Match rates vary 0.49–0.88 across agencies.
- **Size axis is RICH and externally measurable:** agency FTE headcount, annual budget,
  #rules/year, #comments/rule — all public (OPM FedScope, agency budgets). This is a *real*
  institutional-size variable, unlike GitHub stars.
- **Huge per-group n** → within-agency predictability is high-powered (unlike GitHub's median-0
  contrast). Can even go one level finer: **docket = group** within an agency (thousands of
  dockets), giving a nested size story (agency size × docket salience).
- Metric bank exists (`online-rubrics/{claude,gpt}-parsed/`), autometrics-ready.
- **This is the notice-and-comment idea you had, and the data is already here.**

### ★★ Peer review (venue = group) — works but few groups, confounded
- **Grouping:** venue. Only **9 venues** with ≥100 papers (iclr 32K, neurips 18K, …).
- **Outcome:** accept/reject present (`decision_unified`), but accept rates are wildly
  venue-defined (neurips .95, elife .00, iclr .34) — venue *is* the decision policy, so
  "size" and "outcome regime" are collinear. n=9 groups kills correlational power.
- Better framing here: **area/track within a venue** (ICLR areas), or **year** as a within-venue
  size proxy (submission counts exploded 2017→2024) — "did ICLR get less predictable as it
  scaled?" is a clean longitudinal version of the same question, and we have per-year ICLR.

### ★★ Grant funding (study section / funder = group) — promising, needs assembly
- **NIH RePORTER** (`nih_exporter/`, FY1985–present): grouping = **study section (CSR review
  panel)** or **institute (NCI, NIGMS, …)**. Study sections vary in size/load; institutes vary
  massively in budget. Funded/not is derivable. Hundreds of study sections = good group count.
  But: need to assemble funded-vs-applied (RePORTER is funded-only; unfunded applications aren't
  public → outcome label is the blocker, same problem as always).
- **ERC** (`erc/`): grouping = panel (LS/PE/SH domains). Fewer panels, funded-only again.

### ✗ Others — no viable grouping+outcome+count combo
- **Legal-outcome** (CourtListener): court/circuit *is* a natural group with a size axis (court
  caseload), and outcome (grant/deny motion) exists — but the canonical dockets we've built are
  single-outcome-type slices (title_vii, flsa, ss_disability) and court metadata isn't joined
  in yet. **Assemblable** (courts table is in bulk_data) but not turnkey. Medium-term ★★.
- **Patents:** natural group = **art unit / examiner** (examiner-size / art-unit-load varies,
  and this is a famous formalization axis in the patent-law literature — "examiner leniency").
  But our local patents data has **no art-unit/examiner column** — would need a re-pull from
  PatentsView/PEDS. High value if we fetch it (★★ potential).
- **Humor / creative-writing / press-releases / news-homepages:** grouping units are
  authors/subreddits/publishers, but "size" isn't an institutional-formalization variable and
  the outcome isn't an accept/reject gate. Not this test.
- **Math / stackoverflow / competition:** no org/institution grouping with a size axis.

## Recommendation

Run it on **notice-and-comment next** — it's the cleanest replication and a genuine upgrade:
real institutional size (agency FTE/budget, not a popularity proxy like stars), massive
per-group n (so within-group predictability is well-estimated), and a nested agency→docket
structure. The predicted pattern from the GitHub result: **big agencies have more elaborate
written comment-response procedures but their comment-adoption decisions are no more predictable
per-comment** — and small agencies may apply a more consistent (idiosyncratic) filter. If that
replicates across a totally different domain (federal rulemaking vs open-source code review), the
"formalization-on-paper ≠ formalization-in-practice" finding becomes a cross-domain claim.

Second-best, cheap: the **ICLR-by-year longitudinal** version ("did the venue get less
predictable as it scaled?"), which needs no new data.

## Method note
"Predictability" = within-group 5-fold CV AUC of accept/reject over the task's VAT feature bank
(and, as validity check, over model-free TF-IDF text — the two agreed ρ=+.34 on GitHub, and the
size finding replicated on both). Same recipe transfers directly to any candidate above.
