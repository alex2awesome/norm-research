# N&C `comment_responses_V2.jsonl` Quality + Coverage Assessment

**Date:** 2026-06-01
**Source file:** `/lfs/skampere3/0/alexspan/regulations-demo/data/bulk_downloads/scripts/data/comment_responses_V2.jsonl` (254 MB, 113,485 lines), now mirrored to `v2_existing/comment_responses_V2.jsonl`.
**Compared against:** local `rtc_extracted/rtc_sections.parquet` (3,644 docs).

---

## Schema confirmed

Each `responses[i]` object has these fields (none of them is the raw verbatim source text — they are LLM-summarized):

```
content_of_comment            (paraphrased commenter argument)
summarized_content_of_comment (shorter paraphrase)
response_to_comment           (paraphrased agency response)
quoted_or_paraphrased         (mostly "paraphrased")
type_of_response              (accepted / rejected / acknowledged / partial / …)
reference_scope               (single_comment / multiple_comments / category)
commenter_identifiers_text    (raw names from the RTC)
response_engagement_type      (substantive / procedural_response / acknowledgement / …)
rule_change_outcome           (no_change_made / minor_change / substantive_change / …)
```

`raw_outputs[i]` is the LLM's raw JSON string output per chunk (often `"[]"` when the chunk produced nothing). The structured `responses[]` is the parsed/concatenated version.

**Important caveat:** `content_of_comment` and `response_to_comment` are themselves LLM paraphrases, not verbatim text from the rule. So V2 is two abstractions removed from raw rule text.

---

## Task 1 – Basic stats

| Metric | Value |
|---|---|
| Total docs | 113,485 |
| Docs with `n_responses > 0` | 15,061 (13.3%) |
| Total (comment, response) pairs | 112,321 |
| Median n_responses (responsive docs) | 3 |
| Mean n_responses (responsive docs) | 7.5 |
| p90 n_responses | 15 |
| Max n_responses | 1,627 |

**87% of V2 docs have zero extracted responses.** This is a much harsher coverage figure than the headline 113K suggests — the useful corpus is ~15K docs / 112K pairs.

### Top 15 agencies (V2)

| agency | docs | docs_w_resp | median n_resp | mean n_resp |
|---|---:|---:|---:|---:|
| SEC   | 32,207 | 1,859 | 3   |  4.6 |
| EPA   | 25,911 | 3,962 | 2   |  5.8 |
| FDA   | 14,695 | 1,761 | 1   |  3.9 |
| FAA   | 13,306 | 2,823 | 2   |  3.7 |
| NOAA  |  3,927 |   709 | 4   |  8.5 |
| FCC   |  2,328 |   528 | 8   | 12.7 |
| ED    |  2,191 |   140 | 8   | 11.8 |
| NHTSA |  1,951 |   481 | 3   |  6.8 |
| DOL   |  1,854 |    38 | 4   | 10.7 |
| APHIS |  1,505 |   271 | 3   |  8.2 |
| FWS   |  1,422 |   390 | 8   | 10.6 |
| AMS   |  1,416 |   394 | 2   |  3.8 |
| CDC   |  1,321 |    72 | 2.5 | 11.7 |
| CMS   |  1,159 |   283 | 14  | 84.0 |
| IRS   |  1,037 |   270 | 7   | 10.3 |

Notable: SEC has the most docs but mostly thin extraction (med 3). FCC, FWS, ED, CMS show the richest per-doc extraction (med 8–14).

---

## Task 2 – Coverage diff vs RTC parquet

### Headline

| Metric | Count | % of RTC |
|---|---:|---:|
| RTC docs | 3,644 | — |
| RTC docs present in V2 (any state) | 3,643 | 100.0% |
| RTC docs with V2 `n_responses > 0` | 2,487 | 68.2% |
| RTC docs MISSING from V2 | 1 | 0.0% |
| V2 docs with responses NOT in my RTC | 10,287 | (separate set) |

V2 is **a near-perfect superset** of my RTC parquet at the document level (only 1 missing: `NHTSA-2017-0085-0001`). But V2's extractor fired on only 68% of my RTC docs — so for ~32% of the docs where I found a real RTC section by regex, V2's LLM-pass returned an empty list.

### Per-agency overlap (top 10 RTC agencies)

| agency | RTC n | in V2 | % | V2 w/ resp | % w/ resp |
|---|---:|---:|---:|---:|---:|
| EPA  | 1512 | 1512 | 100.0% | 1224 | 81.0% |
| FAA  |  299 |  299 | 100.0% |  280 | 93.6% |
| NOAA |  290 |  290 | 100.0% |  201 | 69.3% |
| FCC  |  284 |  284 | 100.0% |  204 | 71.8% |
| CMS  |  241 |  241 | 100.0% |   97 | 40.2% |
| FWS  |  144 |  144 | 100.0% |   69 | 47.9% |
| SEC  |  133 |  133 | 100.0% |   20 | 15.0% |
| IRS  |  117 |  117 | 100.0% |   54 | 46.2% |
| HHS  |   86 |   86 | 100.0% |   22 | 25.6% |
| ED   |   69 |   69 | 100.0% |   45 | 65.2% |

**SEC is a disaster zone** in V2: 15% of my RTC SEC docs have V2 extractions. CMS, FWS, HHS also <50%. EPA and FAA are the strongest, both >80%.

### V2 caught 10,287 docs my RTC regex missed

Sample of V2-only docs (my regex did NOT pick these up but V2 did extract responses):

- `EPA-HQ-OAR-2014-0866-0021` (EPA, 8 resp) — but R is "No response necessary." → low value
- `NOAA-NMFS-2020-0149-0013` (NOAA, 3 resp) — R: "not discussed further as not relevant" → low value
- `FWS-HQ-MB-2021-0057-0030` (FWS, 1 resp) — R: real substantive Tribal-rights discussion
- `FCC-2022-0126-0001` (FCC, 2 resp) — R: real substantive SHLB/SECA filing-window response
- `FAA-2019-0701-0008` (FAA, 4 resp) — R: real substantive AD applicability explanation

Mixed bag — the V2 extractor casts a wider net but a portion is throwaway acknowledgements.

---

## Task 3 – Manual classification of 30 stratified samples

Stratified 5 per agency across top 6 = EPA, FAA, SEC, FDA, NOAA, FCC; `n_responses >= 2`.

**The original heuristic classifier was too strict** (only 2/30 NORMATIVE-RICH because it required dense statute citations). On manual re-read using the user's definitions:

### Manual distribution

| Class | N | % |
|---|---:|---:|
| NORMATIVE-RICH | 16 | 53.3% |
| SUBSTANTIVE-DRY | 7 | 23.3% |
| PERFUNCTORY | 7 | 23.3% |

### Per-agency tilt

| agency | NORM | DRY | PERF |
|---|---:|---:|---:|
| EPA  | 2 | 1 | 2 |
| FAA  | 3 | 1 | 1 |
| SEC  | 3 | 1 | 1 |
| FDA  | 2 | 2 | 1 |
| NOAA | 2 | 1 | 2 |
| FCC  | 4 | 1 | 0 |

Tilts:
- **FCC is the most reliably normative** (4/5 NORM, 0 PERF): broadcast/spectrum debates force the Commission to articulate cost-benefit and statutory reasoning.
- **FAA and SEC are mostly NORM-leaning** when extracted, but contain a lot of technical/safety reasoning that uses CFR citations — these read as principled even when not invoking "policy" language.
- **EPA and NOAA have notable PERF tails** because their RTCs include many "we received support from X commenters" pro-forma acknowledgements that V2 happily extracts as separate response items.

### 3 verbatim NORMATIVE-RICH examples (the kind we want)

**1. EPA — Cross-State Air Pollution (`EPA-R07-OAR-2015-0356-0013`)**
> COMMENT: The commenter stated that EPA must take action on Missouri's submission regarding interstate transport. The commenter asserted that the Cross State Air Pollution Rule (CSAPR) update does not cover all sources of interstate transport and that in EPA's own words is only a "partial remedy" for transport pollution.
>
> RESPONSE: EPA stated it was not taking action on the good neighbor provisions in section 110(a)(2)(D)(i)(I) as Missouri did not address these requirements in its infrastructure SIP submission. EPA acknowledged the commenter's concerns and noted it has already taken steps to address interstate transport with the CSAPR update and will take further steps in a separate action.

Why: Cites statute (CAA §110(a)(2)(D)(i)(I)), articulates a procedural-substantive principle (state must raise it in their submission for EPA to act), references prior action and future plan.

**2. SEC — FICC done-away clearing (`SEC-2024-1712-0001`)**
> COMMENT: Several commenters suggest that the Proposed Rule Change's failure to include a requirement that FICC's direct participants offer done-away clearing services would not sufficiently provide for a workable done-away model.
>
> RESPONSE: The Commission disagrees, stating that a done-away mandate could expose FICC and its participants to unique risks, such as liquidity risks, and could be counterproductive, ultimately discouraging Netting Member intermediaries from providing clearing services to customers.

Why: Explicit principle balancing (risk vs. participation incentives), names the specific harm pathway (liquidity, intermediary disincentive), articulates a value tradeoff.

**3. FCC — Connect America scoring (`FCC-2018-0139-0001`)**
> COMMENT: Hughes contends that low-latency, high-speed bids will always necessarily win. Bids will be scored relative to the reserve price and therefore bids placed for lower speeds and high latency will have the opportunity to compete for support, but will have to be particularly cost-effective to compete with low-latency, high-speed bids.
>
> RESPONSE: The Commission disagrees, stating that bids will be scored relative to the reserve price, allowing lower-speed, high-latency bids to compete if they are cost-effective. The Commission also notes that adopting minimal weights, as Hughes proposes, could deprive rural consumers of higher-speed, lower-latency services.

Why: Articulates a welfare/equity principle (rural consumer access), shows the design-mechanism reasoning behind the scoring system, identifies who bears the harm of an alternative.

### 3 verbatim PERFUNCTORY examples (the kind we don't want)

**1. EPA (`EPA-HQ-OAR-2016-0598-0016`)**
> COMMENT: Four commenters provided substantive comments
> RESPONSE: The Agency's responses to the principal comments are provided below. The remaining comments are addressed in the Response to Comments document available in the docket for this action.

Pure pointer-to-document; no content.

**2. FAA (`FAA-2018-0957-0004`)**
> COMMENT: Air Line Pilots Association, International (ALPA) stated that it supports the NPRM.
> RESPONSE: The FAA acknowledged ALPA's support and determined that air safety and the public interest require adopting the final rule as proposed, except for minor editorial changes.

Boilerplate acknowledgement of support.

**3. NOAA (`NOAA-NMFS-2017-0150-0089`)**
> COMMENT: There were 74 unique comments submitted in favor of the action. Of these, 13 were from recreational fishing/diving organizations and 61 were from individuals…
> RESPONSE: NMFS acknowledges the support and has implemented the SMZs as proposed, with some corrections to the coordinates for the Ocean City and Shark River Reef Sites.

Aggregated-support acknowledgement; the only "substance" is a coordinate correction.

---

## Verdict + Recommendation

### Verdict

V2 is **genuinely useful but noisy**. About **half** of the substantive multi-response docs (n_responses ≥ 2) produce real normative reasoning — invocations of statute, cost-benefit balancing, articulation of equity/welfare values, named harm pathways. The other half is roughly evenly split between dry "we updated X" descriptions and pure acknowledgements ("we received support and finalized as proposed"). The LLM-paraphrase nature of `content_of_comment`/`response_to_comment` is a real liability if downstream training requires verbatim agency language — for that you'd want my RTC parquet's raw text or a re-extraction with verbatim preservation.

Coverage-wise, V2 is essentially a superset of my RTC corpus at the document level (3,643 / 3,644) but only fires on 68% of them with content; conversely V2 found ~10K docs with responses my regex missed entirely. SEC is anemic in V2 (15% extraction rate on RTC SEC docs), suggesting the V2 prompt doesn't handle SEC's exchange/SRO-filing format well.

### Recommendation

**Combine, don't pick one.** Concretely:

1. **Use V2 as the primary feedback corpus**, filtered to:
   - `n_responses >= 2` (drops trivial), and
   - exclude rows where `response_engagement_type == "acknowledgement"` or `rule_change_outcome == "no_change_made"` AND response length < 200 chars (these are the PERF tail).
   - This should give ~10K docs / 60K–80K substantive pairs after filtering.

2. **Bias agency mix toward** EPA, FAA, FCC, NOAA, FWS, CMS, IRS (high per-doc richness, mixed normative content). Use SEC sparingly — V2 only catches 15% of SEC RTCs, and the SEC corpus tilts substantively but uniformly procedural-financial (less generalizable normative reasoning).

3. **Backfill from my RTC parquet** for the 1,156 RTC docs where V2 has no responses (32% of my RTC) — particularly for SEC, CMS, FWS, HHS where V2 is weakest. Run a focused extraction on just those docs.

4. **Future:** If verbatim text matters (it does for STaR/judge-rationale work), do a second-pass extraction on V2's responsive docs that pulls the *verbatim* RTC sentences corresponding to V2's paraphrased pairs — V2 gives you a useful index ("here's where the substantive responses are") but the paraphrases lose the agency's actual rhetorical move.

---

## Files

- `/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/comment_responses_V2.jsonl` (downloaded)
- `/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/analyze_v2.py` (analysis script)
- `/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/analysis_output.json` (structured numeric results + 30-doc sample with manual labels)
- `/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/analysis_run.log` (full run log including all 30 verbatim samples)
