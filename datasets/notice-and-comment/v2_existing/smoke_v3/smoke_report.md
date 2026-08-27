# Smoke v3 Report — Aggressive Verbatim Norm Extraction

- Model: **claude-sonnet-4-5**
- 20 RTC sections (stratified across EPA/NOAA/FAA/FCC/CMS/FWS), 5K-30K chars each
- 4 parallel workers (5 docs each), wall-clock 533s (~9 min)
- All 20 docs extracted successfully, 0 retries, 0 JSON parse errors
- Input tokens: 82,202, Output tokens: 76,510
- Cost: **$1.39** (Sonnet 4.5 pricing $3/M in, $15/M out)

## 1. Per-doc table (V2 vs new)

| doc_id | ag | rtc_len | pairs_v2 | pairs_new | resp_len_v2 | resp_len_new | norms | norms/resp | norms/v2_pair |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CMS-2019-0111-41965 | CMS | 5,533 | 976 | 5 | 265 | 284 | 7 | 1.4 | 0.01 |
| CMS-2021-0057-0054 | CMS | 26,315 | 0 | 3 | - | 1986 | 15 | 5.0 | - |
| CMS-2025-0028-0696 | CMS | 11,580 | 0 | 1 | - | 2595 | 6 | 6.0 | - |
| EPA-HQ-OAR-1992-0101-0006 | EPA | 12,541 | 9 | 8 | 252 | 1189 | 33 | 4.12 | 3.67 |
| EPA-HQ-OAR-2020-0560-0043 | EPA | 29,080 | 17 | 11 | 331 | 1489 | 39 | 3.55 | 2.29 |
| EPA-HQ-OLEM-2021-0466-0005 | EPA | 13,485 | 8 | 9 | 222 | 790 | 33 | 3.67 | 4.12 |
| EPA-R03-OAR-2020-0528-0008 | EPA | 12,106 | 4 | 3 | 333 | 2817 | 19 | 6.33 | 4.75 |
| FAA-2017-0495-0007 | FAA | 10,027 | 3 | 4 | 141 | 440 | 9 | 2.25 | 3.0 |
| FAA-2018-0161-0004 | FAA | 10,027 | 1 | 1 | 283 | 486 | 2 | 2.0 | 2.0 |
| FAA-2018-0301-0007 | FAA | 10,029 | 2 | 2 | 146 | 428 | 3 | 1.5 | 1.5 |
| FCC-2016-0264-0002 | FCC | 22,232 | 0 | 9 | - | 891 | 30 | 3.33 | - |
| FCC-2018-0404-0001 | FCC | 6,725 | 8 | 8 | 222 | 747 | 22 | 2.75 | 2.75 |
| FCC-2024-0264-0001 | FCC | 20,209 | 9 | 8 | 237 | 608 | 19 | 2.38 | 2.11 |
| FWS-HQ-MB-2018-0047-0014 | FWS | 11,842 | 0 | 7 | - | 631 | 19 | 2.71 | - |
| FWS-HQ-MB-2018-0080-0007 | FWS | 5,558 | 3 | 1 | 214 | 567 | 3 | 3.0 | 1.0 |
| FWS-R4-ES-2013-0010-0037 | FWS | 22,998 | 7 | 23 | 247 | 770 | 76 | 3.3 | 10.86 |
| NOAA-NMFS-2016-0096-0292 | NOAA | 16,233 | 21 | 4 | 268 | 2408 | 22 | 5.5 | 1.05 |
| NOAA-NMFS-2017-0111-0025 | NOAA | 17,665 | 5 | 4 | 300 | 2058 | 24 | 6.0 | 4.8 |
| NOAA-NMFS-2018-0132-0007 | NOAA | 13,925 | 3 | 17 | 213 | 663 | 44 | 2.59 | 14.67 |
| NOAA-NMFS-2018-0133-0024 | NOAA | 8,535 | 3 | 8 | 218 | 808 | 29 | 3.62 | 9.67 |

## 2. Aggregate statistics

- **Pairs per doc**: V2 mean=54.0, new mean=6.8
  - V2 is *finer-grained* — it splits each agency response into sub-pairs by individual commenter point; the new prompt groups one full response into one pair (per-doc ratio median: 0.94×).
- **Per-pair response length**: V2=243 chars (paraphrased), new=1133 chars (verbatim) → **4.66× longer per pair**
- **Total response-text chars across all 20 docs**: V2=285,203, new=131,695 (ratio 0.46×). V2's fine-grained fragmentation produces MORE total paraphrase characters; new captures coarser units but verbatim.
- **Total norms extracted (new)**: 454
- **Norms per response (new)**: mean=3.55, median=3.31
- **Norms per V2-equivalent pair**: mean=4.27, median=2.88 (V2 had zero norm extraction)

### Norm type distribution

| norm_type | count | pct |
|---|---:|---:|
| principle | 128 | 28.2% |
| statutory | 113 | 24.9% |
| procedural | 97 | 21.4% |
| value | 55 | 12.1% |
| balancing | 31 | 6.8% |
| philosophy | 21 | 4.6% |
| cost_benefit | 9 | 2.0% |

All 7 categories represented. Top three (principle, statutory, procedural) account for 74% — these are the agency's substantive policy stances, statutory citations, and process-of-decisionmaking rules. The rarer categories (cost_benefit, philosophy, balancing) are exactly the ones we'd expect to be sparser in any given response.

## 3. Side-by-side examples (V2 paraphrase vs new verbatim+norms)

### FWS-R4-ES-2013-0010-0037 (FWS)

**V2 (7 fragmented pairs, paraphrased)** — first response:

> **comment**: There are three areas under candidate conservation agreements with assurances (CCAAs) specifically designed for the spring pygmy sunfish (Belle Mina Farms Ltd., McDonald Farms, and Horton Farm), all in proposed Unit 1. One peer reviewer and five public commenters stated that these areas should not be excluded from the critical habitat designation, because exclusion would be less protective of the sunfish and its habitat.
>
> **response (paraphrase)**: The Service finds that the areas under the three CCAAs meet the criteria for exclusion. Under the CCAAs, landowners implement conservation measures to address threats to the species' habitat, which outweighs the benefits of designation as critical habitat.

**NEW (23 pairs, verbatim+norms)** — first pair:

> **comment_verbatim**: An additional benefit of inclusion of CCAA-enrolled lands in critical habitat is that the critical habitat (and its incremental benefit under section 7) will remain in place regardless of whether or not the CCAAs persist.
>
> **response_verbatim**: Final critical habitat designation becomes Federal regulation, while these CCAAs can be terminated with 30-days' written notice. If the CCAAs are terminated, the associated permit would no longer be valid, and the full protection of sections 7 and 9 of the Act would be in effect in the areas currently covered. However, there would nonetheless be a slight incremental benefit to having critical habitat in this scenario through the benefits critical habitat provides under section 7 of the Act.
>
> **norms (3)**:
> - [statutory] *"Final critical habitat designation becomes Federal regulation, while these CCAAs can be terminated with 30-days' written notice."* — Critical habitat designations are permanent federal regulations, whereas CCAAs are voluntary agreements that can be terminated on short notice.
> - [statutory] *"If the CCAAs are terminated, the associated permit would no longer be valid, and the full protection of sections 7 and 9 of the Act would be in effect"* — Termination of CCAAs triggers the application of full statutory protections under sections 7 and 9.
> - [balancing] *"there would nonetheless be a slight incremental benefit to having critical habitat in this scenario through the benefits critical habitat provides under section 7 of the Act"* — The agency acknowledges a slight incremental benefit from critical habitat designation even when CCAAs exist.

### NOAA-NMFS-2018-0132-0007 (NOAA)

**V2 (3 fragmented pairs, paraphrased)** — first response:

> **comment**: CDFW submitted a comment recommending final recreational fishing season dates for the 2019 season. CDFW hosted an online survey following the IPHC annual meeting. Based on public comments received on California halibut fisheries and fishing performance in recent years, CDFW recommended season dates of May 1-October 31, or until quota has been attained, whichever comes first.
>
> **response (paraphrase)**: NMFS concurs that the CDFW-recommended season dates are appropriate. The Area 2A catch limit is significantly higher than in the recent past and the season structure recommended by CDFW should allow California to fully utilize its allocation.

**NEW (17 pairs, verbatim+norms)** — first pair:

> **comment_verbatim**: Comment 1 above
>
> **response_verbatim**: NMFS changed season dates off of California in this final rule.
>
> **norms (1)**:
> - [procedural] *"NMFS changed season dates off of California in this final rule"* — The agency modified season dates in response to a comment.

### EPA-HQ-OAR-2020-0560-0043 (EPA)

**V2 (17 fragmented pairs, paraphrased)** — first response:

> **comment**: One commenter remarked that the EPA must revise MACT standards when it finds there have been developments in processes, products, or control technologies under CAA section 112(d)(6) to reduce emissions to the maximum achievable degree. They further stated that it is achievable for facilities to switch to membrane cell technology, as demonstrated by the number of facilities that have already made this switch since adoption of the 2003 rule, which would eliminate emissions of mercury.
>
> **response (paraphrase)**: We agree that it is technically achievable for facilities to switch from mercury cell to membrane cell technology, as there are many instances of successful switches spanning the last three decades. We also agree that it is technologically achievable, as a section 112(d)(2) and (3) beyond-floor measure, to require elimination of mercury emissions from the single remaining operating existing source.

**NEW (11 pairs, verbatim+norms)** — first pair:

> **comment_verbatim**: one commenter agreed with our assessment that emissions were low and that risks were low and at acceptable levels.
>
> **response_verbatim**: As noted above, in 2021, we proposed that the 2003 Mercury Cell Chlor-Alkali NESHAP provides an ample margin of safety to protect public health without any revisions. Other than a general agreement with the results, there were no specific comments submitted on the risk review approach, results, or decision. Therefore, we are finalizing the proposed determination that the risks are acceptable, that the 2003 rule provides an ample margin of safety to protect public health and that no additional standards are necessary to prevent an adverse environmental effect.
>
> **norms (3)**:
> - [principle] *"the 2003 Mercury Cell Chlor-Alkali NESHAP provides an ample margin of safety to protect public health"* — The agency maintains that the existing 2003 NESHAP standard provides an ample margin of safety to protect public health.
> - [value] *"the risks are acceptable"* — The agency concludes that current risks to public health are at acceptable levels.
> - [principle] *"no additional standards are necessary to prevent an adverse environmental effect"* — The agency determines that existing standards are sufficient to prevent adverse environmental effects without further regulation.

## 4. Verdict

**Verdict: GOOD on the two design goals; the comparison frame matters.**

- **Goal 1: verbatim responses.** ACHIEVED. Per-pair response text is 4.7× longer than V2's paraphrase and quotes the agency directly. V2 lost the agency's reasoning by collapsing each pair to a one-sentence summary.
- **Goal 2: more norms per response.** ACHIEVED. Mean **3.55 norms per response** (V2 had zero). The norms span all 7 categories, with a healthy split between substantive (principle/value/balancing/cost_benefit) and procedural/statutory.

**Caveat on pair count:** new prompt yields *fewer pairs per doc* (median 0.94× V2). V2 fragments one agency response into ~3 sub-pairs (one per commenter argument). This isn't necessarily worse — it's a different unit of analysis. If we want V2's fine-grained commenter-argument-level resolution, the prompt would need an explicit splitting instruction. **For norm extraction the coarser unit is arguably better** because it keeps the full reasoning chain together.

**Scaling cost (Sonnet 4.5):**

- 3,644 RTCs from parquet: ~$254
- 15K V2-responsive docs: ~$1046 (these scale roughly linearly with token volume)

**Recommendation:** scale to all **3,644 RTC sections** from the parquet first — the RTC text is already cleanly extracted there, and the per-doc avg of ~14.9K chars matches our smoke distribution. Run the same 4-worker pattern at 16-32 concurrency to keep wall-clock under 2 hours.

**Two prompt revisions worth trying before the full scale-up:**

1. Add an instruction to split when a response addresses multiple distinct commenter arguments (would restore V2's finer pair count without sacrificing verbatim/norm extraction).
2. For norm.norm_verbatim, enforce that the quoted span be findable via `in response_verbatim` substring match — currently this is asserted by the prompt but not validated. Easy to add a post-hoc validator.