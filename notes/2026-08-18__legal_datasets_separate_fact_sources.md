# Legal datasets where the facts come from a DIFFERENT document than the decision

Date: 2026-08-18. Written on user request: a consolidated account of exactly which
legal corpora we hold whose fact patterns (x) are authored **separately from and
before** the decision document that supplies the label (y) — i.e., the corpora
immune to the same-document circularity critique. Everything below was verified
against the per-domain READMEs and the sk3 artifacts on 2026-08-18; counts are
exact row counts of the assembled files, not estimates.

Companion docs (already in repo, narrower scope):
- `datasets/legal-outcome-prediction/EX_ANTE_PIPELINE.md` — the unified Stage 0–4
  schema/guards these corpora are built to (x_docs strictly predate decision_date,
  hard assert; y_raw never collapsed; stable-hash splits; funnel strata never pooled).
- Per-domain READMEs under `datasets/legal-outcome-prediction/{nlrb,dol,mspb,cavc,
  bva,ttab,ptab,ss_exante,trademark}/` — source maps, parser validation rounds.
- `datasets/legal-outcome-prediction/VERIFIABILITY_SCORECARD.md` — per-era doctrine
  accounting for the V metrics.

## Why this cut matters (the criterion)

In the district-court slices (Title VII / FLSA / SS §405(g)), "facts" are
LLM-extracted from the **judge's own opinion** — the document that announces the
outcome. Six rounds of de-leaking killed literal disposition leakage, but the deeper
problem is endogeneity of *framing*: the tone probe on those slices measured a
merits-blind framing-only AUC of 0.55–0.63, localized to judge authorship (trial
findings 0.66–0.72 vs. party-authored MTD complaints 0.54–0.57). That is the
Aletras/Medvedeva critique, and no prompt engineering removes it. The corpora below
avoid it **by construction**: x is a document written by a different author
(a lower tribunal, or a party) at an earlier time, and y is read from a separate
decision document or a coded docket. The assembly guard asserts every x-doc date
strictly precedes the decision date; violations are dropped and logged.

Two structural flavors:
- **A. Appellate-review pairs** — x = the lower tribunal's full written decision;
  y = the reviewing body's disposition (direction always coded RELATIVE to the
  lower decision). The x-author did not know the outcome; the leakage risk is
  remand-round contamination (an ALJ writing *after* a first remand), which the
  temporal guard + `remand_round` stratum handles.
- **B. Party-filing pairs** — x = adversarial filings (petition, notice of
  opposition, merits brief); y = the tribunal's decision. Fully ex-ante; the
  x-author is *motivated* to win, so framing is advocacy, not judicial hindsight.

---

## A. Appellate-review pairs (x = lower tribunal's decision)

### A1. NLRB: ALJ decision → Board disposition — **2,995 pairs, ladder run**
- **Provenance**: nlrb.gov server-rendered decision tables (ALJ list with Board-
  outcome column; Board list searchable by case number), PDFs via
  apps.nlrb.gov/link/document.aspx; join on lead case number, validated on 15
  cross-year samples. No API; Drupal GET pagination. Docket pages supply confound
  metadata (who filed exceptions, allegations, participants).
- **Level**: federal administrative — ALJ (trial-type hearing) → the five-member
  **National Labor Relations Board** (administrative appellate).
- **Law**: National Labor Relations Act unfair-labor-practice provisions —
  §8(a)(1) interference, §8(a)(3) discrimination (Wright Line burden-shifting),
  §8(a)(5) refusal to bargain; §10(b) limitations. Credibility-heavy hearing
  records; thin-ish doctrine over thick inputs.
- **x/y separation**: x = full ALJ decision text (written before Board review);
  y = Board decision text, template-parsed (86.7% hard parse) + listed-column
  reconciliation; 430 residuals adjudicated by Llama-3.3-70B. y_raw: adopt_full
  561 / affirm_modify 1,344 / affirm_in_part 537 / reverse 144 / remand 39.
  y_binary = ALJ affirmed-in-full. "Adopted/no exceptions filed" (352) is a
  separate funnel stratum — never reviewed, different label meaning.
- **Guards run**: 0 temporal violations; 54 short-x quarantined; consolidated-case
  dup groups keyed for grouped splits; dedup keeps the EARLIEST Board decision.
- **Ladder so far** (pool n=2,572, pos .216): V (15 span-grounded regex features)
  .575 · lexical TF-IDF .654 · Llama-8B LoRA ≤.630 (data-bound, not evidence of
  low articulability). A-layer judge pass not yet run.

### A2. DOL: OALJ ALJ decision → BRB / ARB disposition — **8,840 pairs (v2), vetted**
- **Provenance**: the public Azure Cognitive Search index behind DOL's OALJ
  decision search (oalj-search-prod), full text, 431,945 docs dumped 2026-06-12
  (OALJ 174,511 / BRB 18,731 / ARB 6,927). Join = ALJ case number regex from the
  appellate decision, with six validated parser fixes (caselist-page junk,
  citation-fragment mis-joins, surname-agreement guard, etc.).
- **Level**: federal administrative — DOL Office of Administrative Law Judges →
  **Benefits Review Board** (BRB) or **Administrative Review Board** (ARB), both
  administrative appellate.
- **Law**: BRB reviews Black Lung Benefits Act and Longshore & Harbor Workers'
  Compensation Act claims; ARB reviews whistleblower statutes (SOX, AIR21, ERA,
  STAA…) and other Secretary-delegated matters. Medical/eligibility-fact-heavy.
- **x/y separation**: x = latest ALJ decision STRICTLY predating y (the v2 rebuild
  exists precisely because v1 leaked: post-remand ALJ decisions written after a
  Board ruling inflated lexical AUC to .825; fixed by parsing BRB issue dates —
  now "remand_round" is its own stratum, 1,512 rows). Primary pool =
  board_reviewed (BRB 4,321 / ARB 1,414).
- **Vetting**: post-fix lexical .686 (BRB) / .655 (ARB) with doctrinally genuine
  top features; two recorded confounds — expert-witness identity (frequent
  pulmonary experts; group-split candidate) and ARB posture words
  (settlement-approval affirms; stratify). V/A rungs not yet run.

### A3. MSPB final order → Federal Circuit disposition — **598 pairs (honest max)**
- **Provenance**: MSPB precedential + nonprecedential manifests and full-text
  search shards (2005+); CAFC metadata via its wpDataTables POST endpoint
  (origin=MSPB, 3,648 docs); join by docket regex (alphanumeric case-type segment
  fix recovered 1,206 captions). 1,354 CAFC appeals have NO public x (AJ initial
  decisions are FOIA-only; pre-2005 gap) — 598 is the structural ceiling from
  public data.
- **Level**: the only **Article III appellate** pair we hold — MSPB (final
  administrative order on federal-employee adverse actions) → **U.S. Court of
  Appeals for the Federal Circuit**.
- **Law**: Civil Service Reform Act (5 U.S.C. ch. 43/75 adverse actions),
  Whistleblower Protection Act; CAFC review standard 5 U.S.C. §7703.
- **y**: affirmed 439 (73%, Rule 36 summary affirmances included via metadata) /
  dismissed 63 / vacated+remanded 38 / reversed 31. Heavy affirm imbalance —
  README rule: report ladder *gaps* with CIs, never raw AUC comparisons.
- Doctrine bank exists (mspb_cafc, 18 metrics, 10 eras). No ladder yet.

### A4. BVA decision → CAVC disposition — **y-side DONE (20,770), x-side pending**
- **Provenance**: y from CAVC (**U.S. Court of Appeals for Veterans Claims**, an
  Article I federal court) — 142,007 docketed cases 2001–2026 scraped; 20,770
  decision PDFs; dispositions span-grounded-parsed with two 30-case validation
  rounds + 917 LLM-adjudicated residuals → **99.9% labeled**: vacated_remanded
  9,689 / affirmed 7,644 / affirmed_in_part 1,041 / reversed 572 / dismissed 496
  (EAJA + petitions = separate strata). x = the **Board of Veterans' Appeals**
  decision text: va.gov/vetapp file store, ~1M decisions 1992–present, slot-probe
  enumeration (no index pages); scrape state on sk3 (`bva/data`, 5.8G of
  sitemap/state so far) — the corpus was ~12 days out when parked.
- **Level**: administrative appellate (BVA) → Article I court (CAVC).
- **Law**: veterans-benefits law, 38 U.S.C./38 C.F.R. (service connection,
  rating schedule, TDIU §4.16), "benefit of the doubt" doctrine.
- **Status**: the highest-volume appellate pair we would hold (~20K joinable),
  blocked only on finishing the BVA text scrape + case-number join.

### (A5. BVA as its own first-instance domain — designed, not built)
x = claimant filings + Statement of the Case, y = BVA grant/denial. Listed in
EX_ANTE_PIPELINE's domain map; x-docs need OCR and are only partially public.
No assembly exists. Mentioned for completeness.

---

## B. Party-filing pairs (x = adversarial filings, fully ex-ante)

### B1. TTAB inter partes: notice of opposition → Board decision — **2,093 pairs**
- **Provenance**: USPTO Open Data Portal bulk XML (TTABYR backfile, 647,190
  proceedings 1951–present with full coded prosecution history) for labels;
  TTABVUE per-proceeding PDFs for x (notice of opposition / petition to cancel
  + answer; ESTTA cover-sheet dates parse at 99.8%).
- **Level**: **Trademark Trial and Appeal Board** — first-instance administrative
  tribunal inside USPTO (adversarial, trial-like).
- **Law**: Lanham Act — §2(d) likelihood of confusion under the 13-factor
  *DuPont* framework, plus dilution/descriptiveness grounds.
- **x/y separation**: x is party-authored months-to-years before decision; y from
  the coded docket (`BD DECISION: OPP SUSTAINED` etc.). y_binary = plaintiff
  (opposer/petitioner) wins: 1,339 / 754. Selection funnel is the story: merits
  decisions are ~2% of filings (defaults/withdrawals/settlements dominate) —
  funnel quantified, strata never pooled. 116 records quarantined for
  docket-table x-leak risk (prior-proceeding printouts attached as evidence).
- 0 temporal violations; 30-case guard inspection before scale.

### B2. TTAB ex parte appeals: appeal brief → affirm/reverse — **6,908 assembled, 1,688-row modeling pool, lexical vetted**
- x = applicant's appeal brief + examiner's statement (+ reply brief when filed);
  y = Board affirms or reverses the examiner's refusal. Balanced modeling pool
  1,688 rows, appellant-entity-grouped splits (0 straddlers). Lexical group-split
  AUC .676; length-only .493 (clean); reply-brief-rate confound recorded
  (win .666 vs lose .536). `CONFOUND_exa.txt` holds the feature audit.
- Level/law: same tribunal, but *ex parte* review of examination refusals
  (§2(d), descriptiveness, specimen/disclaimer practice).

### B3. PTAB AIA trials: petition → institution / final written decision — **19,239 assembled**
- **Provenance**: USPTO Open Data Portal keyless UI backend (the official v3 API
  needs a key; the UI POST endpoints do not) — ~19.3K proceedings, ~20.6K
  decision records, 1.52M document rows; petition/ID/FWD PDFs via signed
  CloudFront redirects.
- **Level**: **Patent Trial and Appeal Board**, first-instance adversarial
  administrative trials (IPR / PGR / CBM).
- **Law**: America Invents Act, 35 U.S.C. §§311–328; invalidity grounds §102
  anticipation / §103 obviousness; era shifts recorded (BRI→Phillips claim
  construction 2018-11-13; Fintiv discretionary denials ~2020, rescinded 2022,
  reinstated 2025).
- **Two labels**: (1) institution granted/denied; (2) FWD all-claims-invalidated
  vs any-claim-upheld (title-boilerplate parse; 2013–16 "Final Decision" titles
  need body parsing). x = the petition, filed before any board action exists —
  ex-ante by construction. Repeat-player confounds (Apple/Samsung/Google
  petitioners, firm identity) recorded in labels.
- V-metric candidates are unusually good here: claims challenged, grounds count,
  prior-art reference counts/dates, §102/§103 mix, expert-declaration presence.

### B4. SS §405(g) ex-ante briefs — **pilot only (57 briefs downloaded)**
- x = the **claimant's merits brief** (and Commissioner's response) from
  CourtListener/RECAP docket entries — because for Social Security appeals the
  complaint is boilerplate, the brief is the real ex-ante fact document. y = the
  district-court disposition already held in the ex-post slice (44,058 canonical
  dockets, earliest-opinion rule for the strictest date guard).
- **Level**: U.S. **district court** (Article III trial-level) reviewing SSA;
  **law**: 42 U.S.C. §405(g), five-step sequential evaluation 20 C.F.R.
  §404.1520.
- The exclusion regex battery is the interesting artifact: every excluded entry
  type maps to a named leak (EAJA/fees = post-win; objections/R&R = quoted
  recommended disposition; stipulations = the outcome itself; replies = quote
  interim rulings). Enumeration ran to DONE; bulk download never scaled — this is
  the natural companion to the ex-post SS slice if we want a same-y,
  different-x-author contrast (judge-authored vs claimant-authored facts).

---

## C. Examination/prosecution outcomes (x = application as filed)

Not court adjudication, but the same separation property: the applicant writes x,
an examiner produces y, different documents entirely.

### C1. Trademark prosecution — **79,936 balanced (39,968/39,968)**
- USPTO Case Files 2023 bulk; filings 2003–2021-09 (right-censoring handled);
  y=1 registered vs y=0 abandoned-after-office-action (602 status confirmed by an
  examiner OA event); 57 year×basis strata, owner-name-grouped stable-hash
  splits. x = mark text + goods/services statements + Nice classes + basis.
- Vetting caught and fixed a real leak (intent-to-use basis annotations surviving
  only in dead applications); post-decision fields (supp_reg, amendments) are
  AUDIT-ONLY; known artifact: registered GS text is post-amendment (true as-filed
  x needs a ~22h TSDR crawl). Lexical .697 with doctrinally genuine features.
- Level: examiner (agency first instance); law: Lanham Act registrability
  (§2(d) confusion, §2(e) descriptiveness, surname doctrine, ID-Manual
  definiteness), with era-gated doctrine shifts (Tam 2017, Brunetti 2019,
  Booking.com 2020).

### C2. Utility-patent prosecution (in `datasets/patents/`, cross-referenced)
- 579,084 applications (granted 290,198 / not 288,886), claim-level rejection
  labels on a nested 59,937-claim corpus; x = application text as filed, y =
  examiner outcome; examiner-leniency (examiner-LOO) controls required — see the
  patents pipeline memory/notes. Law: 35 U.S.C. §§101/102/103/112.

---

## Explicitly EXCLUDED from this document (same-document extraction)

These are our other legal corpora; they are fine for what they are, but their
facts are parsed from the deciding document itself, which is what the user asked
to exclude:

| Corpus | Why excluded |
|---|---|
| District Title VII / FLSA / SS §405(g) v2 slices (32,005 / 7,140 / 51,101) | facts LLM-extracted from the district opinion; tone endogeneity measured (.55–.63) |
| The 12-domain district ladder in paper-3's appendix (600/domain, incl. ERISA LTD) | same construction |
| ERISA slice (`build_erisa_slice.py`) | same construction |
| Law SE answer votes (4,192 / 72,369-post pool) | community forum quality, not an outcome; single document |
| r/supremecourt commentary votes | crowd preference over commentary, not adjudication |
| prior-norms rule scans (Chandrasekharan, AIRules, 11-domain) | rules corpora, no outcome pairs |

## Status summary

| Domain | x author | Level | Pairs | State |
|---|---|---|---:|---|
| NLRB ALJ→Board | ALJ | admin trial→admin appellate | 2,995 | guards ✓, V+lexical+dense run; A-layer pending |
| DOL OALJ→BRB/ARB | ALJ | admin trial→admin appellate | 8,840 | guards ✓ (v2 temporal fix), lexical vetted; V/A pending |
| MSPB→CAFC | MSPB | admin appellate→**Art. III appellate** | 598 | assembled; heavy affirm imbalance; no ladder |
| BVA→CAVC | BVA | admin appellate→Art. I court | 20,770 y-labeled | **x-side scrape unfinished** — largest blocked win |
| TTAB inter partes | parties | admin tribunal, 1st instance | 2,093 | assembled, guards ✓; no ladder |
| TTAB ex parte | applicant+examiner | admin appeal of examination | 6,908 (pool 1,688) | lexical .676 vetted |
| PTAB IPR/PGR | petitioner | admin trial, 1st instance | 19,239 | assembled; doc PDFs partial; no ladder |
| SS ex-ante briefs | claimant | Art. III district (agency review) | 57 (pilot) | enumeration DONE, download never scaled |
| Trademark prosecution | applicant | examiner | 79,936 | vetted; V/A pending |
| (Patents prosecution) | applicant | examiner | 579,084 | own pipeline (datasets/patents) |

**The two highest-leverage unfinished items**: (1) finish the BVA text scrape and
join → ~20K appellate pairs with a real court on the y side; (2) scale the SS
ex-ante brief download → the only corpus where the *same outcome* can be
predicted from judge-authored vs claimant-authored facts, which directly measures
how much of the ex-post slices' signal is framing endogeneity.
