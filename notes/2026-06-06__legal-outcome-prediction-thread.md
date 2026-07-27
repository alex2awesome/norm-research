# Legal-outcome prediction — thread summary (2026-06-06)

Detailed write-up of the conversation thread re: focusing the legal-outcome-prediction dataset on a "rule + discretion" slice (analog to a friend's "housing law in Denver" framing). Includes all findings, file locations (local + sk3), and next steps to pick up.

## Context

The legal-outcome-prediction task is currently modeled with a broad federal-district-court corpus (108K cases after extraction) and dense Llama-8B baselines of 0.818 AUC (facts only) / 0.831 (facts + statutes). Concerns raised in conversation:

1. **Is the dataset too broad?** Federal district courts handle everything — civil rights, employment, immigration, IP, criminal, habeas, securities. A model could be learning surface topical features rather than legal reasoning.
2. **Is the 0.818 dense AUC overfitting?** TF-IDF / LR baseline check requested.
3. **A friend's framing**: focus narrowly on one area of law (e.g., "housing law in Denver"), pull all relevant statutes, then look at variation between cases. The hypothesis: where statutes are explicit and applied strictly, the model is doing rule-application; where there's genuine discretion in applying the rule, that's the residual we want to study.
4. **Framing refinement**: legal isn't a strong V (verifiable-via-code) domain — you can't really code-score pretext analysis. But legal IS exceptionally rich in *explicit articulations* of doctrinal rules. So the probe becomes A-heavy: how much of the LLM's signal can explicit, human-articulated doctrinal rules recover, with the residual being genuine T (taste/discretion)?

## Findings

### F1. Starting-set funnel (lots of room to filter)

| Stage | Rows | Source |
|---|---:|---|
| CourtListener total dockets | 68M | bulk_data download |
| District-court dockets kept | 22.7M | filtered in `build_merit_decisions.py` |
| Opinion clusters mapped to district courts | 1.15M | step 3 of build |
| District-court opinions scanned | 1.16M | step 4 |
| **Merit decisions found** (v1 build) | **332,423** | merit_decisions.csv.gz |
| Final v1_dataset (with valid facts + binary label) | 108,129 | LLM-extracted facts + outcomes |
| Binary-labeled subset (drops -1/None) | 88,766 | what dense sweep trained on |

So there's a 22.7M → 332K → 108K → 88K funnel. Going back to the 332K merit pool gives substantial scale-up room for any topical slice.

### F2. The "broad" dataset is actually all federal district courts

Top 15 court_ids in the v1_dataset are all federal trial courts: nysd (SDNY = 10,169 cases), ind, dcd (DC), paed (E.D. Pennsylvania), ilnd, nyed, caed, mad, njd, cand, mied, flsd, pawd, moed, laed.

Federal courts don't do much pure housing law — that's state/municipal. The "housing law in Denver" analog at the federal level is § 1983 civil rights cases (police/prison/housing discrimination), Title VII employment discrimination, FHA, ADA, etc.

### F3. court_id is NOT a major confound

| Model | AUC |
|---|---:|
| court_id only (one-hot LR) | 0.556 |

So venue alone is barely above chance. Court isn't carrying the label.

### F4. TF-IDF baseline on the full 88K dataset

| Model | AUC |
|---|---:|
| TF-IDF facts only (LR, bigrams, 200K features) | **0.742** |
| TF-IDF facts + statute_context | **0.746** |
| TF-IDF facts + court_id prefix | 0.742 |
| Dense Llama-8B facts only (memory claim) | 0.818 |
| Dense Llama-8B facts + statutes (memory claim) | 0.831 |

**Interpretation:** ~75 percentage points of the dense AUC is in surface lexicon (bigrams). Dense beats TF-IDF by ~7 points. So the "deeper than lexical" residual is real but small. The 0.818 is not overfitting in the leakage sense — it's a legitimate model — but it's lexically dominated.

### F5. TF-IDF on the Title VII slice (topically pure)

| Model | Full dataset (88K) | Title VII slice (5,455) | Δ |
|---|---:|---:|---:|
| TF-IDF facts only | 0.742 | **0.726** | -0.016 |
| TF-IDF facts + statutes | 0.746 | 0.718 | -0.028 |
| court_id only | 0.556 | 0.556 | 0.000 |
| TF-IDF facts + court_id | 0.742 | 0.727 | +0.001 |

**Two key results:**
1. AUC barely drops when narrowing to Title VII (-0.016). So TF-IDF's signal is NOT primarily "topic identification" — it's picking up within-topic signal (case-strength patterns, opinion-writing style, fact-pattern features). This is non-obvious and useful: it means we don't need a focused slice to *avoid* the lexical-topic-leak failure mode.
2. statute_context HURTS in the focused slice (-0.028). Confirms statute_context was carrying topical signal in the full dataset, not legal-reasoning signal. In a single-statute slice, it's noise.

### F6. Rule + discretion archetypes in the v1 dataset

Scanned 88K binary-labeled rows for explicit-rule-with-discretionary-tail domains:

| Domain | n in v1 | pos_rate | Rule-vs-discretion fit |
|---|---:|---:|---|
| **Title VII (2000e)** | 5,455 | 0.391 | McDonnell-Douglas burden-shift = textbook structured discretion: prima facie → legit reason → pretext |
| **FLSA (201)** | 3,645 | 0.474 | Explicit overtime/min-wage math + "primary duty" exemption discretion. Most balanced. |
| Bankruptcy (523/727) | 1,400 | 0.314 | Statutory exceptions + "actual fraud" / "willful and malicious" discretion |
| ADA (12112) | 866 | 0.342 | Statutory definition + "reasonable accommodation" / "undue hardship" discretion |
| Habeas / AEDPA (2254) | 838 | 0.101 | "Clearly established law" + "unreasonable application" — but 10% pos is highly imbalanced |
| ERISA (1132) | 619 | 0.397 | "Arbitrary and capricious" review + plan documents |
| **SS disability** | 510 | 0.300 | Textbook 5-step sequential eval + RFC determination. Small here. |
| FMLA (2615) | 373 | 0.273 | Mechanical eligibility + "serious health condition" discretion |
| **FHA (3601)** | 307 | 0.508 | Literal "housing law" analog. Balanced. Small. |
| Sentencing (3553) | 185 | 0.178 | Guidelines + departures — cleanest framing, smallest N |
| Qualified immunity | 176 | 0.358 | "Clearly established right" test |
| Asylum (1158) | 158 | 0.525 | Persecution categories + credibility |

### F7. Merit-pool scan (332K pool) — real upper bounds

Critical finding: substantial scale-up available for every domain.

| Domain | v1 dataset (88K) | Merit pool (332K) | Scale-up |
|---|---:|---:|---:|
| FLSA | 3,645 | 41,121 | 11× |
| Title VII | 5,455 | 30,536 | 5.6× |
| Habeas | 838 | 20,919 | 25× |
| **SS disability** | **510** | **15,873** | **31×** |
| ERISA | 619 | 7,777 | 12.5× |
| Qualified immunity | 176 | 7,358 | 42× |
| ADA | 866 | 7,148 | 8.2× |
| Sentencing | 185 | 2,566 | 14× |
| FHA | 307 | 1,465 | 4.8× |

**The SS disability scale-up is the headline.** It was 510 in v1 (too small for modeling), but it's 16K in the merit pool. SS disability appeals are the textbook "rigid rules + structured discretion" case in federal court: the 5-step sequential evaluation in 20 CFR § 404 is one of the most rule-bound things in federal practice; RFC and credibility determinations are the discretionary residual. The v1 pipeline probably filtered them out because the procedural structure (appeal of ALJ decision) didn't fit the LLM extractor's templates.

## Where the data lives

### Local (Mac)
- **Dataset dir**: `/Users/spangher/Projects/stanford-research/norm-research/datasets/legal-outcome-prediction/`
  - Collection / preprocessing scripts (mostly Mar-Apr 2026): `build_courtlistener_dataset.py`, `build_merit_decisions.py`, `extract_facts_and_outcomes.py`, `run_full_extraction.py`, `extract_citations.py`, `assemble_v1_dataset.py`, `clean_facts_prompt.py`, `parse_findings_of_fact.py`, `build_statute_db.py`, `build_state_statute_index.py`, `build_statute_lookup.py`, `scrape_justia.py`, `fetch_cfr.py`
  - Subdirs: `bulk_data/` (bulk CSV pointers), `online-rubrics/` (legal commentary corpus), `sample_opinions/` (sample data)
  - `README.md` (~190 lines, written 2026-06-05 in this refactor pass)
  - `research_notes.md` (Mar 2026 build-time notes)

### sk3
- **Working dir**: `/lfs/skampere3/0/alexspan/norm-research/datasets/legal-outcome-prediction/`
- **Bulk source**:
  - `bulk_data/courts-2025-07-02.csv.bz2` — 81 KB
  - `bulk_data/dockets-2025-07-02.csv.bz2` — 4.6 GB
  - `bulk_data/opinion-clusters-2025-07-02.csv.bz2` — 2.4 GB
  - `bulk_data/opinions-2025-07-02.csv.bz2` — 53.4 GB
- **Merit-decision intermediates**:
  - `merit_decisions.csv.gz` — 3.3 GB, **332,423 rows** (v1 build, Mar 24)
  - `merit_decisions_v2.csv.gz` — 1.3 GB, 118,025 rows (v2 build, Mar 25; tighter merit filter)
  - Schema: opinion_id, cluster_id, docket_id, court_id, case_name, date_filed, has_fjc_link, token_count, text
- **Canonical dataset**:
  - `v1_dataset.jsonl` — 1.77 GB, **108,129 rows**
  - Schema: opinion_id, court_id, case_name, date_filed, outcome, outcome_confidence, outcome_summary, binary_label, facts, facts_chars, num_citations, num_resolved, num_unresolved, resolved_citations, statute_context, statute_context_chars
- **Cleaned facts**: `cleaned_facts.jsonl` — 167 MB, 88,766 rows (binary-labeled subset)
- **Citations**: `citations.jsonl` (49 MB), `citations_v2.jsonl` (48 MB)
- **CFR**: `cfr_texts.jsonl` (19 MB)
- **State statutes** (Justia scraper, 20+ states): `justia_statutes.jsonl`, `justia_output/{state}.jsonl`
- **Dense reward model runs**:
  - `runs/legal_outcome_facts_only_sweep_llama8b/` — facts-only sweep
  - `runs/legal_outcome_facts_statutes_sweep_llama8b/` — facts + statutes sweep
  - Each contains `subset_*/validation_metrics.csv` with per-fraction AUC

### Memory entries
None currently. After this thread, worth writing:
- `project_legal_outcome_modeling_2026_06_06.md` — capture the 0.742 / 0.818 TF-IDF/dense gap, court_id non-confound, and the merit-pool scale-up table

## Recommendations / next steps

### Recommendation: SS disability (16K cases) is the strongest probe

Given the user's "discretion = bigger gap" framing and the V→A reframing (legal is A-heavy, not V-heavy):

**SS disability** is the cleanest match:
- 16K cases in merit pool (plenty of N)
- 20 CFR § 404.1520 sequential evaluation is one of the most rule-bound frameworks in federal court (clean A layer to articulate)
- RFC (residual functional capacity) and credibility determinations are the discretionary residual (clean T to measure)
- pos_rate is balanced enough (~0.30 in v1; probably similar in merit pool)
- Closest federal-court analog to the friend's "narrow legal domain, deep into facts" framing

**Title VII** is the strong second choice:
- 30K cases in merit pool
- McDonnell-Douglas is the canonical structured-rule-with-discretion framework
- 39% pos_rate (balanced-ish)
- Heavy doctrinal-commentary corpus available as articulation ground truth

**FLSA** as a fallback:
- 41K cases (biggest N)
- Cleanest V layer (overtime arithmetic)
- Discretion concentrated in one point ("primary duty test") — possibly too narrow for the probe

### Concrete next steps (in order)

1. **Pull the SS disability slice from the 332K merit pool.** Filter merit_decisions.csv.gz with the regex used in F7. Save as `ss_disability_slice.jsonl.gz`. ~5 min CPU.
2. **Re-run the LLM facts/outcome extraction** on the SS disability slice. The v1 pipeline dropped these because the procedural structure didn't fit. May need a SS-disability-specific extraction prompt that knows about ALJ decisions, sequential evaluation steps, RFC.
3. **Pull the explicit doctrinal articulation corpus**: 20 CFR § 404.1500-1599 (sequential evaluation regs), POMS DI 24500 series (SSA program operations manual), Hallex (SSA hearings, appeals, and litigation law manual), key Supreme Court cases (Sims, Bowen, Sullivan). These are the ground-truth "A layer" articulations.
4. **Run TF-IDF + LR + dense baselines on the SS disability slice** to establish the lexical floor and dense ceiling for this specific domain.
5. **Build the A-layer scoring system**: encode the explicit doctrinal rules into a programmatic scorer (e.g., does the opinion engage all 5 steps; what's the RFC determination; is the credibility analysis present). Measure AUC of this rule-based scorer.
6. **Measure the gap**: dense AUC − (lexical AUC + rule-scorer AUC) = the residual that's neither surface lexicon nor explicit doctrinal articulation = the genuine T tail.

### Alternative path: Title VII probe

If SS disability extraction proves hard (procedural-structure mismatch in v1 pipeline suggests it might), the fallback is Title VII:
- 5,455 cases already in v1_dataset, expandable to 30K from merit pool
- McDonnell-Douglas factors are textbook
- TF-IDF baseline on the v1 Title VII slice is 0.726 — established
- Doctrinal articulation corpus: EEOC compliance manual, Title VII case-law commentary, ABA labor-employment treatises

### Tracking

If we pursue this, recommend:
- New task in TaskList: "Build SS disability A-layer probe" (or Title VII alternative)
- Memory entry: `project_legal_outcome_a_layer_probe.md` capturing the probe design
- Update `running-research-notes.md` legal section to reflect this scope shift
- Update `datasets/legal-outcome-prediction/README.md` with the slice plan in the open-questions section

## Conversation context — what was decided vs what's still open

**Decided:**
- Legal probe is A-heavy (not V-heavy); the contribution is "can explicit doctrinal articulation recover the dense model's signal?"
- court_id is not a major confound (0.556 AUC)
- TF-IDF lexical baseline is ~0.74; dense ceiling is ~0.82; the gap to attack with A-layer rules is ~7 AUC points
- Title VII slice shows AUC barely changes when topic is held constant — so the lexical signal is within-topic, not topic identification
- statute_context becomes noise in a topically pure slice
- SS disability is the strongest candidate domain (31× scale-up to 16K cases)

**Still open (your call):**
- Which slice to actually pursue: SS disability vs Title VII vs FLSA vs FHA
- Whether to do the LLM re-extraction for SS disability (needs SS-specific prompt) or use Title VII (already extracted)
- Whether to build the explicit rule-scorer first, or train a dense model on the slice first to get the ceiling
