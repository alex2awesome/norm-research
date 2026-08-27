# Layer-3 articulation closure — CONFIRMATORY campaign, N&C RESPONDED cell

Date: 2026-08-06. Status: **CONFIRMATORY, run under the FROZEN protocol.**
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` (FREEZE DECLARATION 2026-08-06 +
FREEZE ADDENDUM + FREEZE ADDENDUM 2). Pilot precedent: the peer-verdict rounds 1-4
(`notes/2026-08-05__layer3_round{1,2,3,4}_peer_verdict.md`) and the missing-mass
robustification battery (`notes/2026-08-06__missing-mass-robustification.md`).
Cell profile: `notes/2026-08-06__layer2_robustness.md` §1, §3.
Code + artifacts: `methods/taste_decomposition/closure/nc_responded/`.

Terminology, spelled out on first mention per the standing rule.
**N&C** = notice-and-comment, the United States federal rulemaking process in which
agencies publish a proposed rule and the public files comments on it.
**RESPONDED** = the cell's outcome label: a comment received a government response
(1) versus was genuinely unmatched (0).
**V** = 27 programmatic surface features (length, punctuation, keyword counts);
**A** = the articulated-criterion bank, here the pre-GEPA 198-rubric Gemma bank;
**VA** = V and A concatenated; **lin** / **nl** = the frozen linear (standardise +
logistic regression) / gradient-boosted (HistGradientBoosting) aggregation of the
same score matrix; **T** = the dense readout (Llama-3.1-8B LoRA on raw comment text);
**Δ_beyond** = T − VA_nl, the unarticulated residual; **Δ_r** = that residual at
mining round r; **ε** = .005, the per-round saturation threshold; **AUC** = area
under the ROC curve; **OOF** = out-of-fold; **GKF** = GroupKFold; **GEPA** = the
prompt-iteration standard the A bank is normally held to; **Good-Turing missing
mass** = the estimated probability that the next independent proposal names a
species not yet seen; **species** = an equivalence class of proposals judged to name
one concept; **P** = number of proposers in a sealed fleet round; **k_A / k_B** =
criteria scored per round on Track A / Track B; **MIXED** = a Track-B channel whose
conjectured upstream cause plausibly also causes real quality.

---

## 0. Why this cell is the hard one (priors carried in from Layer 2)

The task brief and `notes/2026-08-06__layer2_robustness.md` both flag this cell as
the program's most structurally awkward, and every one of those flags survived
contact with the data:

| prior | Layer-2 value | consequence for this campaign |
|---|---|---|
| **SEVERE docket-identity structure** | docket-identity-alone AUC **.916**, far above VA_nl's pooled .726; within-docket AUC drops to .675 | every split is docket-grouped; the mining slice, MONITOR and the dense splits are all whole dockets |
| **worst length sensitivity in the program** | VA_nl length-stratified drop **−.073**, the largest failure in all 9 cells | Track B starts with strong priors; length is a *declared* channel from round 1, not a discovery |
| **dense chain selected on TEST** | protocol deviation recorded at build time | **T = .8167 is quoted with that flag attached, always** |
| **V dominates A** | Layer-1: V_nl .709, A_nl .658, VA_nl .724; Δ_interact **+.089, V-driven** | the articulated bank adds ~.015 AUC over surface features alone; the closure question here is unusually stark |

---

## 1. Frozen setup, and the two implementation decisions recorded before round 1

### 1.1 Population and splits

Population = the cell's A/V evaluation population: **9,521 comments, 1,814 dockets,
78.1% responded**, identical rows to `nc_layer1_stack.load_cell("responded")` under
the FROZEN GKF design (`methods/taste_decomposition/results/nc_responded_layer1.json`).

| split | rule | n | dockets |
|---|---|---:|---:|
| FIT+MINE | sha256(docket)/2²⁵⁶ < .80 | 7,629 | 1,457 |
| MONITOR_FULL | hash ≥ .80 | 1,892 | 357 |
| **MONITOR** | hash ≥ .80 **and** dense-held-out | **377** | 207 |
| unused (monitor-side, dense-TRAIN) | hash ≥ .80 and dense-train | 1,515 | — |
| mining slice M | FIT+MINE and dense-held-out | 1,527 | — |
| honest population | all dense-held-out rows | 1,904 | — |

FIT+MINE and MONITOR_FULL are docket-disjoint by construction (asserted in code).
The 1,515 monitor-side dense-TRAIN rows are used by **nothing**: not fit on, and not
read as MONITOR because T is contaminated there. That is strictly more conservative
than folding them back into FIT+MINE.

**DECISION 1 (recorded before any round ran).** The freeze fixes "MONITOR ⊂
dense-held-out". On this cell the dense chain held out only 20% of rows, so that
intersection is n=377 — too thin to carry the saturation statistic (its VA_nl seed
spread is **.0365**, versus **.0074** on MONITOR_FULL). The stopping rule turns on the
**VA_nl gain**, which requires VA honesty and *not* T honesty. So the readout is split
and both halves are reported every round:

- **saturation statistic** = VA_nl gain on **MONITOR_FULL** (n=1,892, VA-honest) —
  the exact analogue of the pilot's MONITOR (n=1,192) on which the rule was applied;
- **Δ_r level** = T − VA_nl on **MONITOR** (n=377, both honest);
- **honest level** = Δ on all 1,904 dense-held-out rows (OOF inside FIT+MINE,
  refit-predict on the monitor side) — better powered, mildly mining-contaminated and
  therefore conservative, exactly as in the pilot.

**DECISION 2 (recorded before any round ran).** The freeze fixes k_A=15 / k_B=10 as
the per-round **scored** budget while the sealed fleet produces far more (P×15 and
P×10). A label-blind selection rule is therefore required and is stated in
`select_round_set.py`: per round, **2 planted probe pairs** (one substantive member to
A, its shape-only look-alike to B) plus **13 A species** (8 consensus + 5 diversity)
and **8 B species** (5 consensus + 3 diversity). *Consensus* = most distinct
proposers naming the species, ties by stable sha256; *diversity* = stable-sha256 draw
from species named by exactly one proposer. Both strata are recorded per criterion so
each round's gain can be attributed to either.

### 1.2 GEPA: what was actually done

The freeze requires GEPA-iterated phrasing before any confirmatory Δ is quoted. The
repo's GEPA machinery optimises rubric wording against a **labelled** signal, which
would break this campaign's hard label-blindness rule. So what runs here is the
label-blind half of that standard, named honestly (`phrasing_pass.py`):

1. a frontier rewriter puts all 25 selected criteria into A-bank house style (one
   scorable property, explicit 0 and 10 anchors, no outcome reference, and
   specifically no instruction to score document *shape* in place of merit — the
   pilot's twice-caught authoring failure mode);
2. a **blind fidelity gate**: two sealed judges see original-vs-rewrite as an
   unlabelled X/Y pair, interleaved with the authored anchor battery, and answer
   "same underlying concept?". A rewrite judged DIFFERENT is **rejected** and the
   original phrasing is scored instead.

So the pass can sharpen phrasing but cannot silently swap a concept. Every quoted
number below is post-phrasing-pass; the rejection count is reported per round. This
is a **deviation from the letter of the freeze** (no label-driven iteration) and is
flagged wherever a number is quoted.

---

## 2. ROUND 0 — baseline and the incoming bank's concept census

### 2.1 The bank census (freeze: "concept census of the incoming bank at round 0")

Cheapest decisive test first; embedding used only to shortlist, and only within one
register (bank rubric vs bank rubric); identity decided by two sealed blind judges.

| level | instrument | count |
|---|---|---:|
| L0 | rubrics delivered | **198** |
| L1 | distinct names (normalised, exact) | **198** |
| L2 | columns surviving the frozen degeneracy screen (fit on FIT+MINE only) | **171** |
| L3 | value clusters after collapsing \|Pearson r\| ≥ .98 columns | **171** |
| L5 | **effective concepts** after blind pairwise adjudication (strict rule) | **157** |
| L5′ | same, loose rule (either judge says SAME) | 155 |

Instrument health: judge raw agreement **.9375**; both judges **4/4** on the anchor
battery. The SAME anchors were **authored paraphrase pairs**, per the missing-mass
note's PART-4 fix 5 — the pilot's cosine-derived SAME anchors were rejected by both
judges and taught nothing; these were accepted by both, so the instrument now has a
demonstrated positive as well as negative control.

**The headline contrast with the peer bank is large and it matters.** The peer bank
collapsed 154 delivered → 54 effective concepts (−65%). This bank collapses 198 → 157
(−21%), with **zero** value-level near-duplicates (max column \|r\| = .908; only .07%
of the 14,535 column pairs reach .90). The ten merged clusters are the obvious ones —
three PRA/ICR burden-estimation rubrics, four Federal-Register/NPRM-notice rubrics,
three distributional-analysis rubrics — i.e. topical families the bank authored more
than once, not a pervasive redundancy.

**But low redundancy has not bought informativeness.** Alone-AUC over the 171
surviving columns, computed on FIT+MINE only: max **.553**, median **.501**, median
absolute deviation from chance **.0031**, exactly **one** column at ≥ .55 and none at
≤ .45. The peer bank's best single concept reached .607. So this bank is 157 nearly
orthogonal, nearly-null measurements — the aggregation, not any member, carries the
signal.

Also recorded per the freeze's register requirement: the bank is written in
**federal regulatory-analysis language** (RIA, E.O. 12866, CFR framing, agency-facing
comment-quality rubrics) while the corpus is **public comments** ranging from law-firm
filings to one-line form letters. That mismatch governs every similarity-based
measurement downstream and is why no cosine threshold is allowed to decide identity
anywhere in this campaign.

### 2.2 Round-0 readouts

All three readouts, one estimator, one split, 198 features after the screen:

| readout | n | VA_lin | VA_nl | Δ_interact | seed spread | T | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MONITOR_FULL (saturation statistic) | 1,892 | .6222 | **.7257** | +.1035 | .0074 | — | — |
| MONITOR (T-honest) | 377 | .7189 | .7737 | +.0548 | .0365 | .7940 | **+.0203** |
| honest (all dense-held-out) | 1,904 | .6888 | **.7808** | +.0920 | .0093 | **.8167** | **+.0359** |

Rank agreement with the dense model at round 0: **ρ = .579** (honest population).

**Three things to notice, and one caveat that must travel with the numbers.**

1. **Δ_interact is enormous and it reproduces Layer 1.** +.1035 on MONITOR_FULL,
   +.0920 on the honest rows, against Layer 1's +.089. The nonlinear aggregation is
   doing far more work here than the criteria are: VA_lin .622 → VA_nl .726 on the
   same matrix.
2. **The closure-protocol Δ is much smaller than the Layer-1 matched Δ_beyond.**
   Layer 1 reports Δ_beyond = +.084 pooled and the brief carries a matched +.092.
   Under the closure protocol, on the same rows where T is honest, Δ = **+.036**
   (n=1,904) or **+.020** (n=377). The reason is not a discrepancy in T: it is that
   VA_nl rises to .781 on the dense-held-out subpopulation, from .724 pooled. This is
   the pilot's "closure-split Δ levels are protocol-specific and NOT comparable to
   Layer-1 Δ_beyond" amendment biting harder than it did on peer. **Only
   round-over-round changes and these honest levels are quotable for this cell.**
3. **The residual to be mined is genuinely small.** A +.036 residual against ε = .005
   means the whole campaign has roughly seven ε-units of headroom to work in. That is
   a much tighter target than the peer pilot's +.092.

**CAVEAT, carried per the brief:** the N&C dense chain selected on TEST (a recorded
protocol deviation). T = .8167 on the 1,904 held-out rows, T = .808 eval / .825 test.
Every Δ quoted for this cell carries that flag.

### 2.3 Swap baseline

Pair algebra on the honest population (w₊ = .816 of (responded, unresponded) pairs
ordered correctly by the dense model, w₋ = .183 ordered backwards):

| quantity | round 0 |
|---|---:|
| C₊ = P(bank concordant \| dense correct) | **.8244** |
| C₋ = P(bank concordant \| dense wrong) | **.5865** |
| bank/dense agreement on label-discordant pairs | .7490 |
| Spearman ρ(VA_nl, dense) | .5791 |

C₋ = .587 is the load-bearing baseline number for the swap question: on the 18% of
pairs the dense model gets backwards, the articulated bank is *already right* 59% of
the time. Any round that raises C₊ while pushing C₋ toward .5 is trading independent
signal for dense-imitation, and the (ΔC₊, ΔC₋) pair per round will show it.

---

## 3. ROUND 1

### 3.1 Fleet — and a recorded degradation

Slice: top \|dense percentile − VA_nl percentile\| inside the mining slice (\|M\| =
1,527), 30 per direction, median \|rank gap\| **.618**. Label-blind: the slice carries
text, dense probability and VA_nl only.

Sealed fleet, k=15 Track A and k=10 Track B per proposer, separate contexts, separate
briefs, per-proposer stable-hash slice reordering (12 distinct orderings), 153 KB
prompts:

| slot | family | model | Track A | Track B |
|---|---|---|---|---|
| claude_sonnet | claude | Claude Sonnet (sealed subagent) | 15/15 | 10/10 |
| claude_opus | claude | Claude Opus (sealed subagent) | 15/15 | 10/10 |
| codex_luna_a | openai | gpt-5.6-luna, `codex exec`, effort high | 15/15 | 10/10 |
| codex_luna_b | openai | gpt-5.6-luna, independent call | 15/15 | 10/10 |
| glm_a, glm_b | glm | glm-5.2 thinking | **MISSING** | **MISSING** |

**DEGRADATION RECORDED: P = 4 across 2 families, not the target P = 6 / 3 families.**
Both GLM keys returned HTTP 429 code **1308** — *"Usage limit reached for 5 hour, your
limit will reset at 2026-08-07 14:24:07"* — a hard subscription window limit, not the
per-request 1302 rate limit the pilot hit. This is above the freeze's stated
degradation floor (P ≥ 4, ≥ 2 families) and is the same fleet size the peer M3
replicates ran at. The runner (`nc_run_glm.py`) resumes by output file and was left
retrying; if GLM lands, it is added to the species pool as a supplementary
higher-P missing-mass readout for that round and never used to change a scored set
after the fact.

Fleet pool: **100 proposals** (60 Track A, 40 Track B).

### 3.2 Species, and the round-1 missing-mass readout (both tracks)

Species by blind pairwise adjudication over an in-register embedding shortlist. The
shortlist instrument is IN RANGE here, unlike the peer M3 cross-register case: max
off-diagonal cosine inside the fleet is **.912** (Track A) and **.937** (Track B),
comfortably above the τ band, versus .72 when the peer fleet was compared against a
differently-registered bank. Identity is still decided by the judges, never by cosine.

| | Track A | Track B |
|---|---:|---:|
| proposals N | 60 | 40 |
| shortlisted pairs adjudicated | 56 | 14 |
| merge edges (both judges SAME) | 18 | 3 |
| **species S_obs** | **46** | **37** |
| judge raw agreement | .964 | .929 |
| anchor battery (both judges) | 4/4, 4/4 | 4/4, 4/4 |
| f1 / f2 | 36 / 6 | 34 / 3 |
| **Good-Turing missing mass M̂** | **.600** | **.850** |
| jackknife 95% CI (leave-one-proposer-out) | [.459, .741] | [.737, .963] |
| cross-proposer recapture | **.40** | **.15** |
| species named by ≥ 2 families | 4 | **0** |

**The first substantive finding of the campaign is the A/B asymmetry.** The
quality-criterion space is partially converged — four independent proposers agree
often enough that 40% of Track-A proposals recapture a species someone else also
named, and Good-Turing puts the unseen mass at .60. The **spurious-channel space is
not converged at all**: recapture .15, missing mass **.85**, and *zero* species named
by both families. Naming what makes a comment good is a shared skill; naming what
makes a comment look good is idiosyncratic. Track-B conclusions on this cell
therefore carry much wider "unfound channels" uncertainty than Track-A ones — which
is exactly what the FREEZE ADDENDUM's B-side missing-mass requirement was added to
expose, and it is visible in round 1.

For calibration: the peer fleet's Track-A M̂ was .42–.55 at P=4–6 with recapture
.20–.32. This cell's Track A is *more* convergent (.60 mass but .40 recapture on a
smaller pool), its Track B much less.

### 3.3 Selection, phrasing pass, and audit

Selection under the pre-declared rule: Track A took 8 consensus + 5 diversity;
**Track B had only 3 consensus species available** (a direct consequence of the .15
recapture above), so 2 slots were filled from the diversity stratum — recorded as a
substitution in `round1_selection.json`. Composite count asserted in code: 0 of 15
Track-A criteria are lexically composite.

Phrasing pass: **25/25 rewrites accepted** by the blind fidelity gate (both judges
SAME on all 25 real pairs), **0 rejected**, anchors 4/4 and 4/4 — including both
authored DIFFERENT anchors correctly refused. All round-1 numbers are therefore
post-phrasing.

Blind routing audit (fresh Sonnet auditor, 25 provenance-stripped criteria):

| | round 1 |
|---|---|
| proposed A / B | 15 / 10 |
| **misrouting rate** | **0.0%** |
| disputes → arbiter | 0 |
| final routing | 15 A / 10 B |
| **planted probes** | **both pairs separated correctly** |

Probe pair P1 — "identifies a specific analytical error in the agency's own analysis"
(→ A) vs "bare mention of the phrase *cost-benefit analysis*" (→ B) — and P2 —
"states who would be affected and how, with concrete detail" (→ A) vs "numbered or
bulleted list formatting" (→ B) — were each split by the auditor as designed.

### 3.4 The round-1 criteria

Track A (13 fleet species + 2 planted): verifiable attributable sourcing;
detail-level concreteness of a proposed alternative; causal mechanism linking action
to goal; engages a named provision or solicited question; comparative tradeoff
analysis; temporal and cumulative effects; concrete distributional effects;
monitoring and verification design; quantified scale of real-world impact;
implementation burden realism; differentiated identification of affected stakeholder
groups; substance survives deletion of the position statement; compliance feasibility
grounded in how the work is done.

Track B (8 fleet species + 2 planted), each carrying its FREEZE-ADDENDUM-2
`upstream_parent` tag and MIXED flag:

| id | channel | conjectured upstream parent | MIXED |
|---|---|---|---|
| B01 | coordinated campaign residue | coordinated advocacy campaign | no |
| B02 | scanned/faxed physical-document artifacts | mail/fax submission channel and era | no |
| B03 | OCR and attachment provenance | professional document production + docket ingestion | **yes** |
| B04 | reads as a graded course assignment | produced as a classroom exercise, not by a stakeholder | no |
| B05 | pseudo-quantified authority | professional advocacy research and briefing support | **yes** |
| B06 | professional regulatory drafting | law-firm / regulatory-affairs drafting resources | **yes** |
| B07 | credentialed insider self-positioning | submitter's sector standing or seniority | **yes** |
| B08 | first-hand local exposure | submitter's direct personal or local exposure | **yes** |
| B09 | bare mention of "cost-benefit analysis" (**planted probe**) | surface-only | no |
| B10 | numbered/bulleted list formatting (**planted probe**) | professional drafting / template use | **yes** |

**6 of 10 channels are MIXED**, so the discount readouts are reported as a band
(all-channels vs mixed-excluded) throughout, per FREEZE ADDENDUM 2. The upstream mode
did what it was added to do: the fleet reasoned from *unseen* causes (submission
channel, law-firm drafting, sector standing, campaign coordination, even classroom
assignments) to *textual* fingerprints, rather than only pattern-hunting the surface.

### 3.5 Scoring

Gemma-4-31B offline-batch vLLM on sk3, 9,521 comments + 150 anchor rows × 25 criteria
= **241,775 prompts**, temperature 0, one-token 0-10 readout, 4,000-char truncation
(matching the bank's own `score_va_gemma_nc.py` convention), regulatory-analyst system
framing. GPU discipline: race-free launcher, ledger-claimed GPUs of other agents
excluded by name, utilisation sized from actually-free memory (landed on GPU 1 at
util .73 alongside a co-tenant, which was never touched); nothing killed.

Anchor battery at the freeze's **K = 50 per class** (the pilot could only afford 12):

| anchor class | mean |
|---|---:|
| positive (responded) | 2.181 |
| negative (unresponded) | 1.952 |
| scrambled | 0.928 |
| **coherent vs scrambled AUC** | **.8167 PASS** |
| positive vs negative AUC | **.537** |

The scrambled gate passes decisively — the instrument is reading text. The
positive/negative separation is **.537**, i.e. barely above chance. At K=50 the
standard error on this AUC is ≈.058, so .537 is not distinguishable from .5. Read
correctly this is not an instrument failure but a **restatement of the cell's
difficulty**: a set of 25 fresh criteria, 15 of them quality-relevant, separates
responded from unresponded comments almost not at all on its own. It is the same
anomaly family as the peer pilot's round-4 inversion (.361 at K=12), now measured at
the sample size the pilot recommended, and it points the same way.

Collapse gate: **1 collapse**, `P11` — *"bare mention of the phrase 'cost-benefit
analysis'"*, modal fraction .995. That is a **planted probe** on the B side: the
auditor caught it, and then the corpus killed it (the phrase essentially never
appears in public comments). As in the peer pilot, a collapsed probe tests the
auditor but cannot test the discount. Overall NA rate .0004.

### 3.6 Round-1 readouts

| | round 0 | round 1 | gain | 95% CI | P(>0) | sub-ε? |
|---|---:|---:|---:|---|---:|---|
| features after screen | 198 | 213 | | | | |
| **VA_nl, MONITOR_FULL (n=1,892)** | .7257 | **.7269** | **+.0012** | [−.0077, +.0104] | .635 | **YES** |
| VA_nl, MONITOR (n=377) | .7737 | .7813 | +.0076 | [−.0119, +.0282] | .776 | no |
| VA_nl, honest (n=1,904) | .7808 | .7858 | +.0050 | [−.0035, +.0142] | .861 | no |
| VA_lin, MONITOR_FULL | .6222 | .6244 | | | | |
| Δ_interact, MONITOR_FULL | +.1035 | +.1025 | | | | |
| VA_nl seed spread, MONITOR_FULL | .0074 | .0044 | | | | |
| **Δ, MONITOR (T = .794)** | **+.0203** | **+.0127** | −.0076 | | | |
| **Δ, honest (T = .8167)** | **+.0359** | **+.0309** | −.0050 | | | |
| ρ(VA_nl, dense), MONITOR_FULL | .6169 | .6391 | +.0222 | | | |
| ρ(VA_nl, dense), honest | .5790 | .5863 | +.0073 | | | |

**Saturation flags so far: [YES] — trailing run 1 of the required 2.**

The round-1 gain is sub-ε on the frozen saturation statistic (+.0012) but the level
readouts move more: the honest residual falls **+.0359 → +.0309**, a **14% reduction
of the whole residual in one round**, which is a much larger *relative* step than
anything the peer pilot produced (its largest single-round Δ decrement was 5% of a
+.092 residual). Small absolute numbers on a small residual are not the same as
nothing happening, and the two readouts disagree in a way worth watching: the gain is
concentrated on the dense-held-out rows.

### 3.7 The swap pair

| | round 0 | round 1 | Δ |
|---|---:|---:|---:|
| C₊ = P(bank concordant \| dense correct) | .8244 | .8297 | **+.0053** |
| C₋ = P(bank concordant \| dense wrong) | .5866 | .5901 | **+.0035** |
| bank/dense agreement on discordant pairs | .7490 | .7527 | +.0037 |
| Spearman ρ vs dense | .5791 | .5863 | +.0072 |

**No swap signature.** The swap hypothesis predicts C₊ up and C₋ *down* — the bank
buying rank agreement with the dense model by taking on its errors. Here **both rose**
(error-minus-insight inheritance = −.0089, i.e. the bank gained genuine ordering
faster than it gained dense-imitation). Round 1's criteria added independent signal
rather than dense mimicry. This is a cleaner result than the peer pilot got, where
rank agreement climbed while label AUC did not.

### 3.8 Track-B discount — and the MIXED band

Nine channels survive the collapse gate (of 10; the planted P11 collapsed).

| readout | ALL 9 channels | MIXED excluded (3 channels) |
|---|---:|---:|
| spurious-alone AUC, linear | .6249 | .5412 |
| **spurious-alone AUC, HistGB** | **.6724** | .5444 |
| pooled T | .8167 | .8167 |
| pooled VA_nl | .7858 | .7858 |
| pooled Δ | +.0309 | +.0309 |
| decile-stratified T_adj | .7843 | .8100 |
| decile-stratified VA_adj | .7358 | .7750 |
| **Δ_adj** | **+.0484** | **+.0350** |
| matched-sampling Δ (triggered, 1,454 pairs) | **+.0450** | not triggered |
| **stacked increment: dense over ALL named channels** | **+.1387** | +.2651 |
| stacked increment: bank over ALL named channels | +.1191 | +.2440 |

**The discount is null at both ends of the MIXED band — the residual does not shrink.**
Δ_adj = +.048 (all channels) and +.035 (mixed excluded) both sit at or above the
undiscounted +.031. The matched-sampling estimator, triggered in round 1 because
spurious-alone already exceeds .65, agrees: +.045. As in the pilot, **do not quote
Δ_adj as an effect size** — stratifying on a .67-AUC nuisance score costs VA more than
it costs T. The defensible claim is the negative one: *nine named nuisance channels,
six of them derived by reasoning from unseen upstream causes, do not explain the
residual.*

The **stacked-increment readout** (freeze addendum, stratification-free) says the same
thing without any stratification at all: after a logistic stack has absorbed the joint
nuisance model, the dense score still adds **+.139 AUC**. That number cannot degenerate
as the nuisance set grows, which is exactly why it was added.

Per-channel alone-AUC on the honest population, strongest first:

| channel | alone AUC | upstream parent | MIXED |
|---|---:|---|---|
| first-hand local exposure | **.593** | submitter's direct personal/local exposure | yes |
| credentialed insider self-positioning | **.574** | submitter's sector standing or seniority | yes |
| **coordinated campaign residue** | **.431** | coordinated advocacy campaign | no |
| professional regulatory drafting | .538 | law-firm / regulatory-affairs resources | yes |
| numbered/bulleted list formatting (probe) | .470 | professional drafting / template use | yes |
| OCR and attachment provenance | .514 | professional document production | yes |
| reads as a graded course assignment | .507 | classroom exercise | no |
| pseudo-quantified authority | .507 | advocacy research/briefing support | yes |
| scanned/faxed document artifacts | .501 | mail/fax submission channel | no |

**Two findings here are worth carrying forward.**

1. **The upstream-reasoning mode earned its place immediately.** The two strongest
   channels in round 1 (.593, .574) are both fingerprints of *unseen* causes —
   personal exposure and sector standing — that a pure surface-pattern hunt would not
   have proposed. Both are MIXED, which is why they are reported in a band rather than
   silently discounted away.
2. **Coordinated-campaign residue is strongly ANTI-predictive (.431).** Comments
   bearing the marks of an organised mass campaign are *less* likely to receive a
   government response. This is a substantive regulatory finding, not a nuisance
   artifact, and it is the mirror image of the peer pilot's bipolar
   artifact-availability channel.

---

## 4. ROUND 2 — and the stopping rule fires

### 4.1 Fleet, species, audit

Slice mined against the round-1 bank (213 features), rounds-1 rows excluded, median
\|rank gap\| **.618 → .536**. Fleet again **P = 4 / 2 families** at selection time
(GLM's 5-hour usage window had not yet reset).

| | Track A | Track B |
|---|---:|---:|
| proposals | 60 | 40 |
| shortlisted pairs | 75 | 12 |
| merge edges | 14 | 4 |
| **species S_obs** | **48** | **36** |
| judge agreement | .933 | 1.000 |
| anchors | 4/4, 4/4 | 4/4, 4/4 |
| **Good-Turing M̂** | **.633** | **.800** |
| cross-proposer recapture | .367 | .200 |
| species named by ≥2 families | 3 | 1 |

Phrasing pass: **25/25 accepted, 0 rejected**, anchors 4/4 and 4/4.

Routing audit: **1 misrouting (4.0%)**, and it produced the round's first genuine
**dispute → frontier arbiter**. The Track-B proposer offered *"implementation-workflow
specificity"* as a spurious channel (upstream parent: "operational proximity to
regulated implementation", MIXED). The blind auditor routed it to A with high
confidence. **The Opus arbiter upheld the auditor** and gave the reason that matters
for this study:

> the instruction scores propositional content about how the rule would actually be
> executed … it references nothing about the submitter, credentials, letterhead,
> length, formatting or form-letter status … the identity load is real but derivative:
> operational detail correlates with sophisticated filers because those filers more
> often possess the content, and routing every merit criterion that correlates with
> filer sophistication to Track B would empty the bank of exactly the criteria that
> define comment quality.

The arbiter also flagged a precision limit worth carrying: as written the criterion
rewards *presence* of specificity without checking the detail's relevance. Final
routing 16 A / 9 B. Both planted probes separated correctly.

Scoring: 241,775 prompts, **0 collapses**, NA rate .0000, anchors pos 2.002 / neg
1.840 / scrambled 0.878, coherent-vs-scrambled **.8145 PASS**, pos-vs-neg **.521**
(again at chance, K=50).

### 4.2 THE CURVE, and the stopping rule

| | round 0 | round 1 | round 2 |
|---|---:|---:|---:|
| features after screen | 198 | 213 | 229 |
| **VA_nl, MONITOR_FULL** | .7257 | .7269 | **.7216** |
| VA_lin, MONITOR_FULL | .6222 | .6244 | .6185 |
| Δ_interact, MONITOR_FULL | +.1035 | +.1025 | +.1031 |
| VA_nl, MONITOR (n=377) | .7737 | .7813 | .7782 |
| **VA_nl, honest (n=1,904)** | .7808 | .7858 | **.7885** |
| **Δ, honest (T=.8167)** | **+.0359** | **+.0309** | **+.0282** |
| Δ, MONITOR (T=.794) | +.0203 | +.0127 | +.0157 |
| ρ(VA_nl, dense), MONITOR_FULL | .6169 | .6391 | .6408 |

Round-over-round gains, docket-level paired bootstrap (2,000 draws):

| round | MONITOR_FULL gain | 95% CI | P(>0) | MONITOR | honest |
|---|---:|---|---:|---:|---:|
| r0→r1 | **+.0012** | [−.0077, +.0104] | .635 | +.0076 | +.0050 |
| r1→r2 | **−.0053** | [−.0136, +.0037] | .124 | −.0030 | +.0027 |

**A rule ambiguity I resolved in favour of the frozen text, and report both ways.**
The prereg says saturation = "2 consecutive rounds with MONITOR VA_nl gain < ε". Read
literally, `gain < ε` is **signed**, so a *negative* gain is sub-ε. My implementation
initially used the magnitude variant `|gain| < ε`. The two readings diverge here for
the first time and I am recording both rather than choosing after the fact:

| reading | r1 | r2 | trailing run | verdict |
|---|---|---|---|---|
| **signed, `gain < ε` (FROZEN prereg text)** | YES | YES | **2** | **SATURATION DECLARED at round 2** |
| magnitude variant, `\|gain\| < ε` (conservative) | YES | no | 0 | keep mining |

The signed reading is the frozen one and it governs. Substantively it is also the
right one: a round whose criteria made the monitor readout *worse* is not evidence
that mining is still buying anything.

**CONFIRMATORY DECLARATION (fixed here, not revisable by later rounds):**
saturation at round 2; **Δ_plateau = +.0282 on the honest population** (T = .8167 vs
VA_nl = .7885, n = 1,904), carrying the dense-chain-selected-on-TEST flag, the
pre-GEPA-substitute phrasing flag, and the remaining-mass estimate below.

**Rounds 3+ are run as a labelled POST-SATURATION EXPLORATORY EXTENSION** — they
expand the spurious map (whose B-side missing mass is still .80), and they test the
stopping rule's stability, which is the pilot's own biggest methodological worry. They
**cannot** change the declared plateau.

### 4.3 The swap pair — round 2 is where it turns

| step | ΔC₊ | ΔC₋ | error − insight inheritance | swap signature? |
|---|---:|---:|---:|---|
| r0→r1 | **+.0053** | **+.0035** | −.0089 | **no** |
| r1→r2 | **+.0046** | **−.0056** | +.0010 | **YES** |

This is the diagnostic earning its place. Round 1 raised concordance on *both* the
pairs the dense model gets right and the pairs it gets wrong — genuine independent
signal. Round 2 raised C₊ by +.0046 while **losing** .0056 on the pairs the dense
model orders backwards: the bank moved toward the dense model's ranking (ρ .6391 →
.6408) by inheriting its errors at essentially the rate it inherited its insights.
That is exactly the swap the algebra was written to detect, and it is the mechanism
behind the −.0053 MONITOR_FULL gain that fired the stopping rule. Mining a
disagreement slice against a .82-AUC teacher eventually teaches the student the
teacher's mistakes.

### 4.4 Track-B discount at 18 channels

| readout | ALL 18 | MIXED excluded (6) |
|---|---:|---:|
| spurious-alone, linear / HistGB | .6731 / **.6883** | .5853 / .6122 |
| pooled Δ | +.0282 | +.0282 |
| decile-stratified **Δ_adj** | **+.0365** | **+.0374** |
| matched-sampling Δ (1,454 pairs) | **+.0454** | not triggered |
| **stacked increment: dense over all named channels** | **+.1274** | +.2028 |

**Null again, at both ends of the band and by all three estimators.** Doubling the
nuisance set from 9 to 18 moved spurious-alone .672 → .688 (+.016) and left the
residual larger, not smaller, under every discount. The stratification-free stacked
increment is the cleanest statement: after a logistic stack absorbs all eighteen named
channels, the dense score still adds **+.127 AUC**.

Strongest channels across both rounds:

| channel | alone AUC | direction |
|---|---:|---|
| first-hand local exposure | .593 | + |
| institutional credential display | .580 | + |
| credentialed insider self-positioning | .574 | + |
| **campaign template reuse** | **.417** | **−** |
| **coordinated campaign residue** | **.431** | **−** |
| agency-relationship disclosure | .448 | − |

**The campaign finding replicates and strengthens.** Two independently proposed
channels — round 1's "coordinated campaign residue" (.431) and round 2's "campaign
template reuse" (.417) — are the two most anti-predictive features in the whole map.
Mass-campaign comments are markedly *less* likely to draw a government response. The
mirror-image positive channels are all fingerprints of an individual, credentialed,
locally-exposed submitter. Both poles came out of the FREEZE-ADDENDUM-2 upstream mode.

### 4.5 SUPPLEMENTARY: the P = 6 / three-family fleet, and what fleet size buys

GLM's 5-hour window reset while round 2 was scoring, and both GLM slots landed on
Track A (one on Track B) **after** the round-2 scored set had been frozen. Per the
recorded rule they were **not** used to change any scored set; they were banked and the
species clustering was re-run on the enlarged pool as a supplement, with fresh judges.
The P=4 pool actually used for selection is preserved verbatim in `fleet_r2.json`; the
enlarged pool is `fleet_r2_P6_supplement.json`.

This gives the campaign something the freeze wanted and the peer battery never got: a
**direct within-round comparison of the fleet's own size and family count**, same
slice, same round, same instrument.

| | Track A P=4 / 2 fam | **Track A P=6 / 3 fam** | Track B P=4 / 2 fam | **Track B P=5 / 3 fam** |
|---|---:|---:|---:|---:|
| proposals N | 60 | **90** | 40 | **50** |
| species S_obs | 48 | **63** | 36 | **44** |
| f1 / f2 | 38 / 8 | **45 / 11** | 32 / 4 | **39 / 4** |
| **Good-Turing M̂** | .633 | **.500** | .800 | **.780** |
| cross-proposer recapture | .367 | **.500** | .200 | **.220** |
| species named by ≥2 families | 3 | **9** | 1 | **2** |
| judge agreement | .933 | .927 | 1.000 | .917 |
| anchors | 4/4, 4/4 | 4/4, 4/4 | 4/4, 4/4 | 4/4, 4/4 |

**Adding a third family and two proposers converges the QUALITY space and barely
touches the SPURIOUS space.** Track-A missing mass falls .633 → **.500** and recapture
rises .367 → **.500**; Track-B missing mass moves .800 → .780 and recapture .200 →
.220. The A/B asymmetry reported in round 1 is therefore not an artifact of a thin
fleet — it survives the fleet reaching the freeze's target size.

The species-accumulation curve replicates the peer fleet almost exactly: the **sixth**
independent proposer still contributes **6.3 new Track-A species from 15 proposals — a
42% novelty rate**, against the peer fleet's 43% at the same P. Two different corpora,
two different registers, the same "the concept pool is not exhausted" signature.

### 4.6 Round-2 remaining-mass statement

Odds form, the only quotable one (the species/Chao1 form is non-identified at f2 ≤ 11
and is never quoted): with M̂ = .500 (P=6 Track A) and the last realised MONITOR_FULL
gain of −.0053, the odds-form bound on what further proposal buys is **≤ 0** — the
series has turned negative, so the estimator's own answer is that continued mining of
this kind subtracts. Using round 1's positive gain (+.0012) instead as the last
positive step gives R̂ ≈ **+.0012 to +.0018**, i.e. **4–6% of the +.0282 residual**.
Both routes agree the recoverable remainder is small; neither is a licence to say the
remainder is zero.

---

## 5. ROUND 3 — post-saturation extension

Slice mined against the round-2 bank (229 features), rounds-1-and-2 rows excluded.
Mining-slice median \|rank gap\|: **.618 → .536 → .473** — the disagreement the miner
is shown shrinks monotonically, exactly the pilot's pattern.

| | Track A | Track B |
|---|---:|---:|
| proposals | 60 | 40 |
| shortlisted pairs | 61 | 9 |
| merge edges | 21 | 5 |
| species S_obs | **44** | **36** |
| judge agreement | .967 | 1.000 |
| anchors | 4/4, 4/4 | 4/4, 4/4 |
| **Good-Turing M̂** | **.533** | **.825** |
| cross-proposer recapture | .467 | .175 |

Track-A missing mass falls again at fixed P=4 (.600 → .633 → **.533**) and recapture
rises (.400 → .367 → **.467**): by round 3 the proposers are converging on each other.
Track B does not converge at all (.850 → .800 → **.825**).

Phrasing pass: **25/25 accepted, 0 rejected**, anchors 4/4 and 4/4.

Routing audit: **1 misrouting (4.0%)**, again escalated to the arbiter — and this time
**the arbiter overturned the proposing track in the other direction**. The Track-A
proposer offered *"ask serves more than the commenter's own burden or exposure"*. The
auditor called it incidental. The arbiter agreed, and the reasoning is the sharpest
methodological moment of the campaign:

> The instruction scores beneficiary orientation *"as the comment itself frames it"*,
> with no evidentiary predicate anywhere in the anchors: a bare, unsupported claim that
> a change helps third parties earns a 10, while a data-rich, technically rigorous
> demonstration that a deadline is infeasible earns a 0 because the relief sought is
> the submitter's. That is scoring the stance the comment adopts about who benefits …
> close to collinear with submitter identity (advocacy organisation vs. regulated trade
> association) … A merit-preserving version would score whether the comment
> *substantiates* effects on third parties; this instruction as written does not.

Final routing 14 A / 11 B. Both planted probes separated correctly.

**Across three rounds the audit has now caught one shape-feature attempt in each
direction** — a nuisance-proposed criterion that was really merit (round 2) and a
merit-proposed criterion that was really stance (round 3) — with the arbiter siding
with the blind auditor both times. The audit is load-bearing here exactly as it was in
the pilot.

### 5.1 Round-3 scoring and readouts

241,775 prompts. Anchors pos 2.031 / neg 1.808 / scrambled **0.305**, coherent vs
scrambled **.9823** (the campaign's best), pos-vs-neg **.5616**. NA rate .001.

**Three collapses, and all three are planted probes** — including both round-3 A-side
probes:

| id | route | criterion | modal frac |
|---|---|---|---:|
| P13 | B | text is entirely upper case (probe) | .981 |
| P16 | **A** | states the conditions under which its own recommendation would fail (probe) | .990 |
| P22 | **A** | identifies an internal inconsistency between parts of the proposal (probe) | .980 |

Those are corpus facts, not decoding failures: public comments essentially never state
the conditions under which their own recommendation would fail, and essentially never
identify an internal inconsistency in the proposal. **Self-limiting and
inconsistency-hunting behaviours simply do not occur in this corpus.** That is a
substantive finding about regulatory comments as a genre, and it echoes the peer
pilot's round-4 discovery that ML abstracts never report seed variability. Round 3
therefore contributes **12 A** and **10 B** criteria, not 14/11.

| | round 2 | round 3 |
|---|---:|---:|
| features after screen | 229 | 241 |
| **VA_nl, MONITOR_FULL** | .7216 | **.7210** |
| VA_nl, MONITOR (n=377) | .7782 | .7735 |
| VA_nl, honest (n=1,904) | .7885 | .7853 |
| **Δ, honest** | **+.0282** | **+.0314** |
| ρ(VA_nl, dense), MONITOR_FULL | .6408 | .6264 |

| round | MONITOR_FULL gain | 95% CI | P(>0) | MONITOR | honest |
|---|---:|---|---:|---:|---:|
| r2→r3 | **−.0006** | [−.0091, +.0074] | .406 | −.0048 | −.0032 |

Sub-ε under **both** readings. Signed flags [YES, YES, YES] → trailing 3; magnitude
flags [YES, no, YES] → trailing 1.

Swap: ΔC₊ = **−.0029**, ΔC₋ = **−.0048**, Δρ = −.0061. Round 3 lost ground on *both*
pair classes and moved *away* from the dense model's ranking. Not a swap — just noise
and mild over-fitting. The curve is oscillating around its floor.

Track B at 28 channels: spurious-alone **.712** (all) / .668 (mixed excluded); Δ_adj
+.0433 / +.0402; matched-sampling Δ +.0485 / +.0636; stacked increment of dense over
all 28 named channels **+.105** (all) / +.148 (ex-mixed). **Null discount for the
third round running**, now at a nuisance set whose joint score reaches .712 — 87% of
the way from chance to T.

---

## 6. ROUND 4 — the audit turns on the campaign's own probe

Slice mined against the round-3 bank (241 features), rounds 1-3 rows excluded, median
\|rank gap\| **.618 → .536 → .473 → .441**.

| | Track A | Track B |
|---|---:|---:|
| species S_obs | **40** | **34** |
| judge agreement | **1.000** | .917 |
| anchors | 4/4, 4/4 | 4/4, 4/4 |
| **Good-Turing M̂** | **.450** | **.725** |
| cross-proposer recapture | **.550** | .275 |
| species named by ≥2 families | 6 | 1 |

**Track-A missing mass falls monotonically at fixed P = 4 across the campaign:
.600 → .633 → .533 → .450, with recapture rising .400 → .367 → .467 → .550.** By round
4 more than half of every Track-A proposal is a concept some other proposer also
named. Track B never converges: .850 → .800 → .825 → .725, recapture .150 → .200 →
.175 → .275.

Phrasing pass: 25/25 accepted, 0 rejected, anchors 4/4 and 4/4.

### 6.1 The audit catches the campaign runner

**Misrouting rate 8.0% (2 of 25) — the campaign's highest — and one of the two is a
planted probe that the campaign runner authored wrongly.**

| id | proposed | auditor | arbiter | criterion |
|---|---|---|---|---|
| P09 | **A (planted probe)** | B | **B — auditor upheld** | ties a requested change to a specific paragraph of regulatory text |
| P24 | B | A | **A — auditor upheld** | operational implementation microdetail |

P09 was authored as the **substantive** member of probe pair P7. The phrasing pass
sharpened its closing clause from *"Judge the anchoring of the request"* into
*"Score the anchoring, not the merit of the change requested"* — and in doing so made
the merit-disclaimer explicit. The blind auditor caught it; the arbiter agreed:

> a one-line unsupported demand citing a paragraph number scores 10 while a data-rich
> comment that names only the topic scores 5, so the induced ordering tracks citation
> convention rather than substantive merit. Provision-level citation is a
> professional-drafting fingerprint heavily confounded with whether the commenter had
> counsel … Admitting it to the A bank would credit the articulated scorecard with
> document shape and inflate apparent closure.

So **probe pair P7 failed to separate** (`all_probes_separated: false`) — but it failed
because the probe's *substantive* member was itself a shape feature, not because the
auditor missed anything. That is the audit working, not the audit failing, and it is
the fourth time in this campaign that the score-the-form failure mode has been caught
(round 2 nuisance→merit, round 3 merit→stance, round 4 twice). It is also a real
finding about the phrasing pass: **sharpening a criterion can sharpen it into a shape
feature**, which is exactly why the fidelity gate checks concept identity and the
routing audit runs *after* the rewrite rather than before.

P24 went the other way: the arbiter restored a Track-B-proposed channel to the bank
because "an insider letter with no particulars scores 0 while any commenter supplying
real operational particulars scores 10" — identity-correlated but not identity-scoring.
That is the same distinction the round-2 arbiter drew, applied consistently.

Final routing 15 A / 10 B.

---

## 7. Robustness of the residual LEVEL — and where enlargement was and was not possible

Driver: `delta_robustness.py`, `round3_delta_robustness.json`,
`within_docket_full_population.json`. All on saved predictions; no refits.

### 7.1 The design already prefers enlargement over small-n

The coordinator's cross-campaign instruction (from CW's Stage-0 correction) is to
prefer enlargement over a thin readout. **This campaign applies that rule by
construction, decided before round 1 ran**: the saturation statistic is read on
MONITOR_FULL (n=1,892) rather than the T-honest MONITOR (n=377), and the level is read
on all 1,904 dense-held-out rows. The choice is worth a great deal — VA_nl seed
spread on the thin set is **.0365** versus **.0074** on the enlarged one, a 5× noise
reduction — and the thin set would have been useless for a ±.005 threshold.

Further population enlargement is **not available** on this cell without breaking the
frozen definitions: the population is exactly the rows the 198-rubric bank scored, and
T exists honestly only on the 20% of rows the dense chain held out. Both limits are
structural, not choices.

### 7.2 The residual level, three ways

| readout | n | T | VA_nl (r0) | Δ (r0) | Δ (r1) | Δ (r2) | Δ (r3) |
|---|---:|---:|---:|---:|---:|---:|---:|
| honest (all dense-held-out) | 1,904 | .8167 | .7808 | **+.0359** | +.0309 | **+.0282** | +.0314 |
| **eval-only** | 952 | .8084 | .7954 | **+.0130** | +.0159 | +.0155 | +.0152 |
| test-only | 952 | .8250 | .7662 | **+.0589** | +.0457 | +.0411 | +.0473 |
| MONITOR (T-honest) | 377 | .7940 | .7737 | +.0203 | +.0127 | +.0157 | +.0205 |

Docket-cluster bootstrap on the honest Δ: **+.0359 [+.0069, +.0651], P(>0) = .992** at
round 0, still **+.0314 [+.0030, +.0604], P = .982** at round 3. The residual is
positive with docket-clustered uncertainty, but the CI is wide (±.029) — docket
clustering costs most of the nominal n.

**The eval/test split is the sharpest caveat this cell carries, and it is bigger than
the T flag suggests.** T differs across the two halves by only .0166 (.8084 vs .8250),
but VA_nl differs by **.029** (.7954 vs .7662), so Δ is **4.5× larger** on the test
half. The dense chain selected on TEST, so the test half is the contaminated one; the
**eval-only Δ ≈ +.013–.016 is the selection-free reading**, and it is roughly **half**
the pooled honest figure.

And on the selection-free half, **mining moved Δ not at all** (+.0130 → +.0159 →
+.0155 → +.0152). Every bit of the pooled Δ decrement across rounds 1-2 came from the
test half (+.0589 → +.0411). That is a strong internal consistency check on the
saturation verdict: where the readout is cleanest, there was never anything to close.

### 7.3 Is the residual a between-docket effect?

This cell's Layer-2 profile (docket-identity-alone AUC .916) makes this the question.
Per-docket AUC is undefined here — only 6 dockets in the honest population clear even
a 5-row bar. So the readout was **enlarged to pair level**: AUC on a binary label is
exactly pair concordance, so pool every (responded, unresponded) pair that lies inside
a single docket and read both instruments on exactly those pairs. That is ~30× more
information than the per-docket-AUC route.

| readout | pairs | dockets | T | VA_nl | Δ |
|---|---:|---:|---:|---:|---:|
| within-docket, honest population | **258** | 20 | .674 | .60–.74 (unstable) | −.016 / −.070 / +.070 / −.054 by state |
| within-docket, FULL population (bank only) | **21,928** | 166 | n/a | **.689** | — |
| pooled, FULL population (bank only) | — | — | n/a | .733 | — |

Two honest conclusions:

1. **The bank's signal is mostly NOT a docket artifact.** On 21,928 within-docket pairs
   the bank scores **.689** against a pooled .733 — it keeps 81% of its edge over
   chance within docket, and the within-docket series tracks the pooled one across
   states (.6890 → .6908 → .6988 → .6859). This reproduces Layer 2's .675-vs-.726
   independently.
2. **The within-docket Δ is NOT estimable on this cell, and must not be quoted.** T
   exists on only 1,904 rows, which yield just 258 within-docket pairs; the Δ estimate
   flips sign across bank states. The pooled Δ therefore cannot be decomposed into
   within- and between-docket components here. Enlargement was pushed as far as the
   data allows and stopped short of an answer — that is a limitation of the cell's
   dense-split geometry, not of the estimator.

### 6.2 Round-4 readouts — and saturation becomes rule-invariant

241,775 prompts, **0 collapses**, NA .001, anchors pos 1.889 / neg 1.578 / scrambled
0.495, coherent-vs-scrambled **.8889 PASS**, pos-vs-neg .547.

| | r0 | r1 | r2 | r3 | **r4** |
|---|---:|---:|---:|---:|---:|
| features after screen | 198 | 213 | 229 | 241 | **256** |
| **VA_nl, MONITOR_FULL** | .7257 | .7269 | .7216 | .7210 | **.7250** |
| VA_nl, MONITOR (n=377) | .7737 | .7813 | .7782 | .7735 | **.7862** |
| VA_nl, honest (n=1,904) | .7808 | .7858 | .7885 | .7853 | **.7905** |
| **Δ, honest (T = .8167)** | **+.0359** | +.0309 | +.0282 | +.0314 | **+.0262** |
| ρ(VA_nl, dense), MONITOR_FULL | .6169 | .6391 | .6408 | .6264 | .6194 |

| round | MONITOR_FULL gain | 95% CI | P(>0) | MONITOR | honest |
|---|---:|---|---:|---:|---:|
| r3→r4 | **+.0040** | [−.0023, +.0111] | .888 | +.0128 | +.0052 |

| reading | r1 | r2 | r3 | r4 | trailing | verdict |
|---|---|---|---|---|---|---|
| **signed (FROZEN)** | YES | YES | YES | YES | **4** | saturated (declared at r2) |
| magnitude variant | YES | no | YES | YES | **2** | **now also saturated** |

**Saturation is now rule-invariant.** The ambiguity flagged at round 2 no longer
matters: by round 4 both readings of "2 consecutive sub-ε rounds" are satisfied. The
confirmatory declaration stays fixed at round 2 (the first round the frozen signed
rule fired); the magnitude variant would have declared at round 4, and the plateau
value is the same either way to within .005.

**Round 4 was, on the honest level, the campaign's best round** — Δ fell to **+.0262**,
its minimum, a **27% reduction of the round-0 residual**. And the swap pair explains
why:

| step | ΔC₊ | ΔC₋ | Δρ | swap? |
|---|---:|---:|---:|---|
| r0→r1 | +.0053 | +.0035 | +.0072 | no |
| r1→r2 | +.0046 | **−.0056** | +.0093 | **YES** |
| r2→r3 | −.0029 | −.0048 | −.0061 | no (both fell) |
| r3→r4 | +.0033 | **+.0136** | +.0016 | **no — the best round** |

Round 4 gained **+.0136 on the pairs the dense model gets backwards** — four times its
gain on the pairs the dense model gets right, and the largest single-round gain on
dense-wrong pairs in the campaign — while barely moving rank agreement with the dense
model (Δρ = +.0016). That is the opposite of the round-2 swap: round 4's criteria added
signal the dense model does **not** have. The pattern across four rounds is that
mining a disagreement slice produces both kinds of round, and only the swap pair tells
them apart.

Track B at 38 channels: spurious-alone **.707** (all) / .689 (mixed excluded); Δ_adj
+.0347 / +.0332; stacked increment of dense over all 38 named channels **+.109**.
Fourth consecutive null discount.

---

## 7A. FREEZE ADDENDUM 4 — position-in-container, and the biggest single finding of the campaign

Addendum 4 landed after round 4 and required Track-B proposers to consider
POSITION-IN-CONTAINER channels, citing the patents audit (claim-ordinal alone-AUC .754,
found only by manual audit; no proposer in the program has ever named this family
unprompted). This cell allows an unusually direct test, because the N&C `doc_id` is
`<AGENCY>-<YEAR>-<DOCKET#>-<SEQ>` — **the container position is recoverable exactly,
for free**. Driver: `position_audit.py`, `position_audit_full_population.json`,
`position_discount.json`.

Alone-AUC on the FULL 9,521-row population (grouped-OOF for the joint model):

| position variable | alone AUC |
|---|---:|
| raw sequence number | .5273 |
| **within-docket rank** | **.6862** |
| within-docket percentile | **.4366** (early comments favoured) |
| **docket size** | **.7102** |
| **JOINT position model (4 variables)** | **.7216** |

Response rate by docket-size quintile: **.640 → .608 → .866 → .885 → .952**. A comment
filed into a 187-comment docket is responded to 95% of the time; in a 2-comment docket,
64%.

**Put that next to the mined nuisance map.** After four rounds, 160 sealed Track-B
proposals and 38 scored channels, the joint mined nuisance model reaches **.707** and
**no single mined channel exceeds .436/.564** on the full population. Four programmatic
container variables, requiring **zero judge calls**, reach **.722** — and the stacked
increment of position over the *entire* named Track-B map is **+.079** on the honest
population (.707 → .786). The mined map does not contain this family at all.

**This replicates the patents result the addendum was written from, on a completely
different corpus, with the same signature: strongly predictive, and invisible to every
proposer.** Across this campaign the fleet proposed campaign residue, letterhead, OCR
artifacts, credential displays, emotional register, course assignments — and never
once, in 160 proposals across four rounds, "which docket is this in and how big is it".

**Does it threaten Δ? No, and the reason is exactly the one the addendum's
interpretation note gives.** Position is not in the text: a judge reading a comment
cannot see it, and neither can the dense model. Spearman correlations with T are .06
to .11 and with VA_nl .05 to .14 — both instruments are nearly blind to it. And
stratifying on the joint position score does **not** shrink the residual:

| bank state | pooled Δ | Δ stratified on position deciles |
|---|---:|---:|
| round 0 | +.0359 | +.0578 |
| round 2 | +.0282 | +.0541 |
| **round 4** | **+.0262** | **+.0478** |

So the correct reading is the H3 one: **position is a large, real, non-textual
determinant of whether a comment gets a response, which lowers the achievable ceiling
for every text-based instrument equally rather than biasing the gap between them.**
T = .8167 is not "82% of the way to explaining agency response" — a substantial part
of what any text instrument cannot reach is container position, and that is not
tacit knowledge in a comment, it is a fact about the docket.

For the record: this is an **audit of the corpus, not a bank metric**. The position
variables were never added to V or A, never judged, and never used in any fit that
feeds the closure curve.

---

## 7B. A correction: per-channel Track-B strengths were read on too small a population

While ranking channels for the Addendum-3 decomposition I recomputed every Track-B
channel's alone-AUC on three populations. The per-channel figures reported for rounds
1-3 above were read on the **honest population (n=1,904)**, and they do not hold up:

| population | max \|alone-AUC − .5\| over 38 channels | median |
|---|---:|---:|
| honest (n = 1,904) | .093 | .039 |
| FIT+MINE (n = 7,629) | .054 | .014 |
| **FULL (n = 9,521)** | **.064** | **.012** |

Mean absolute per-channel disagreement between the honest and full populations is
**.038**, maximum **.083**. So "first-hand local exposure .593" and "credentialed
insider self-positioning .574" are **sub-population noise**; on the full population
they sit near .5. This is the same enlargement lesson the coordinator flagged from CW,
appearing a second time inside this campaign, and the corrected statement is:

**Individually, the mined nuisance channels are near-null on the full population. Only
the joint model reaches .707, and only one family has a stable sign:**

| channel (full population) | alone AUC | direction |
|---|---:|---|
| formal salutation and signature block present | **.436** | − |
| coordinated campaign residue | **.450** | − |
| bare presence of a numeric figure (probe) | .452 | − |
| pseudo-quantified authority | .453 | − |
| formal institutional filing apparatus / letterhead | .464 | − |
| NGO action-alert templated phrasing | **.464** | − |
| campaign template reuse | **.469** | − |

The **campaign/form-letter family replicates across three independent proposals in
three different rounds** and is anti-predictive in all of them, on every population.
That finding stands. The credential/exposure "positives" do not, and are withdrawn.

---

## 8. ROUND 5 — the cap round, spent on the two new addenda

Freeze Addenda 3 and 4 landed mid-campaign, so the cap round was spent implementing
them rather than running a sixth ordinary mining round (recorded deviation: the
author-proposed split is 10 A / 15 B rather than k_A=15/k_B=10; the scored total is
unchanged at 25, and routing is decided by the blind audit as always).

Composition: **16 Addendum-3 decomposition components** (8 MIXED parents × 2) +
**5 Addendum-4 position fingerprint channels** + 2 planted probe pairs.

Parents selected by \|alone-AUC − .5\| **on FIT+MINE only** (a design decision never
reads MONITOR or the honest set): formal salutation/signature block; pseudo-quantified
authority; litigation-brief posture; formal itemized rebuttal structure;
agency-relationship cues; agency-relationship disclosure; formal institutional filing
apparatus; organisational-scale self-credentialing.

Phrasing pass: 25/25 accepted, anchors 4/4 and 4/4. Routing audit: **1 misrouting
(4.0%)**, escalated to the arbiter, and it is the most informative dispute of the
campaign.

**The decomposition of one parent FAILED, and the machinery caught it.** For the
"formal salutation and signature block" parent, the decomposer's *candidate-real*
component was "identified accountable author with stated standing". The blind auditor
routed it to B; the arbiter upheld and went further:

> the instruction scores a costless, unverifiable self-assertion of identity and
> occupation rather than anything the comment establishes about the rule … a
> boilerplate form letter carrying one occupational clause is scored identically to an
> expert submission … The decomposer's separability argument establishes dissociation
> from the letter-furniture sibling, not capture of merit.

So that parent split into **two surface components and no surviving real one**. Recorded
consequence: its **MIXED flag is resolved to not-mixed** — the channel was pure surface
all along — rather than treated as decomposed. The other seven parents decomposed with
both components surviving independent routing. All eight parents are retired from the
Track-B readouts (`retired_channels.json`, recorded not deleted); final routing
**9 A / 16 B**; both probe pairs separated.

That is Addendum 3 doing precisely what it was added for: it converted a dual-reported
sensitivity band into a per-parent verdict, and in one case out of eight the verdict
was "the MIXED flag was wrong".

### 8.1 Round 5 BREAKS THE PLATEAU — the campaign's headline result

241,775 prompts, 1 collapse (the planted probe "comment is shorter than three
sentences", modal .981), NA .000, coherent-vs-scrambled **.8283 PASS**.

| round | MONITOR_FULL gain | 95% CI | P(>0) | sub-ε? |
|---|---:|---|---:|---|
| r1 | +.0012 | [−.0077, +.0104] | .635 | yes |
| r2 | −.0053 | [−.0136, +.0037] | .124 | yes (signed) |
| r3 | −.0006 | [−.0091, +.0074] | .406 | yes |
| r4 | +.0040 | [−.0023, +.0111] | .888 | yes |
| **r5** | **+.0085** | **[+.0001, +.0178]** | **.976** | **NO** |

**Round 5 is the only round of the campaign to clear ε, and the only round whose
bootstrap CI excludes zero.** Both saturation counters reset to 0. The campaign
therefore **ends at the cap of 5 rounds in a NON-saturated state** — the same way the
CW community campaign ended.

Full curve:

| | r0 | r1 | r2 | r3 | r4 | **r5** |
|---|---:|---:|---:|---:|---:|---:|
| features after screen | 198 | 213 | 229 | 241 | 256 | **265** |
| VA_nl, MONITOR_FULL | .7257 | .7269 | .7216 | .7210 | .7250 | **.7335** |
| VA_nl, MONITOR (n=377) | .7737 | .7813 | .7782 | .7735 | .7862 | **.7997** |
| VA_nl, honest (n=1,904) | .7808 | .7858 | .7885 | .7853 | .7905 | **.7957** |
| **Δ, honest (T = .8167)** | **+.0359** | +.0309 | +.0282 | +.0314 | +.0262 | **+.0210** |
| **Δ, eval-only (selection-free)** | +.0130 | +.0159 | +.0155 | +.0152 | +.0100 | **−.0033** |
| Δ, test-only (selected-on) | +.0589 | +.0457 | +.0411 | +.0473 | +.0418 | +.0448 |
| ρ(VA_nl, dense), MONITOR_FULL | .6169 | .6391 | .6408 | .6264 | .6194 | .6333 |

Swap pair, r4→r5: ΔC₊ **+.0055**, ΔC₋ **+.0042**, Δρ **+.0011**. Both pair classes
gained and rank agreement with the dense model barely moved — the cleanest
non-imitative gain shape available, and the second such round in a row.

**Why did round 5 work when four ordinary rounds did not?** Because it changed the
*kind* of proposal, not the amount of it. Its two strongest contributions are:

| criterion | route | alone AUC (full pop) | origin |
|---|---|---:|---|
| **provision-by-provision engagement with the proposed text** | **A** | **.5392** | Addendum-3 candidate-real component of "formal itemized technical/regulatory rebuttal structure" |
| letter furniture: salutation and signature block | B | .4350 | Addendum-3 surface component |
| density of figures, citations and named sources | B | .4455 | Addendum-3 surface component |
| awareness of the accumulated docket record | B | .4709 | Addendum-4 position fingerprint |

The single strongest A-side criterion of the entire campaign — .5392 on the full
population, against a 198-rubric bank whose best member reaches .553 — was recovered
**by splitting a channel that four rounds of ordinary mining had classified as
spurious**. The merit was sitting inside a MIXED nuisance the whole time; the
decomposition pass is what got it out.

### 8.2 Final Track-B map (45 channels, 8 parents retired)

| readout | ALL 45 | MIXED excluded (25) |
|---|---:|---:|
| spurious-alone, linear / HistGB | .6721 / **.7231** | .6570 / .7207 |
| pooled Δ | +.0210 | +.0210 |
| decile-stratified Δ_adj | **+.0248** | +.0255 |
| matched-sampling Δ | **+.0148** | +.0285 |
| stacked increment, dense over all named channels | **+.0985** | +.1008 |

The nuisance set is at its strongest (.723 alone, 89% of the way from chance to T) and
the discount is **still null** — Δ_adj +.025 against a pooled +.021, and matched
sampling +.015. But note the convergence: the ratio Δ_adj/Δ_pooled fell from **1.57**
at round 1 to **1.18** at round 5. As the nuisance map got better, the discounted and
undiscounted residuals moved together, which is what should happen if the map is
genuinely capturing nuisance rather than eating signal.

**Five rounds, 45 named channels, three estimators, one verdict: the mined shortcut
channels do not explain the residual.**

---

## 9. FINAL SUMMARY

### 9.1 The curve and the two declarations

**Declaration 1 (CONFIRMATORY, frozen protocol, fixed at the time and not revised).**
Under the frozen stopping rule — 2 consecutive rounds with MONITOR VA_nl gain < ε —
saturation fired at **round 2** and held through round 4 (the magnitude variant of the
same rule fired at round 4). The plateau at that point was **Δ = +.0282** (honest
population, n = 1,904, T = .8167 vs VA_nl = .7885).

**Declaration 2 (what actually happened).** The plateau was **an artifact of the
miner, not a property of the residual**. Round 5 changed the kind of proposal — MIXED
decomposition components and position-in-container fingerprints instead of more
free-form criteria — and produced the campaign's only super-ε gain (**+.0085,
P(>0) = .976**), taking Δ to **+.0210**. **The campaign ends at the 5-round cap in a
NON-saturated state, with the residual still falling.**

This is the pilot's own warning realised: the peer pilot noted that round 1's null
"would have been declared a taste bound had the rule allowed stopping at one sub-ε
round". Here **four** consecutive sub-ε rounds would have been declared a bound, and a
fifth round of a different shape moved it.

### 9.2 The residual, honestly stated

| statement | value |
|---|---|
| Δ at round 5, honest population (n = 1,904) | **+.0210**, docket-bootstrap CI [−.0070, +.0487], P(>0) = .921 |
| Δ at round 5, **eval-only** (selection-free half, n = 952) | **−.0033** |
| Δ at round 5, test-only (dense chain selected on this half, n = 952) | +.0448 |
| Δ at round 0, honest population | +.0359 |
| total movement across 5 rounds, honest | **−.0149 (−41% of the round-0 residual)** |

**The load-bearing caveat is the eval/test split, and it is decisive.** This cell's
dense chain selected on TEST (recorded protocol deviation, carried from the brief). On
the **selection-free eval half, the residual is now −.003 — the articulated scorecard
has caught the dense model.** All of the remaining pooled residual lives on the
contaminated half. So the defensible closing statement is not "a +.021 taste residual
survives" but:

> **On the half of this cell where the dense readout was not selected, five rounds of
> dual-track mining closed the articulation gap entirely (Δ = −.003). The +.021 pooled
> figure is carried by the selection-contaminated half and must not be quoted as a
> taste bound.**

Quotable only with all of: the dense-chain-selected-on-TEST flag; the
label-blind-phrasing-pass-not-label-driven-GEPA flag; the P=4/2-family fleet
degradation; and the note that closure-split Δ levels are protocol-specific and not
comparable to Layer 1's Δ_beyond = +.084.

### 9.3 Remaining mass

| quantity | value |
|---|---|
| Track-A Good-Turing M̂, P=4, by round | .600 → .633 → .533 → .450 (monotone fall) |
| Track-A M̂ at P=6 / 3 families (round-2 supplement) | **.500** vs .633 at P=4 |
| Track-A cross-proposer recapture, by round | .400 → .367 → .467 → .550 |
| **Track-B M̂, by round** | **.850 → .800 → .825 → .725** (never converges) |
| Track-B cross-proposer recapture | .150 → .200 → .175 → .275 |
| marginal new Track-A species at the 6th proposer | 6.3 from 15 proposals (**42% novelty**) |
| odds-form remaining-AUC bound after round 4 | ≈ **+.011** (M̂=.45, last gain +.0040, λ̂ fitted) |
| **actual round-5 gain** | **+.0085** — inside that bound, and non-zero |

The remaining-mass estimator earned its keep here: after round 4 it predicted ≈+.011
recoverable and round 5 delivered +.0085. On the peer pilot it was only ever a
conservative upper bound on gains that never materialised; on this cell it bounded a
gain that did.

**The A/B asymmetry is the campaign's most robust structural finding.** Naming what
makes a comment good is a convergent skill — four independent proposers reach 55%
mutual recapture by round 4, and adding a third family drops missing mass to .50.
Naming what makes a comment *look* good is not — recapture never exceeds 28%, missing
mass never falls below .725, and across four rounds **zero** Track-B species were named
by two different model families in three of the four rounds. Track-B conclusions on
this cell carry much wider "unfound channels" uncertainty than Track-A ones — and
Addendum 4 proved that concretely, by finding a .722-AUC channel family that 160
sealed Track-B proposals had entirely missed.

### 9.4 Spurious map — highlights

**1. Position in container is the largest nuisance in the cell and no proposer found
it** (§7A). Docket size alone .710; joint position model .722 on the full population,
versus .707 for the whole 38-channel mined map and .436–.564 for its best single
member. Stacked increment over the entire named map: **+.079**. It does not threaten Δ
(both instruments are blind to it; stratifying on it does not shrink the residual) but
it substantially reframes what T = .817 means: a large part of what no text instrument
can reach here is *which docket the comment landed in*, not anything about the comment.

**2. The campaign/form-letter family is the one stable textual channel, and it is
anti-predictive.** Three independently proposed channels in three different rounds —
coordinated campaign residue (.450), NGO action-alert templated phrasing (.464),
campaign template reuse (.469) — plus mass-campaign petition register (.492) in round
5. Mass-campaign comments are consistently *less* likely to draw a government response,
on every population tested.

**3. Letter furniture is anti-predictive and is pure surface.** "Formal salutation and
signature block present" is the single most predictive mined channel on the full
population (.436), and the Addendum-3 decomposition established that its MIXED flag was
wrong: both its components routed to B, and the arbiter declared the decomposition
failed for lack of any recoverable merit.

**4. Individually the mined channels are near-null** (§7B correction): max
\|alone-AUC − .5\| = .064 on the full population, median .012. The map's power is
entirely in the joint model (.723).

### 9.4a Block decomposition — 67 mined criteria ≈ 198 authored rubrics

Driver `block_decomposition.py`; same split, same estimator, honest population.

| block | features | VA_nl (honest) | Δ vs T=.8167 |
|---|---:|---:|---:|
| V (surface features only) | 27 | .7474 | +.0693 |
| A (198-rubric bank only) | 171 | .7252 | +.0915 |
| VA (round-0 state) | 198 | .7808 | +.0359 |
| **M₁₋₅ (the 67 mined criteria alone)** | 67 | .6719 | +.1448 |
| **V + M₁₋₅ (no authored bank at all)** | **94** | **.7772** | **+.0395** |
| VA + M₁₋₅ (round-5 state) | 265 | .7957 | **+.0210** |

**Sixty-seven mined criteria plus the surface features reach .7772 — within .004 of the
entire 198-rubric authored bank plus the same surface features (.7808) — using less
than half as many columns.** Five rounds of a sealed fleet reading disagreement slices
reconstructed essentially all of what a 198-rubric, professionally authored regulatory
bank contributes on this cell. And the two are not redundant: stacked together they
reach .7957, better than either.

This also puts the campaign's small Δ in context. On this cell the articulated side was
never weak — V alone is .747 and V+A is .781 against a dense readout of .817. The
closure question here was always about the last four AUC points, which is why ε = .005
made the rounds so tight.

### 9.5 Instrument health across the campaign

| | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| misrouting rate | 0.0% | 4.0% | 4.0% | **8.0%** | 4.0% |
| disputes → arbiter | 0 | 1 | 1 | 2 | 1 |
| arbiter upheld the auditor | — | yes | yes | yes ×2 | yes |
| planted probes separated | 2/2 | 2/2 | 2/2 | **1/2** | 2/2 |
| collapses | 1 (probe) | 0 | 3 (all probes) | 0 | 1 (probe) |
| phrasing rewrites accepted | 25/25 | 25/25 | 25/25 | 25/25 | 25/25 |
| phrasing anchors (both judges) | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| coherent-vs-scrambled anchor AUC | .8167 | .8145 | **.9823** | .8889 | .8283 |
| pos-vs-neg anchor AUC (K=50) | .537 | .521 | .562 | .547 | .482 |

**The audit is load-bearing, five times over.** It caught a nuisance-proposed criterion
that was really merit (r2), a merit-proposed criterion that was really stance (r3), a
nuisance-proposed criterion that was really merit and **the campaign's own planted
probe** that was really document shape (r4), and a decomposition component that was
really identity (r5). The arbiter upheld the blind auditor every single time — five
disputes, five upholds, twice against the campaign runner's own authorship.

**The K = 50 anchor battery answered the pilot's open question.** The peer pilot's
round-4 positive-vs-negative inversion (.361 at K=12) could not be resolved at that
sample size. Here, at K=50 across five rounds, positive-vs-negative separation is
**.482 to .562** — i.e. **flat at chance, every round**. Fresh criteria, quality-relevant
or not, do not separate responded from unresponded comments as a set. That is not an
instrument failure (scrambled separation is .81–.98 every round, so the judge is
certainly reading text); it is a direct measurement of how thin the signal in this cell
is, and it explains why the whole campaign operates inside a ±.04 residual.

### 9.6 What travels to the rest of the program

1. **A saturation declaration is a statement about a miner, not about a residual.**
   Four consecutive sub-ε rounds were overturned by one round of a different shape.
   Any plateau this program quotes should be phrased "not discoverable by this miner"
   and should say what kind of miner was used.
2. **The MIXED decomposition pass (Addendum 3) is worth its cost.** It produced the
   campaign's single strongest quality criterion out of a channel four rounds of mining
   had written off as spurious, and it resolved one MIXED flag to "not mixed" outright.
3. **Position-in-container generalises** (Addendum 4). Third corpus, same signature:
   large, predictive, never proposed. It should be audited programmatically in every
   cell that has an ordered container — cheaply, before any mining round.
4. **Read per-channel statistics on the largest population available.** Honest-set
   per-channel AUCs on this cell are off by up to .083 from the full-population values,
   which was enough to make three channels look strong that are not.
5. **Enlargement over small-n, applied to pair-level readouts.** Within-docket AUC was
   undefined here (6 qualifying dockets), but within-docket *pair concordance* used
   21,928 pairs and showed the bank keeps 81% of its edge inside dockets.

### 9.7 Judge-call ledger and compute

| item | count |
|---|---|
| Gemma-4-31B scoring prompts | 5 rounds × 241,775 = **1,208,875** |
| sealed fleet proposals (4 rounds × P=4 × (15+10), +GLM r2) | **440** |
| blind concept-identity judge passes | 22 (census 2, species 16, phrasing gates ×2 per round ×5 = 10 … 28 total judge calls) |
| routing audits / arbiter rulings | 5 / 5 |
| GPU jobs | 6 (5 scoring rounds + 1 duplicate killed) |

GPU discipline: every launch used the race-free retry launcher with utilisation sized
from actually-free memory; the shared ledger was claimed before every launch and
released after. One incident, recorded in the ledger: a timed-out `ssh` left a first
round-2 launcher running while a second started, producing two of my own scoring
processes; I killed **only my own PIDs**, wrapper shell first, then python, then
`VLLM::EngineCore`, and released the duplicate GPU claim. A second incident: the
launcher's ledger-exclusion logic over-excluded (it counted historical CLAIMs whose
RELEASE had already been written), starving round 4 while four GPUs sat idle; fixed to
track the last event per GPU and recompute every attempt. No co-tenant process was ever
touched.

### 9.8 Deviations from the frozen protocol, all recorded

1. **GEPA**: the freeze requires GEPA-iterated phrasing; the repo's GEPA is label-driven
   and would break label-blindness, so a **label-blind house-style phrasing pass with a
   blind concept-fidelity gate** was substituted (§1.2). 125/125 rewrites accepted
   across five rounds, 0 rejected.
2. **Fleet degradation**: P = 4 / 2 families throughout, against a target of P = 6 / 3.
   GLM's 5-hour window blocked rounds 1 and 3-5 and its weekly quota is exhausted until
   2026-08-13. One P = 6 / 3-family supplement was obtained for round 2 (§4.5) and shows
   the effect of the shortfall: Track-A M̂ .633 → .500.
3. **MONITOR split** (§1.1, decided before round 1): saturation statistic on
   MONITOR_FULL (n = 1,892), Δ level on MONITOR (n = 377) and the honest population
   (n = 1,904).
4. **Selection rule** for the round's scored 25 from a larger fleet pool (§1.1,
   Decision 2), consensus + diversity strata, recorded per criterion.
5. **Stopping-rule ambiguity**: `gain < ε` read as signed per the prereg text, with the
   magnitude variant reported alongside (§4.2). Both saturate by round 4; both reset at
   round 5.
6. **Round 5 composition**: the cap round was spent on Addenda 3 and 4 rather than a
   sixth ordinary mining round, with an author-proposed split of 10 A / 15 B instead of
   15/10 (scored total unchanged at 25).

### 9.9 Artifact locations

All under `methods/taste_decomposition/closure/nc_responded/`.

| what | files |
|---|---|
| protocol + splits | `nc_closure_lib.py`, `build_splits_nc.py` → `nc_responded_splits.json`, `nc_responded_population.csv`, `nc_responded_dense_preds_aligned.csv` |
| round-0 bank census | `concept_census.py`, `census_adjudicate_{build,analyze}.py` → `concept_census.json`, `concept_census_final.json` |
| per round r ∈ {1..5} | `round{r}_disagreement_slice.json`, `fleet_r{r}.json`, `round{r}_species_{a,b}.json`, `round{r}_selection.json`, `round{r}_phrasing_rewrites.json`, `round{r}_criteria_final.json`, `round{r}_proposals_blinded.json`, `round{r}_routing_{audit,final}.json`, `round{r}_scores.npz`, `round{r}_score_report.json`, `round{r}_results.json`, `round{r}_swap.json`, `round{r}_track_b_discount.json`, `round{r}_missing_mass.json` |
| **canonical result** | **`round5_results.json`** (full 6-state curve, both stopping-rule readings) |
| drivers | `stage1_slice.py`, `nc_harness.py`, `fleet_species.py`, `select_round_set.py`, `phrasing_pass.py`, `routing_audit.py`, `score_round_gemma.py`, `run_round_when_free.sh`, `stage4_curve.py`, `swap_readout.py`, `track_b_discount.py`, `missing_mass.py`, `run_round_readouts.sh` |
| Addendum 3 | `mixed_decomposition.py` → `mixed_parents_ranked.json`, `round5_decomposition.json`, `retired_channels.json`, `round5_arbiter_ruling.json` |
| Addendum 4 | `position_audit.py` → `position_in_container_audit.json`, `position_audit_full_population.json`, `position_discount.json` |
| robustness | `delta_robustness.py` → `round5_delta_robustness.json`, `within_docket_full_population.json`, `block_decomposition.py` → `round0_block_decomposition.json`, `track_b_per_channel_three_populations.json` |
| P=6 supplement | `fleet_r2_P6_supplement.json`, `round2_species_{a,b}_P6.json`, `round2_missing_mass_P6.json` |
