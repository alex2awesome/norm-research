# B-SIDE leave-out recovery control — the spurious-track mirror of the M3 battery

Date: 2026-08-08. Status: **exploratory, CPU + proposer calls only. No Gemma scoring,
no GPU, `latex/` untouched.** Mirrors PART 2 (M3, "leave-out recovery, the positive
control") of `notes/2026-08-06__missing-mass-robustification.md` on the TRACK-B
(spurious-channel) side of the same peer-review verdict cell, reusing the fleet
harness in `methods/taste_decomposition/closure/robust_mm/` end to end.

Terminology, spelled out on first mention per the standing rule: **Track A** = the
quality-relevant articulated-criterion bank (the subject of the A-side M3 battery);
**Track B** = the DECLARED-NUISANCE bank — channels a sealed proposer fleet flags as
plausibly PREDICTIVE of a paper's outcome but explicitly NOT a mark of quality (surface
style, formatting, topic, or the textual fingerprint of an unseen upstream factor);
**alone-AUC** = a single channel's own ROC-AUC against the verdict label, unaided;
**FIT+MINE / MONITOR** = the 80/20 title-grouped closure splits (design decisions here
use FIT+MINE only, per the standing rule that MONITOR is never read for a design
choice); **P** = number of proposers in a fleet round; **k** = proposals per proposer;
**M3** = the leave-out-recovery arm of the design-note §8 battery; **FREEZE ADDENDUM 2
("upstream mode")** = the Track-B brief's required MODE 2, in which every proposer must
name the unseen factor it thinks is upstream of a fingerprint and tag whether that
factor is `mixed` (plausibly also causal of real quality) or purely a nuisance;
**FREEZE ADDENDUM 4 ("position prior")** = the Track-B brief's required MODE 3, in
which every proposer must propose at least one channel scoring an item's POSITION or
ORDER within its container (submission-cycle era, crowding of the topic wave, etc.).

**Purpose.** The A-side M3 battery measured how often a sealed multi-proposer fleet
rediscovers a QUALITY criterion that was deliberately removed from the bank, and found
33% overall / 56% on high-value holdouts / lift over a retained-channel control
indistinguishable from zero (`notes/2026-08-06__missing-mass-robustification.md` §2.4).
Every B-side (spurious-track) missing-mass or "we have mapped the nuisance space"
claim needs the identical calibration on the other track, or it is an unmeasured
assertion. This note runs it.

Artifacts, all under `methods/taste_decomposition/closure/robust_mm/bside/`:

| file | what |
|---|---|
| `bside_census.py` → `bside_census.json` | 40-channel Track-B census, alone-AUCs, stratified K=6×3 holdout design |
| `harness_b.py` | sealed fleet builder/collector, production TRACK-B instruction (Addendum-2 + Addendum-4 verbatim) |
| `run_codex_b.py`, `run_glm_b.py` | gpt-5.6-luna (`codex exec`) and GLM-5.2 proposer legs |
| `proposals_bside_rep{1,2,3}.json` | every proposal, full provenance (proposer/family/model/pid) |
| `recall_build_b.py` → `bside_recall_manifest.json` | full-recall blind-judge prompt builder |
| `adjudicate_build_b.py` | borderline (judge-disagreement) provenance-stripped third pass |
| `recall_analyze_b.py` → `bside_recall.json` | **primary** rediscovery readout: sensitivity, retained control, lift, breakdowns |
| `fleet_raw/` | every prompt + raw model output + judge/adjudicator output, per replicate |

---

## 1. Design (mirror of A-side M3, §2.1)

### 1.1 The census

The 40-channel Track-B bank is not a separate artifact from the closure pilot — it is
the union of every channel the pilot's own rounds 1–4 sealed fleets proposed and the
pilot's own misrouting audit routed to Track B (`final_route == "B"`) and survived the
collapse gate, cross-checked column-for-column against the published
`round4_results.json:track_B.per_feature_alone_AUC_MONITOR` (**0 mismatches on all 40
channels** — `bside_census.py` output). 3 of the 40 channels were originally *proposed*
on Track A and reclassified "incidental" by the audit (`A05→B`, `A29→B`, `A51→B`); the
census pulls their instruction text from `round{r}_track_a.json` accordingly.

Design decisions, mirroring `m3_concepts.py` exactly except K and the per-stratum
split (A-side: K=8, 3+3+2; B-side: K=6, 2+2+2 — the task-specified scale):

- **channel = a round-tagged blind id** (e.g. `r3:P15`), collapse-gated, final-route-B.
- **alone-AUC computed on FIT+MINE only** for the stratification decision; MONITOR
  alone-AUC is reported descriptively and matches the published numbers exactly.
- **strata = terciles of |alone-AUC − .5|** over the 40 channels (13/13/14 → high/mid/low).
- 3 replicates × (2 high + 2 mid + 2 low) = 18 distinct channels, no channel in two
  replicates, stable sha256 sort over a fixed salt (`bside-leaveout-holdout-v1`).
- Retained-control pool per replicate = the 34 channels NOT in that replicate's own
  held set (same convention as `m3_recall_build.py`; this pool can include channels
  held out in a *different* replicate — rep1's control includes 2 channels that rep2/3
  hold out, rep2's includes 3 — because each replicate is evaluated only against its
  own fleet, exactly as on the A side).
- The held-out channels' declared identities were never exposed to any proposer: the
  fleet is sealed by construction (M1 contract) and never sees the Track-B bank
  regardless of which channels a given detection pass is scoring — so there is no
  "footprint" to remove from what a proposer could infer, unlike the A-side's
  column-removal-from-the-fit-matrix step. B-channels never enter VA_nl in the first
  place (they are declared nuisances, not scorecard members), so **there is no
  depletion/refit/AUC-recovery arm on this side** — this control measures detection
  sensitivity only, which is what "calibrate the spurious-miner's sensitivity" asks for.

The two channels the task's own hint named as "strong" both landed exactly where
expected: **"A public repository URL appears in the abstract"** (`r3:P15`, alone-AUC
.593, rank 2/40) and **"Abstract length and verbosity"** (`r1:P02`, alone-AUC .553,
rank 5/40) — both in the high stratum. The single most informative channel of all 40
is actually **"Non-idiomatic English and grammatical slips"** (`r2:P14`, alone-AUC
**.406**, the largest |AUC−.5| in the census).

### 1.2 The fleet

Per-replicate fleet, sealed on the **same standard disagreement slice**
(`../slice_round5_fullbank.json`, the full-bank round-4 slice — unlike the A-side, no
per-replicate depletion means no per-replicate slice regeneration; only the
proposer-salt ordering differs per replicate, giving 3 independent draws on identical
input):

| slot | family | model | k | status all 3 reps |
|---|---|---|---|---|
| `claude_sonnet` | claude | Claude Sonnet (sealed subagent) | 10 | 3/3, 10/10 distinct each |
| `claude_opus` | claude | Claude Opus (sealed subagent) | 10 | 3/3, 10/10 distinct each |
| `codex_luna_a` | openai | gpt-5.6-luna (`codex exec`, effort high, sandbox read-only) | 10 | 3/3, 10/10 distinct |
| `codex_luna_b` | openai | gpt-5.6-luna, independent call | 10 | 3/3, 10/10 distinct |
| `glm_a` | glm | glm-5.2 (thinking, `budget_tokens=2048/max_tokens=32000`, key `~/.z-ai-api-key.txt`) | 10 | 3/3, 10/10 distinct |

**P = 5 across 3 families in all 3 replicates** — above both the P≥4/≥2-family floor
and the A-side's actual M3 fleet (which landed P=4/2 families; GLM was rate-limited out
of the M3 replicates there). GLM was **not** rate-limited today: all 3 calls returned
clean 10/10 parses on the first attempt (88–189 s each, 26–32K think-chars, well inside
the Lite weekly budget). 150 proposals total (50/replicate), all distinct names, all
carrying the required Addendum-2 `upstream_parent`/`mixed` tags and at least one
Addendum-4 position-in-container channel per proposer (verified in the collect-step
JSON, `n_mixed` / `n_surface_only` counts recorded per proposer in each
`proposals_bside_rep*.json`).

TRACK_B instruction text in `harness_b.py` is copied **verbatim** from the current
production brief (`methods/taste_decomposition/closure/maps_hw_si/harness_maps.py`),
i.e. it already carries FREEZE ADDENDUM 2 (MODE 2, upstream-factor reasoning) and
FREEZE ADDENDUM 4 (MODE 3, position-in-container), adapted only for the peer-abstract
item/construct framing the A-side M1/M3 harness already used.

---

## 2. Detection: full-recall blind pairwise, never embedding-τ

Per the A-side finding that the mechanical embedding detector is out of instrument
range on this corpus (max cross-register cosine ~.72, below the τ≥.78 defensible band —
`notes/2026-08-06__missing-mass-robustification.md` §2.3), the mechanical detector was
**not built at all** on this side. The sole instrument is full-recall blind pairwise
judging: for each replicate, a sealed judge sees **all 50 of that replicate's
proposals** (neutral `K01..K50` ids, hash-ordered, no proposer attribution) and answers
a recall question for **12 targets** (6 held-out + 6 stratum-matched retained,
shuffled together by stable hash, unlabeled): does any candidate name the same
underlying channel?

**Two independent sealed Opus judges per replicate.** Raw agreement: rep1 **91.7%**
(11/12), rep2 **83.3%** (10/12), rep3 **100%** (12/12) — 32/36 = **88.9%** overall.

**Borderline resolution (the provenance-stripped third pass the task spec calls for).**
The 3 targets where the two judges disagreed (rep1/T07 = "Enumeration of specific named
models in a model zoo"; rep2/T02 = "Explicit count-signalling of experimental scale";
rep2/T03 = "Magnitude of the engineering or data resource the work required") were
re-judged by an independent third sealed adjudicator, seeing the identical
provenance-stripped candidate pool and only the disputed targets. The adjudicator's
verdict is the FINAL word for the primary statistic below: T07 → no match, T02 → match,
T03 → match.

---

## 3. Results

### 3.1 Sensitivity table

| stratum | n | sensitivity | 95% CI | retained control | lift | 95% CI | P(lift>0) | P(sens≥.70) |
|---|---|---|---|---|---|---|---|---|
| **overall** | 18 | **.500** | [.278, .722] | .444 | **+.056** | [−.278, +.389] | .57 | .05 |
| high | 6 | .333 | [.000, .667] | **.000** | +.333 | [.000, +.667] | .91 | .02 |
| mid | 6 | .833 | [.500, 1.000] | .833 | .000 | [−.333, +.333] | .34 | .73 |
| low | 6 | .333 | [.000, .667] | .500 | −.167 | [−.667, +.333] | .18 | .02 |

By replicate (count-weighted, primary rule): rep1 **.833** (5/6), rep2 **.333** (2/6),
rep3 **.333** (2/6).

**The overall, decision-relevant number is the pooled lift, and it is null**: +.056,
95% CI spanning zero, P(lift>0) = .57 — a coin flip. This is the same qualitative
finding as the A-side's retained-control test (there: lift −.042, P(>0)=.32): removing
a channel from the declared Track-B bank does not measurably increase the chance the
fleet re-proposes it. The fleet's Track-B proposals are governed by what upstream
factors and surface patterns it independently notices in the slice, not by a hole in a
bank it never saw.

**One stratum-level nuance worth flagging but not leaning on.** Unlike the A-side
(where lift was flat-to-negative in every stratum), the high stratum here shows a
nominally positive lift (+.333) driven by an exact 0/6 retained-control rate — no
proposer, across 3 replicates × 5 proposers × 10 channels, ever independently
re-proposed a high-value RETAINED channel. That is a clean number, but n=6 per stratum
is small and the CI still touches zero (P(lift>0)=.91, short of a decisive threshold),
and it is not corroborated by the mid stratum (lift exactly 0) or the low stratum
(lift **−.167**, i.e. retained low-value channels were rediscovered *more* often than
held-out ones). Read as: worth a second wave with a larger K before it is load-bearing,
not yet a contradiction of the A-side's "no depletion-direction" verdict.

### 3.2 Retained-channel control (the zero-lift test)

Overall retained-control rediscovery rate = **.444** (8/18) — almost identical to the
overall sensitivity (.500). This is the decisive comparison the task asked for: the
fleet names channels it was never pushed toward almost as often as the ones that were
actually removed, exactly the A-side's diagnostic (there: control .375 vs sensitivity
.333). The B-side census's individual channels are collectively weak (max alone-AUC
.593, most near .50 — see §1.1), so the disagreement slice is as weak a pointer for the
spurious-track fleet as it was for the quality-track one, and for the same mechanistic
reason: the fleet is reading the same 60 abstracts and proposing from its own priors
about surface style and upstream factors, not from a hole in a bank it cannot see.

### 3.3 Per-family and per-proposer discoveries

Of the 9 held-out channels rediscovered (of 18):

| family | catches (channel counted once if any proposer in that family matched) | UNIQUE catches (channel no other family also caught) |
|---|---|---|
| openai (codex_luna_a/b) | 6 | **3** |
| claude (sonnet+opus) | 6 | **2** |
| glm (glm_a) | 3 | **0** |

| proposer | catches |
|---|---|
| claude_opus | 5 |
| codex_luna_a | 4 |
| codex_luna_b | 4 |
| glm_a | 3 |
| claude_sonnet | 1 |

4 of the 9 rediscovered channels were caught by ≥2 families independently (cross-family
agreement), the rest by one family only. GLM's zero unique catches is plausibly a power
artefact rather than a quality one — it ran a single proposer per replicate (P=1)
against the other families' P=2, so it had half the draws to land a match; it still
matched 3/9 rediscoveries where present.

### 3.4 What was rediscovered vs missed (held-out set, all 3 replicates)

| replicate | stratum | alone-AUC | rediscovered? | channel |
|---|---|---|---|---|
| rep1 | high | .406 | **yes** | Non-idiomatic English and grammatical slips |
| rep1 | high | .521 | **yes** | Membership in the learning-theory / optimization subfield |
| rep1 | mid | .483 | **yes** | Anonymized-repository or release-on-acceptance code boilerplate |
| rep1 | mid | .516 | **yes** | Mathematical notation density in the abstract |
| rep1 | low | .492 | **yes** | Self-description as novel or new |
| rep1 | low | .507 | no (adjudicated) | Enumeration of specific named models in a model zoo |
| rep2 | high | .593 | **no** | A public repository URL appears in the abstract |
| rep2 | high | .526 | no | Belief-change — a changed understanding vs. a usable artifact |
| rep2 | mid | .521 | **yes** | Branded method name or acronym |
| rep2 | mid | .518 | **yes** (adjudicated) | Magnitude of the engineering or data resource the work required |
| rep2 | low | .493 | no | "Framework" as the contribution noun |
| rep2 | low | .501 | no | Scare-quoted or coined terms in quotation marks |
| rep3 | high | .528 | no | Parenthetical example density |
| rep3 | high | .527 | no | Poses a specific question and states its answer plainly |
| rep3 | mid | .516 | **yes** | Explicit priority or first-ness claims |
| rep3 | mid | .510 | no | Density of percentage figures |
| rep3 | low | .501 | no | Explicit count-signalling of experimental scale |
| rep3 | low | .499 | **yes** | Famous benchmark and dataset name-dropping |

The single strongest channel in the whole census — **repo-URL presence** (rank 2/40,
alone-AUC .593) — was **missed** by all 5 proposers across both judges. That is a
concrete, load-bearing miss: a 5-proposer / 3-family fleet reading the exact abstracts
that carry anonymized-repository links did not independently propose "does a repo URL
appear" as a channel in that replicate, even though 3 other proposals in that same
replicate's pool scored adjacent artifact-availability language (promise-vs-delivered
phrasing, placeholder residue) without naming the URL-presence signal itself.

---

## 4. The QUOTATION CONTRACT

Mirroring the A-side's design-note §8 contract exactly, adapted to the spurious track:
**a claim of the form "the Track-B census covers the corpus's spurious-channel space"
(equivalently: "there are no more nuisance channels left to find," "the map is
complete") requires measured rediscovery sensitivity ≥ 70% on high-value holdouts, with
lift over the retained-channel control distinguishable from zero.**

Measured here: high-stratum sensitivity **.333**, 95% CI [.000, .667], **P(≥.70) =
.018** — an 18-in-1,000 chance of clearing the floor, lower even than the A-side's
already-failing 16%. Overall lift **+.056**, 95% CI [−.278, +.389], **P(lift>0) = .57**
— not distinguishable from zero.

**So: no.** The floor is not met on either of its two required legs, and it is not met
more decisively than on the A side. The correct, quotable wording:

> A sealed 5-proposer / 3-family spurious-mindset fleet reading a dense-disagreement
> slice rediscovers **50% of held-out declared-nuisance channels overall / 33% on the
> highest-value third**, at a lift over channels never removed that is **statistically
> indistinguishable from zero** (+.056, P(lift>0)=.57). The 40-channel Track-B census —
> and any Track-B census produced by this pipeline — is a **lower bound** on the
> spurious-channel space with this measured detection rate attached. It may never be
> quoted as covering, exhausting, or mapping that space.

This is the B-side twin of the A-side's `Δ_plateau` contract line
(`notes/2026-08-06__missing-mass-robustification.md` §4.2): both tracks now carry a
measured miner-sensitivity number, and both numbers say the same thing — a sealed
proposal fleet on this corpus finds roughly a third to a half of what it is shown to be
missing, with no measurable evidence that removal is what drives the finding, on
EITHER track.

---

## Caveats that travel with every number here

1. **Exploratory, pre-GEPA.** No prereg covers this battery. A GEPA-iterated
   Track-B proposer (`FREEZE ADDENDUM 2`/`4` are prompt content, not GEPA output) could
   shift the rediscovery rate in either direction.
2. **n=18 held-out, 18 control.** Stratum cells hold 6 each; every stratum-level CI is
   wide, and the §3.1 high-stratum lift note is exploratory color, not a result.
3. **No depletion/AUC-recovery arm.** Unlike the A-side M3, Track-B channels never
   enter VA_nl, so there is nothing analogous to §2.6's "AUC recovered from
   rediscovered concepts" to compute here — this control measures detection
   sensitivity only, which is what "calibrate the spurious-miner's sensitivity" means.
4. **Judges are the same instrument as the A-side's, same strictness profile expected**
   (88.9% raw two-judge agreement here vs 93.9–100% on the A-side's two full-recall
   instruments); no separate anchor battery with planted probes was re-run on this
   side — the A-side's anchor result (judges reject weak-label lookalikes, i.e. err
   strict) is assumed to transfer since it is the same judge model and the same
   full-recall protocol, but this is an assumption, not a re-measurement.
5. **GLM was NOT rate-limited today** (all 3 calls clean on first attempt), unlike the
   A-side M3 run where GLM was rate-limited out entirely. This is a schedule/quota
   state fact (per `reference_glm_subscription_api`), not a claim that GLM is reliably
   available — do not assume a future run gets the same result.
6. **Control-pool cross-replicate overlap.** A retained-control channel in one
   replicate can be a held-out channel in a different replicate (2 cases in rep1's
   control set, 3 in rep2's, 0 in rep3's) — by design, since each replicate is scored
   only against its own fleet. Same convention as the A-side's `m3_recall_build.py`.
7. **3 of the 40 census channels were originally proposed on Track A** and reclassified
   "incidental" by the pilot's own misrouting audit (`A05→r1:P16`, `A29→r2:P02`,
   `A51→r4:P06`) — and, checked directly, **all 3 landed in held-out sets this round**:
   `r1:P16` ("Magnitude of the engineering or data resource...", rep2 mid, rediscovered
   via adjudication), `r2:P02` ("Belief-change...", rep2 high, missed), `r4:P06`
   ("Poses a specific question and states its answer plainly", rep3 high, missed). All
   three are already in the §3.4 table under their normal rows; flagged here only so a
   reader auditing provenance knows these three originated on the other track.
