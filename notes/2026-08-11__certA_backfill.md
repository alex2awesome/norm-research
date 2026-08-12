# Track-A missing-mass certificate backfill — COMPLETE (6 cells re-judged)

Assignment (coordinator, 2026-08-11): re-run the strict two-judge blind pairwise merge on
the archived A-side proposal pools of nine campaigns, since their Track-A missing masses
were computed with τ-embedding clustering only.

**Status: complete for the six cells whose premise held.** A pre-flight pass raised four
problems; all four were ruled on by the coordinator, and the run below implements those
rulings. Judges: **gpt-5.6-sol + gpt-5.6-luna**. Headline table in §5.

---

## 1. The motivating premise is inverted for Track A

The brief's reasoning is that τ-only A-masses are *inflated* "by exactly the f1 mechanism
the B-side merge was invented to remove". On the one campaign where both tracks have been
strictly merged, the A side moved the other way:

| aops_curation r1 | S_obs | f1 | Ĝ-T missing mass |
|---|---:|---:|---:|
| Track A, τ-only | 83 | 60 | .5000 |
| Track A, strict two-judge | 87 | 67 | **.5583** |
| Track B, τ-only | 54 | 40 | .5000 |
| Track B, strict two-judge | 45 | 27 | **.3375** |

The brief's quoted figures (.50/.50 → .5583/.3375) are exactly right; the *direction* read
off them is not. Strict merging **raised** the A-side mass and **lowered** the B-side.

The mechanism is structural, not incidental. `species_merge.py` does not merge τ clusters —
it restarts from the raw pool as singletons and adds an edge only where a **cross-proposer**
pair is called SAME by **both** judges (`if B[i]["proposer"] == B[j]["proposer"]: continue`).
So it systematically *undoes* the within-proposer merges τ had made. Whether a cell's mass
rises or falls depends on how much of its τ clustering was within-proposer, which is a
per-cell empirical fact. So the backfill is worth doing — but no cell's direction can be
predicted in advance, and the table must not be framed as a deflation.

## 2. Judge substitution would confound the one comparison the table exists to make

Every existing figure-of-record merge used **claude-sonnet-5 on both legs**:

| artifact | judge A | judge B |
|---|---|---|
| `aops_curation_r1_bmergeA_judgeA/B.json` | claude-sonnet-5 | claude-sonnet-5 |
| `aops_curation_r1_bmergeB_judgeA/B.json` | claude-sonnet-5 | claude-sonnet-5 |
| `mathse_vote_r1_bmerge_judgeA/B.json` | claude-sonnet-5 | claude-sonnet-5 |
| `jokes_community_r1_bmerge_judgeA/B.json` | claude-sonnet (sealed blind) | claude-sonnet |

The AoPS .5583/.3375 reference that the whole backfill is calibrated against is a
Sonnet–Sonnet product. Running the backfill on gpt-5.6-luna + glm-5.2 makes every cell's
"old-τ vs strict" delta a *sum* of a merge-rule change and a judge-family change, so the
table would no longer isolate the merge rule — the only thing it is being built to measure.
Standing rule `feedback_judges_sonnet_or_better` puts judging at Sonnet+ independently.

## 3. The cited precedent is proposer-side, and the proposed GLM leg is the one documented to break here

`notes/2026-08-08__maps_hw_si.md` §1.4 (lines 122-129): GLM-5.2 is alive but
**1302/1308-limited at these payload sizes** — the same subscription cap that "killed the
GLM leg in maps_batch1". The degradation recorded there is **P = 4 over 2 families on the
PROPOSER side** (Claude Opus, Claude Sonnet, gpt-5.6-luna ×2 via the Codex companion).
`gpt-5.6-luna` is a *proposer* alias in that fleet; it has never been a judge of record.
So the "cap_finalist r5 degradation precedent" is a proposer-side precedent, and the
specific GLM judge leg proposed is the leg documented to fail on packets of this size.

## 4. Three of the nine named cells are mis-specified

- **`nc_responded` already has the strict A-side merge.** `nc_responded/fleet_species.py`:
  "IDENTITY decided by a blind pairwise judge pass over the shortlist, with an authored
  anchor battery. Cosine never decides"; `finalize` "folds two judges' verdicts into
  single-linkage species". The artifacts exist: `round1_species_a_judge1.json`,
  `round1_species_a_judge2.json`, `round1_species_a_key.json` (60 pool items, anchors
  included). It is not a τ-only cell.
- **`peer_verdict` has no A-side proposal pool in this layout.** Its `round1_track_a.json`
  is a *selected-criteria* file (`track / instruction_given_to_self / scale / pre_gepa /
  criteria`); there is no `*_proposals_fleet.json` and no species file with a `tracks`
  block. Its pool has to be located before anything can be re-merged.
- **`cw_community` uses a different layout** — `round1_fleet_A.json` / `round1_fleet_B.json`
  and a `round1_species.json` with no `tracks` block. Adaptable, but `species_merge.py`
  cannot be pointed at it as-is.

Two further scope items: `maps_batch1` holds **duplicate copies** of peer_curation r1/r2 and
peer_revealed r1/r2 (byte-identical masses to the campaign dirs) — pick one canonical copy
per round or the table double-counts; and `maps_hw_si` contains a second cell,
`style_inv_toptier` (r1/r2), which the brief did not name but which sits in the same
campaign directory.

Also mechanical: the brief assumes an overwrite guard already protects the originals. It
does not. `cmd_apply` writes `<tag>_species.PREMERGE.json` only *once* and then **rewrites
`<tag>_species.json` in place**. Honouring "write NEW `<tag>_species_strictA.json`"
requires adding an out-path option to `species_merge.py`.

---

## The old-τ column (computed, ready)

Track-A τ-only Good-Turing per round. **Terminal round = certificate of record** (bold);
earlier rounds are the trajectory. The six cells below are the ones with a pool layout
`species_merge.py --track A` can run on today.

| campaign | round | S_obs | f1 | old-τ Track-A M̂ |
|---|---|---:|---:|---:|
| peer_curation_ext | r1 | 54 | 49 | .8167 |
| | r2 | 46 | 38 | .6333 |
| | r3 | 62 | 49 | .5444 |
| | r4 | 65 | 52 | .5778 |
| | **r5** | **78** | **70** | **.7778** |
| peer_revealed | r1 | 50 | 45 | .7500 |
| | r2 | 50 | 42 | .7000 |
| | r3 | 77 | 70 | .7778 |
| | r4 | 65 | 57 | .6333 |
| | **r5** | **59** | **51** | **.5667** |
| maps_hw_si · hashtagwars_verdict | r1 | 56 | 53 | .8833 |
| | r2 | 48 | 41 | .6833 |
| | r3 | 46 | 39 | .6500 |
| | **r4** | **52** | **44** | **.7333** |
| press_verdict | r1 | 48 | 39 | .4333 |
| | **r2** | **66** | **53** | **.5889** |
| mathse_vote | r1 | 57 | 48 | .5333 |
| | **r3** | **53** | **34** | **.2833** |
| mathse_accepted | r1 | 55 | 40 | .3333 |
| | **r2** | **47** | **39** | **.3250** |
| *(out of scope, same dir)* style_inv_toptier | r1 / r2 | 50 / 49 | 42 / 44 | .7000 / .7333 |
| *(reference, already strict)* aops_curation | r1 | 83→87 | 60→67 | .5000 → **.5583** |
| *(already strict A)* nc_responded | r1 | — | — | strict merge already on disk |

Strict column: computed — see §5.

---

## 5. RESULTS — strict two-judge Track-A merge, six cells

Judges **gpt-5.6-sol + gpt-5.6-luna**, `--track A`, 150 shortlisted cross-proposer pairs
+ 2 authored anchors per packet, edge added only where BOTH legs say SAME. Every cell's
anchor battery passed on both legs (AS1→SAME, AD1→DIFFERENT, 6/6 cells).

| campaign | round | τ-only M̂ | **strict M̂** | Δ | S_obs τ→strict | f₁ τ→strict | edges | anchors |
|---|---|---:|---:|---:|---|---|---:|---|
| peer_curation_ext | r5 | .7778 | **.5889** | −.1889 | 78→66 | 70→53 | 24 | PASS |
| peer_revealed | r5 | .5667 | **.6000** | +.0333 | 59→68 | 51→54 | 22 | PASS |
| maps_hw_si · hashtagwars_verdict | r4 | .7333 | **.3667** | −.3667 | 52→36 | 44→22 | 24 | PASS |
| press_verdict | r2 | .5889 | **.6222** | +.0333 | 66→68 | 53→56 | 22 | PASS |
| mathse_vote | r3 | .2833 | **.4750** | +.1917 | 53→76 | 34→57 | 44 | PASS |
| mathse_accepted | r2 | .3250 | **.5583** | +.2333 | 47→86 | 39→67 | 34 | PASS |
| *(anchor)* aops_curation | r1 | .5000 | .4917 | −.0083 | 83→84 | 60→59 | 36 | PASS |

**Which quoted numbers move: all six, and three of them a lot.** Direction splits 3 down /
3 up, magnitudes .033–.367. The largest single move is hashtagwars_verdict r4
(.7333 → .3667, halved); the largest upward moves are the two math.SE cells
(+.19, +.23). No cell is unchanged, so every Track-A mass quoted from these campaigns is
superseded.

### Structural observation

`S_obs_strict = N_proposals − n_merge_edges` in **7 of 7** cells. The union-find almost
never chains: each accepted edge joins two otherwise-distinct groups. So the strict mass is
driven almost entirely by how many cross-proposer pairs clear the both-judges-SAME bar, and
the τ→strict delta is mostly a statement about how many of τ's clusters were *within*-proposer
merges that the strict rule refuses to make. This is the mechanism behind the framing in §3
and it is why the direction is not predictable from the τ number alone.

### Judge-pair behaviour

| campaign | N | P | sol %SAME | luna %SAME | sol/luna agreement |
|---|---:|---:|---:|---:|---:|
| peer_curation r5 | 90 | 6 | 30.0% | 24.0% | 90.0% |
| peer_revealed r5 | 90 | 6 | 20.0% | 30.7% | 88.0% |
| hashtagwars_verdict r4 | 60 | 4 | 22.7% | 22.7% | 93.3% |
| press_verdict r2 | 90 | 6 | 22.7% | 20.0% | 94.7% |
| mathse_vote r3 | 120 | 8 | 48.7% | 45.3% | 86.0% |
| mathse_accepted r2 | 120 | 8 | 38.7% | 30.7% | 90.7% |
| aops_curation r1 (anchor) | 120 | 8 | 32.0% | 45.3% | 86.7% |

## 6. Anchor gate — PASSES AS SPECIFIED, but the specified test is too weak

The gate was: re-run aops_curation r1 Track A on sol+luna over the **identical packet** the
Sonnet legs saw; if it reproduces .5583 inside the LOO band, deltas against Sonnet numbers
are readable.

| aops r1 Track A | S_obs | f₁ | M̂ | LOO band |
|---|---:|---:|---:|---|
| τ-only | 83 | 60 | .5000 | [.4857, .5619] |
| strict, claude-sonnet-5 ×2 | 87 | 67 | .5583 | — |
| strict, sol+luna | 84 | 59 | **.4917** | [.4667, .5905] |

Sonnet's .5583 **is** inside the sol+luna LOO band, so the literal criterion passes. But the
band is a leave-one-proposer-out stability band on 8 proposers (±.06), wide enough to
swallow the entire effect being measured, and the point estimates say something the gate
does not capture:

- judge-family delta = **−.0667** (.5583 → .4917);
- the Sonnet τ→strict correction it must be small against = **+.0583**;
- so under Sonnet the AoPS A-side correction is +.058, and under sol+luna it is −.008 — the
  two families disagree on the **sign and size of the correction itself**.

Per-judge concordance says the same thing: the historic pair agreed 95.3% (sonnetA vs
sonnetB), the new pair agrees 86.7%, and luna is systematically the most liberal SAME-caller
(45.3% vs sol 32.0%, sonnetA 34.0%, sonnetB 32.0%).

One piece of good news: the AND-of-two-judges rule is partially self-stabilising. Across all
six pairings of the four judges on the AoPS packet, both-SAME edge counts stay in 41–49
despite single-judge SAME rates spanning 32–45%. The *structure* the rule recovers is far
less judge-sensitive than any single judge's threshold — it is the resulting f₁ that moves,
because which specific pairs merge changes even when how many does not.

**Recommendation, and how these certificates are written:** report the backfill as the
coordinator's fallback branch — a **NEW-INSTRUMENT RE-SURVEY**. Quote the strict figures as
sol+luna Track-A masses in their own right; do **not** difference them cell-by-cell against
Sonnet-era numbers. aops_curation r1 carries both instruments side by side as the calibration
record. Every `certA_strict.json` states this in its `cross_instrument_calibration` field.

## 7. Scope resolution — three of the nine named cells are out, for three different reasons

| cell | ruling | evidence |
|---|---|---|
| `nc_responded` | **already strict** — dropped | `fleet_species.py`: "IDENTITY decided by a blind pairwise judge pass over the shortlist… Cosine never decides"; `round1_species_a_judge1/judge2/key.json` on disk |
| `cw_community` | **already strict** — dropped, adapter not needed | `notes/2026-08-06__closure_cw_community.md` l.188: "Species from the blind full-recall partition above, **so no embedding threshold enters the estimate**". Its A-masses (.300/.333/.383/.383/.283/.367 over r1–r6) are blind-partition figures, not τ figures |
| `peer_verdict` | **A-POOL UNRECOVERABLE** — certificate stays τ-era with caveat | `round1_proposals_provenance.json` is 25 entries carrying only `blind_id/src_id/track/name` (15 Track-A). No `proposer` or `family` field exists anywhere in the root round files. Without proposer identity the cross-proposer merge constraint, f₁-over-proposers, and the LOO-proposer jackknife are all undefined. Not rebuilt by re-mining, per instruction |

The cw_community adapter was therefore **not written** — the schema work would have produced
a second copy of an estimate that already meets the strict standard.

`maps_batch1` duplicates confirmed **byte-identical** (`cmp`) to the record dirs for
peer_curation r1/r2 and peer_revealed r1/r2; the `peer_curation_ext` / `peer_revealed` dirs
are the record and maps_batch1's copies are ignored, so no double-counting.

Premise re-verified for the six cells that ran: each terminal round's `tracks.A.good_turing`
carries a `tau` field and no `blind_merge.A` block — i.e. genuinely τ-derived, never
previously A-merged.

## 8. Code changes (synced)

- `species_merge.py` `cmd_apply` now writes a **NEW `<tag>_species_strict<T>.json`** by
  default and never touches the cited τ-era species file; `--inplace` restores the legacy
  behaviour. The old PREMERGE sidecar did **not** protect these files — it is written on
  every apply, so a second pass would have overwritten the τ-era copy with no survivor.
  `merged_gt` now also carries the **LOO-proposer jackknife**, computed exactly as
  `species.py` computes it for the τ table (drop one proposer, recount species over the
  survivors, Good-Turing f₁/N), so strict and τ figures come off the same estimator.
- Synced to every campaign copy. The generalised both-tracks version (from `aops_curation`)
  was placed in `peer_curation_ext`, `peer_revealed`, `maps_hw_si`, `press_verdict` (no
  prior copy) and in `mathse_vote`, `mathse_accepted` (whose Track-B-only variants are kept
  as `species_merge.LEGACY_TRACKB.py` — nothing deleted). `cap_crowd`, `cap_finalist`,
  `jokes_community` run a third lineage with a `--tracks A,B` CLI; those were **patched in
  place for the new-file default only**, preserving their own CLI and file naming. Divergence
  recorded here per the code-sync rule.
- New: `closure/run_bmerge_judges.py` (scripted judge pass — the Sonnet-era merges had no
  runner and no recorded bmerge prompt; the preamble is reproduced verbatim from the
  `cap_finalist_r5` / `cap_crowd_r4` bmerge prompt files on disk), `closure/write_certA.py`,
  `closure/run_certA_backfill.sh`.

## 9. Artifacts

`certA_strict.json` in `peer_curation_ext/`, `peer_revealed/`, `maps_hw_si/`,
`press_verdict/`, `mathse_vote/`, `mathse_accepted/`. Per campaign, alongside:
`<tag>_species_strictA.json` (new), `<tag>_bmergeA_packet.json`, `<tag>_bmergeA_key.json`,
`<tag>_bmergeA_judge_sol.json`, `<tag>_bmergeA_judge_luna.json`. Anchor:
`aops_curation/aops_curation_r1_species_strictA.json` + its two sol/luna judge files.
Original `<tag>_species.json` files untouched throughout.

---

## 10. certB queue reconciliation (pre-work for the queued Track-B re-survey)

Run before any judging, per the cw_community lesson ("never redo one that exists"). Result:
**the queue is 4 cells, not 9.**

### Root cause of the resolver's misclassification

The five campaigns the F2 resolver reads as "τ-era" store their Track-B strict merge under
the top-level key **`b_merge`**, not `blind_merge`, and store the superseded τ figures under
`tracks.B.good_turing_PREMERGE_tau_only`. A resolver looking for `blind_merge` therefore sees
the *current* (already strict) `good_turing` and mislabels it τ-era. This is a one-line
resolver fix, not a re-judging job.

### Already strict — DO NOT redo (7)

| cell | round | τ-only M̂ | strict M̂ already on disk | existing judge A |
|---|---|---:|---:|---|
| mathse_vote | r3 | .3625 | **.3500** | claude-sonnet-5 |
| mathse_accepted | r2 | .3500 | **.2625** | *(unlabelled)* |
| jokes_community | r5 | .4000 | **.3000** | claude-sonnet |
| cap_finalist | r5 | .5400 | **.3800** | GPT-5 |
| cap_crowd | r4 | .4200 | **.3000** | GPT-5 |
| nc_responded | r1–r4 | — | strict | `round{1..4}_species_b_judge1/2.json` |
| cw_community | r1–r6 | — | strict | blind full-recall partition (closure note l.188) |

Note for the new-instrument framing: the existing strict-B corpus is **already
judge-heterogeneous** — Sonnet on jokes and mathse_vote, **GPT-5 on cap_finalist and
cap_crowd**, unlabelled on mathse_accepted. So a sol+luna B re-survey is not introducing
family heterogeneity to this column; it is already there and undocumented.

### Genuinely τ-only Track-B, proposer fields present — the real queue (4)

| cell | round | τ-only B M̂ | S_obs | f₁ | pool |
|---|---|---:|---:|---:|---|
| peer_curation_ext | r5 | .6667 | 48 | 40 | same fleet file as the Track-A run |
| peer_revealed | r5 | .4333 | 36 | 26 | same |
| maps_hw_si · hashtagwars_verdict | r4 | .6500 | 33 | 26 | same |
| press_verdict | r2 | .4500 | 40 | 27 | same |

All four are the cells whose Track-A ran today, so their pools, packets machinery and judge
runner are already in place — the B pass is the same commands with `--track B`.

### peer_verdict

Not a re-survey case. It has no Track-B pool at all, which is what the queued retroactive
Track-B fleet round creates; its `certB_strict.json` comes out of that round natively.
