# mathse_vote Layer-3 closure campaign — runbook

Cell: math.StackExchange ANSWERS, y = raw vote score strictly above the median answer
score on its own question (ties dropped). 11,629 rows / 4,960 questions.
Note: `notes/2026-08-10__closure_mathse_vote.md`.
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + freeze addenda 1–4.
Pipeline shape inherited from `../press_verdict/` and `../code_v3/RUNBOOK.md`.

## Conventions

* sk3 home must be pinned: `export HOME=/lfs/skampere3/0/alexspan` in every remote shell.
* GPU ledger `/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt` — CLAIM before
  launching, RELEASE after. Never touch a co-tenant's process; kill only your own PIDs.
* All GPU scoring is **offline batch vLLM**, never an HTTP server.
* `latex/` is untouched by this campaign.
* The A/V matrix is SHARED with the math.SE accepted-verdict cell. The two y's are never
  merged and never differenced.

## The alignment gate (mandatory, runs inside every `cells.load()`)

`AUC(y, mathse_vote_score_va_nl_oof_seed0.npy in assembled row order)` must equal the
ledger's `nonlinear.VA["0"].auc` = **.624849045069194** to < 1e-9. Row order is the bank
`item_ids` filtered by `isfinite(ys["vote_score"])` ("kept-subset order"). Set
`MATHSE_SKIP_GATE=1` only for a deliberate diagnostic, never for a readout.

## Round 0

```bash
cd methods/taste_decomposition/closure/mathse_vote

python3 fetch_dense.py                # pull per-seed dense probs off sk3 (positional
                                      # join asserted on judgement AND group)
python3 oof_alignment_gate.py         # -> oof_alignment_gate.json  (must PASS)
python3 build_splits.py               # FIT+MINE / MONITOR by salted question hash
OMP_NUM_THREADS=6 python3 round0.py   # gate table + closure r0 + swap + jackknife
python3 position_line.py              # FREEZE ADDENDUM 4 answer-position audit
python3 census.py stage1              # L0–L3 + blind adjudication packet
#   -> two sealed Sonnet judges write census_verdicts_judge{A,B}.json
python3 census.py finish --verdicts census_verdicts_judgeA.json,census_verdicts_judgeB.json
```

```bash
python3 length_stratification.py      # length / LaTeX strata + bank ablations
python3 position_matched.py           # matched-sampling discount on the position family
```

**GATE.** Seeds 1 and 2 come either from the scaleupC dense chain or from the accelerated
stacked run (`sk3:logs/mathse_vote_seeds12_gpu4.log`, canonical output dirs, so the chain
skips them). Gate quantity = mean over seeds {42,1,2} of the EVAL AUC − VA_nl_mean3
(.6242101942973988). If it is ≤ .02 the campaign STOPS at round 0 and the seed verdict is
the terminal result. Once the seeds land:

```bash
bash refresh_round0_3seed.sh          # re-runs every DENSE-side round-0 readout at 3 seeds
```

## Rounds 1..5

```bash
python3 stage1_slice.py --cell mathse_vote --round R      # disagreement slice in FIT+MINE
python3 harness_maps.py build --cell mathse_vote --round R
python3 run_fleet.py codex --tags mathse_vote_rR --tracks A,B
python3 run_fleet.py glm   --tags mathse_vote_rR --tracks A,B
#   Claude legs are sealed subagents reading only their own prompt file
python3 harness_maps.py collect --cell mathse_vote --round R
python3 species.py --cell mathse_vote --round R           # species + Good-Turing, both tracks
python3 audit.py build --cell mathse_vote --round R       # blind routing + 2 planted pairs
python3 arbiter.py --cell mathse_vote --round R           # only on disputed routes
python3 audit.py finalize --cell mathse_vote --round R
# GPU (sk3): ./gpu_runner.sh mathse_vote_rR <log> $HOME/envs/gemma4/bin/python \
#            score_gemma_maps.py --jobs mathse_vote_rR
python3 readout.py --cell mathse_vote --round R
```

Stopping rule: 2 consecutive rounds with MONITOR VA_nl gain < ε = .005 (signed reading),
or the cap of 5 rounds. Matched sampling is triggered once spurious-alone AUC exceeds .65.

## Cell-specific deviations from the press/maps pipeline, all recorded

| file | deviation |
|---|---|
| `cells.py` | new loader; kept-subset row order; alignment gate wired into `load()`; observed covariates `answer_position`/`n_answers`/`answer_year`/`primary_tag` carried but never banked |
| `build_splits.py` | the closure cut is SALTED (`"mathse-vote-closure\|"`) because the dense arm's own split hashes the same key; collision check reported |
| `score_gemma_maps.py` | truncation matched to the A bank's deterministic HEAD-3000 + TAIL-2000 middle omission, not head-only |
| `audit.py` | four corpus-matched math.SE probe pairs replace the press ones |
| `harness_maps.py` | MODE 3 = position of the answer under its question; MODE 4 = math.SE upstream priors (answerer standing, markup habit, question kind) |
| `position_line.py` | container = the question, order = answer arrival; adds the within-question AUC readout |
| `round0.py` | readout tiers declared in advance; no A-imputation fork (this cell's Layer-1 already median-imputes) |

## Fleet

Target P = 6 across 3 families (Claude ×2 sealed subagents, gpt-5.6-luna ×2 via the Codex
companion, GLM-5.2 ×2, budget_tokens 2048 / max_tokens 32000). GLM is tried once per
round; on a 429 the round is recorded as degraded to P = 4 / 2 families, above the
freeze's floor.

## Scale note

The population is 11,629 rows, ~4× the press cell. One round's corpus-wide Gemma pass is
11,629 × 25 = **290,725 prompts**. Prefix caching carries most of it (the item text is the
shared prefix across a round's 25 criteria), but budget ~1.5–2.5 h of one B200 per round.
