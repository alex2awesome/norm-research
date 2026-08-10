# jokes_community Layer-3 closure campaign — runbook

Cell: **r/Jokes posts**, y = crowd upvote quartile (top 25% vs bottom 25% of raw score
inside a `length_bin × format × topic` stratum; middle 50% dropped upstream).
16,000 posts / 50 LDA topics / pos-rate .496.
Note: `notes/2026-08-09__closure_jokes.md`.
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + freeze addenda 1–4.
Queue: `notes/2026-08-09__full_sweep_queue.md` — **LANE A, GPU 5**.
Pipeline shape inherited from `../mathse_vote/`.

## Conventions

* sk3 home must be pinned: `export HOME=/lfs/skampere3/0/alexspan` in every remote shell.
* GPU ledger `/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt` — CLAIM before
  launching, RELEASE after. Never touch a co-tenant's process; kill only your own PIDs.
  `gpu_lane_runner.sh` pins the lane's card (GPU 5) instead of wandering to whichever
  card has the most free memory.
* All GPU scoring is **offline batch vLLM**, never an HTTP server.
* `latex/`, the strict list, the registry and the frozen notes are untouched by this
  campaign.

## The alignment gate (mandatory, runs inside every `cells.load()`)

`AUC(y, jokes_community_va_nl_oof_seed0.npy in assembled row order)` must equal the
ledger's `nonlinear.VA["0"].auc` = **.7321856098790323** to < 1e-9. Row order is the bank
`item_ids` order, unfiltered (all 16,000 y are finite). Set `JOKES_SKIP_GATE=1` only for
a deliberate diagnostic, never for a readout.

## Two T conventions — never mixed in one figure

| convention | HONEST (3,163 rows) | where |
|---|---|---|
| mean over dense seeds of the AUC (**campaign T**) | .7377 | `round0.py`, `cells.T_by_seed` |
| AUC of the seed-mean prediction (seed **ensemble**) | .7469 | `readout.py`, `era_line.py`, the master ledger |

The stratified and stacked readouts need ONE score column, so they necessarily use the
ensemble vector; every such block says so in its own `T_note` / `T_convention` key.

## Round 0

```bash
cd methods/taste_decomposition/closure/jokes_community

python3 build_covariates.py           # created_utc join off the raw scrape (86.1% matched)
python3 fetch_dense.py                # per-seed dense probs off sk3 (positional join
                                      # asserted on judgement AND group)
python3 oof_alignment_gate.py         # -> oof_alignment_gate.json  (must PASS)
python3 build_splits.py               # FIT+MINE / MONITOR by salted topic hash
OMP_NUM_THREADS=6 python3 round0.py   # tiers + closure r0 + swap + topic jackknife
OMP_NUM_THREADS=6 python3 era_line.py # FREEZE ADDENDUM 4 posting-stream audit
```

No round-0 STOP gate applies to this cell (the mathse_vote runbook's `.02` gate was that
cell's dispatch condition). What the round-0 residual DOES decide is the freeze roster
tier: Δ_beyond = +.0143 < .02 puts this cell on the **map-focused** side of the roster —
both tracks run at full budget, and the spurious map is the headline.

## Rounds 1..5

```bash
OMP_NUM_THREADS=6 python3 stage1_slice.py --cell jokes_community --round R
python3 harness_maps.py build --cell jokes_community --round R
python3 run_fleet.py codex --tags jokes_community_rR --tracks A,B    # nohup, NOT setsid
python3 run_fleet.py glm   --tags jokes_community_rR --tracks A,B    # (macOS has no setsid)
#   Claude legs are 6 sealed subagents reading only their own prompt file (DISPATCH.md)
python3 harness_maps.py collect --cell jokes_community --round R
python3 species.py --cell jokes_community --round R        # tau-only species + Good-Turing
python3 species_merge.py build --cell jokes_community --round R --tracks A,B
#   -> two sealed blind judges write bmerge verdict files
python3 species_merge.py apply --cell jokes_community --round R --tracks A,B \
        --verdicts <judgeA>.json,<judgeB>.json               # STRICT two-judge merge
python3 audit.py build --cell jokes_community --round R    # blind routing + 2 planted pairs
#   -> fresh blind Sonnet-class auditor writes <tag>_audit_verdicts.json
python3 audit.py finalize --cell jokes_community --round R
python3 arbiter.py build --tags jokes_community_rR --out jokes_community_rR  # if disputes
#   -> frontier arbiter writes <tag>_arbiter_raw.json
python3 arbiter.py apply --raw <tag>_arbiter_raw.json --default-round R
python3 audit.py finalize --cell jokes_community --round R
bash launch_score.sh R                                     # sk3 GPU 5, offline batch Gemma
OMP_NUM_THREADS=6 python3 readout.py --cell jokes_community --round R
```

**Stopping rule.** 2 consecutive rounds with MONITOR VA_nl gain < ε = .005 (signed,
TIER 1), or the cap of 5 rounds. A sub-ε round counts toward stopping **only if it was a
PROPOSING round** (registered 2026-08-08); a decomposition-only round never does.
Matched sampling is triggered once spurious-alone AUC exceeds .65.

**Two-tier rule.** Only the sealed fleet (TIER S) feeds Good-Turing / missing mass. Any
taxonomy-directed sweep is TIER D, is excluded from every estimator quantity, and any
table quoting mass names the tier it counted.

**species.py overwrite guard.** Once a round is merged, audited or scored, `species.py`
refuses to rebuild it. Never pass `--force` on a live round.

## Cell-specific deviations from the mathse_vote pipeline, all recorded

| file | deviation |
|---|---|
| `cells.py` | new loader; unfiltered bank item_ids order; alignment gate wired into `load()`; `created_utc` carried as the only observed covariate (`score` deliberately omitted — it defines y) |
| `build_covariates.py` | new; recovers `created_utc` from the raw scrape by exact sha1(title+" "+selftext) join, 86.1% matched, unmatched carried as NaN |
| `build_splits.py` | closure cut SALTED (`"jokes-community-closure|"`); granularity caveat recorded (only 6 MONITOR topics → group bootstrap is coarse, item band printed beside it) |
| `era_line.py` | replaces `position_line.py`: the container is the subreddit's posting stream, the ordinal is `created_utc`; also owns `within_topic_auc`, this cell's TIER-2 readout |
| `harness_maps.py` | Track-B MODE 3 rewritten for a corpus with no sibling container (era + retelling chain); MODE 4 rewritten for a joke forum (poster practice, sourcing vs origination, audience-management furniture, delivery-format habit, taboo register) |
| `audit.py` | four corpus-matched joke probe pairs replace the math.SE ones |
| `score_gemma_maps.py` | item view matched to the A bank EXACTLY — `JOKE:\n"<text>"`, no truncation (corpus max 3,508 chars); persona = "an expert comedy writer performing a measurement task"; 4,096-token context is ample |
| `species_merge.py` | STRICT blind pairwise merge extended to **both tracks** (`--tracks A,B`), because this campaign reports missing mass on both sides; tau-only tables preserved under `*_PREMERGE_tau_only` |
| `gpu_lane_runner.sh` | new; pins the lane's GPU (5) instead of picking the emptiest card |
| `round0.py` | tiers declared in advance; TIER 2 is within-TOPIC; both group and item bootstrap bands; no A-imputation fork (Layer-1 already median-imputes) |

## Fleet

P = 8 across 3 families (Claude ×3 sealed subagents, gpt-5.6-luna ×3 via the Codex
companion, GLM-5.2 ×2 on the Lite subscription endpoint, budget_tokens 2048 /
max_tokens 32000). Both GLM keys were smoke-tested live before round 1. Degradation
below P = 8 / 3 families is recorded in the round's `_proposals_fleet.json` report.

## Scale note

16,000 rows × 25 criteria + a 150-text anchor battery ≈ 404k prompts per round, but the
items are short (median 77 chars) so prefix caching and small sequences carry it; budget
~1–2 h of one B200 per round.
