# CW-community closure campaign — runbook

Frozen protocol: `notes/2026-08-05__layer3-closure-prereg.md` (FREEZE DECLARATION +
ADDENDUM + ADDENDUM 2). Every step below is scripted; nothing is done by hand.

## Stage 0 — enlarge the honest population (once)

```
python stage0_build_population.py                    # CW_TARGET_NEW_ROWS=6600
scp stage0_score_ext_gemma.py cw_ext_to_score.csv cw_honest_population.csv \
    stage0_dense_rescore.py gpu_runner.sh sk3:<cw_community>/
ssh sk3 './stage0_driver.sh'                         # waits for a FREE GPU, claims it
scp sk3:<cw_community>/{cw_ext_scores.npz,cw_ext_scores.report.json,\
    cw_honest_dense_preds.csv,cw_honest_dense_preds.report.json} .
python stage0_readout.py                             # -> round0_state.npz, round0_results.json
```

`gpu_runner.sh` is the safety layer: it polls until a GPU has **zero** compute
processes, claims it in `/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt`,
re-verifies after 5 s, runs, and releases. Scoring is chunk-checkpointed
(`*_parts/`), so a SIGTERM mid-run costs one chunk, not the run.

## Per round r

```
python stage1_slice.py --round r                     # 40-row disagreement slice, label-blind
python fleet_cw.py build --slice round{r}_slice.json --tag r{r}
./run_round_fleet.sh r{r}                            # codex(luna)x2 + glm x2 legs
#   Claude legs: 2 sealed subagents per track, prompt inline, AGENT_PROMPTS.md S1
python fleet_cw.py collect --tag r{r} --track A
python fleet_cw.py collect --tag r{r} --track B
#   blind species partition per track (AGENT_PROMPTS.md S2)  -> round{r}_species.json
#   author 2 planted probes                                  -> round{r}_probes.json
python select_and_route.py --round r                 # -> round{r}_audit_blind.json
#   blind auditor (S3) -> round{r}_audit_verdicts.json ; arbiter (S4) on disputes
python finalize_routing.py --round r                 # -> round{r}_criteria.json
scp round{r}_criteria.json cw_population_with_splits.csv score_round_gemma.py sk3:...
ssh sk3 './gpu_runner.sh r{r}_score round{r}_score.log <gemma-python> score_round_gemma.py --round r'
scp sk3:<cw_community>/round{r}_scores.npz* .
python round_readout.py --round r                    # -> round{r}_results.json, round{r}_state.npz
python missing_mass_cw.py --rounds 1..r
```

## Stopping and reporting

Stop at 2 consecutive rounds with MONITOR VA_nl gain < ε = .005, or at round 5.
TEST is quoted exactly once, at the declared stopping round.

```
python gepa_phrasing.py targets --rounds 1..R        # bounded, label-blind phrasing pass
python build_report.py --rounds 1..R                 # markdown tables for the note
```

## Invariants asserted in code

* row order: `round{r}_scores.npz.ids` must equal `round{r-1}_state.npz.ids`
* criteria identity: extension npz `a_names` must equal the frozen bank's `a_names`
* MONITOR is never read by any fit, imputer, column screen, or design decision
  (the sign-contradiction trigger uses FIT+MINE alone-AUC)
* the incoming state's readouts are reconstructed from its saved per-seed held-out
  predictions, so Δ_r compares the exact fit the previous round reported
