# code_v3 Layer-3 closure campaign — runbook

Cell: GitHub pull-request MERGE outcome, v3 enriched text, Gemma-4-31B 83-criterion bank.
Note: `notes/2026-08-09__closure_code.md`. Prereg: `notes/2026-08-05__layer3-closure-prereg.md`.

Everything here is **within-repo**; pooled AUC is never a residual on this cell, and the
historical `V+A = .592` is retired (different input AND different instrument, and it no
longer reproduces).

## Conventions

* sk3 home must be pinned: `export HOME=/lfs/skampere3/0/alexspan` in every remote shell
  (the AFS home is unreadable and breaks login shells).
* GPU ledger `/lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt` — CLAIM before
  launching, RELEASE after. Never touch a co-tenant's process; kill only your own PIDs.
* All GPU scoring is **offline batch vLLM**, never an HTTP server.
* `latex/` is untouched by this campaign.

## Round 0 (done)

```bash
cd methods/taste_decomposition/closure/code_v3

# 1. dense v3 seeds 1 and 2, CHAINED on one ledger GPU (recipe byte-identical to seed 42)
#    launched from sk3:
#      GPU=6 setsid nohup bash dense_standard_v3/run_code_v3_seeds12.sh \
#            > dense_standard_v3/runner_v3_seeds12.log 2>&1 &
#    per-seed scorer: dense_standard_v3/abank_rescore/score_one_seed_v3.py
#    (merges into eval_pass_results.json; never rewrites seed 42's entry)

python3 build_splits_code.py          # FIT+MINE / MONITOR by repository hash at .80
python3 round0_code.py                # gate table + closure round-0 + position + swap
HF_HUB_OFFLINE=1 python3 census_code.py shortlist     # L0-L4 + blind adjudication packet
#   -> two sealed Sonnet judges write census_verdicts_judge{A,B}.json
python3 census_code.py finalize --verdicts census_verdicts_judgeA.json census_verdicts_judgeB.json
```

**GATE.** If the 3-seed within-repo residual is ≤ .02 the campaign STOPS at round 0 and
the seed verdict is the terminal result. Otherwise rounds 1..5 run.

> **If `round0_results.json` is missing**, the closure block of `round0_code.py` did not
> finish (it takes ~40 min wall; the HistGB grid thrashes OpenMP on this box — prefer
> `OMP_NUM_THREADS=6 nohup python3 round0_code.py > round0.log 2>&1 &`). Everything else
> in round 0 is already on disk and independent of it: `census_code.json`,
> `position_line.json` + `position_line_ext.json`, `length_stratification.json`,
> `fused_check.json`, `gate_uncertainty_seed42.json`, `splits_report.json`,
> `parents_code.json`. Only the closure-protocol r = 0 anchor (VA refit on FIT+MINE) is
> in that one file, and it is not needed until a round actually runs.

## Decomposition pass (FREEZE ADDENDUM 3, runs BEFORE round 1)

```bash
python3 select_parents_code.py        # SHAP interaction screen on FIT+MINE + brief parents
python3 decompose_code.py build       # sealed decomposer prompt -> scratchpad/code_v3/code_v3_rd/
#   -> a sealed frontier decomposer writes out_decomposer.txt
python3 decompose_code.py collect     # -> code_v3_rd_species.json
python3 audit.py build    --cell code_v3 --round d     # blind routing audit + 4 corpus-matched probes
#   -> fresh blind Sonnet auditor writes code_v3_rd_audit_verdicts.json
python3 arbiter.py --cell code_v3 --round d            # only if the audit disputes a route
python3 audit.py finalize --cell code_v3 --round d
# GPU (claim a ledger GPU first):
#   python3 score_round_code.py --tag code_v3_rd --util .90
```

## Rounds 1..5

```bash
python3 stage1_slice_code.py --round R        # disagreement slice inside FIT+MINE
python3 harness_code.py build  --cell code_v3 --round R   # sealed per-proposer prompts
python3 run_fleet.py codex --tags code_v3_rR --tracks A,B  # gpt-5.6-luna legs
python3 run_fleet.py glm   --tags code_v3_rR --tracks A,B  # GLM legs (try once; may 429)
#   Claude legs are sealed subagents reading only their own prompt file
python3 harness_code.py collect --cell code_v3 --round R
python3 species.py --cell code_v3 --round R    # species + Good-Turing (both tracks) + selection
python3 audit.py build --cell code_v3 --round R
python3 arbiter.py --cell code_v3 --round R
python3 audit.py finalize --cell code_v3 --round R
# GPU: python3 score_round_code.py --tag code_v3_rR --util .90
python3 readout_code.py --round R              # within-repo Δ_r, swap, discount, stacked increment
```

Stopping rule: 2 consecutive rounds with the MONITOR within-repo VA_nl gain < ε = .005
(signed reading, as frozen), or the cap of 5 rounds. Matched-sampling discount is
triggered once spurious-alone AUC exceeds .65.

## Resuming after the dense seeds land

```bash
# on sk3: confirm the chain finished
ssh sk3 'export HOME=/lfs/skampere3/0/alexspan; \
  cat $HOME/norm-research/datasets/code-review/dense_standard_v3/runner_v3_seeds12.log; \
  cat $HOME/norm-research/datasets/code-review/dense_standard_v3/eval_pass_results.json'

# pull the new per-seed preds into the campaign dir (cells_code.load picks them up
# automatically once dense_seed1/ and dense_seed2/ exist)
for S in 1 2; do
  mkdir -p dense_seed$S
  rsync -a sk3:/lfs/skampere3/0/alexspan/norm-research/datasets/code-review/dense_standard_v3/rm_out_seed$S/preds_{eval,test}.csv dense_seed$S/
done

python3 round0_code.py --gate-only     # cheap: recomputes ONLY the 3-seed gate table
python3 gate_readout.py                # THE gate readout -- see below
```

### `gate_readout.py` — the one command that decides the campaign

Seed-count agnostic. It runs the **mandatory OOF alignment gate first and refuses to
report if it fails** (registry 2026-08-10 landmine: `*_va_nl_oof_*.npy` are in bank
`item_ids` order, and a misaligned join reads AUC ≈ .50; the check includes a shuffled
counterfactual so a pass is a real pass, not a vacuous one). Then it emits:

* per-seed within-repo T and the across-seed mean / SD / range — the round-0 item-1
  deliverable ("3-seed within-repo T ± spread");
* Δ = T − VA_nl under **both** protocols (LAYER-1: VA fit within split — the ruler the
  published .0576/.0390 were measured on; CLOSURE: VA fit on FIT+MINE only, read straight
  from `round0_state.npz`, no refit);
* three tiers: eval, test, and **both splits combined at the repository level** (the
  best-powered honest statement — legitimate because the readout is repo-centred and the
  two dense splits contribute *disjoint* repositories, so this adds repos to a within-repo
  average rather than pooling rows across repos);
* n-weighted **and** equal-repo Δ, repo-cluster bootstrap CI, leave-one-repo-out
  jackknife, paired Wilcoxon;
* the GATE verdict against the frozen .02, carrying **`BINDING: true` only at 3 seeds**.
  With fewer it self-labels INTERIM and states that rounds stay held.

Output: `gate_readout_<n>seed.json`.

**Rounds fire only when BOTH hold: `BINDING: true` (3 seeds) AND a free ledger GPU.**
Reason for the second condition beyond compute: the mining slice is defined as
top |dense percentile − VA_nl percentile|, and the dense score is the **seed ensemble** —
so a slice built before all three seeds land is a *different* slice, and the sealed
fleet's one blind look would be spent on the wrong one.

## RESUME CHECKLIST after the 2026-08-10 jump-host outage

sk3 became unreachable via the `whale` ProxyCommand at ~18:15 on 2026-08-10
(`ssh whale` → `Connection reset by 171.64.75.72 port 22`). sk3 itself was never shown to
be down, and both GPU jobs run at ppid = 1 under `setsid --fork`, so they continued. On
reconnect, in this order:

```bash
# 1. what survived
ssh sk3 'D=/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure/code_v3; \
  grep -E "^\[shard|SCORE_DONE" $D/rd_score.log | tail; \
  ls $D/code_v3_rd_scores*.npz; \
  tail -5 $D/../../code_competitions/ac_dense.log 2>/dev/null; \
  grep claude-closure-code-v3 /lfs/skampere3/0/alexspan/norm-research/gpu_ledger.txt | tail -4'

# 2. RE-SYNC the code fixed during the outage -- sk3 still has the PRE-FIX copies
rsync -a readout_code.py stage1_slice_code.py cells_code.py cells.py species.py \
      score_round_code.py \
      sk3:/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure/code_v3/

# 3. pull the decomposition scores and run the readout (path already dry-run proven)
rsync -a sk3:.../code_v3/code_v3_rd_scores.npz sk3:.../code_v3/code_v3_rd_scores.report.json .
python3 readout_code.py --round d
```

**Fixed during the outage, so sk3's copies are stale**: the `sys.path` shadowing fix
(`maps_hw_si` must be inserted *before* `HERE`), the non-integer round-label tolerance in
`current_blocks`, the `_expand` fix in `readout_code.discount`, the named shape guard in
`cells_code.within_repo_auc`, the token-budgeted `render_card`, and the both-track blind
merge in `species.py`.

**Readout path status: DRY-RUN PROVEN.** `readout_code.py` was executed end-to-end on
synthetic scores during the outage; all four tiers, the spurious map, both discount blocks,
the matched-sampling trigger, the stacked increment and the swap algebra produced correct
structure, and the enforced collapse gate correctly dropped a planted constant column
(n_B 10 → 9). All dry-run artifacts were deleted.

## Fleet

Target P = 6 across 3 families (Claude ×2 sealed subagents, gpt-5.6-luna ×2 via the Codex
companion, GLM-5.2 ×2). GLM is tried once per round; on a 429 the round is recorded as
degraded to P = 4 / 2 families, which is above the freeze's floor.

## Files

| file | what |
|---|---|
| `cells_code.py` | cell loader (A shards, V matrix, dense preds, position covariates) + the within-repo AUC/Δ estimators |
| `cells.py` | maps-contract shim (CELL_META, slice-card renderer, sk3 text fetch) |
| `build_splits_code.py` | FIT+MINE / MONITOR by repository, `splits.npz`, `splits_report.json` |
| `round0_code.py` | gate table, closure round-0 baseline, position line, swap baseline |
| `census_code.py` | round-0 concept census, L0→L5 |
| `select_parents_code.py` | SHAP interaction screen → MIXED parents to decompose |
| `decompose_code.py` | Addendum-3 decomposition pass (code-specific prompt) |
| `harness_code.py` | sealed dual-track proposer prompts (code-specific MODE 3) |
| `audit.py` | blind routing audit, 4 corpus-matched PR probes |
| `score_round_code.py` | Gemma-4-31B offline-batch scorer for a round's criteria (sk3) |
| `abank_rescore/` | the incoming instrument: 83-criterion scores, V matrix, VA_nl OOF, dense seed-42 preds |
