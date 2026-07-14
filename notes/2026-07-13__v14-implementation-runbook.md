# v14 decoder-tuning campaign: implementation and execution runbook

Implementation entry point:
`methods/metric_implementer/experiments/run_v14_value_campaign.py`.

The current v13.1 Tier B run remains authoritative and must finish, consolidate to 280 rows,
stop, and be reported before any v14 GPU phase starts. CPU-only v14 design work may be prepared
in parallel. The v13 runbook is `notes/2026-07-13__v13-1-value-bound-campaign-runbook.md`.

## Hard scheduling rule

On `sk3`, physical GPUs **1, 2, 3, and 4 are forbidden**. Use only the guarded wrapper:

```bash
scripts/run_v14_campaign_sk3.sh 0 ...
scripts/run_v14_campaign_sk3.sh 5 ...
scripts/run_v14_campaign_sk3.sh 6 ...
scripts/run_v14_campaign_sk3.sh 7 ...
```

The wrapper rejects every other `sk3` device and takes a per-device process lock before exposing
CUDA. The Python launcher independently verifies the declared physical IDs. Do not invoke a GPU
phase directly on `sk3`.

## Finite execution order

Set `ROOT` to a new immutable v14 output root, `CERTIFIED` to the 35-metric breadth manifest, and
`ALL_R3` to a manifest containing enough non-certified R3 metrics across all seven tasks.

1. CPU-only design freeze:

   ```bash
   python -m methods.metric_implementer.experiments.run_v14_value_campaign \
     --phase design --metrics-manifest "$CERTIFIED" --out-root "$ROOT" --run-sha "$RELEASE_SHA"
   python -m methods.metric_implementer.experiments.run_v14_value_campaign \
     --phase prepare-dev --dev-metrics-manifest "$ALL_R3" --out-root "$ROOT" --run-sha "$RELEASE_SHA"
   python -m methods.metric_implementer.experiments.run_v14_value_campaign \
     --phase prepare-sentinel --sentinel-metrics-manifest "$ALL_R3" \
     --out-root "$ROOT" --run-sha "$RELEASE_SHA"
   ```

2. Run one-panel decoder mini-qualification. For each family, run the primary constructor once,
   then the fixed executor once. If it fails, run exactly the declared same-lineage fallback once;
   never search additional models.

3. Run exactly three bounded tuning jobs: MCQ, behavioral unconstrained, and behavioral
   no-verbatim. Each has eight candidates per round and at most four rounds. The implementation
   batches a full round by resident model and stops on the preregistered gain/transfer/residual
   conditions.

4. Freeze the three traces and qualified decoder revisions with `--phase freeze`. This writes
   `template_freeze.json` and `preregistration.json`; no scientific mutation is allowed afterward.

5. Run the six-metric sentinel with the frozen instrument. For each qualified family, invoke
   `sentinel-constructor`; then invoke `sentinel-executor` once and `sentinel-aggregate` once.
   These phases deliberately bypass the not-yet-created gate only inside `ROOT/sentinel`; their
   scientific values are recorded and never used as a stopping rule. Next run
   `liveness-constructor` for each family and `liveness-executor` once. The latter refuses to run
   until the 36 sentinel result rows exist, constructs the planted/degenerate/blind/annotated
   controls, and writes `sentinel_report.json`. Only structural failure or control-defined
   instrument death may block fan-out.

6. Fan out by resident decoder family. For each of `qwen`, `llama`, and `mistral`, run one
   `constructor` phase across all metrics, passing the successful sentinel report. Then run one
   fixed `executor` phase. Completed induction, MCQ, and rule/probe cells are append-only and
   reused after crashes.

7. Generate the pure audit with three model-resident `audit-proposer` phases (`phi4`, `qwen14`,
   `llama8`), then one fixed-executor `audit-score` phase. The per-metric total is exactly 400,
   quotas are 134/133/133 with the extra draw rotated by metric hash, and every accepted draw has
   an immutable independent seed and is never absorbed.

8. Run `aggregate`, then `report`. The release audit requires 35 metrics, two instruments, three
   reported channel/arms (210 rows), complete finite 50×256 state tables, exact cap ≥ achieved,
   all audit artifacts, valid certificate hashes, and every required file.

## Completion artifacts

Per metric/instrument/channel-arm:

- `design_manifest.json`
- `state_tables.npz`
- `prompt_values.parquet`
- `novelty_curves.parquet`
- `certificate.json`

Campaign-wide:

- `template_freeze.json`
- `preregistration.json`
- `sentinel_report.json`
- `results.parquet`
- `campaign_manifest.json`
- `run_status.json`
- `artifact_checksums.json`
- `release_report.json`
- `report.md`

Do not begin a follow-up experiment after `report`. The terminal action is to stop and report the
untuned/tuned results, decoder-family variation, MCQ/behavioral disagreement, fidelity/legibility
distortion, exemplar-arm gap, structural caps, achieved values, and all valid future-gain bounds.
