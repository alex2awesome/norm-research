# v14 decoder-tuning campaign: implementation and execution runbook

Implementation entry point:
`methods/metric_implementer/experiments/run_v14_value_campaign.py`.

## Live execution state (2026-07-13 19:56 PDT)

- Staged code root: `/lfs/skampere3/0/alexspan/cr3-v14/code`.
- Campaign output root: `/lfs/skampere3/0/alexspan/cr3-v14/outputs/campaign`.
- Source population manifest:
  `/lfs/skampere3/0/alexspan/cr3-v13.1/assets/tier_b/tier_b_metrics.json` (220 raw
  R3 metrics; the v14 design freezes the certified subset).
- Scientific design code and deterministic panel seed: `5db95c9442f08aed2a664a8ad6d0f7bd2106ffac`.
  Operational launcher hardening is commit `2406c804adbd4852f32fcf719c741b4866725ea6`;
  it does not change or restart completed scientific design cells.
- CPU-only certified design construction is running on sk3. At this snapshot 7 of 35 metric
  design manifests existed and no v14 GPU process had started. Check with:

  ```bash
  ssh sk3 'find /lfs/skampere3/0/alexspan/cr3-v14/outputs/campaign/designs \
    -name design_manifest.json | wc -l'
  ```

- The hard 3--5 label-balance feasibility audit found only four eligible legal metrics. The
  frozen 35-metric quota is therefore legal 4, creative writing 6, and 5 for each of the other
  five tasks. The design index records all exclusions and the deterministic quota reallocation.
- V14 GPU work remains blocked on the verified 280-row v13.1 Tier B consolidation. Tier B is
  currently using only sk3 devices 0, 5, 6, and 7; the V14 wrapper will not be invoked until those
  lanes exit and consolidation succeeds.

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
CUDA. It pins CUDA enumeration to PCI-bus order, forces vLLM workers to use `spawn`, and moves
`HOME` to durable `/lfs` storage. It also uses the pinned sk3 miniconda interpreter because a
noninteractive sk3 shell has no bare `python` on `PATH`; `V14_PYTHON` may override that path.
Before spawning vLLM, it changes into its own `/lfs` code root and pins that root on `PYTHONPATH`,
so an expired AFS working directory cannot poison a spawned EngineCore.
It pins the private Hugging Face cache and supplies immutable shared-cache snapshot overrides for
Llama-3.3-70B and Mistral-Small-24B, avoiding duplicate model downloads.
The Python launcher independently verifies the declared
physical IDs. Do not invoke a GPU phase directly on `sk3`.

The campaign has a hard **four-GPU total cap across all hosts**. Using sk3 GPUs 0, 5, 6, and 7
therefore consumes the entire allowance: while those four lanes run, no v14 GPU lane may run on
sk1 or sk2. This is a cross-host scheduling invariant and cannot be enforced by the per-host
wrapper alone.

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

   The frozen primary → fallback pairs are:

   - Qwen2.5-14B-Instruct → Qwen2.5-32B-Instruct.
   - Llama-3.3-70B-Instruct → Llama-3.1-8B-Instruct.
   - Mistral-Small-24B-Instruct-2501 → Mistral-7B-Instruct-v0.3.

   Qualification freeze rejects any other model path or more than one fallback.

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
