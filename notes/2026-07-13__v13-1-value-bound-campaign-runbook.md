# v13.1 value-bound campaign: live runbook

Last operational snapshot: **2026-07-13 20:39 PDT**.

## 19:40 PDT recovery update

This update supersedes the PID/state tables later in this note; the experiment definition,
completion markers, stopping point, and artifact paths below remain current.

- Commit `1f162e5` adds a deterministic fail-closed redaction for the residual case where a
  no-verbatim induced rule still contains a 12-word demo shingle after both model repair passes.
  Only the copied span is removed, the fallback is recorded in the induction cache payload, and
  the final leakage assertion remains active. The focused test passed, and the code is staged on
  both sk2 and sk3.
- The sk2 Tier B lanes are running as PIDs `347907` (`llama31_8b`, resumed from cache), `2031614`
  (`qwen25_14b`), and `2031615` (`phi4`). The Llama lane is on **sk2** device 4; the prohibition
  on devices 1--4 applies to **sk3**.
- The sk3 launcher now pins `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`, and `HOME=/lfs/skampere3/0/alexspan`. Tier B 70B shards
  0, 3, 4, and 5 are running as PIDs `1441737`, `1457789`, `1597568`, and `1597569` on sk3
  devices 0, 5, 6, and 7 respectively. No campaign work is permitted on sk3 devices 1--4.
- Recovery supervisor PID `1121368` remains queued to resume shard 1 on device 6 after shard 4
  completes, and shard 2 on device 7 after shard 5 completes. All resumes preserve the existing
  content-addressed caches.
- The old sk2 completion orchestrator cannot authenticate onward to sk3 and has exited. Final
  copying and consolidation will instead be performed explicitly from the authenticated local
  session after all nine lane manifests exist. No Tier A upgrades will be launched.
- V14 remains ordered after the verified 280-row Tier B consolidated artifact. Its guarded sk3
  wrapper may use only devices 0, 5, 6, and 7, and nothing may run on sk1/sk2 while those four v14
  lanes are resident.

### 19:56 PDT follow-up

- Shards 4 and 5 completed successfully. Their manifests triggered the cached shard 1 and shard 2
  resumes on sk3 devices 6 and 7. At this snapshot the active 70B PIDs were `1441737` (shard 0),
  `1457789` (shard 3), `1968952` (shard 1), and `1844392` (shard 2), corresponding only to sk3
  devices 0, 5, 6, and 7.
- Spawned EngineCore processes also inherit the parent's current directory. The restricted sk3
  launcher now changes into `/lfs/skampere3/0/alexspan/cr3-v13.1/code` before launching; this
  prevents an expired AFS current directory from killing a `spawn` worker. The resumed sk2 Llama
  lane was likewise launched from its `/lfs` code root as PID `1224489`.

### 20:11 PDT follow-up

- Shard 0 reached its fixed-executor handoff and its pre-fix process failed EngineCore startup.
  Its cache remains intact. A corrected retry found sk3 device 0 occupied by a separate live
  `methods.codability.experiments.score_fresh_name_arms` calibration job (parent PID `2261554`),
  not an orphan; do not terminate or interfere with that job.
- Shard 0 is therefore paused with no retry loop. Resume it only after device 0 is free, or on
  another allowed device after its current Tier B shard completes:

  ```bash
  ssh sk3 'V13_SHARD_IDS=0 V13_DEVICE_OVERRIDE=0 \
    /lfs/skampere3/0/alexspan/cr3-v13.1/scripts/run_v13_llama70_shards_sk3.sh'
  ```

### 20:39 PDT follow-up

- Shards 4 and 5 remain complete. Shards 1 and 2 are constructing on allowed sk3 devices 6 and
  7. Shard 3 finished all constructor cells, hit the expected pre-fix executor-start failure, and
  was resumed from its content-addressed cache as PID `3393889` on allowed device 5 through the
  corrected launcher. Its live environment was verified to contain
  `CUDA_VISIBLE_DEVICES=5`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, spawn mode, and the durable `/lfs`
  home.
- Shard 0 remains paused with its constructor cache complete. The unrelated codability job still
  owns device 0, so no shard-0 retry is queued. After shard 3 completes, shard 0 may instead be
  resumed on device 5 if device 0 is still occupied.
- The sk2 Llama, Qwen, and Phi lanes remain live. The Llama lane has completed all 35 MCQ metrics
  and is filling fixed-executor behavioral cells; Qwen and Phi are finishing their last MCQ and
  constructor metrics. None has a campaign manifest yet.
- In parallel, v14 is performing CPU-only design preparation. It has no exposed CUDA device and
  cannot begin a GPU phase until this Tier B run has nine lane manifests, consolidates to exactly
  280 rows, and is reported.

This note is the durable handoff for the currently running v13.1 campaign. PIDs and row counts
below are a point-in-time snapshot; the paths, completion markers, restrictions, and monitoring
commands are the durable source of truth.

## Experiment and stopping point

The campaign measures two distinct value-bound channels and does not combine them numerically:

- **Multi-panel Reconstruction-MCQ:** target-choice lift after blind and shuffled control
  subtraction.
- **Behavioral induce-and-execute reconstruction:** held-out plug-in mutual information in bits,
  also control-subtracted, with balanced agreement as a diagnostic.

The executor is fixed at `Llama-3.1-8B-Instruct` with constrained binary readout. The constructor
models are `Llama-3.1-8B-Instruct`, `Qwen2.5-14B-Instruct`, `Phi-4`, and
`Llama-3.3-70B-Instruct`. The frozen implementation is release **v13.1**, commit `75f7662`;
the current scheduling/guard commit is `05b1460`.

The active broad run is **Tier B**: 35 deterministically selected R3 metrics, five each from
humor, creative writing, code review, news, peer review, legal, and math. Four constructors by
two channels yields **280 Tier B results**. The preceding six-humor-metric Tier A wave yields
48 results and is being allowed to finish because some Tier B 70B lanes depend on its GPUs.

The user-directed terminal condition is:

> Finish and consolidate Tier B, then stop and report. Do not run the ten automatic Tier A
> upgrades.

The completion orchestrator is running with `V13_STOP_AFTER_TIER_B=1`, and every Tier B lane was
launched with `--disable-auto-upgrade`.

## Hard GPU restriction

On **sk3**, GPU IDs **1, 2, 3, and 4 must not be used by this campaign again**. Only sk3 GPUs
0, 5, 6, and 7 are permitted.

`scripts/run_v13_llama70_shards_sk3.sh` enforces this structurally: it rejects device IDs 1--4,
including explicit `V13_DEVICE_OVERRIDE` values. The guard was committed in `05b1460` and staged
to sk3. Do not bypass or weaken it.

As of the snapshot, campaign work is only on sk3 GPUs 0, 5, 6, and 7. Activity on GPUs 3 and 4
is an unrelated DPO job owned by `ahmedah`. The earlier GPU 1 `norm-scraper` cron process had
already exited by 16:24. Never kill or attribute unrelated processes based only on GPU usage;
check owner and full command line first.

## What is running

Canonical remote root on sk2:
`/lfs/skampere2/0/alexspan/cr3-v13.1`

Canonical remote root on sk3:
`/lfs/skampere3/0/alexspan/cr3-v13.1`

### sk2

Tier B small-constructor lanes:

| Lane | Snapshot PID | Constructor |
|---|---:|---|
| `llama31_8b` | 2031611 | Llama-3.1-8B-Instruct |
| `qwen25_14b` | 2031614 | Qwen2.5-14B-Instruct |
| `phi4` | 2031615 | Phi-4 |

Supervisors:

- PID 2121595: `schedule_v13_tier_b_after_assets.sh`; launches dependency-bound 70B shards only
  on allowed devices.
- PID 2787878: `orchestrate_v13_completion.sh`; consolidates Wave 1, then Tier B, then exits due
  to `V13_STOP_AFTER_TIER_B=1`.

Wave 1 Qwen and Phi lanes were also still finishing on sk2 at the snapshot. The Wave 1 8B lane
was complete.

### sk3

| GPU | Lane | Snapshot PID | State |
|---:|---|---:|---|
| 0 | Tier B `llama33_70b_shard_0` | 1441737 | running, 6 metrics |
| 5 | Wave 1 `llama33_70b_shard_11_12` | 895750 | running; frees GPU 5 for Tier B shard 3 |
| 6 | Tier B `llama33_70b_shard_4` | 1124843 | running, 6 metrics; shard 1 resumes here afterward |
| 7 | Tier B `llama33_70b_shard_5` | 1137193 | running, 5 metrics; shard 2 resumes here afterward |

PID 1121368 is the restricted-device recovery supervisor. It waits for shard 4 and shard 5,
then resumes the interrupted content-addressed caches for shard 1 on GPU 6 and shard 2 on GPU 7.
Shard 3 has 6 metrics and starts on GPU 5 after its Wave 1 dependency finishes. No completed
cache is discarded.

At 16:18 PDT, useful cache snapshots were:

| Lane | MCQ state rows | Behavioral induction rows |
|---|---:|---:|
| sk2 `llama31_8b` | 89,088 | 20,998 |
| sk2 `qwen25_14b` | 51,200 | 11,780 |
| sk2 `phi4` | 50,176 | 11,780 |
| sk3 70B shard 4 | 12,032 | 1,538 |
| sk3 70B shard 5 | 11,008 | 1,538 |
| sk3 70B shard 1, saved | 4,608 | 0 |
| sk3 70B shard 2, saved | 4,352 | 0 |

For scale, a complete Tier B metric contributes 18,432 MCQ state rows and 3,072 behavioral
induction rows before rule deduplication. These counts are progress indicators, not formal
completion markers, because channel phases can advance at different rates.

## Read-only monitoring

### Process and GPU ownership

```bash
ssh sk2 'ps -eo user,pid,ppid,etimes,stat,args | rg "run_v13_value_campaign|schedule_v13_tier_b|orchestrate_v13_completion"'
ssh sk3 'ps -eo user,pid,ppid,etimes,stat,args | rg "run_v13_value_campaign|resume_v13_tier_b"'
```

The process command line, owner, and `CUDA_VISIBLE_DEVICES` are more reliable than a process
name alone. Do not use a global `nvidia-smi` enumeration on sk3 for this campaign; inspect only a
known campaign PID's environment. To inspect the assigned device for one of our main PIDs:

```bash
ssh sk3 'tr "\000" "\n" </proc/PID/environ | rg "^CUDA_VISIBLE_DEVICES="'
```

### Logs and crash scan

```bash
ssh sk2 'tail -n 80 /lfs/skampere2/0/alexspan/cr3-v13.1/logs/tier_b/scheduler.log'
ssh sk2 'tail -n 80 /lfs/skampere2/0/alexspan/cr3-v13.1/logs/completion_orchestrator.log'
ssh sk2 'rg -n "Traceback|CUDA out of memory|exited without campaign manifest" /lfs/skampere2/0/alexspan/cr3-v13.1/logs/tier_b'
ssh sk3 'rg -n "Traceback|CUDA out of memory|exited without campaign manifest" /lfs/skampere3/0/alexspan/cr3-v13.1/logs/tier_b'
```

Per-lane logs are under `logs/tier_b/<lane>.log` on the host running that lane. Active cache
files are under `outputs/tier_b/lanes/<lane>/cache/value_cells.sqlite`.

### Completion markers

A lane is complete only when its `campaign_manifest.json` exists. Tier B expects three small
lanes on sk2 and six 70B shards on sk3:

```bash
ssh sk2 'find /lfs/skampere2/0/alexspan/cr3-v13.1/outputs/tier_b/lanes -name campaign_manifest.json -print'
ssh sk3 'find /lfs/skampere3/0/alexspan/cr3-v13.1/outputs/tier_b/lanes -name campaign_manifest.json -print'
```

After all nine markers exist, the orchestrator copies the non-cache 70B artifacts to sk2 and
writes the consolidated Tier B outputs here:

```text
/lfs/skampere2/0/alexspan/cr3-v13.1/outputs/tier_b/consolidated/results.parquet
```

The consolidated file must contain 280 Tier B rows: 35 metrics by 4 constructors by 2 channels.
Per metric/model/channel directories retain `design_manifest.json`, `state_tables.npz`,
`certificate.json`, and `prompt_values.parquet`.

## ETA and final handoff

At the snapshot, the best ETA for complete Tier B consolidation was **2026-07-14 around 13:00
PDT**, with a reasonable **10:00--15:00 PDT** window if throughput remains stable. The bottleneck
is the two sequential 70B resume chains on allowed GPUs 6 and 7. This is an estimate, not a
deadline; crashes or unusually long behavioral generations can move it.

After consolidation, verify the 280-row cardinality, required columns/artifacts, finite state
tables, and cap-versus-achieved structural assertions. Then stop all campaign scheduling and
report Wave 1 plus Tier B results. Do not select or launch the automatic Tier A upgrades.
