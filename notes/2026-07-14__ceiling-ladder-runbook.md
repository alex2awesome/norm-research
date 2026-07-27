# Independent fidelity audit and C0/C1/C2 ceiling ladder

Status: v14.1 implementation freeze in progress. The old `frozen_8b_bootstrap` comparison is
invalid for executor selection and is retained only as a cross-executor disagreement diagnostic.

## Scientific freeze

- Population: the same 35 Tier-B metrics. Ninety same-corpus probes are appended without changing
  the original 300 indices. The independent audit covers all 390 probes and the ladder uses the
  frozen 120/30/240 teaching/development/held-out split.
- C0: execute each metric's exact frozen description-form orbit and compare with three blind
  Sonnet passes (majority reference). A six-criterion mechanically labeled planted suite runs
  through the same C0/C1/C2 pipeline.
- C1: Llama-3.3-70B selects from the complete task-local description bank using eight labeled
  demonstrations; a deterministic size-11 menu is the sensitivity arm. Both use blind and
  shuffled-label controls, eight counterbalanced menu orders, and the picked description is then
  executed by the fixed Llama-3.1-8B executor.
- C2: Llama-3.3-70B freely induces a rule from the same eight demonstrations; the fixed 8B
  executor applies it to H. Blind and shuffled-label controls are subtracted.
- C3: the same decoder sees four-level quantized source `P(YES)` values for the eight examples.
  Canonical and shuffled-label C3 cells are run and evaluated against the same independent H.
  The observed C3 rung is cheap; an exact 65,536-state C3 structural cap is not present in the
  binary v13 cache and is explicitly reported unavailable. Obtaining it would require 65,536 new
  decoder inductions per panel, so it is not represented as a CPU-only re-slice.
- Native-v13 robustness: saved six-demo state/rule/execution cells are re-evaluated against the
  independent reference. The report leads with the exact 4,096-pattern structural cap, and adds
  a selection-preserving 10,000-shuffle null, Miller--Madow sensitivity, raw-lift sign/Wilcoxon
  tests, ordinary/balanced accuracy, majority baseline, and an exact structural accuracy cap.
- A reference survives only when all hidden anchors pass and the stable 10,000-bootstrap one-sided
  95% lower bound for pairwise agreement above pooled-marginal chance is positive. Failed metrics
  receive explicit void artifacts. Fleiss kappa is always reported with agreement and base rates.
  Dawid--Skene attenuation is a sensitivity analysis only,
  because repeated passes from one judge model are not independent raters.

The ladder is Phase A/B. Controlled scaling, unconditional bounded GEPA tuning, and nested
`|Omega|={1,2,3,5,8}` composition follow under the same release commit.

## Code and artifacts

- Driver: `methods/metric_implementer/experiments/ceiling_ladder.py`
- Guarded launcher: `scripts/run_ceiling_ladder_sk3.sh`
- Focused tests: `methods/metric_implementer/tests/test_ceiling_ladder.py`
- Remote code root: `/lfs/skampere3/0/alexspan/cr3-v14.1-roadmap/code`
- Remote output root: `/lfs/skampere3/0/alexspan/cr3-v14.1-roadmap/outputs`
- Tier-B manifest: `/lfs/skampere3/0/alexspan/cr3-v13.1/manifests/tier_b.json`
- Native v13 root: `/lfs/skampere3/0/alexspan/cr3-v13.1/outputs/tier_b/lanes`

The output root is resumable at whole-metric granularity. Completion markers are
`constructor/manifest.json`, `executor/manifest.json`, `native_v13_robustness_report.json`, and
the final `report.json`.

## Execution order

1. CPU freeze on sk3.
2. Copy only the frozen design bundle to the local machine, run the independent Sonnet reference,
   and copy the reference bundle back.
3. Run the native-v13 CPU robustness audit before any new model inference.
4. Ground-truth allowed devices once, then run the guarded 70B constructor on physical GPU 5.
5. Run the fixed-8B executor on physical GPU 7; GPUs 1/2/3/4 remain untouched.
6. Run CPU aggregation and issue the A/B decision report before launching any later phase.

## Ordered continuation after the A/B report

The A/B result chooses the headline rung but does not stop the later phases.

7. Phase C measures the controlled decoder ladder with identical panels, executor, templates,
   probes, and metric rows: Llama-3.1-8B, Mistral-Small-24B, Qwen2.5-32B, and
   Llama-3.3-70B, in both MCQ and behavioral channels. The clean within-family comparison is
   Llama 8B→70B. The previously quoted unmatched 13.7× comparison remains withdrawn.
8. Phase D runs the bounded v14 GEPA implementation unconditionally.
   Tune MCQ as well as both behavioral arms, use C3 only if its measured gain over C2 is
   supported, and always report tuned and untuned values.
9. Phase E composes targets at `|Omega| = 1, 2, 3, 5, 8` under both declared conjunction and
   weighted-sum compilers. Report exact cap, achieved tuned/untuned values, C0/C1/C2/C3,
   selection-preserving permutation percentile, and accuracy at each size; fit `gamma_V` from
   existing ledgers on CPU in parallel.

## Hard GPU rule

sk3 physical GPUs 1, 2, 3, and 4 are permanently forbidden. The launcher accepts exactly one of
0, 5, 6, or 7 for a GPU phase, pins PCI-bus ordering and vLLM spawn mode, uses durable `/lfs`
HOME/HF cache paths, and holds a per-device `flock` across `exec`. Never use `pkill` or `killall`;
if intervention is necessary, inspect ownership and terminate only specifically verified PIDs.
The two-lane run uses GPUs 5 and 7 only; nothing runs concurrently on sk1 or sk2.

## Sparse monitoring

Use one combined SSH observation rather than polling loops. Check the selected PID's command,
elapsed time, state, physical-device environment, recent log tail, and completed metric count in
one call. Do not touch or enumerate prohibited devices.

```bash
ssh sk3 'pid=$(cat /lfs/skampere3/0/alexspan/cr3-v14.1-roadmap/run.pid); \
  ps -o user,pid,ppid,etimes,stat,args -p "$pid"; \
  tr "\000" "\n" </proc/"$pid"/environ | rg "^CUDA_VISIBLE_DEVICES="; \
  find /lfs/skampere3/0/alexspan/cr3-v14.1-roadmap/outputs/constructor \
    -name constructor.json 2>/dev/null | wc -l; \
tail -n 40 /lfs/skampere3/0/alexspan/cr3-v14.1-roadmap/run.log'
```

# FAST/CERT scoring lanes

The two lanes always use separate output roots. FAST is screening-only and emits
`results.parquet`, `screening_summary.parquet`, and `fast_permutation_nulls.npz`;
it never emits certificates. CERT is a fresh measurement population and remains
the only input accepted by the release audit.

CPU freeze for a wide FAST population:

```bash
python -m methods.metric_implementer.experiments.run_v14_value_campaign --phase design \
  --scoring-lane fast --metrics-manifest "$METRICS" \
  --probe-extension-root "$EXTENSIONS" --out-root "$FAST_ROOT" \
  --run-sha "$RELEASE_SHA"
python -m methods.metric_implementer.experiments.run_v14_value_campaign --phase seed-freeze \
  --out-root "$FAST_ROOT" --template-freeze "$FAST_ROOT/template_freeze.json"
```

Run the three constructor-family stages, the fixed executor stage, then CPU
aggregation using the existing phase entry points. FAST aggregation uses 200
selection-preserving permutations, K=6, four menu permutations, and only states
realized by the frozen candidate bank (plus required shuffled-control states).

Freeze promotion from the one-row-per-metric screening summary:

```bash
python -m methods.metric_implementer.experiments.run_v14_value_campaign \
  --phase promote --out-root "$PROMOTION_ROOT" --run-sha "$RELEASE_SHA" \
  --fast-results "$FAST_ROOT/screening_summary.parquet" \
  --promotion-manifest "$PROMOTION_ROOT/promotion_manifest.json" \
  --top-k-per-task 3 --figure-metric-keys "${FIGURE_METRICS[@]}"
```

The CERT design must be created beneath a different root. It consumes only the
promoted metric identities; no FAST score, cache, state table, or reference is
copied:

```bash
python -m methods.metric_implementer.experiments.run_v14_value_campaign \
  --phase design --scoring-lane cert --metrics-manifest "$METRICS" \
  --probe-extension-root "$EXTENSIONS" --out-root "$CERT_ROOT" \
  --promotion-manifest "$PROMOTION_ROOT/promotion_manifest.json" \
  --run-sha "$RELEASE_SHA"
```

CERT then follows the independent-reference, hidden/planted-anchor,
preregistration, 10K-permutation, exact-cap-where-enumerable workflow below.
