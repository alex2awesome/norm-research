# Technical retrospective replay v2

This additive lane selects expensive technical artifacts as seeded pipeline decisions and
measures their utility without pretending that a manual, mocked, oracle-conditioned, or
historical decomposition was originally discovered automatically. It uses no new external
supervision. The comparison reference remains the frozen LLM judgement already present in the
metric-seam tasks; certificate-only cases do not substitute accept/reject outcomes for it.

The implementation uses the canonical vocabulary in
`methods/metric_seam/reconstruction_v2.py` and records the reproducibility fingerprint from
`methods/metric_seam/environment_v2.py`.

## Four separate questions

1. **Articulability** is prompt-based: can an LLM prompt operationalize the articulated
   relation? Prompt repeatability is evidence on this axis.
2. **Verifiability** is code-based: can executable code check the relation and emit a scoped,
   replayable certificate or honest abstention?
3. **Isomorphic reconstruction** is agreement with the frozen LLM reference when both sides
   receive the same represented evidence. It can be measured for prompt, code, or hybrid
   programs.
4. **Constructive extension** is a different outcome: code directly certifies a relation on
   cases where the LLM reference disagrees or lacks the executable evidence. Higher
   correlation alone is never an extension certificate.

## Initial cases

The manifest replays seven bounded cases across four technical domains:

- Math a150: agent integration of a manually supplied SymPy capability, replayed as a
  relation-mismatch result. The algebra checker works on synthetic examples but covers zero
  real condition-to-operation licensing relations.
- Legacy code-review prototype a104: a mocked lookup of previously computed repository/test
  transitions from `f2p_mock/`. It measures what execution evidence would add without
  claiming that this module executes a repository. This is not the active coding census,
  whose h0 fleet is still pending.
- Legacy transplant prototype replay: the current probe classifies a frozen 800-row artifact
  produced by earlier repository/test executions; it does not launch 800 executions now. The
  frozen telemetry contains 65 pinned/partial/vacuous behavioral certificates across Python,
  Go, and Java, 571 indeterminate rows, and one duplicated row ID. This is likewise not the
  active coding census.
- Patents a34: an oracle-conditioned prior-art payload. It can establish reconstruction of
  an evidence-aware prompt reference, not autonomous prior-art discovery or pure-code
  verification.
- Peer-review a214: an abstract-only release-claim grounding program. It cannot certify that
  a repository exists or reproduces results.
- Peer-review claim/body checking: an explicit corpus guard. The existing deep design needs
  a results body, so it is ineligible on the current abstract-only corpus.
- Peer-review full-paper claim/body checking: a selected 2,400-paper evidence surface with
  1,957 non-empty bodies. A conservative code probe issues 429 positive numeric-recurrence
  certificates and treats zero matches as unresolved.

These corpus limits are part of the result. In particular, the historical diff-only code task
does not imply live compilation, Math.SE prose is not a formal-proof corpus, peer-review
abstracts do not support statistical recomputation, and the patent prior-art payload is marked
`oracle`. The frozen prior-execution telemetry and full-paper cases explicitly add the missing
evidence surfaces rather than silently attributing them to the old inputs.

## Provenance is not pipeline selection

Every case now has two orthogonal records:

- `discovery_mode` says how the historical artifact originated:
  `agentic|manual|mock|oracle|replay`.
- `pipeline_status=selected` and `selection_mode=retrospective_seed` say that this experiment
  selected it as a seeded pipeline decision for utility measurement.

Thus a manual SymPy decomposition, mocked execution interface, or oracle patent payload can
have measurable selected-pipeline utility. The provenance still limits the wording: seeded
selection is not historical automatic discovery, mock utility is mock-conditioned, and oracle
utility is oracle-conditioned.

Relation depth is reported on a frozen ladder:

0. surface lexical;
1. parsed document structure;
2. cross-span or cross-section relation;
3. formal solver or evidence graph;
4. environment/world execution.

## Run

From the repository root:

```bash
python -m methods.metric_seam.technical_replay.evaluate --check
python -m methods.metric_seam.technical_replay.fullpaper_probe
python -m methods.metric_seam.technical_replay.code_execution_probe
python -m methods.metric_seam.technical_replay.evaluate
python -m unittest methods.metric_seam.technical_replay.test_replay
```

The full run writes only to `outputs/metric_seam_pilot/technical_replay_v2/`:

- `manifest.snapshot.json`: exact evaluated manifest;
- `results.json`: resolved measurements, artifact hashes, provenance taints, canonical v2
  records, and environment fingerprint;
- `REPORT.md`: compact human-readable readout.
- `fullpaper_probe.json`: unlabeled science evidence coverage and claim/body certificates;
- `code_execution_probe.json`: unlabeled classification of frozen prior-execution telemetry.

The source manifest is `initial_manifest.json`. Each artifact has one of the frozen discovery
modes: `agentic`, `manual`, `mock`, `oracle`, or `replay`. The evaluator refuses missing modes,
missing objective axes, unknown artifact references, path escapes, and invalid derived
measurements. Missing corpus sections make an otherwise positive axis ineligible.

The full-paper source and execution CSV both contain historical accept/reject fields. The two
new probes explicitly ignore them. Utility means coverage, nondegeneracy, reconstruction
marginal against an existing frozen LLM reference, or code-native certificate yield—not
outcome prediction.

## What this replay does and does not claim

The initial replay supports selected-pipeline utility and several constructive evidence
extensions even where isomorphic reconstruction is imperfect or unavailable. It still permits
zero historical automatic-selection claims and zero confirmatory isomorphic-reconstruction
claims. The distinction is intentional: the machinery can be useful and deep without a false
story about how it was originally selected.

The report distinguishes an **evidence-surface extension** (code obtains a replayable relation
unavailable from the old text surface) from canonical **verifier-dominant constructive
extension** (a code certificate adjudicates item-level disagreement with the frozen LLM
reference). The prior-execution telemetry and full-paper cases support the former and code
verifiability; they do not invent the latter without a frozen disagreement set. As required by the Collins
asymmetry, no negative result permits a tacitness claim.

A future automatic-selection replay can freeze the candidate catalog, expose only criteria,
unlabelled `ctext`, contracts, and capability metadata to an agent, and then ask whether its
selection recovers the useful decomposition. That would measure discovery under the new
protocol. It would still not retroactively make the original manual/mock/oracle construction
agentic.

For new math runs, use the corrected `hybrids/ops_capability_v2.py`; the historical a150 replay
continues to hash the exact earlier capability it actually used.

Historical full-paper scoring scripts reference three generated NPZ files that are not present
in this checkout. The replay hashes and uses the available 2,400-paper JSONL evidence and code
scripts, but does not invent NPZ-derived reconstruction results. Once those matrices are
restored, they can be added as a separate frozen-reference readout without changing the current
certificate results.
