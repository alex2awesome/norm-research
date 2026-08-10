# E-S1 planted end-to-end kill-switch — DESIGN + pre-registered pass criteria

*2026-07-03. Written BEFORE any pipeline verdicts were computed (codegen agents launched
blind the same hour; nothing evaluated yet). The point of this file is that the pass bars
below cannot be accused of being fitted to results.*

## What is being tested

The seam pipeline claims to *measure placement* — which medium (code / LLM / op-typed tool)
a criterion channel needs. Every wild-corpus verdict so far is unfalsifiable in the
uncomfortable direction ("a105 is A-layer" could mean "our codegen is weak"). This run seeds
the pipeline with seven criteria of KNOWN placement, laundered as ordinary aspects p901–p907,
and scores the pipeline's placement verdicts against design truth.

## The plants

| id | name (as agents see it) | truth type | ground truth |
|---|---|---|---|
| p901 | Quantitative support | code | numeric-token density /1000w, decile-ranked (ρ with length −0.20) |
| p902 | Temporal anchoring | code + computation op | distinct day-precision dates, reference multi-format parser, min(n,5)×2 |
| p903 | Corpus distinctiveness | code + EVIDENCE op | 10 − decile(max TF-IDF sim to rest of corpus), via ops.retrieve_similar itself |
| p904 | Voice diversity | mixed (LLM field + code predicate) | Claude-labeled distinct quoted named speakers, map {0:0,1:4,2:7,3+:10} |
| p905 | Authentic authorship | a-layer (directional) | Claude holistic 0–10 |
| p906 | Persuasive cadence | null | channel = pure noise behind a plausible description |
| p907 | Comprehensive detail | code + KNOWN noise | word-count deciles + noise calibrated to rel1≈0.60 |

Two arms share the descriptions:
- **Arm S** — synthetic channels: pass1/pass2 = truth + calibrated noise (rel1≈0.85 for
  p901–p904, 0.80 p905, 0.60 p907, ~0 p906). Tests codegen→hybrid→gates→certificates→S6
  with the generating function exactly known.
- **Arm J** — the descriptions judged by the real Gemma-4-31B 2-pass protocol
  (results_judge.jsonl). Tests the full loop including judge operationalization slack;
  ρ(judge, truth) is itself a readout per truth type.

Blinding: codegen and improver agents receive the identical pack format used for real bank
aspects; nothing marks these as plants; truth.json is never exposed to any agent.

## Pre-registered pass criteria (Arm S = hard; Arm J = hard only where marked)

- **p901 (code):** best code rung (description-compiled or evolved, no LLM fields) reaches
  ≥85% of the S1 ceiling on the held-out split; S6 verdict CODE; if a hybrid is evolved, its
  LLM Shapley share ≤0.05. **Arm J hard too** — if the pipeline cannot recover THIS from the
  real judge, the pipeline is broken.
- **p902 (code+comp-op):** placement CODE (same 85% bar). Conditional: if the winning
  program calls computation ops (normalize/extract_dates), the S5 ablation marginal must be
  ≥0 with P≥.9; any used op must classify as COMPUTATION under the
  stronger-executor-without-tool test (an LLM can count dates unaided — no new information
  in Z). Inlining the parsing instead of calling ops is a legitimate CODE path, not a fail.
- **p903 (evidence-op):** code-only rungs must NOT certify at ≥85% ceiling (if one does →
  ALARM: the plant's information is corpus-external by construction, so a within-document
  certification means the harness leaks or the gate over-passes). Hybrid using
  ops.retrieve_similar must reach ≥85% ceiling; S5 op ablation certified positive (P≥.95);
  op-type test must come out EVIDENCE (executor-without-tool stays low).
- **p904 (mixed):** code-only plateaus below ceiling; hybrid with an LLM field must beat the
  best code-only rung by ≥0.10 test ρ AND reach ≥70% of ceiling (bar is softer than 85%
  because the recovery-time extractor is Gemma while truth labels are Claude — extractor
  noise is part of the design); S6 verdict MIXED (both media material in the Shapley split).
- **p905 (a-layer, directional):** no CODE certification at any rung (code-only must fail
  the G1 gate); if any hybrid passes, its LLM share must dominate (≥0.5). We do NOT
  pre-commit that the hybrid must pass — only that the pipeline must not call this codable.
- **p906 (null):** measured rel1 ≈ 0 → attenuation ceiling degenerate; the pipeline must
  flag the channel unreliable and certify NOTHING (all rungs and hybrids: P(gate) < .5;
  expected ≈ 0). Any certified pass = alarm. **Arm S only** (the Gemma channel for this
  description is whatever Gemma thinks cadence is — real, not null).
- **p907 (ceiling formula check):** the ORACLE program (the generator itself, run by us —
  never given to agents) must land within |Δ| ≤ 0.05 of the S1 predicted ceiling against the
  noisy channel; no rung or hybrid may EXCEED the ceiling beyond bootstrap noise (0
  violations at 95%); placement CODE.

Kill-switch verdict = the conjunction of the hard criteria; report the full per-plant
placement-accuracy table either way, including any alarms. All gate probabilities via the
Rung-3 bootstrap (B=2000), never point estimates.

## Protocol stages (mirrors the real pipeline exactly)

1. build_killswitch.py phase 1/2 → channels_synth.jsonl, truth.json, prompts_judge.jsonl,
   label_chunks (Claude labels → claude_labels.json, merged by merge_labels.py).
2. Gemma Arm-J batch on sk3 (queue job 60) → results_judge.jsonl.
3. Blind description-compiled codegen: 3 rungs × 7 plants → killswitch/codegen/ (agents,
   same contract as codegen_claude).
4. run_killswitch_flavors.py → code scores on the 250 canonical texts.
5. Improver packs (same format as gen_packs_v2) → blind hybrid agents → programs_ks/.
6. LLM-field extraction batch for declared fields (queue).
7. eval_killswitch.py: seam tables (both arms), gates with bootstrap, S1 ceilings,
   S5 ablations, oracle rungs (p901/902/903/907 truth functions run as programs),
   op-type tests, S6 placement verdicts → killswitch_report.json.
8. Placement-accuracy table vs this file's criteria → notebook §6.

## Blinding incident + clean-room protocol (2026-07-03, before any evaluation)

The first codegen fleet's p907 agent disclosed that it had read `plants.py` (ground-truth
definitions, same package directory) while orienting; the original prompts forbade judge
scores/metric outputs but not repository exploration. Treat the ENTIRE first fleet's output
as potentially truth-exposed. Remediation, decided before any program was evaluated:
- first-fleet programs preserved in `codegen_disclosed/` (never evaluated as the primary rung);
- all 21 rungs regenerated by a CLEAN-ROOM fleet: prompt-only context, explicit instruction
  to read/list NO files, smoke tests on self-constructed strings, output = only the three
  module files (written to `codegen/`);
- the improver round inherits the same clean-room rule (pack content passed inline).
The kill-switch verdicts use the clean-room programs exclusively. (A disclosed-vs-clean-room
comparison is a free robustness readout, reported descriptively.)

## Notes

- Items = the v1 250 canonical texts (head5000+tail2500), same scoping channel as v1.
- p901 was switched from URL-count to numeric density BEFORE any channel was built
  (URL truth was 66% zeros — degenerate); recorded here for full disclosure.
- p906's "truth_distinct=1" builder warning is by design (dummy constant truth; channel
  is noise).
