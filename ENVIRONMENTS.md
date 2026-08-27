# Environments

Source: cross-box code-integrity audit, `outputs/analyses/code_integrity_20260808/divergence_report.md`
(environment inventory table, captured 2026-08-08).

| box | env | purpose | python | vllm | torch | transformers | dspy | numpy |
|---|---|---|---|---|---|---|---|---|
| sk3 | `envs/ai_usage` | primary metric_implementer offline-batch scoring env | 3.11.15 | 0.17.0 | 2.10.0 | 4.57.6 | 3.1.3 | 2.2.6 |
| sk3 | `envs/gemma4` | dedicated Gemma-4 scoring env (metric_seam pilot, taste_decomposition closure) | 3.11.15 | 0.23.0 | 2.11.0 | 5.12.1 | — | 2.3.5 |
| sk2 | `datasets/prompt-optimality-test/.venv` | orchestration/client only — no local torch/vllm/transformers (confirmed via import), calls out to a running vLLM OpenAI-compatible server | 3.11.14 | — | — | — | 3.2.1 | 2.4.6 |
| sk2 | `envs/vllm0251` | server-side env that actually hosts inference (notes/2026-07-19: "server env is envs/vllm0251, record for all future sk2 serves"); `pip show` returns nothing (non-pip-tracked install) — versions via `import` | 3.11.14 | 0.25.1 | 2.11.0+cu130 | 5.14.1 | absent | 2.3.5 |
| sk2 | `envs/gemma4` | dedicated Gemma-4 scoring env, matches sk3's gemma4 exactly | 3.11.15 | 0.23.0 | 2.11.0 | 5.12.1 | — | 2.3.5 |
| sk1 | `datasets/prompt-optimality-test/.venv` | self-contained env — client + server-capable stack bundled together (unlike sk2's split) | 3.12.3 | 0.25.1 | 2.11.0 | 5.14.1 | 3.2.1 | 2.3.5 |

## Notes (from the audit)

- sk3's `envs/gemma4` and sk2's `envs/gemma4` are version-identical — consistent Gemma-4 stack
  across both boxes.
- sk1's venv and sk2's `envs/vllm0251` are package-version-identical (vllm 0.25.1 / torch 2.11.0
  / transformers 5.14.1) but on **different Python minors** (3.12.3 vs 3.11.14) and packaged
  differently (bundled vs split client/server).
- sk3's `envs/ai_usage` (metric_implementer's main scoring env) runs a full major version behind
  the prompt-optimality-test stack: vllm 0.17.0 vs 0.25.1, and **transformers 4.57.6 vs 5.12–
  5.14.x** — a major-version jump (4→5) that changes some tokenization/chat-template defaults.
  Not necessarily a bug (metric_implementer may not need the newer stack) but worth knowing if
  any code is ever shared across the two campaigns.
- The sk2 vLLM server env is `envs/vllm0251` — the campaign `.venv` is client-only and contains
  no vllm.

## Freeze rule

Environments are measurement instruments. They are pinned and documented here; they are never
upgraded or modified mid-campaign. New needs get a NEW env, added to this table. The sk2 vLLM
server env is envs/vllm0251 — the campaign .venv is client-only and contains no vllm.
