# T0 / UNTRAINED-T FUSION ARM — all 16 master-ledger cells

Frozen design: `notes/2026-07-27__vat-run-registry.md`, entry
**"2026-08-08 — FROZEN DESIGN (before any scoring): UNTRAINED-T FUSION ARM"**.
This note is the run record. Logging into the registry / strict list is the
coordinator's, not this run's.

## What T₀ is

T₀ = the **base `meta-llama/Llama-3.1-8B`** checkpoint — the exact checkpoint the
program's LoRA dense T trains from (`methods/dense/train_reward_model.py`
`--base_model` default) — **zero-shot, no LoRA, no chat template** (the base
checkpoint carries `chat_template: false`), scored with **offline batch vLLM**
(one GPU, `envs/vllm_latest` = vLLM 0.24.0, `gpu_memory_utilization` 0.93,
`max_model_len` 1280, spawn + `CUDA_DEVICE_ORDER=PCI_BUS_ID`).

Elicitation, frozen ex ante in
`methods/taste_decomposition/fusion/t0_templates.json`
(sha256 `50c1a5a98f8ff506033e1f1fe2ab5644b97c668ff45c86f995a9dafbfc18a080`,
committed **before any score existed**, commit `a47d8fc`):

```
{question}

{document}

Answer Yes or No.
```

* one fixed one-sentence question per cell, naming that cell's preference
  variable; questions were derived from each cell's **label definition**
  (builder code + manifests), never from any label value or any score;
* the **document** is the string the trained T read, truncated to **1024
  tokens** (T's own `--max_length`) with the Llama-3.1-8B tokenizer, right side;
* score = **P(Yes)** = Yes-mass / (Yes-mass + No-mass) over the first generated
  token with the logits masked (`allowed_token_ids`) to the Yes/No variant set.

Realised single-token variant set (recorded per the freeze):
POS `Yes 9642 · " Yes" 7566 · yes 9891 · " yes" 10035 · YES 14331 · " YES" 14410`;
NEG `No 2822 · " No" 2360 · no 2201 · " no" 912 · NO 9173 · " NO" 5782`.

**No prompt was iterated after any score was seen.** Where a cell's score
distribution collapsed, that is recorded below and the template was left alone.

## Rows, fusion, readouts

* **Rows** = each cell's E population exactly as the master fused ledger's VAT
  arm used it, rebuilt by importing the ledger's own loaders
  (`direction1_mirror.py`, `direction1_mirror2.py`, the closure cell adapters)
  — never reimplemented. `n_E`, `n_groups_E` **and T itself** are asserted
  against `results/vat_fullgrid_<cell>.json` for all 16 cells; all 16 matched to
  the printed precision of the ledger.
* **Fusion** — `direction1_mirror.fit_arm` is called **twice** per cell on the
  identical `(family, VA_raw, y, groups)`, once with the trained dense column and
  once with the T₀ column. `L1.outer_folds` is a pure function of `(n, groups)`,
  so both calls share byte-identical folds; the run asserts the two calls'
  VA arms are bit-identical, which is what makes the paired bootstrap
  `VAT_nl − VAT₀_nl` legitimate. Grouped OOF `GroupKFold(5)` on the cell's
  canonical grouping unit, HistGB seeds {0,1,2} mean, same bank family
  (`clean_once` / `impute_perfold`) per cell as the ledger.
* **Bootstraps** — group-level paired, 2,000 draws, on the cell's grouping unit:
  `VAT₀−VA_nl`, `VAT_nl−VAT₀`, `T₀−T` (plus `VAT₀−T₀`, recorded).
* **Platform** — the ledger's own landmine (GroupKFold fold *membership* is
  sklearn-version *and* architecture dependent) applies. Every cell was fused on
  **both** boxes (mac Darwin-arm64 / sk3 Linux-x86_64 `envs/ai_usage`) and the
  box kept per cell is the one that **reproduces that cell's published VA_nl and
  VAT_nl within 1e-4**. The per-box deviations and the kept box are recorded in
  `results/t0_untrained_arm.json` under `box_choice`.

## Inventory first (reuse-before-rebuild)

No zero-shot base-model scores existed for any of the 16 cells anywhere in the
repo or on sk3 — searched for `t0`/`zero-shot`/`untrained`/`no-LoRA`/`p_yes`/
first-token-logprob artifacts, and for a base-model score column in the v2/v2_va
population files. The nearest thing,
`methods/metric_seam/llama_base_score_sk3.py`, does score a base checkpoint via
offline batch vLLM but reads **greedy text**, not logprobs, and is not keyed to
these cells. Scores were therefore generated fresh; **code** was reused:
`metric_implementer/vllm_backend.py`'s constrained-binary readout as the pattern
(with its chat-template call dropped, since the base checkpoint has none) and
the 1024-token truncation from `dense/score_eval_dense_v4.py`.

## T₀ score distributions (collapse check)

Analogue of the judge all-min check. `distinct` = number of distinct P(Yes)
values; `argmaxYes` = fraction of rows whose single most likely allowed token is
a Yes variant (a *different* statistic from P(Yes) > .5, because the Yes and No
masses are each summed over six surface forms).

| cell | n | median P(Yes) | min | max | distinct | argmaxYes | collapse |
|---|---:|---:|---:|---:|---:|---:|---|
| `peer_verdict` | 1244 | .4688 | .0081 | 1.0000 | 192 | 1.000 | no |
| `peer_curation` | 1571 | 1.0000 | .0004 | 1.0000 | 117 | 1.000 | no |
| `peer_revealed` | 478 | .4378 | .0091 | 1.0000 | 116 | 1.000 | no |
| `nc_responded` | 1904 | .5000 | .0000 | 1.0000 | 423 | .998 | no |
| `nc_outcome` | 1417 | .5312 | .0000 | 1.0000 | 344 | .999 | no |
| `nc_agree` | 1009 | .5467 | .0000 | 1.0000 | 241 | .997 | no |
| `cw_community` | 7008 | .5622 | .0009 | 1.0000 | 878 | 1.000 | no |
| `hashtagwars_verdict` | 924 | 1.0000 | .0000 | 1.0000 | **7** | .752 | **YES** |
| `cap_finalist` | 1055 | .3346 | .0000 | 1.0000 | 78 | .717 | no |
| `cap_crowd` | 2190 | .3486 | .0000 | 1.0000 | 85 | .722 | no |
| `jokes_community` | 3163 | .3923 | .0000 | 1.0000 | 152 | .761 | no |
| `mathse_accepted_verdict` | 2600 | .6654 | .0000 | 1.0000 | 399 | 1.000 | no |
| `mathse_vote_score` | 2326 | 1.0000 | .0373 | 1.0000 | 156 | 1.000 | no |
| `aops_curation` | 5202 | 1.0000 | .0203 | 1.0000 | 495 | 1.000 | no |
| `code_v3` | 11452 | .6077 | .0000 | 1.0000 | 1191 | 1.000 | no |
| `press_verdict` | 605 | .5320 | .0169 | 1.0000 | 126 | 1.000 | no |

**One collapse: `hashtagwars_verdict`** — 924 rows take only **7** distinct
P(Yes) values (short tweets inside an identical contest header; the base model
returns near-identical mass for almost every entry). Per the freeze the template
was **not** rewritten. Read that cell's T₀ and VAT₀ as tie-dominated.

Four further cells saturate at the top of the scale (median P(Yes) = 1.0000:
`peer_curation`, `mathse_vote_score`, `aops_curation`, and `hashtagwars_verdict`)
— leading-question saturation, not collapse, since they retain 117–495 distinct
values below the ceiling.

## Results

<!-- TABLE -->

## Artifacts

| what | where |
|---|---|
| frozen templates | `methods/taste_decomposition/fusion/t0_templates.json` |
| E-row exporter | `methods/taste_decomposition/fusion/t0_build_rows.py` |
| E rows + texts + checksums | `methods/taste_decomposition/fusion/t0_rows/<cell>.{npz,texts.jsonl.gz,meta.json}` |
| vLLM scorer | `methods/taste_decomposition/fusion/t0_score_vllm.py` |
| raw T₀ scores | `methods/taste_decomposition/fusion/t0_scores/<cell>.{jsonl.gz,meta.json}` |
| fusion | `methods/taste_decomposition/fusion/t0_fuse.py` → `t0_results/<cell>.{mac,sk3}.json` |
| merger | `methods/taste_decomposition/fusion/t0_merge.py` |
| **merged result** | `methods/taste_decomposition/results/t0_untrained_arm.json` |
| scoring logs (sk3) | `logs/t0_score.log`, `logs/t0_score_nc.log` |
| GPU ledger | GPU 6 claimed `2026-08-09T02:47:46Z`, released `02:59:35Z` |
