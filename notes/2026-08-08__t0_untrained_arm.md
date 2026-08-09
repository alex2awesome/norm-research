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

**Kept-box reproduction.** 14 of 16 cells reproduce their published `VA_nl` *and* `VAT_nl` to within 1e-4 on the kept box (mirror cells on the mac, the scale-up-wave-C / mirror-2 cells on sk3) — the two-box design recovered the ledger exactly. Two cells fail on **both** boxes and are flagged in `box_choice`: `press_verdict` (its ledger was produced under scikit-learn 1.9.0 on Darwin-arm64; neither available box is that combination — kept sk3, |Δ VA_nl| .0015) and `code_v3` (kept the closer box; its canonical readout is within-repo, and its pooled row is marked ‖ = POOLED_DO_NOT_QUOTE in the master ledger too). Neither flag touches the T₀ comparisons, which are all computed **within one run on shared folds**.

| field | cell | n_E | T₀ | T | VA_nl | VAT₀_nl | VAT_nl | (VAT₀−VA) est [CI] P | (VAT−VAT₀) est [CI] P | (T₀−T) est [CI] P |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| Peer review | `peer_verdict` | 1244 | .5573 | .7769 | .6684 | .6717 | .7415 | +.0090 [−.0060,+.0234] 0.89 | +.0604 [+.0334,+.0881] 1.00 | −.2196 [−.2583,−.1785] 0.00 |
| Peer review | `peer_curation` | 1571 | .5251 | .5936 | .5286 | .5302 | .5542 | +.0077 [−.0101,+.0245] 0.79 | +.0263 [−.0030,+.0559] 0.96 | −.0685 [−.1093,−.0251] 0.00 |
| Peer review | `peer_revealed` | 478 | .4988 | .8842 | .6554 | .6518 | .8478 | +.0042 [−.0157,+.0238] 0.66 | +.2010 [+.1507,+.2508] 1.00 | −.3855 [−.4410,−.3289] 0.00 |
| Regulatory (N&C) | `nc_responded` | 1904 | .4310 | .8167 | .7912 | .7947 | .8319 | +.0052 [−.0046,+.0149] 0.83 | +.0365 [+.0170,+.0557] 1.00 | −.3857 [−.4259,−.3444] 0.00 |
| Regulatory (N&C) | `nc_outcome` | 1417 | .5379 | .6238 | .6121 | .6125 | .6227 | +.0070 [−.0070,+.0209] 0.82 | +.0153 [−.0028,+.0337] 0.95 | −.0859 [−.1259,−.0431] 0.00 |
| Regulatory (N&C) | `nc_agree` | 1009 | .5637 | .6034 | .5627 | .5615 | .5713 | +.0195 [−.0032,+.0409] 0.95 | +.0135 [−.0189,+.0482] 0.78 | −.0397 [−.0941,+.0236] 0.09 |
| Creative writing | `cw_community` | 7008 | .5211 | .7921 | .6652 | .6668 | .7869 | +.0016 [−.0015,+.0048] 0.85 | +.1216 [+.1098,+.1334] 1.00 | −.2710 [−.2876,−.2542] 0.00 |
| Humor | `hashtagwars_verdict` | 924 | .5096 | .7315 | .5290 | .5270 | .6454 | +.0007 [−.0216,+.0193] 0.56 | +.1107 [+.0434,+.1776] 1.00 | −.2219 [−.2950,−.1503] 0.00 |
| Humor | `cap_finalist` | 1055 | .4824 | .6124 | .5806 | .5897 | .6077 | +.0374 [+.0126,+.0616] 1.00 | −.0025 [−.0349,+.0306] 0.46 | −.1299 [−.1979,−.0633] 0.00 |
| Humor | `cap_crowd` | 2190 | .5008 | .5554 | .5831 | .5815 | .5920 | −.0019 [−.0120,+.0076] 0.33 | +.0067 [−.0065,+.0205] 0.83 | −.0546 [−.0866,−.0219] 0.00 |
| Humor | `jokes_community` | 3163 | .4884 | .7469 | .6888 | .6887 | .7375 | +.0021 [−.0030,+.0104] 0.71 | +.0469 [+.0290,+.0602] 1.00 | −.2585 [−.2817,−.2339] 0.00 |
| Math | `mathse_accepted_verdict` | 2600 | .4957 | .6439 | .5737 | .5736 | .6196 | −.0024 [−.0109,+.0059] 0.29 | +.0443 [+.0241,+.0642] 1.00 | −.1482 [−.1765,−.1203] 0.00 |
| Math | `mathse_vote_score` | 2326 | .4992 | .6538 | .6107 | .6130 | .6558 | +.0005 [−.0075,+.0078] 0.55 | +.0433 [+.0258,+.0609] 1.00 | −.1546 [−.1793,−.1291] 0.00 |
| Math | `aops_curation` | 5202 | .5727 | .7806 | .7705 | .7685 | .7851 | −.0010 [−.0039,+.0022] 0.28 | +.0171 [+.0063,+.0279] 1.00 | −.2079 [−.2409,−.1737] 0.00 |
| Software code | `code_v3` ‖ | 11452 | .5153 | .6933 | .7043 | .7031 | .7537 | −.0095 [−.0189,−.0012] 0.01 | +.0545 [+.0319,+.0833] 1.00 | −.1781 [−.2698,−.0744] 0.00 |
| Journalism/press | `press_verdict` | 605 | .4935 | .7744 | .6795 | .6713 | .7459 | +.0019 [−.0138,+.0215] 0.60 | +.0807 [+.0472,+.1296] 1.00 | −.2809 [−.3551,−.1724] 0.00 |

‖ = `POOLED_DO_NOT_QUOTE` (carried over from the master ledger).

### How much of the fused gain needs the community's labels?


| cell | VAT₀−VA_nl | VAT−VA_nl | share of the fused gain reached WITHOUT training | T₀ score collapse? | box |
|---|---:|---:|---:|---|---|
| `peer_verdict` | +.0033 | +.0731 | +5% | no | mac |
| `peer_curation` | +.0016 | +.0256 | +6% | no | mac |
| `peer_revealed` | −.0035 | +.1924 | -2% | no | mac |
| `nc_responded` | +.0035 | +.0407 | +9% | no | mac |
| `nc_outcome` | +.0005 | +.0107 | +4% | no | mac |
| `nc_agree` | −.0011 | +.0086 | -13% | no | mac |
| `cw_community` | +.0016 | +.1217 | +1% | no | mac |
| `hashtagwars_verdict` | −.0020 | +.1164 | -2% | YES | mac |
| `cap_finalist` | +.0091 | +.0271 | +34% | no | mac |
| `cap_crowd` | −.0016 | +.0089 | -18% | no | mac |
| `jokes_community` | −.0002 | +.0487 | -0% | no | mac |
| `mathse_accepted_verdict` | −.0001 | +.0460 | -0% | no | sk3 |
| `mathse_vote_score` | +.0023 | +.0451 | +5% | no | sk3 |
| `aops_curation` | −.0020 | +.0146 | -13% | no | sk3 |
| `code_v3` | −.0012 | +.0494 | -2% | no | mac |
| `press_verdict` | −.0082 | +.0664 | -12% | no | sk3 |

### Cross-cutting read

1. **The untrained base model is at chance on every community-preference variable in
   the grid.** T₀ spans **.4310 – .5727** across 16 cells, mean **.5108**; it clears
   .53 on only four cells and never reaches .58. This holds on cells where the trained
   T is very strong (`peer_revealed` T .8842 / T₀ .4988; `nc_responded` T .8167 /
   T₀ .4310; `cw_community` T .7921 / T₀ .5211). **T₀ − T is negative on 16/16**, with
   P(>0) = 0.00 on 15 of 16 (`nc_agree`, the weakest-T cell, is the lone exception at
   P=.09). Zero-shot Llama-3.1-8B reading the same document under the same token budget
   simply does not know what these communities reward.

2. **Fusion without training is worth essentially nothing.** VAT₀ − VA_nl spans
   **−.0095 to +.0374**, median **+.0003**. It is positive at P≥.95 on exactly **one**
   cell (`cap_finalist` +.0374 [+.0126,+.0616] P=1.00; `nc_agree` +.0195 P=.95 is the
   only other one at threshold) and **significantly negative on one** (`code_v3`
   −.0095 [−.0189,−.0012] P=.01 — adding a chance-level column to the bank actively
   costs you). On the other 14 the CI straddles zero. Contrast VAT − VAT₀, which is
   positive at P≥.95 on **13/16** and P=1.00 on 11.

3. **So the fused gain is bought by label-training, not by the LLM prior.** Across the
   16 cells the untrained share of the fusion gain (VAT₀−VA)/(VAT−VA) has median
   **≈0%**; it is under 10% on 12 cells and negative on 8. The single real exception is
   `cap_finalist` at **+34%** — and that is the cell whose documents are one-line
   captions, i.e. the one place where "is this a good contest entry?" is answerable
   from generic prior alone. `cap_crowd`, the *same corpus* under a crowd-median rather
   than an editor label, gets **−18%**: the prior helps pick what an editor would
   shortlist, not what the crowd voted up.

4. **This is a clean falsification of the "the dense arm is just an LLM reading the
   text" reading of Δ_beyond.** Every cell's Δ_beyond (T − VA_nl) survives with the
   *trained* column and evaporates with the *untrained* one, at identical rows, folds,
   bank, and token budget. The only thing that changed is whether the 8B encoder saw
   the community's labels. Taste, as this program measures it, is not latent in a
   frontier-adjacent prior waiting to be prompted out — it is learned from the
   community.

5. **Scale caveat, stated plainly.** T₀ is one 8B base checkpoint with one frozen
   question per cell. This bounds *that* instrument, not "LLM priors" in general; a
   larger base model or a prompt-optimised elicitation could move T₀. What the arm does
   establish is that the fusion gains in the master ledger are **not** attributable to
   generic document-reading capacity at the dense arm's own scale, which is the
   confound the arm was registered to kill.

6. **One instrument-quality caveat.** `hashtagwars_verdict` collapsed (7 distinct
   P(Yes) over 924 rows) and four cells saturate at P(Yes)=1.0000 in the median. Where
   T₀ is tie-dominated its AUC is pulled toward .5 mechanically, so those cells' T₀
   values are a floor, not a measurement. This *strengthens* rather than weakens
   conclusion 2 for the non-collapsed cells, but `hashtagwars_verdict`'s T₀ .5096
   specifically should be read as "uninformative", not "chance".

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
