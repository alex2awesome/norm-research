# reference_gepa_pr.py -- mechanics summary

Source: `sk3:/lfs/skampere3/0/alexspan/scripts/llama_norm_extraction/gepa_pr.py` (736 lines,
fetched read-only 2026-07-06, copied verbatim into `reference_gepa_pr.py` for study). This is
the production GEPA loop for the norm-extraction pipeline (Gemma-31B extractor + Qwen/GLM
judge + GLM mutator) across ~20 corpora (press_releases, legaladvice_uk, humor_multi, math_se,
mathlib, various legal/patent corpora, ...). Our harness borrows its GEPA-loop shape and a few
specific conventions but is a much simpler, single-purpose reimplementation (one Gemma call
per document producing a 0-10 score, not multi-signal extraction+judging).

## The GEPA_CORPUS env trap (read this before ever running the reference script)

```python
CORPUS = os.environ.get("GEPA_CORPUS", "press_releases")   # line ~30
```

Corpus selection is **exclusively** via the `GEPA_CORPUS` env var, resolved at **import time**
(module-level global, not inside a function). The `mode` positional argument that several
subcommands take (`article_only` / `article_first` / `combined`) looks like it might select the
corpus -- it does not. It only controls how press_releases' dual-text (article vs. press
release) is framed into the prompt for that one corpus. If you run `python3 gepa_pr.py gen ...
some_mode ...` without first exporting `GEPA_CORPUS=math_se`, it silently loads the
**press_releases** config/data/eval-pairs/judge-sys-prompt regardless of what `mode` string you
pass, and every downstream comparison is cross-contaminated with the wrong corpus. `cmd_run`
(the `run` driver subcommand) protects itself by doing `os.environ["GEPA_CORPUS"] = mode` right
at the top -- i.e. for `cmd_run` specifically, whatever you pass as the "mode" positional IS
also (re)used to set `GEPA_CORPUS` before spawning the `gen` subprocess, which is a slightly
confusing overload of the same string for two different roles (dual-text framing tag most of
the time, corpus-selector for `cmd_run`'s own bookkeeping) -- but every other subcommand
(`gen`, `judge`, `judge_corpus`, `mutate` run standalone) requires you to export `GEPA_CORPUS`
yourself beforehand; it is never inferred from any other CLI argument.

Our harness has no analogous trap: `state.json` stores task/aid explicitly per criterion and
every script (`build_round.py`, `ingest_round.py`, `propose.py`, `eval_final.py`) resolves the
task from that, never from an ambient env var.

## Loop shape (`cmd_run`, the top-level driver)

Round 0 = baseline: `gen` (Gemma vLLM, offline batch, temperature 0.0) over the small eval
slice, then `judge` (either Qwen vLLM offline-batch for full runs, or `cmd_judge_glm` -- the
same I/O contract via the GLM-5.2 subscription API, 0 GPU -- for the GEPA eval loop
specifically, since the eval slice is small and an API judge is cheap/fast there). Each
subsequent round: `mutate` (GLM API call, in-process, 0 GPU) -> `gen` -> `judge` -> accept-if-
better. `_drain_gpu()` polls `nvidia-smi` between the alternating Gemma/Qwen vLLM subprocess
steps because vLLM's teardown releases GPU memory asynchronously and the next engine's init
can OOM on the stale allocation -- not relevant to us (no live judge model in our loop; we
score once per round against the ALREADY-COMPUTED, frozen `ctx["judge"]` verdict, no GPU judge
step at all).

Accept-if-better: keep the new mutant only if `precision >= FLOOR` AND `F1(coverage,
precision)` improves over the current best (or, in `OBJECTIVE=yield` mode, `n_good` count
improves subject to a looser floor). This monotone best-tracking is the same idea as our
`state.json["criteria"][key]["best"]` (argmax dev rho across rounds, tracked incrementally by
`ingest_round.py`), just on a different metric (F1 of two judge-derived rates vs. our single
Spearman rho).

## Reflective feedback (`cmd_mutate`) -- what's worth borrowing, what we tightened

`cmd_mutate` shows the GLM mutator: (a) the CURRENT prompt's semantic sub-fields only (`role`,
`inline_evidence_example`, `polarity_hint` -- everything else in the cfg, including the `task`
name and structural JSON, is held fixed and the code re-merges only the touched keys back in);
(b) aggregate stats (coverage, validity-precision); (c) up to `MIN_EXAMPLES=8` concrete FAILURE
examples: `{signal_text, passage_text, faithful, valid, reason}` -- **this does include the
judge's own per-signal binary verdicts** (faithful=0/1, valid=0/1) as labels on synthetic
extracted candidates, not a continuous ground-truth score on the source document itself. A
COLD-START branch (extractor returned literally zero signals) swaps the failure block for raw
positive-example texts instead, since "here are 8 failures" is meaningless when there's nothing
extracted yet.

**Our constraint is stricter**: no judge-derived number, binary or continuous, may appear in
the proposer prompt at all -- only rank-order language ("scored too HIGH/LOW relative to
peers") plus a doc snippet. `ingest_round.py` computes this by comparing the model's own score
ranks to the judge's score ranks (`certificates.ranks`) over the fixed 40-item TRAIN dev set,
and only ever surfaces the sign/magnitude of the RANK gap, never a judge value. The dev-rho
*history trend* (an aggregate correlation coefficient, not a per-item label) is shown, matching
the seam-note spec's explicit instruction to include "rho history" in the feedback -- this
reads as sanctioned because it is exactly analogous to `cmd_mutate`'s aggregate coverage/
precision numbers, not a per-document label.

Robust JSON parsing worth keeping: strip `<think>...</think>`, strip markdown code fences,
`json.loads` first with a `json_repair` fallback (we skip the fallback dependency entirely --
"dependency-light" constraint -- and instead just discard-and-keep-old-prompt on a parse
failure, logging it; this is safe because our prompt state is idempotent/resumable turn to
turn, unlike the reference's single mutable cfg file per round).

`glm_call`: raw `urllib.request` POST to `https://api.z.ai/api/anthropic/v1/messages`,
`x-api-key` header (not `Authorization: Bearer`), `anthropic-version: 2023-06-01`, JSON body
`{model, max_tokens, messages:[{role:user,...}], system, temperature}`. On HTTP 429 (quota) it
rotates between two key files (`KEYFILES` list, primary + `-alexander-spangher` account) so a
quota exhaustion on one account fails over to the other. Our `propose.py::glm_call` mirrors the
request shape exactly (same endpoint/headers/body) but reads a single ordered fallback list of
LOCAL key file candidates (`common.KEY_PATHS`) rather than doing live 429-triggered rotation --
appropriate for our much smaller call budget (<=48 total vs. the reference's per-corpus-deploy
scale).

## Structural discipline worth calling out explicitly

The reference NEVER lets the mutator touch structural/identity fields (it merges only
`role`/`inline_evidence_example`/`polarity_hint` back into the existing cfg dict, and always
sets `cfg2["task"] = cfg["task"] + "_mut"` itself rather than trusting the model to preserve
naming). We copied this discipline as our #1 correctness safeguard: the proposer is asked to
revise only the scoring-guidance BODY text; the literal `<<<DOCUMENT>>>` marker and the
`SCORE: <integer 0-10>` reply-format instruction are contract elements the harness enforces
itself (`common.build_doc_prompt` always appends the footer; `propose.py::parse_prompt_json`
rejects -- and falls back to keeping the unchanged prompt for -- any GLM reply that drops the
marker). A GEPA rewrite can shrink dev rho, but it can never break the scorer's parse regex.
