# GEPA-H2H harness (seam note Sec 9)

Head-to-head: **arm G** (a single GEPA-optimized Gemma-31B scoring prompt, one call/doc,
0-10 int) vs **arm H** (the frozen certified hybrid program: code + <=2 Gemma field
extractions, already scored in `../agentic_cert.json`), on the SAME 12 criteria / SAME
held-out test split / SAME judge.

12 criteria (battery/agentic_cert.json's `CANDS` minus `math.a132`):
press_releases a119 a115 a87 | creative_writing a90 a72 a99 a342 | math a198 a42 |
humor a351 a135 a153.

This directory is DESIGN + DRAFT: every script here is local-only python3 (json/pathlib/
random + battery_common; `propose.py` additionally uses stdlib `urllib` for the one live
API call type). **Nothing in this directory launches a GPU job or calls an LLM API except
`propose.py` without `--dry-run`.** GPU scoring is a separate step the orchestrator runs on
sk3 via the existing `queue2` job queue (`methods/metric_seam/pilot/gpu_queue2.sh`) --
these scripts only build the prompts files queue2 consumes and ingest the results it
produces.

Constraints this harness honors throughout: TRAIN-only until `eval_final.py` (the ONLY
script that reads `ctx["test"]`); judge scores from `ctx["judge"]` (own-verdict
reconstruction) never external human labels; no raw per-item judge number is ever placed
in an LLM prompt, only rank-residual language + aggregate rho history; GEPA calls capped
at 1/criterion/round.

## Files

| file | role |
|---|---|
| `reference_gepa_pr.py` | read-only copy of sk3's production GEPA loop (study reference) |
| `NOTES.md` | its mechanics + the GEPA_CORPUS trap + what we borrowed/tightened |
| `common.py` | shared constants (12 criteria, dev-set seed/size, prompt marker/footer, key paths) + `load_ctx` re-export |
| `init_state.py` | builds round-0 `state.json` (seed prompts + 40 fixed TRAIN dev ids/criterion) |
| `build_round.py` | `state.json` -> `gepa_round<r>_prompts.jsonl` (480 rows at any round) |
| `ingest_round.py` | scored results -> dev rho + rank-residual feedback, written into `state.json` |
| `propose.py` | GLM-5.2 reflective proposer (1 call/criterion/round); `--dry-run` prints, doesn't call |
| `eval_final.py` | `build` (freeze best-dev prompt -> TEST prompts file) / `eval` (test rho G vs H, paired boot, cost) |
| `state.json` | the loop's only persistent state (checked in per run, not gitignored -- it's small) |

## Orchestrator runbook

All commands run from this directory (`methods/metric_seam/battery/gepa_h2h/`) unless noted.
`$QDIR2` = `/lfs/skampere3/0/alexspan/norm-research/outputs/metric_seam_pilot/queue2` (sk3).

**Step 0 -- one-time init (local, already done this pass):**
```
python3 init_state.py          # -> state.json, round 0 for all 12 criteria
```
Refuses to run if `state.json` already exists (delete it first to hard-restart).

**Per round `r` (r = 0, 1, 2, ...), repeat:**

```
# 1. LOCAL: build this round's prompts file from state.json's current prompt/dev-ids
python3 build_round.py <r>
#    -> gepa_round<r>_prompts.jsonl  (480 rows: 12 criteria x 40 TRAIN dev items)

# 2. sk3: submit to the existing queue2 job queue (job format: one line
#    "<prompts_path> <out_path> [scorer] [pybin]"; omit scorer/pybin for the Gemma default)
scp gepa_round<r>_prompts.jsonl sk3:/lfs/skampere3/0/alexspan/norm-research/methods/metric_seam/battery/gepa_h2h/
ssh sk3 "echo '/lfs/skampere3/0/alexspan/norm-research/methods/metric_seam/battery/gepa_h2h/gepa_round<r>_prompts.jsonl \
/lfs/skampere3/0/alexspan/norm-research/methods/metric_seam/battery/gepa_h2h/gepa_round<r>_results.jsonl' \
> \$QDIR2/$(printf '%03d' <r>)_gepa_round<r>.job"
#    scorer defaults to gemma_score_v1.py in the gemma4 env (queue2's $GEMMA_SCORER/$GEMMA_PY) --
#    exactly right here: it expects "SCORE: <int|NA>" in the reply, which our footer guarantees.
#    Wait for queue2 to process it (watch $QDIR2/queue.log; job lands in $QDIR2/done/ on success).
scp sk3:/lfs/skampere3/0/alexspan/norm-research/methods/metric_seam/battery/gepa_h2h/gepa_round<r>_results.jsonl .

# 3. LOCAL: ingest -- computes dev rho + rank-residual feedback per criterion
python3 ingest_round.py <r> gepa_round<r>_results.jsonl

# 4. LOCAL: inspect the feedback that WOULD be sent (no API call, no state.json write)
python3 propose.py <r> --dry-run

# 5. LOCAL: the real GLM-5.2 call (1 per pending criterion, <=12 calls this round)
python3 propose.py <r>
#    -> state.json: round -> r+1, prompt -> GLM's revision, for every criterion that had
#       a round-r history entry and hadn't already been proposed through round r.
```

Then go back to step 1 with `r+1`.

## STOP conditions

- **Round budget**: run rounds `r = 0, 1, 2, 3` (4 dev-scoring rounds total: seed P0 through
  P3). After `ingest_round.py 3` + `propose.py 3`, **STOP the round loop** -- do not build a
  round 4 prompts file. That is exactly 4 proposer calls x 12 criteria = 48 GLM-5.2 calls in
  the worst case (the round-3 proposal produces a P4 that is deliberately never scored, since
  scoring it would be a 5th dev round beyond the stated cap -- it exists only because the loop
  structure calls the proposer unconditionally at the end of every round it processes).
  **Cheaper alternative, recommended by default** ("be sparing with GLM" -- monthly z.ai quota
  is shared across projects): skip `propose.py 3` entirely. Round 3's dev-scored prompt (P3)
  is then simply the last candidate considered; `eval_final.py build` still does the right
  thing (argmax dev rho over rounds 0-3, whichever that is). This caps spend at 3
  proposals/criterion = 36 calls total and forfeits nothing but a possible-but-unscored P4.
- **Per-round early stop**: if `ingest_round.py`'s printed table shows a criterion whose dev
  rho has been flat or declining for 2 consecutive rounds, it's fine to stop proposing for
  that criterion specifically (skip it in that round's `propose.py` invocation isn't
  supported as a flag today -- simplest is to just let it run; `state.json["best"]` already
  tracks the argmax so a wasted round costs a GLM call, not correctness).
- **API failures**: `propose.py` retries transport errors 3x with backoff and falls back to
  keeping the CURRENT prompt unchanged (still advances state to `r+1`) if GLM's reply is
  unparseable or drops the `<<<DOCUMENT>>>` marker -- never crashes the loop, never writes a
  broken template.

## Final held-out comparison

```
# 1. LOCAL: freeze best-dev prompt/criterion, render against TEST (only place test is read)
python3 eval_final.py build
#    -> gepa_final_prompts.jsonl (12 x 100 = 1200 rows), gepa_final_frozen_prompts.json (audit)

# 2. sk3: same queue2 pattern as above, one final Gemma job
scp gepa_final_prompts.jsonl sk3:.../gepa_h2h/
ssh sk3 "echo '.../gepa_final_prompts.jsonl .../gepa_final_results.jsonl' > \$QDIR2/900_gepa_final.job"
scp sk3:.../gepa_h2h/gepa_final_results.jsonl .

# 3. LOCAL: test rho G vs H (recomputes H's raw column the same way cert_agentic.py does,
#    picks whichever of {cand,h0} is already certified as best in agentic_cert.json, paired
#    bootstrap B=2000 on the identical test items, cost column)
python3 eval_final.py eval gepa_final_results.jsonl
#    -> gepa_h2h_final.json
```

## Deviations from the seam-note text (and why)

1. **Reply-format wording.** The note's literal seed template says "reply with ONLY the
   integer." The actual offline scorer already wired into this repo's queue2 convention
   (`methods/metric_seam/pilot/gemma_score_v1.py`) parses replies with
   `re.search(r"SCORE:\s*(NA|\d+)")`, matching the convention already used for LLM-judge
   scoring prompts elsewhere in the battery (`build_seampos_prompts.py`'s `CCL_T`: "Reply
   with exactly one line: SCORE: <integer 0-10>"). Using the note's literal wording would
   make every round's scores silently parse to `None` under the actual scorer. This harness
   uses "Reply with exactly one line: SCORE: <integer 0-10>" instead, appended as a FIXED
   FOOTER the proposer never sees/touches (see `common.FOOTER`).
2. **aspect_id format.** The note writes `aspect_id="<task>|<aid>.g<r>"`. Implemented as
   `f"{task}.{aid}.g{r}"` (`.` throughout, no literal `|`), matching the `f"{task}.{aid}"` key
   convention `cert_agentic.py` already uses for `agentic_cert.json` (so `eval_final.py` can
   look up arm H's certified rho directly by the same key).
3. **Backend file location.** The note points at
   `methods/verification_library/backends.py` for the `zai_anthropic` backend; that file
   doesn't define it -- the actual backend registry (incl. `zai_anthropic`, glm-5.2,
   subscription-free) lives in `methods/metric_implementer/backends.py`. Rather than import
   that module (heavier dependency chain than the "json/pathlib/random + battery_common"
   constraint wants for these scripts), `propose.py` implements the same request shape
   directly with stdlib `urllib` (mirrors `reference_gepa_pr.py`'s own `glm_call`, which does
   the same thing for the same reason). `common.KEY_PATHS` is the fill-in point for local
   key file location/order.
4. **GLM call budget interpretation.** "max 4 rounds, 1 proposer call per criterion per
   round (48 GLM calls total)" is implemented as an upper bound, not a mandatory count --
   see the STOP-conditions section above for the recommended 36-call default.
5. **`propose.py` does one internal retry** on a transport error (up to 3x, exponential
   backoff) and falls back to "keep prompt unchanged" (not a crash) on unparseable JSON /
   missing `<<<DOCUMENT>>>` marker. This is a robustness addition, not a spec requirement;
   it does not increase the steady-state 1-call/criterion/round budget (extra calls fire
   only on transport-level failure, not on a normal successful response).
6. **`eval_final.py` folds "build the held-out prompts file" and "compute the comparison"
   into one script with two subcommands** (`build` / `eval`) rather than a 7th separate
   script, since the note's numbered workflow describes this as one terminal step and the
   deliverables list names 6 scripts total.
