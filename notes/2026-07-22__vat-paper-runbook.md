# VAT Paper (Paper #3) — Execution Runbook

*2026-07-22. For the executing agent (Opus): keep the pipeline moving, do NOT re-solve or
re-design in real time. Every step below is pre-scripted. If a step needs NEW code or a
design decision, STOP and surface it to the user — do not improvise a new pipeline.*

Plan + master table: `notes/2026-07-22__vat-paper-plan.md`. Tracker: memory `project_vat_paper`.

---

## Where things stand (2026-07-22 ~22:40)

Academia three-y VAT scoring is running on sk3. All scripts + data are already written and
validated. The ONLY live task for the next 1–2 cycles is: **wait for scores → harvest →
aggregate → record.**

- sk3 job: `score_va_gemma_3y.py`, GPU 1, launched ~22:38, pid was 102092 (pid may differ if
  relaunched — match by process name, not pid).
- Output being written: `sk3:/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/vat_3y/union_scores.npz`
- Log: same dir, `acad3y_score.log`. ETA ~2.5–3h (2.2M prompts). Done ≈ 01:00–01:30.
- Local mirror of scripts/data: `datasets/peer-review/vat_3y/`.

---

## CYCLE 1 — harvest + aggregate (the one live task)

### Step 1.1 — each wake, check whether scoring finished
```
ssh -o ConnectTimeout=10 sk3 'D=/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/vat_3y; \
  echo "--- proc ---"; pgrep -af score_va_gemma_3y | grep -v pgrep; \
  echo "--- npz ---"; ls -la $D/union_scores.npz 2>/dev/null; \
  echo "--- log ---"; tail -4 $D/acad3y_score.log'
```
Decide from the output:
- **`SCORE_DONE` in log AND `union_scores.npz` exists** → go to Step 1.2.
- **process alive, no npz yet** → still scoring. Note progress if a `Processed prompts` line
  shows a %, then STOP for this cycle (check again next wake). Do nothing else.
- **process GONE and no npz** → FAILURE. Go to the Failure Playbook.
- **sk3 unreachable (NAT64 timeout)** → transient. STOP; retry next wake. Do not escalate on a
  single miss.

### Step 1.2 — harvest scores to laptop
```
scp -o ConnectTimeout=20 sk3:/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/vat_3y/union_scores.npz \
  /Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/vat_3y/
```

### Step 1.3 — aggregate (already-written script, no edits)
```
cd /Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/vat_3y
python3 aggregate_3y.py
```
Prints two markdown tables (nested ladder + apples-to-apples) and writes `vat_3y_results.json`.

### Step 1.4 — sanity gate BEFORE trusting the numbers
Check these against expectations (from memory `project_peer_review_va`):
- **A-bank NA rate ≈ 0.55–0.70** (peer abstracts genuinely lack evidence for many rubrics).
  If NA > 0.85 → likely a parse/scoring failure; the numbers are suspect. Surface to user.
- **verdict rung: V ≈ 0.60–0.62, A ≈ 0.65–0.68** (must roughly reproduce the known
  ICLR V .611 / A .676). If verdict A < 0.58 or V < 0.55 → something is wrong (wrong bank,
  degenerate scores, bad join). DO NOT report as a finding — surface to user with the log.
- curation / revealed AUCs are the UNKNOWN (that's the experiment) — no expectation, just report.
- If the sanity gate passes → the numbers are trustworthy; proceed to Step 1.5.

### Step 1.5 — record (only if sanity gate passed)
1. Append a results block to `notes/2026-07-22__vat-paper-plan.md` under "## Progress log":
   the two tables + the one-line reading (does A−V widen up the selectivity ladder? does the
   seam differ between curation-y and revealed-y on the SAME 2,202 papers?).
2. Update memory `project_vat_paper.md` status line with the headline numbers.
3. Report the two tables to the user in the reply, with a 2–3 sentence reading. Flag any
   number that contradicts the master-table story. Then STOP and await direction.

**Do NOT** start Cycle 2 experiments automatically — they need a design confirmation. After
Cycle 1 results are reported, hold for the user unless they've pre-authorized the next step.

---

## CYCLE 2+ — queued, but each needs a go-ahead (do NOT author these solo)

These are the planned next experiments. They require NEW code / design, so per the standing
instruction the executing agent should **surface them and hand off**, not build them live:

- **Robustness — topic-stratified revealed-y.** Re-attach the citation label using the
  within-topic-cluster balanced version (sk3 `..._v3_topicstrat.csv.gz`, per memory
  `project_nc_vat_run`/taste-grid notes). Guards against the citations topic/era confound.
- **Hole (a) heterogeneity — reviewer IRT + noise ceiling.** Needs the per-reviewer
  OpenReview scores (multiple reviewers/paper). First a SCOPING read to locate them
  (`datasets/peer-review/` — unified_papers / berenslab-iclr-dataset / casimir), then an IRT /
  mixed-effects fit → noise ceiling → community(venue) vs personal Taste split.
- **Fidelity-optimized A replication (H6).** Run the N&C GEPA fidelity↑/AUC↓ test on the peer
  bank (Sonnet-via-CLI proposer, unmetered).

When you reach these: post the one-liner "next queued: <name> — needs code/design, ready to
scope on your go" and wait.

---

## FAILURE PLAYBOOK (Cycle 1)

Read `acad3y_score.log` tail first; match the symptom:

1. **CUDA OOM / init OOM.** Relaunch once with lower util. Edit the launch script's `--util`
   from 0.85 → 0.70 and rerun it:
   `ssh sk3 'bash -s' < /private/tmp/.../scratchpad/launch_acad3y.sh` (script path in the plan
   note; or recreate from the recipe in §"launch recipe" below). If it OOMs again → escalate.
2. **GPU 1 no longer free** (fleet expanded). Pick another 0-MiB GPU:
   `ssh sk3 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'` → choose a GPU
   with ~0 MiB, **avoid GPU 0** (co-resident-oscillation flag). Set `CUDA_VISIBLE_DEVICES` to
   that index in the launch script and relaunch.
3. **flashinfer / sampler JIT crash.** Already mitigated (`VLLM_USE_FLASHINFER_SAMPLER=0`). If
   it recurs, the log will name the op — escalate, do not hand-patch vLLM.
4. **NAT64 / connection reset.** Retry the same ssh once, gently. If it fails twice in one
   cycle, STOP and retry next wake. Never disable IPv6, never thrash.
5. **Anything else** (traceback you don't recognize) → STOP, quote the log tail to the user,
   do NOT attempt a fix.

Relaunch at most ONCE per cycle without user sign-off. Never `pkill`/`killall`; if you must
kill, target the specific pid AND its `VLLM::EngineCore` child.

### launch recipe (if the scratchpad script is gone)
```
cd /lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/vat_3y
export HOME=/lfs/skampere3/0/alexspan
GEMMA=/lfs/skampere3/0/alexspan/envs/gemma4/bin/python
D=$PWD
setsid env HF_HUB_OFFLINE=1 HF_HOME=/lfs/skampere3/0/shared_hf_cache \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8 \
  $GEMMA "$D/score_va_gemma_3y.py" --util 0.85 \
    --input "$D/union_toscore.jsonl" --out "$D/union_scores.npz" \
  < /dev/null > "$D/acad3y_score.log" 2>&1 &
echo "LAUNCHED pid=$!"
```

---

## STANDING RULES (do not violate while executing)
- Never re-quote superseded numbers (memory never-quote flags: peer .759, PR .734/.749,
  press-release .71 dense, patents .756).
- One GPU only; stack nothing without a free-GPU check; scoring = offline batch vLLM (never an
  HTTP server).
- Verify GPU/job claims with unfiltered `pgrep`/`nvidia-smi` before asserting state.
- Report results descriptively; no sweeping verdicts mid-experiment.
- If the sanity gate (1.4) fails, the correct action is to SURFACE, not to explain the number
  away.
