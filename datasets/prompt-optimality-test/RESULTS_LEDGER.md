# RESULTS LEDGER — prompt-optimality campaign (Paper #2)

**This file is the single source of truth for every quotable number.** Created 2026-07-25 after
the user flagged result churn. Rules of the ledger:

1. A number may be quoted (in the paper, to the user, in notes) ONLY if it appears here with its
   provenance block. The HB entries in `notes/2026-07-19__gepa-consolidation-and-momega-upgrade-plan.md`
   are the journal (how we got here); this file is the state (what is true now).
2. Every canonical number lists: artifact path, box, date, server fingerprint (or "pre-fingerprint"
   for measurements before 2026-07-25), max_tokens, effective k, and pairing status.
3. Update this file in the same commit as any measurement that changes it.

## The five factors that made results move today (checklist for any new number)

| factor | what it does | detection | mitigation (status) |
|---|---|---|---|
| F1 cross-session comparison | arms measured in different sessions differ by u_session (pupa ±.07, aime ±.18) | compare same-session re-read | same-session paired protocol (MANDATORY) |
| F2 serving config | max_tokens truncation + vLLM reasoning-parser (`text: None`) span .35–.53 on aime | truncation warnings in log; `text: None` in responses | server fingerprint row in every rescore (SHIPPED 2026-07-25); dual-column 8k/24k for truncation-sensitive benches |
| F3 dspy response cache | repeated passes returned identical completions → "k=3" was k≈1 | bit-identical pass_means | `cache=False` in rescore_k3 (SHIPPED); paper reports pairing, not passes, as the remedy |
| F4 test-panel selection | candidate chosen by its test score (livebench draw #86) | selection logs / phase provenance | select on train panels only; verified for all canonical candidates |
| F5 run-dir overwriting | 07-23 rerun silently replaced sk1 `aime/official/result.json` (.3667 → .5333) | result.json mtime vs run log content | NEVER reuse a run dir; always `--run-tag`; this ledger pins which file is canonical |

| F6 fd exhaustion | long multi-eval processes leak http fds -> `OSError 24` mid-run (killed P0 at 31/90, 2026-07-26; same class as the old ifbench fd-crash era) | rc!=0 or "Too many open files" in log | RLIMIT_NOFILE bump + checkpoint/resume in livebench_reselect_placebo.py; orchestrator auto-retries (SHIPPED 2026-07-26) |

Also relevant, found and fixed earlier: missing `python-Levenshtein` hard-zeroed 19% of livebench
for every run before 2026-07-24; pupa judge rate-limits scored items 0 before the patient retry stack.

---

## CANONICAL SCOREBOARD (what we build on)

All canonical cells: same-session paired, paper-exact test splits, Qwen3-8B, primary inference =
paired bootstrap on mean item delta. "Effective k≈1" = passes were cache-collapsed (F3); validity
rests on pairing, which F3 does not touch.

**LOCK-IN STATUS (user directive 2026-07-26): each bench below carries a LOCKED line = the
highest-confidence (and where defensible, highest-M_ω) result we build on. Anything not LOCKED
is explicitly provisional.**

### hotpot — WIN, CONTENT-AUDITED, CLOSED (CONFIRM cell ⇒ no follow-up permitted; init replicates were cache-degenerate, effective n=1 — disclosed, verdict unaffected)
**LOCKED addendum: foreign-content control (HB113): real draws .5936 vs foreign-clause draws
.3326 vs init .4133 — own-pool content carries the entire gain (θ=1.45, z=7.28, p≈3e-13), real
min .500 > init, and foreign bulk actively HURTS (−.08, tail to .000). The pool-content thesis
stands on the flagship cell; same foreign clauses that lift livebench +.11 sink hotpot −.08.**

### hotpot — original win record (the flagship cell)
**LOCKED: M_ω .6333 vs GEPA .4133 (+.220, P=.0000).** Nothing pending can move this cell.
- **GEPA .4133 vs M_ω .6333, Δ=+.2200, sign W75-L9-T216 p=4.3e-14, bootstrap P(Δ≤0)=.0000, n=300**
- Provenance: sk3 GPU7, 2026-07-25 21:42–21:47Z, fresh Qwen3-8B server port 8078 (ctx 32768),
  max_tokens 8000, effective k≈1 (pre-fingerprint).
  Artifacts: `runs_paperexact/hotpot/Qwen3-8B/{official,unitrecomb_v5sk2}/rescore_k3.jsonl` (sk3).
- Candidate provenance: selected on train panels (select 100 / confirm 350); test touched once (HB98).
- Pool: 0/68 leakage-flagged; judge anchor-validated recall 1.00 / FPR 0.00 (HB96).
- Threats: none known. Answers are short → truncation (F2) not plausible. STABLE.

### aime — NOT CONFIRMED (delta fails cross-server replication, 2026-07-26)
**LOCKED: the honest label — "delta not stable across serving environments."** Two internally-valid
same-session paired measurements of the SAME candidates at the same nominal 8k config: sk3 Δ=+.091
(P=.0012) vs sk1 Δ=+.0044 (P=.40, W16-L18-T116). The session effect on aime is not arm-symmetric,
so no single paired session can carry the cell. aime does NOT count in W. The 24k column (running)
is informational only — a same-config replication failure cannot be overridden by a
different-config win. aime still supports the "never worse" clause (no measurement shows GEPA
beating M_ω).
- Paper-exact-config paired reading: **GEPA .3067/.3000 (two configs, same session) vs M_ω .3978,
  Δ=+.091, bootstrap P=.0012, n=150** — sk3 GPU7 session 2026-07-25 21:29–21:33Z, 8k, effective k≈1.
  Artifacts: `runs_paperexact/aime/Qwen3-8B/{official_sk1cfg,official_sk2cfg,unitrecomb}/rescore_k3.jsonl` (sk3).
- WHY CONTESTED: at 8k, ~40% of GEPA's answers truncate (578 warnings); the same GEPA candidate
  reads **.5267 at 24k (0 truncations)** (sk1 cert session, `runs_paperexact/aime/Qwen3-8B/rank_certificate/draws.jsonl`
  init row). The 8k win may partly measure truncation-robustness of M_ω's prompt (F2).
- Historical readings, all explained (HB100): .5333 = 07-23 stair rerun (overwrote sk1 result.json, F5);
  .3667 = 07-20 run; .5267 = 24k; .3533/.3067/.3000 = 8k. Candidate byte-identical throughout
  (md5 e018f7779184); code byte-identical across boxes.
- **CLOSED (HB110/HB112).** All measurement lanes dead (sk1 reboot); the two 24k GEPA passes
  (.5800 ×2, pass-stable, cache off) are truncation-mechanism evidence only. Arm×session
  interaction z=2.49, p=.013 (HB114) — the arm-symmetric session model is REJECTED.

### ifbench — DIRECTIONALLY POSITIVE, NOT CONFIRMED (final; no re-roll)
**LOCKED: +.0198 (P=.233), reported with the pre-registered label. Closed cell.**
- **GEPA .4337 vs M_ω .4535, Δ=+.0198, bootstrap P=.233, n=294** — same sk3 session 21:33–21:42Z, 8k.
  Artifacts: `runs_paperexact/ifbench/Qwen3-8B/{official,unitrecomb_v6ctx32k}/rescore_k3.jsonl` (sk3).
- Superseded reading: +.044 single-pass (F1). Pre-registered label applies verbatim; the permitted
  k-widening is deliberately NOT spent (advisor ruling, HB98).

### pupa — TIE (final)
**LOCKED: tie (Δ=−.0009, P=.52). Closed cell.**
- **GEPA .8825 vs M_ω .8817, Δ=−.0009, sign W28-L30-T163 p=.90, bootstrap P=.52, n=221** —
  sk2, 2026-07-25, k=5 requested / effective k≈1 (F3), same-session.
  Artifacts: `runs_paperexact/pupa/Qwen3-8B/{official,unitrecomb_v8failmine}/rescore_k3.jsonl` (sk2, passes=5 rows).
- Superseded: "M_ω .8938 vs GEPA .8835 win" was cross-session (F1). One-shot rule: no further arms.

### hover — WIN, CONFIRMED-STABLE (2026-07-26; passed the cross-session test aime failed)
**LOCKED addendum: second-session stability pair (fresh server, 24k, vLLM 0.16.0): Merge .5267 vs
M_ω-ablated .5622, Δ=+.0356, P=.0086 — deltas +.051 (s1, 8k) and +.036 (s2, 24k) agree across a
serving-config change. Most fortified cell on the board.**

### hover — original win record (2026-07-26, ablation-clean)
**LOCKED: M_ω-ablated .5656 vs GEPA+Merge .5144 (+.051, P=.0009); full M_ω .5689 (+.054, P=.0004);
the flagged clause's causal contribution is NULL (+.0033, P=.41).** One session, 4 arms, cache off,
n=300, sk2 2026-07-26 ~01:00-02:30Z. The ablated candidate is the cell of record (beats the CLEAN
strongest comparator with the flagged clause removed — the leakage objection is empirically dead).
Artifacts: runs_paperexact/hover/Qwen3-8B/{official,official_merge_gepamerge,unitrecomb_stair,unitrecomb_stair_ablated}/rescore_k3.jsonl (sk2).
SUPERSEDED prior state: The deciding run (official + GEPA+Merge + unitrecomb_stair + ablated, one
session, fingerprinted, cache off) started 2026-07-26 ~01:00Z on sk2 (pid 2182963); mid-run
readings in the .56-.58 range are visible but UNATTRIBUTED (arm order unknown mid-flight) — quote
nothing until the paired table lands.
- NOT YET CANONICAL. Best available readings (single-pass, cross-session — quote nothing from these):
  M_ω stair .5833 (sk1, 07-24) > GEPA+Merge .5333 > MIPRO .48 > GEPA .45/.4567.
- Complications to be settled by the queued same-session run (`sk2_hover.sh` pid 39174):
  (a) our best candidate carries 1 test-hit item_hint unit; GEPA+Merge is clean → the run scores
  official, GEPA+Merge, unitrecomb_stair, AND unitrecomb_stair_ablated (clause removed) in ONE session;
  (b) purged-pool certificate (164→160) runs after. hover loads ONLY on sk2 (datasets version).

### livebench — COMPARISON-CONFIRMED / CONTENT-FALSIFIED (HB106, final; excluded from W)
**LOCKED: draw #88 beats GEPA +.0923 (P=.0001, same-session k3, select-panel-provenanced) — but
the foreign-content control reproduces the entire gain (placebo .7342 ≥ real .7055 n.s.; init
.6213), the 120/120 claim is DEAD (kill switch: init max = real mean), and livebench is a
metric-pathology exhibit, not a content win.** Mechanism sentence gated on the per-item
zero-rate decomposition (HB107b); third arm (shuffled text) mandated before submission.
Artifacts: runs/reselect_placebo_livebench_Qwen3-8B.json (sk2), HB106.

### livebench — superseded pre-registration state (kept for the record)
**LOCKED so far: (a) the rank certificate P(fresh > .7914) ≤ .0083 (protocol-relative, per-pool);
(b) the reachability cap .9048; (c) draw #88 as the candidate of record (select-promoted, .7869
on the held-out select panel).** TERMINAL STATE: P0 complete (HB106), all livebench lanes closed.
- NOT YET CANONICAL. Void readings (never quote): GEPA .6956 (Levenshtein-broken era + idle box);
  all pre-2026-07-24 numbers; topdraw +.058 (draw #86 was test-selected, F4).
- Pre-registered decision (committed to git BEFORE data, commit 45e5137): Phase 3 Δ of draw #88
  (select-promoted, select .7869) vs GEPA, same session, thresholds in HB91. Placebo grid decides
  whether the effect is content or padding. Running: sk2 pid 490085, Phase 2 ~30/90, polled by
  count only.
- Certificate (protocol-relative, valid): P(fresh draw from declared 48-unit generator > .7914)
  ≤ .0083, N=120. `runs/bound_rank_certificate_livebench_Qwen3-8B.json` (sk2).
- Certified metric-reachability cap: **.9048** (12/126 items unreachable through the metric).

## Bounds & certificates in hand
| object | value | artifact | status |
|---|---|---|---|
| livebench reachability cap | .9048 | sk2 `runs/` (bound_metric_reachability) | certified (declared output family) |
| livebench rank certificate | P(>.7914)≤.0083 | sk2 `runs/bound_rank_certificate_livebench_Qwen3-8B.json` | certified, protocol-relative, per-pool |
| hover rank certificate | — | purged-pool version queued (sk2) | unpurged version NEVER RAN (sk1 lane died) |
| aime rank certificate | — | KILLED (150–200h projected; unviable at 24k) | only its init reading (.5267) survives, used in HB100 |
| EVT endpoints | — | — | RETRACTED (degenerate); never quote |
| DPI all-prompt cap | — | — | does not transfer (vacuity theorem); never quote |

## Leakage audit (settled)
- 360 units, 5 pools; judge anchor-validated (recall 1.00, FPR 0.00, answer-grade 3/3) —
  `runs/leakage_anchor_validation.json` (sk1).
- No answer-grade unit in any shipped candidate; hover 4 test-hit units (pool purged for certs;
  1 appears in our best hover candidate → ablation queued); hotpot 0/68 clean; livebench 0 test hits.
  `runs/unit_leakage_audit.json`, `runs/leakage_holes.json`.
- hover's answer unit traces to a HoVer train/test near-duplicate (benchmark defect; disclose).

## Standing measurement rules (all new numbers)
1. Same-session paired, all comparators in one invocation; paired bootstrap primary.
2. `rescore_k3.py` only (it now writes a `session_fingerprint` row: host, vLLM version,
   max_model_len, CLI config, cache=False). A number without a fingerprint is provisional.
3. Unique `--run-tag` per invocation; NEVER write into an existing run dir (F5).
4. Truncation-sensitive benches (aime, livebench): dual-column, 8k paper-exact + 24k
   instrument-clean, same server.
5. Decision rules are written down BEFORE the data is read (HB91, HB100) and applied mechanically.
