# FULL SWEEP QUEUE (user /goal 2026-08-09) — cell × stage matrix, 3 GPU lanes (5/6/7)

User goal: (1) full metric-discovery for all cells, (2) full spurious-feature discovery
for all cells, (3) full VA for all cells, (4) fused (V+new)+(A+new)+T_trained AFTER
DECONFOUNDING, (5) same with T_untrained. Discovery = the frozen dual-track closure
program (notes/2026-08-05__layer3-closure-prereg.md + Addenda), run under the CURRENT
standards: P=8 sealed fleets across 3 families, two-tier rule (directed sweeps never
feed Good-Turing), realized-draw probe chaining, species.py overwrite guard, 8192-ctx
for LaTeX-heavy cells.

NEVER-REPEAT RULE: stages marked DONE below are not rerun. Source of truth =
notes/2026-08-08__vat-3xN-decomposition-grid.md + registry.

## Stage matrix (as of 2026-08-09 morning)

| cell | VA | discovery (dual-track) | deconf fused T / T₀ |
|---|---|---|---|
| peer_verdict | DONE | DONE (r4 terminal) | QUEUE F2 |
| peer_curation | DONE | DONE (cap-5 terminal) | QUEUE F2 |
| peer_revealed | DONE | DONE (saturated terminal) | QUEUE F2 |
| nc_responded | DONE | DONE (r5 terminal) | QUEUE F2 |
| nc_outcome | DONE | PARTIAL (2-round map) → QUEUE lane B | after closure |
| nc_agree | DONE | PARTIAL (2-round map) → QUEUE lane B | after closure |
| cw_community | DONE | DONE (r8 terminal) | QUEUE F2 |
| hashtagwars | DONE | DONE (r4 terminal) | QUEUE F2 |
| cap_finalist | DONE | PARTIAL (2-round map) → QUEUE lane A | after closure |
| cap_crowd | DONE | PARTIAL (2-round map) → QUEUE lane A | after closure |
| jokes_community | DONE | NOT RUN → QUEUE lane A (first) | after closure |
| mathse_accepted | DONE | NOT RUN → QUEUE lane B (first) | after closure |
| mathse_vote | DONE | LIVE r2 readout missing → lane C first item; then r3 P=8 fleet | after closure |
| aops_curation | DONE | NOT RUN → QUEUE lane B | after closure |
| code_v3 | DONE | r0 DONE; 3-seed gate readout MISSING (seeds trained) → lane C; rounds if gate holds | after closure |
| press_verdict | DONE | DONE (r2 terminal-at-resolution) | QUEUE F2 |
| — new cells — | at build | at build (queue when ledger lands) | at build |
| V6 SO votes | **DONE 2026-08-10 (articulable, Δ −.03)** | **QUEUED — required before F2** (no Track-B map yet; F2 correctly blocked on it) | after discovery |
| V7 patents fwd-cites | building (T+A overnight) | queue on landing | — |
| V8 co-signing | DONE | BLOCKED (gate underpowered — Track-B route only) | T₀ pending |
| V9 tweets | building | queue on landing | — |
| CW RoyalRoad/Wigleaf | rebuilding | queue on landing | — |
| homepage | T done; bank rebuild needed | queue after bank rebuild | — |
| mathlib | seeds 1-2 + regime pending | queue if cell resolves | — |

## Lane assignments (GPUs 5/6/7; ledger-claimed per campaign; sequential within lane)
- LANE A (GPU 5): jokes_community → cap_finalist (r3→) → cap_crowd (r3→)
- LANE B (GPU 6): mathse_accepted → aops_curation → nc_agree (r3→) → nc_outcome (r3→)
- LANE C (GPU 7): mathse_vote r2-completion + r3 fleet; code gate readout (CPU) + rounds
- F2 deconfounded-fusion recomputes are CPU + existing scores; run continuously off-lane.

## F2 = DECONFOUNDED FUSED LEDGER (frozen spec, registered before any F2 run)
Per cell, on the SAME E rows and frozen stacks as the master ledger:
- bank_enriched = terminal bank INCLUDING promoted Track-A criteria from its campaign.
- nuisance block = the cell's Track-B spurious channels (Gemma-scored columns) +
  declared STRUCT columns where the cell has them (code position, patents num_claims).
- Arms: (a) VA_enr_nl = stack on [bank_enriched]; (b) NUIS = stack on [nuisance only];
  (c) VA_enr+NUIS; (d) VAT_dec_trained = stack on [bank_enriched + nuisance + T
  column]; (e) VAT_dec_untrained = same with the T₀ column (already scored,
  results/t0_untrained_arm.json).
- PRIMARY readout (§13 certified stack): stacked increment (d)−(c) — the taste
  residual conditioned on everything nameable including named nuisance — with group
  bootstrap; matched-sampling sign check on the cell's top nuisance channel;
  Westfall-Yarkoni reliability band. Secondary: (e)−(c) (should be ≈0 per the T₀
  result; a positive here flags nuisance-prior interaction).
- NEVER quote (d)−(c) against the old Δ_beyond without naming both designs.

## Standing rules in force
Fused-beats-bank auto-audit; log every landing registry+strict-list before new
launches; pre-kill checklist before any dead verdict; anchors K≥50 every judging
batch; threshold-free readouts; sub-ε stopping needs a PROPOSING round (decomposition
rounds don't count — registered 2026-08-08); slower agent pace (≤3-4 concurrent,
staggered resumes, milestone-only reporting).

## PRIORITY AMENDMENT (user, 2026-08-09): JOURNALISM/PRESS FIRST
User: "focus specially on getting all the journalism and press release ones finished."
- press_verdict: discovery TERMINAL; remaining = F2 deconfounded arm → FIRST in the F2
  batch when it launches.
- homepage curation: dense T DONE (3 seeds .706-.717 eval / .736-.743 test,
  story-grouped); completion agent DISPATCHED (bank rebuild w/ genre-detector ablation,
  Layer-1, T₀ column, samerows_T record); then jumps the discovery queue ahead of
  lane B's remaining cells.
- V9 tweets: build agent RESUMED (was mid-Layer-1 at session kill); its dense arm gets
  the next free GPU 0-3 window; then discovery-queued with same priority.
- Journalism cells enter discovery AHEAD of nc_agree/nc_outcome in lane ordering.

## F2 RESUMPTION STATE (2026-08-11, battery closed)
F2 complete 11/11 terminal cells; parked resumable (hooks in
notes/2026-08-11__f2_deconfounded_fusion.md, commit 3484e3eb0). Pickups in order:
(1) peer_verdict Z after its sealed Track-B round (MASS_DIR gotcha: closure ROOT);
(2) strict-mass Z refresh per certA/certB backfill cell (X/Y/RR mass-independent —
flips are bookkeeping, not new evidence); (3) NEW TERMINAL CELLS (SO, AoPS,
cap_crowd, homepage) — PREREQUISITE: build each cell's T₀ column FIRST
(t0_build_rows → t0_score_vllm) before the five-command F2 sequence; queue SO's T₀
ahead of its terminal date. Four binding operating rules recorded in the note
(one-cell-per-process; no id-dict joins to E; missing STRUCT raises; run on the
ledger-reproducing box).
