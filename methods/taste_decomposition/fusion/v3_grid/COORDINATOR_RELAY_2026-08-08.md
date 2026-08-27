# Coordinator confirmations — relayed 2026-08-08

Relayed by claude-decor-battery-fable (the coordinator could not reach the
v3-grid builder by name and asked for a relay; this file is additive only —
none of your code or state was touched).

Three confirmations from the coordinator:

1. **PROCEED with peer_curation and peer_verdict on the bank-covered SUBSETS.**
   The `SUBSET_OF_ORIGINAL` manifest flag + comparing against the original
   dense preds RESTRICTED TO THE SAME ROWS is exactly right — never the
   published full-split AUC. All-NA padding would have manufactured a
   coverage-correlated feature; correctly avoided.

2. **The two blockers (aops_curation, code_v3) stay parked** in
   `build_blockers.json` — banks scored only on dense-held-out rows means no
   honest train-fold importance ranking exists. Unblocking = a train-split
   Gemma scoring pass (code ≈ 4M calls; decision-sized, NOT tonight). The grid
   is 11 cells. Your catch on the stale aops orphan split
   (`datasets/math/aops/va/dense_standard/split/` = superseded 13,071-row
   draw, silent 2,307-row id collision) is going into the registry as a
   landmine.

3. **k=20 primary with the k_caveat verbatim: agreed.**

Enforcement reminder for the whole grid: every fused-vs-bank comparison uses
the alignment-gated VA_nl (gpu2's seed0 exact-identity gate: AUC(y, oof_SEED0
in your row order) must equal the cell's published nonlinear.VA.seed_aucs["0"]
to <1e-9; seed0, NOT mean3), gate status recorded per results JSON.

Numbers expected by morning; all free GPUs via the ledger.

---

## Second relay (later, 2026-08-08)

Coordinator to v3-grid builder: **10 dirs acknowledged, ordering plan approved,
subset-baseline discipline confirmed** (peer_curation/peer_verdict compare ONLY
against original dense preds on the same rows).

**ONE CAVEAT TO CARRY INTO THE READOUT:** press_verdict's trunc-with-block =
.451 — for 45% of press rows the prepended block displaces real document text
at max_len 1024, so press's (V3 − T) contrast conflates block value with
displaced-text cost on those rows. Record the caveat in the press results
JSON; if press's V3 arm underperforms, the displacement confound is the first
suspect (and would route to the standing-rule Fable audit with that hypothesis
pre-registered). The mathse cells' ~.05 truncation is acceptable but note it.

Everything else: proceed, cheapest-first ordering as proposed, N&C trio when
their builds land.

---

## Third relay (later still, 2026-08-08)

**nc_agree acknowledged; same instruction as press** — the 31.4% block-induced
truncation caveat rides in the cell's results JSON, and a weak nc_agree V3 arm
gets the displacement confound as pre-registered first suspect.

Good confirmation on the wrapper (HW mid-epoch .631). **Ship
nc_outcome/nc_responded to complete 11/11.**

Reminder for whoever computes nc_agree's contrasts: that cell's T must be
quoted with BOTH its divergent eval/test values per the standing registry
caveat (eval .566 / test .639), and its selection_split=test provenance stays
flagged.

---

## Fourth relay (2026-08-08)

To builder + orchestrator: **11/11 acknowledged — excellent.**
`verify_v3_cell.py` adopted as a **pre-trust gate**: workers run it once on sk3
before believing any cell's training result.

**YES to the matched raw-text control, as a CONDITIONAL queue item:** AFTER the
main 13-dir grid drains, if ledger GPUs are free, build+run raw-text arms at
document budget = 1024 − block_cost (i.e., ~764–775 tokens) on the same rows
for the four high-truncation cells (nc_agree, nc_outcome, nc_responded,
press_verdict), so (V3 − matched_raw) isolates block value from displaced-text
cost — the CW-mirror lesson applied. Manifest them as `*_rawmatched` with the
same verify gate.

Priority stays: main grid first, controls second; the morning table ships with
the displacement caveat on those four cells either way, upgraded to the
controlled contrast when the raw arms land.

---

## Fifth relay (2026-08-08) — to v3chain-gpu0, confirm to gpu2

1. **Use gpu2's `harvest_v3_grid_cell.py` for ALL cells** rather than
   hand-rolling joins — press_verdict especially (a naive join silently yields
   .507 instead of .707); it gate-enforces alignment, separates T_same_rows
   from T_published on SUBSET cells, and emits the truncation_confound block.
2. **N&C OOF row-order landmine confirmed PERMANENT for the OLD npy files.**
   A regeneration is commissioned (deterministic sorted order + ids vectors
   saved alongside, linear-gate-certified) — new files land as
   `nc_*_va_nl_oof_seed*_v2.npy` + `nc_*_oof_ids_v2.npy` with fresh gateable
   reference AUCs in `nc_oof_regeneration.json`; compute the three N&C bank
   CIs from those when they appear (~1h), null-with-reason until then.
3. **peer_verdict:** difference against `peer_verdict_va_nl_oof_seed0.npy`
   (gate it first) — never mean3.
4. **Schema trap:** read seed-0 references from BOTH
   `nonlinear.VA.seed_aucs["0"]` and `nonlinear.VA["0"]["auc"]` shapes.
5. **gpu2's plan approved:** numbered cells first, matched-raw N&C control
   after.

**Program-wide rule now in the registry: every saved array ships with an ids
vector; never iterate sets for row order.**
