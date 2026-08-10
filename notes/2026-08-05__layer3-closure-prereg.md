# Layer-3 articulation-closure — preregistration (dual-track design)

Date: 2026-08-05. Parent design: notes/2026-08-05__taste-decomposition-design.md (§3).
User approvals: 2026-08-05 ("do this on two tracks"; "continue with our dual design:
identifying new helpful features AND new spurious features as a dual design mode").
Status: PILOT = peer verdict, declared EXPLORATORY. Confirmatory cells (CW community;
N&C responded conditional on its Layer-1 result; peer revealed conditional on
topic-strat robustness) run only after the freeze checklist at the bottom is closed.

## Question
Does the taste residual Δ_beyond = T − VA_nl survive an active, well-resourced attempt
to articulate it away (Track A), after discounting mined spurious channels (Track B)?
The closure curve Δ_r (r = mining round) is the deliverable; its plateau is the
defensible taste bound.

## Splits (stable-hash, title-grouped; never seeded-shuffle)
- Population = the cell's A/V-scored evaluation population (peer verdict: 6,030 rows).
- FIT+MINE (80%) / MONITOR (20%): sha256 hash of group key, threshold at .80.
- Mining slice M ⊂ FIT+MINE: rows that are ALSO held-out for the dense model (its own
  eval split), so dense scores are honest — never in-sample train predictions.
- Δ_r is computed on MONITOR only. MONITOR is never read by any proposer.
- Pilot has no third split (exploratory). Confirmatory design adds a TEST split quoted
  once at the declared stopping round.

## Round structure (each round runs BOTH tracks — the dual design)
1. Disagreement slice: top-|dense_prob − VA_nl_OOF| rows within M (up to 60 read).
2. **Track A proposer** (k=15/round): reading the slice, propose quality-relevant
   criteria that explain what the dense model perceives and the bank misses.
   Composite/interaction criteria allowed ("X together with Y").
3. **Track B proposer** (k=10/round, separate context, separate instruction): propose
   SUSPECTED-SPURIOUS predictive features — length/format proxies, boilerplate, venue/
   community style markers, topic markers, temporal tells. "Predictive but not quality."
4. **Blind routing audit**: pool the 25 proposals, strip provenance, an independent
   Sonnet-or-better judge classifies each quality-relevant vs incidental; frontier
   arbiter adjudicates disagreements with the proposing track. Misrouted proposals are
   re-routed per the audit (and the misrouting rate reported).
5. **Scoring**: all 25 criteria scored corpus-wide (full A/V population) by Gemma-4-31B
   offline-batch vLLM on sk3 — never an HTTP server; blinded anchor battery
   (pos/neg/scrambled) EVERY batch; guided-JSON collapse check on every criterion's
   score distribution before use.
6. **Update**: A-routed criteria join the bank → refit VA_lin/VA_nl (frozen Layer-1
   protocol, grouped OOF within FIT+MINE, seeds 0/1/2 mean) → Δ_r on MONITOR.
   B-routed criteria join the declared-nuisance set → recompute discounts (below).

## Track A stopping rule
Saturation = 2 consecutive rounds with MONITOR VA_nl gain < ε = .005. Hard cap B = 5
rounds. Report the full curve regardless of where it stops.

## Track B discounting (uses existing balancing machinery; no new estimators)
- Spurious-alone AUC: linear + HistGB on B-features only.
- Discounted readouts: decile-stratified AUC (n-weighted mean of within-stratum AUCs)
  of T and VA_nl, stratified by (i) each B-feature, (ii) the joint B-model score.
  T_adj, VA_adj, Δ_adj = T_adj − VA_adj reported per round.
- Threshold-free readouts only; no residual-regression AUCs.

## Label-blindness (hard constraints)
- Proposers see: text, dense score, VA_nl prediction. NEVER y. Criterion text must not
  reference the outcome variable.
- All measurement by LLM judges (no coded proxies). Judges Sonnet-or-better class
  (Gemma-4-31B is the scoring judge per the A-bank standard).
- Pilot criteria are fidelity-phrased but PRE-GEPA; confirmatory rounds require
  GEPA-iterated phrasing per the A-bank standard. Pilot results carry a pre-GEPA flag.

## Reporting per round (descriptive until freeze)
round r: n proposals A/B, misrouting rate, anchor pass rate, per-criterion score
distributions, VA_lin_r, VA_nl_r (±seed spread), Δ_r, spurious-alone AUC,
T_adj/VA_adj/Δ_adj, cumulative judge-call count.

## AMENDMENT from pilot round 1 (2026-08-05)
- MONITOR must be defined INSIDE the dense-held-out rows (round 1 found 943/1,192
  MONITOR rows in the dense TRAIN split → T-on-MONITOR contaminated at .857; the
  honest same-rows readout used the 1,244 dense-held-out population rows instead).
- Closure-split Δ levels are protocol-specific and NOT comparable to Layer-1
  Δ_beyond; quote only round-over-round changes plus the same-rows honest level.
- Round-2 proposal instruction may steer toward interaction-shaped/composite
  criteria (pilot: composite P05 strongest by ~2×) — recorded as an explicit,
  allowed steer, not a silent change.

## AMENDMENT 2 from pilot round 2 (2026-08-05)
- **Proposal shape must be FIXED in advance for confirmatory cells** (round 1 free-form
  vs round 2 interaction-shaped produced a ~30× difference in closure gain; the round-1
  null would have been declared a taste bound had the rule allowed stopping at one
  sub-ε round). Pilot rounds ≥3 hold the interaction-shaped steer CONSTANT so
  consecutive-round comparisons are apples-to-apples.
- Open questions parked for the freeze: (a) a trigger for sign-contradicting criteria
  (alone-AUC direction opposite the stated quality rationale) — currently report-only;
  (b) the nuisance-vs-merit boundary for fluency-like channels is a substantive
  decision to be made explicitly per cell, not by default routing.

## Freeze checklist before confirmatory
[ ] ε and B confirmed or revised from pilot; [ ] k_A/k_B confirmed; [ ] mining-slice
rule confirmed (dense-held-out ∩ FIT+MINE); [ ] audit protocol + arbiter model named;
[ ] GEPA phrasing pass integrated; [ ] same-rows T rescore done for every confirmatory
cell (design freeze #2); [ ] TEST split defined per cell; [ ] confirmatory cell list
final.

## FREEZE DECLARATION — 2026-08-06 (user go: "do this on ALL runs"; all-GPU authorization)

Frozen parameters (supersede pilot values where they differ):
- Splits: FIT+MINE/MONITOR stable-hash on group key; **MONITOR ⊂ dense-held-out rows**.
- Rounds: k_A=15, k_B=10, ε=.005, saturation = 2 consecutive sub-ε rounds AND the report
  carries the remaining-mass estimate; cap 5 rounds. No composite quota; composite count
  asserted in code.
- Proposers: sealed multi-family fleet, target P=6 (Claude ×2, gpt-5.6-luna ×2 via Codex
  companion, GLM-5.2 ×2 budget_tokens=2048/max_tokens=32000; degrade gracefully to P≥4 /
  ≥2 families under GLM rate limits, recorded). Proposers never see y, the bank, or each
  other.
- Audit: fresh blind Sonnet-class auditor per round + 2 planted probe pairs; arbiter on
  disputes; sign-contradicting criteria → re-audit trigger; collapse gate programmatic.
- Scoring: Gemma-4-31B offline batch, anchors K≥50/class, per-criterion collapse check.
- Missing-mass: fleet-based Good-Turing with leave-one-proposer-out jackknife, odds-form
  remaining-AUC bound; species-form Chao1 never quoted; concept identity by full-recall
  blind pairwise (NEVER embedding-τ across registers); concept census of the incoming
  bank at round 0.
- Discount: matched sampling once spurious-alone >.65 (decile stratification below).
- Readouts per round: Δ_r, (ΔC₊, ΔC₋) swap pair, remaining-mass, discount table.
- Claim discipline: plateau = "not discoverable by this miner" at the measured M3
  sensitivity (A-side .333/.556, zero targeted lift); B-SIDE recovery control queued to
  calibrate spurious-mining sensitivity symmetrically; GEPA phrasing pass on surviving
  criteria before any final quoted number.
- ROSTER (user: ALL cells get mined maps): full dual-track where matched Δ_beyond>.02
  (CW community +.176, N&C responded +.092, peer revealed +.104 topic-caveat; HashtagWars/
  SI/patents/code-v3 when their T/bank land); map-focused dual-track (Track-B emphasis,
  Track A still run) on the rest (peer curation, N&C outcome/agree, cap crowd/finalist,
  press post-gate). math.SE excluded (user), mathlib deferred (T unverified).
- Compute: all free sk3 GPUs authorized (user 2026-08-06); co-tenant jobs never touched;
  shared GPU ledger before claiming.

## FREEZE ADDENDUM (2026-08-06, additive readouts — decision rules unchanged)
- **B-side missing mass** (user): the fleet species machinery runs on Track-B proposals
  too — per-round Good-Turing mass over spurious-channel species (jackknife widths).
  Value-weighting carries the stated assumption that unfound channels resemble found
  ones in influence; reported as species-mass + assumption-flagged value bound.
- **Stacked-increment readout** (stratification-free control): per round, AUC(joint
  B-model) vs AUC(logistic stack of B-model + dense score) — the dense increment over
  ALL named channels in one scalar; same for the bank. Complements matched sampling;
  does not degenerate as the nuisance set grows.
- B-side leave-out recovery control (calibrates spurious-miner sensitivity) runs once
  on peer, alongside the campaign.

## FREEZE ADDENDUM 2 (2026-08-06, Track-B upstream-factor mode — additive)
Per user (prompted by Dan H.'s unseen-factors question): Track-B proposers get an
explicit UPSTREAM-REASONING mode alongside surface pattern-hunting: (1) enumerate
unseen factors BEYOND the text that could causally affect the outcome (author
reputation/seniority, institution/resources, timing, review-process dynamics, social
networks, professional editing); (2) for each, ask what textual FINGERPRINT it would
leave; (3) propose those fingerprints as nuisance channels. Every B channel is TAGGED
with its conjectured upstream parent (or "surface-only" if none).
MIXED-CHANNEL RULE: a channel whose conjectured parent plausibly causes REAL quality
too (e.g., senior labs both get favored AND write better) gets a mixed flag and is
reported in BOTH the discounted and undiscounted readouts (a sensitivity band), never
силently routed to one side.
Interpretation note for all reports: traceless unseen factors cannot inflate T or bias
Δ (they lower the noise ceiling for every instrument equally — the H3 rater-model line
owns them); only TRACED unseen factors (textual proxies) threaten Δ, and this mode is
their search strategy.

## FREEZE ADDENDUM 3 (2026-08-07, user: decompose MIXED channels)
MIXED channels are no longer merely dual-reported — they get a DECOMPOSITION PASS:
author ≥2 refined criteria isolating the components (e.g., staccato lineation →
"deliberate one-line beats serving tension/pacing, judged in context" [candidate-real]
vs "editor-default blank-line/whitespace habits" [surface]), score each separately,
route each through the blind audit independently. The parent channel is retired from
readouts once its components are scored (recorded, not deleted). This generalizes the
arbiter's salvage-rewrite pattern into standard handling. Decomposed components count
toward their round's k budgets.

## FREEZE ADDENDUM 4 (2026-08-08, additive Track-B prior)
POSITION-IN-CONTAINER channels: verified across CW (125 criteria) and peer (8 B-files)
that no proposer has ever named an ordinal/position channel, yet two of the program's
strongest spurious findings are exactly that family (patents claim ordinal .754; code
repo-recency, repo-local). Track-B proposer instructions now explicitly include: "consider
the item's POSITION or ORDER within its container (position in docket/thread/claim-set/
contest/repository timeline) and any textual fingerprint of it." Applies from each
campaign's next round.
