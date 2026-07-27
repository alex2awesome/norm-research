# v13.1-B Multi-behavioral-reconstructor value bounds: the induce-and-execute recipe (handoff spec)

**2026-07-13. Companion to `2026-07-13__v13-multi-mcq-value-bound-recipe.md` (v13.1-A, the MCQ channel). Same author, same audience, same certification standards. This channel is the canonical R2 recovery readout (`run_r2_recovery` lineage) rebuilt with the pool/panel/capture-recapture machinery.**

## 0. Why this channel, and why now

MCQ identification failed reconstructor qualification twice today (Gemma-4-31b, GLM-5.2: near-one-hot posteriors + canonical anti-identification), with the likely root cause being **menu-gold infidelity** (the gold is a written description; the demos carry executor behavior; strong rule-inferencers follow the behavior). The behavioral channel has no menu at all:

- Reconstructor ρ reads K labeled demo texts → **generates** a criterion description M̂ (free text).
- A frozen execution model E′ **executes** M̂ on a frozen held-out probe set H → verdicts ŷ ∈ {0,1}^|H|.
- Value compares ŷ against M_ω|_H (the metric's own frozen verdicts on H).

Properties that dissolve today's failure modes: (1) the gold is M_ω's behavior itself — exactly the operational target per the two-ladders doctrine, no description-fidelity gap possible; (2) no forced-choice posteriors — the RLHF/reasoning posterior-collapse pathology is irrelevant, so strong models (incl. the ones that failed MCQ) are candidate inducers again; (3) misleading demos produce LOW value, never confident anti-identification; (4) direct comparability with the project's canonical transmission metric I(M_ω; ·).

## 1. Estimand (exact, frozen, no sampling noise anywhere)

Fix: teaching pools/panels as in v13.1-A (G=6 disjoint pools of S=12 design texts; finite declared panel family; behavioral tier uses **R_g = 4 panels/pool, K = 6 demos/panel** — see §4 for why K=6). Fix a frozen eval set H (60–100 probes, disjoint from all pools; stable-hash split). Fix ρ (the inducer) with **temperature-0 deterministic induction** and E′ = the campaign executor (Llama-8B, constrained two-token readout) so M̂ competes on exactly the footing of a mined prompt.

Per panel T and demo-label state s ∈ {0,1}^K:
- M̂_{T,s} = ρ(T's texts, labels s) — the induction prompt contains ONLY panel texts + labels (never the candidate prompt body; same non-disclosure discipline as MCQ, string-asserted in tests). Deterministic ⇒ one M̂ per (T, s).
- ŷ_{T,s} = σ_{E′}(M̂_{T,s})|_H — deterministic constrained readout ⇒ exact.
- **v_T(s) = clip( TR(ŷ_{T,s}) − max(TR_blind, TR_shuffled), 0, H₂(M_ω|_H) )** where TR(·) = plug-in mutual information I(M_ω|_H ; ·) in bits on the frozen 2×2 confusion table (report balanced agreement alongside; MI is the headline per the report-recovery-metric-only rule). Controls: TR_blind = same pipeline with no demos; TR_shuffled = permuted labels (both per panel, deterministic seeds).
- Prompt value: **V̄(p) = (1/G) Σ_g (1/R_g) Σ_{T∈g} v_T(σ(p)|_T)** — identical structure to v13.1-A; exact finite mean; variance decomposition (within-pool vs across-pool) reported identically.

Because induction and execution are both deterministic and H is frozen, every v_T(s) is an **exact number** — no binomial corrections, no per-state confidence penalties. (Optional declared sensitivity arm: re-induce a 5% subsample of (T,s) pairs at 2 extra seeds, temp 0.9, and report the generation-variance spread as a descriptive robustness row — never inside the certified quantities.)

## 2. The factorization, and everything it buys

M̂ depends on p only through (T, σ(p)|_T). Therefore V_T(p) = v_T(state), exactly as in the MCQ channel — and the entire v13.1-A certified stack transfers verbatim:

- **Trust hierarchy for V_unseen:** level 0 (new, free, unavailable to MCQ): **v ≤ H₂(M_ω|_H) bits — the exact entropy ceiling of the target on H**. Level 1: exact per-pool cap by enumerating all 2^K states per panel. Level 2: MILP dual for overlapping designs (same toolbox; unnecessary while pools are disjoint). Level 3: free recombination. Level 4: the clip ceiling. **Extrapolation still banned.**
- **Gain bounds:** identical formulas — pool-decomposed `gain_g(m) ≤ (1−(1−U0_g)^m)·max(0, U_g − A_g)` with measured recombination slack Δ_recomb; joint-species bound; DKW on observed V̄. CP first-contact runs on the same 12-bit pool-pattern species (species definitions are executor-side and channel-independent — the SAME c/r counts serve both channels).
- Premise flags identical: exchangeability for CP, disjoint-pool separability, no smoothness/submodularity/independence anywhere.

## 3. Deduplication — the cost trick that makes this tractable

Unlike MCQ (1 cheap logit query per state), each state costs 1 induction + |H| executor scorings. Three exact (lossless) dedups:
1. **State-level** (inherited): prompts sharing a state share everything. Observed distinct states ≤ ~40/panel in practice.
2. **Rule-level:** different (T,s) often induce byte- or semantically-identical M̂ (degenerate states especially). Execute each distinct M̂ **once**, keyed by content hash; the (T,s) → M̂-hash map is stored. Expect heavy collapse (the t8c tables showed 6–41 distinct values per 256 states).
3. **Batch execution:** all distinct M̂ × H scorings are one vLLM batch matrix — the same batching the mining loop already does.

## 4. Cost and tiering (why K=6)

K=8 gives 256 states/panel — full enumeration at behavioral prices (~200 queries per state) is ~3.7M queries/metric: intractable. **K=6 gives 64 states/panel** and demo informativeness barely drops (6 labeled examples vs 8). Costs per metric, |H|=60, before rule-level dedup:

| tier | scope | queries |
|---|---|---|
| **Tier A (sentinel + headline)** | 24 panels × 64 states, full enumeration | ~1.5K inductions + ≤94K executor scorings ≈ 1–1.5 GPU-h |
| **Tier B (fan-out)** | observed states everywhere + full 64-state slice on 2 pools; caps level 1 there, level 0 elsewhere | ~25–40K ≈ 15–25 min |

Rule-level dedup typically cuts these 2–4×. Induction can run on a strong local model (Llama-3.3-70B-FP8 or Qwen2.5-72B-FP8 — one GPU, ~1.5K short generations/metric); API inducers are possible for spot-check arms only (volume math: full-tier induction ≈ 1.5K calls/metric — GLM spot-arm on 1–2 metrics is quota-safe, never primary).

## 5. Gates (fewer than MCQ — that's the point)

- **Inducer qualification** (replaces the failed MCQ battery; predeclared): on the 6 humor sentinels, canonical induction (demos = target's own verdicts) must achieve TR lift > 0 over both controls for ≥ 4/6, and the shuffled-label control must land ≈ 0 (instrument nulls correctly). No prior-balance rule — there are no menus. No calibration requirement — no posteriors.
- **Execution-health gate** (existing discipline): σ_{E′}(M̂) std over H ≥ std_floor for ≥ threshold share of induced rules, else FORMAL_ONLY (executor too degenerate to score rules — the PRISMA failure mode, now caught explicitly).
- **Quality-prior guard** (Codex-#4 lineage): the shuffled-label control IS the guard — a generic "good writing" M̂ scores the same under true and shuffled labels and self-subtracts.
- Anchor rows in every induction batch (planted recoverable rule; constant rule) per the standing anchor-test rule.

## 5b. Anti-degeneracy: why M̂ cannot score I≈1 by mapping inputs to outputs

The failure to prevent: M̂ = a lookup table ("YES for these texts, else NO") that trivially reproduces the demo labels and reports perfect transmission. Four stacked defenses; the first is structural and decisive.

1. **Held-out execution (load-bearing).** M̂ is scored ONLY on the frozen eval set H, disjoint from every teaching text and pool (stable-hash split). A memorizing M̂ has zero coverage on H → executes to base-rate → value ≈ 0 after control subtraction. Memorization does not cross the split; the split is frozen before induction.
2. **Capacity bottleneck.** Given the panel, M̂ is a deterministic function of exactly K=6 demo bits. Six bits cannot encode M_ω's |H|=60–100 held-out verdicts. High agreement on H is reachable ONLY if those 6 bits select a genuinely generalizing hypothesis — which is exactly the quantity being measured. The bottleneck IS the anti-degeneracy mechanism.
3. **Controls absorb non-demo success.** blind control M̂₀ (no demos) catches prior-guessing (task noun + inducer priors) and generic "score like a typical judge" rules that ride the executor's base behavior; shuffled-label control catches rules that ignore the labels. Value = TR − max(blind, shuffled), so any success not flowing THROUGH the demo bits self-subtracts to ≈0.
4. **Exemplar-dump is a declared decision, not a leak.** The one real edge case: M̂ that embeds the 6 demos verbatim as few-shot examples. This is NOT H-memorization — if it generalizes to held-out texts it genuinely transmits, and under the two-ladders doctrine any finite prompt is admissible. But it conflates verbal articulation with exemplar-carrying (a distinct decompression rung). Resolution: TWO declared arms — (a) unconstrained M̂ (headline: honest sup over evidence-constructible prompts), (b) no-verbatim-exemplar M̂ (string-enforced: reject/regenerate any M̂ containing a demo text substring above a shingle threshold). The (a)−(b) gap is a reported finding = the exemplar-vs-rule decomposition, measured directly.

Ceiling honesty: TR = plug-in MI of the 2×2 confusion on frozen H, capped by H₂(M_ω|_H) ≤ 1 bit. "I=1" is out of codomain unless M_ω is balanced AND M̂ is perfect on unseen texts — that is success, not degeneracy. Test (CPU, fake backends): a planted lookup-table M̂ that perfectly fits demos must certify ≈0 value on held-out H.

## 6. Cross-channel consistency (free, and it would have caught today's crisis)

Where both channels run (sentinels): per (panel-pool, state), MCQ value and behavioral value should rank-correlate. A metric where MCQ says "identifiable" but behavioral transmission ≈ 0 (or vice versa) flags instrument pathology — menu-gold infidelity produces exactly the first pattern. Report Spearman ρ per metric as a standing diagnostic row in both certificates.

## 7. Relation to the ladders + reporting language

M̂ ∈ Dom(E′): the induced rule is itself a prompt, so behavioral value sits on the SAME scale as mined-prompt agreement — one certificate can honestly show: best mined prompt vs best induced-from-state rule vs H₂(M_ω|_H) ceiling, all in bits on frozen H. The certified claims remain process/instrument-indexed: "under this inducer, executor, pools, and H" — never articulability-absolute. Statuses, 95/90 tiers, immutability, prereg locking: all inherited unchanged from v13.1-A.

## 8. Implementation mapping

Reuses from branch `cr3-sampled-v13`: pool/panel builder (K parameterized — the library builder already takes panel size via `n_examples`; verify K=6 flows through), state-keyed caching (key becomes (panel, state) → M̂-hash → ŷ), CP/DKW/gain code verbatim, certificate/tier scaffolding verbatim. New: `behavioral_value_channel.py` — induction prompt builder (port the data-driven induction template from `run_r2_recovery`/recovery_prompts lineage, NOT the generic one), rule dedup store (content-addressed, immutable), execution batcher, TR computation (2×2 plug-in MI, exact). Tests: state→rule determinism; rule dedup correctness; shuffled-control ≈ 0 on synthetic; planted recoverable rule certifies; entropy ceiling enforced; non-disclosure string assertions; fake inducer/executor backends throughout (CPU).

## 9. Recommendation

Run BOTH channels on the humor sentinel (the behavioral channel's Tier-A cost is comparable to MCQ's). If the fidelity diagnostic confirms menu-gold infidelity, the behavioral channel becomes the **primary** value channel for v13 certificates and MCQ demotes to a diagnostic arm pending menu re-grounding; if menus are exonerated, keep MCQ primary (cheaper) with behavioral as the consistency check on sentinels and headline metrics. Either way the c/r spine, pools, species, and gain bounds are shared — the channels differ only in the v_T(s) tables they fill.
