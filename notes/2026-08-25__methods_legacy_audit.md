# METHODS/ LEGACY AUDIT — what the new VA machinery can revive (Addendum F-d)

Date: 2026-08-25 overnight. Charge: review all older methods/ code for ideas
worth updating with the certified VA machinery (frozen banks, Gemma-4-31b
guided judging + anchor batteries, Track-A/B routing, frozen Layer-1 stack,
F2 deconfounding, missing-mass accounting).

## Inventory & verdicts

| package | what it is | verdict |
|---|---|---|
| metric_tree | greedy binary-partition tree, LLM-proposed splits, router-gated inference, NA-aware restructuring (Mar-Apr era, Llama-70B backends, no anchor/nuisance hygiene, base-rate leaves) | **REVIVE — in progress.** Stage 1 (tonight): conditional_bank_tree.py — tree over the CERTIFIED bank matrix, node-local criteria selection + frozen-recipe leaf heads, honest grouped-OOF vs flat bank on same folds. Zero new LLM calls; isolates the ARCHITECTURE question (user: "selectively apply metrics to smaller subsets"). Leaves double as subcommunity candidates. |
| metrics_tree_infilling | newer sibling: MOB tree finds nodes where explicit metrics UNDER-PREDICT, asks an LLM for the missing feature per gap | **REVIVE — designated stage 2.** This is node-conditional criterion MINING — exactly the closure loop applied per-node. Modernization: certified judge + blocks format, Track-B quarantine per node, planted probes, per-node missing mass. Needs GPU judging; cost ≈ one closure round per tree node. The natural instrument for "which subcommunities need criteria the global bank lacks" (user's subcommunity question: heterogeneity = count of node-specific criteria, which the paper's abstract already promises: "quantify heterogeneity by counting new metrics needed for subfield preferences"). |
| articulation_star | STaR loop: small model articulates per-item POS/NEG rationales; information-bottleneck judge reads RATIONALE ONLY to recover the label | **REVIVE as a third articulation architecture.** Directly relevant to the capacity-null debate: a flat bank is a FIXED vocabulary; per-item free-text rationale is an unbounded-vocabulary, bounded-LENGTH articulation channel. Its ceiling vs the bank's tests whether the binding constraint is vocabulary or statability. Modernize: Gemma judge + anchors, grouped splits, honest OOF; compare rationale-only AUC vs bank AUC vs dense on the same rows. Medium cost (one LoRA train + judge passes). |
| autometrics | Algorithm-1 flat discovery + regression | superseded by the closure mining loop; keep as proposer library (its 40+ metric bank + contrastive proposer are dependencies of metric_tree). No independent revival. |
| local_explanations | bottom-up per-example rationales -> clustered concept vocabulary -> L1 bottleneck | partially absorbed (was a proposer arm in the 5-arm system). Revival value: its item-level rationales are the natural PROPOSAL SOURCE for metrics_tree_infilling gap nodes (bottom-up beats top-down in small strata). Fold into stage 2, not standalone. |
| existing_metrics_runner | coded + judge aspect catalogs (V-arm machinery) | active infrastructure, no revival needed. |
| verification_library | per-example generated Python verifiers + refactoring into a shared library | interesting V-side analog of articulation_star (program-articulation vs prose-articulation); park unless the V-frontier becomes a question. |
| codability, tacit_channels, metric_seam, claim_verification | other papers' active lines | out of scope here. |

## Recommended sequence
1. conditional_bank_tree results (tonight) — if tree ≈ flat, architecture-width
   is not the binder at fixed criteria (strengthens the paper); if tree > flat,
   the gap partly reflects GLOBAL aggregation, and stage 2 is mandatory.
2. metrics_tree_infilling modernization (stage 2, node-conditional mining) —
   the strongest response to the ε-tail null AND the subcommunity instrument.
   Needs user sign-off on judge-GPU budget.
3. articulation_star revival as the bounded-length/unbounded-vocabulary arm.
Related: Addendum F (capacity probes), notes/2026-08-25__absolute_va_scoping.md.
