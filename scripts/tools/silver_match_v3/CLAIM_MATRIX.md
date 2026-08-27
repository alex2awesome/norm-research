# Silver-match v3 claim matrix

The final result is not a peer-review-only matching score.  It is a frozen,
cross-domain measurement audit over 8 tasks, 23 naturally occurring corpora,
and 1,732,515 faithful extracted norms.  A claim is reportable only when the
listed evidence artifact exists and its gate passes.

| Claim axis | Required evidence | Reporting rule |
|---|---|---|
| Complete instrument | Manifest-to-final row-order audit for every corpus | Report only after exact UID/task/corpus/row coverage, bank hashes, and all output hashes pass |
| Cross-domain silver matching | Verified exact matches over all 8 tasks and 23 corpora | Report micro, equal-task macro, equal-corpus macro, and full task/corpus ranges; never use the largest corpus as the general conclusion |
| Honest nonmatch behavior | Every typed decision, including family-only, no explicit criterion, context, generic verdict, bank gap, noise, instability, and invalid output | Report counts and rates even when zero, overall and by task/corpus |
| Norm heterogeneity | Canonical polarity, norm kind, and old-valid/recovered-faithful provenance | Report match/nonmatch rates by stratum; do not hide recovered-tail behavior in the aggregate |
| Retriever improvement | Identical frozen labels and query text for BGE, base Nemotron, each task LoRA, and their dev-selected union | Report exact recall at 16/30/50/80, paired rank changes, capture overlap, and source-disjoint external dev/test |
| Candidate-miss control | Diverse encoder/view union plus repeated complementary 50-metric full-bank rescue captures | For rescued abstentions, prove every bank metric—including the primary K—was re-exposed exactly the frozen number of times in shifted system/lane partitions. Capture–recapture is a discovery-diversity diagnostic, not an accuracy bound |
| False-abstention risk | Repeated full-bank rescue plus blind independent labels joined to final decisions | Claim “under 5%” only when the one-sided 95% exact binomial upper bound is <.05; report point estimate and denominator. Do not substitute a capture–recapture estimate for this audit |
| Final exact-MATCH precision | Uniform blind samples from all accepted MATCHes globally and independently within each task | Claim a precision floor only when the one-sided 95% exact lower bound exceeds it; report wrong-leaf and false-positive errors separately |
| Adjudicator quality | Task-specific GEPA selected on dev, frozen test once, swapped-order check, contrastive verification | Report exact match precision/recall, typed decision confusion, invalid rate, and verification attrition |
| Exact-leaf promotion | Exact original/order agreement followed by an independently calibrated contrastive verifier | A task may emit production `MATCH` only from a precision-first dev gate with uncertainty reported; raw agreement is not evidence of correctness. Ambiguous siblings remain `MATCH_FAMILY_ONLY` or unresolved/rescue |
| Bank gaps vs extraction noise | Exhaustive bank rescue plus independent evidence verification | Separate genuine `NO_CANDIDATE_FITS` from `NOISE`, `NO_EXPLICIT_CRITERION`, and `GENERIC_VERDICT` |
| Family robustness | Immutable exact-ID result plus a separately versioned L0→R3 relation artifact | Exact IDs remain primary. New clustering is a sensitivity analysis and may not rewrite teachers or frozen tests |
| MI↔silver external validity | Hash-bound task analysis release plus a pre-existing label-free per-metric MI certificate | Exclude every retriever/GEPA/verifier train/dev/test UID and draw separate uniform blind risk audits from the remaining never-labeled rows. Primary estimand is Spearman between MI and unique-source presence over verified exact MATCHes. Report source-group bootstrap CI, metric-pair permutation p, split-half reliability, partial correlation controlling log leaf count and H_M, raw-norm/equal-corpus/polarity sensitivities, certificate coverage, exclusions, and blind precision/false-abstention status |
| External validity | Final task-level assignment matrix joined after the fact to every canonical outcome leg | Within each task, report expert-verdict, expert-revealed, and community-revealed results with the existing length/venue/exposure controls; never generalize from one label type |
| General conclusion | Cross-task/corpus meta-analysis and label-type interaction | Distinguish direct quality judgment, popularity/uptake, and gatekeeping outcomes; report heterogeneity and sign reversals rather than one pooled coefficient |

## Intended conclusion ladder

1. **Measurement claim:** naturally occurring human evaluative text can be
   converted into a complete task-bank assignment instrument with quantified
   exact-match and typed-abstention error.
2. **Architecture claim:** task-specific retrieval plus GEPA adjudication and
   exhaustive rescue improves that instrument over forced top-1 BGE matching.
3. **Domain claim:** performance and failure modes replicate—or are explicitly
   bounded—across the full task/corpus matrix rather than only peer review.
4. **Scientific claim:** after reconstruction-only matching is frozen, the
   instrument's relationship to outcomes varies predictably by label type:
   direct judged quality, gatekeeping, and popularity are not interchangeable.

The ladder is cumulative.  A downstream scientific conclusion cannot repair a
failed matching audit, and a strong aggregate cannot substitute for missing
task, corpus, revelation-type, or abstention evidence.

A task may enter MI/outcome analysis before global completion only through the
staggered task-release firewall. Once released, that task's matcher is
immutable, and its observed correlations may not influence any unfinished
task's retrieval, prompt, verifier, threshold, or bank.

The exact-ID and family-level instruments must also remain distinct.  A weak
exact-leaf gate cannot be repaired by silently scoring a related family member
as correct; family-only decisions are reported explicitly and the separately
versioned relation graph is used only for sensitivity analyses.

Outcome-source provenance, known canonical paths, missingness restrictions, and
the three-leg reporting design are pinned in `DOWNSTREAM_OUTCOME_CONTRACT.md`.
