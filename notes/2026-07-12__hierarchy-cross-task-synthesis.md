# Hierarchy reconstruction: cross-task synthesis (living draft)

This note separates broad claims supported across tasks from domain-specific results and pending
tests. All semantic outcomes must come from LLM judgments; code-only similarities are candidate or
sampling diagnostics, never truth.

## Current broad claims

### Measurement correction in progress: the first corrected-R1 scores are not global precision/recall

The initial R1 reports used a fixed 450-neighbor + 450-random pair mixture and then computed
unweighted `P(LLM-SAME | partition-co-label)` inside that mixture. That is not a uniform sample of
the partition's predicted-positive pairs, so its value must not be reported as global partition
precision. Humor made the failure concrete: the mixture reported `.916` precision from only 83
co-labeled sampled pairs, while a fresh uniform sample of 300 co-labeled pairs (excluding every
evaluation and build-verifier pair) produced independent strong-LLM SAME rates `.073` and `.077`;
dual-confirmed precision was `.060` (95% Wilson interval `[.038,.093]`). Binary SAME agreement was
`.970` with kappa `.784`. Thus the high number was a sampling artifact, not evidence of coherent
Louvain groups.

The underlying humor arbiter truth was also overbroad: it labeled 230/450 random node pairs SAME,
including intonation/rhythm versus transferred epithets and sentence-level incongruity versus
consistent narrative voice. Optimizing recall against those labels would necessarily destroy
semantic precision. The completed two-judge re-audit plus third-judge adjudication yields only
45/900 SAME (`.050`), versus 584/900 (`.649`) originally. Existing-label versus corrected-truth
agreement is `.258` exact and binary SAME kappa `.048`; 541 old SAME decisions are rejected. Seven
tasks with implausibly high random-pair SAME rates (creative writing,
humor, news, peer review, grant, legal, and notice) are therefore undergoing complete replicated
R1 re-audit under the corrected narrow-construct definition. Existing descendants and R1 recall
claims for those tasks are provisional until two blind judges plus third-judge disagreement
adjudication replace the suspect truth.

The pipeline now adds a direct precision gate: a deterministic uniform sample from all co-labeled
pairs, excluding prior measurement/build pairs, receives two independent LLM judgments. Both
individual judgments are retained and disagreements can be third-judge adjudicated. The user-set
stop rule is now deliberately simple: corrected recall and directly LLM-audited precision must both
exceed `.50`. Once a task clears both, freeze it rather than continuing open-ended optimization.
Every non-singleton community receives LLM coherence refinement, while the
user-requested `>30` rule remains an additional whole-group certification rather than the sole
coherence check. Size does not itself force a split: groups over 30 may remain intact when the special
LLM audit certifies that the full membership is one construct. Recall recovery uses exact member-level
or whole-group proposals with replicated LLM decisions and is retained only when the final candidate
lies on the authorized recall–precision frontier.

### 1. Under-merge repair moves a reproducible recall–precision frontier; it is not automatically a quality win

Across all 11 tasks, current L0v3 versus v6 on the same frozen LLM-adjudicated truth raises macro
recall `.822 -> .897` (+.075) and lowers macro precision `.761 -> .677` (-.083); macro F1 falls
`.788 -> .770`. Recall rises in 11/11 tasks, precision in 0/11, F1 in 2/11. The current system is
therefore a consistent recall intervention, not a Pareto improvement. A separate 1,320-pair blind
coherence audit is measuring whether the precision loss is concentrated in head–tail decisions or
in tail–tail transitivity.

The first four audits identify a concrete mechanism rather than a generic precision tradeoff:
historical L0 confirmation used a prompt literally defining R1 “same construct,” although pipeline
documentation called it L0 “same criterion.” Independent strict-L0 confirmation is only 28–33% for
head–tail pairs and 5–12% for tail–tail pairs in math, creative writing, humor, and news. Most rejected
pairs are related-but-distinct. This predicts that a strict-L0 regrouping can recover precision while
retaining a subset of the recall gains; replication and the other seven tasks remain pending.

Update after all 11 tasks: macro same-criterion confirmation is **.292 head-tail / .065 tail-tail**.
Every task reproduces the collapse; head-tail ranges `.217–.367`, tail-tail `.000–.150`. The task-level
head confirmation rate correlates `.46` with the observed v6→L0v3 precision change (n=11), consistent
with the prompt-boundary mechanism. A second blind judge is replicating the first four domains before
any L0v4 split is authorized.

Math's completed strict global regroup supplies the first causal repair test. L0v4 retains 44 of the
488 v6-source reductions and scores recall/precision/F1 **.847/.812/.829**, versus L0v3
`.881/.742/.806` and v6 `.835/.820/.827`. It recovers seven precision points and 2.3 F1 points from
L0v3 while preserving a small recall gain over v6. This is the first balanced improvement and validates
strict semantic regrouping as the right correction direction; cross-task replication is underway.

News independently replicates and strengthens the result: L0v4 recall/precision/F1
**.881/.670/.761**, versus L0v3 `.913/.587/.715` and v6 `.870/.667/.755`. Unlike math's small
precision trade, news is Pareto-better than v6 on both recall and precision. Across the first two
domains, strict regroup recovers 7–8 precision points and 2.3–4.6 F1 points from L0v3 while preserving
small recall gains over v6. This converts the broad conclusion from “repair inevitably trades precision
for recall” to “the tradeoff was largely induced by applying an R1 relation at L0.”

Humor is the third replication and supplies a boundary condition: L0v4 **.740/.789/.764**, versus
v6 `.722/.785/.752` and L0v3 `.881/.683/.770`. Strict regroup is again Pareto-better than v6, but
F1 is `.006` below the recall-heavy L0v3 because it exchanges `.141` recall for `.106` precision.
Therefore “strict L0v4 dominates v6” holds in 3/3 domains so far; “strict L0v4 always maximizes F1
among all operating points” does not.

Macro over the first three completed domains: v6 recall/precision/F1 `.809/.757/.778`, L0v3
`.892/.671/.764`, strict L0v4 **`.823/.757/.785`**. L0v4 preserves `+.014` recall over v6 with
essentially identical precision and `+.007` F1; versus L0v3 it recovers `+.086` precision and `+.021`
F1 while relinquishing `.069` recall. The prompt correction is therefore a balanced cross-domain gain
so far, not just isolated examples.

Creative writing and peer review extend the pattern to 5/5 domains. CW L0v4 `.855/.700/.770`
versus v6 `.833/.702/.762`; peer `.801/.813/.807` versus `.791/.814/.802`. Both preserve recall
gains (+.022/+.010), essentially restore v6 precision (-.002/-.001), and improve F1 (+.008/+.005).
The strict-L0 correction is therefore a replicated cross-domain improvement, not merely a precision
rollback to v6.

Grant funding is the first neutral exception: L0v4 `.867/.735/.796` versus v6 `.867/.737/.797`
and L0v3 `.922/.610/.734`. Strict regroup removes the large L0v3 precision failure (+.125 precision,
+.062 F1) but yields no extra recall over v6 and is `.001` lower in F1. Current tally: L0v4 improves
v6 F1 in 5/6 domains and ties it practically in one; it improves L0v3 F1 in 5/6 and is `.006` lower
for humor's deliberately recall-heavy operating point.

Legal outcome prediction is a second neutral case: L0v4 **`.843/.709/.770`** exactly matches v6
on the frozen LLM truth, versus recall-heavy L0v3 `.901/.633/.744`. The strict judge retained only
6 of 454 historical v6-source reductions. Macro over the first seven completed domains is now v6
**`.823/.748/.781`**, L0v3 **`.892/.662/.758`**, and strict L0v4 **`.833/.747/.785`**. Thus the
cross-domain effect remains a small recall and F1 gain over v6 (`+.010`/`+.004`) at essentially the
same precision (`-.001`), while repairing `+.085` precision and `+.027` F1 relative to L0v3. Current
tally: F1 improves over v6 in 5/7 and is effectively tied in 2/7; no completed task is worse by more
than `.001` F1.

Press releases extends the result to eight domains: v6 `.805/.724/.762`, L0v3 `.860/.634/.730`,
and L0v4 **`.807/.724/.764`**. The strict judge retained 24 of 567 v6-source reductions, preserving
the small recall gain with no measured precision cost. Eight-domain macro is now v6
**`.821/.745/.778`**, L0v3 **`.888/.658/.754`**, and strict L0v4 **`.830/.744/.783`**. The repaired
partition is `+.009` recall and `+.005` F1 over v6 at `-.001` precision, and `+.086` precision and
`+.029` F1 over L0v3. F1 improves over v6 in 6/8 domains and is effectively tied in 2/8.

Patents is the strongest no-merge boundary condition: the strict LLM retained **zero** of 444 L0v3
v6-source reductions, so L0v4 exactly equals v6 **`.836/.789/.812`**; L0v3 was
`.898/.714/.795`. Across nine completed domains, macro v6 is `.822/.750/.782`, L0v3
`.889/.665/.759`, and L0v4 **`.831/.749/.786`**. The broad improvement remains modest but stable:
`+.009` recall and `+.004` F1 over v6 at `-.001` precision. More importantly, the method is allowed
to conclude that no cross-source L0 merge is defensible in a domain, rather than forcing consolidation.

**All 11 strict-L0 regroupings are now complete.** Macro recall/precision/F1 is v6
**`.822/.761/.788`**, prompt-mismatched L0v3 **`.897/.677/.770`**, and corrected L0v4
**`.834/.761/.793`**. Micro is correspondingly `.821/.750/.784`, `.898/.665/.764`, and
**`.833/.750/.790`**. L0v4 therefore retains `+.012` macro recall over v6 with identical reported
precision and `+.005` F1, while recovering `+.084` precision and `+.023` F1 from L0v3. F1 improves
over v6 in 7/11 domains, is exactly tied in 3/11, and is `.001` lower in grant; it improves over
L0v3 in 9/11, with humor (`-.006`) and code review (`-.017`) as genuine recall-heavy exceptions.
Notice-and-comment is the strongest Pareto gain over v6: `.822/.736/.776` to
**`.873/.742/.802`**. Code review and patents retain zero cross-v6 merges and return exactly to v6,
showing that the procedure does not force a universal consolidation rate.

Using **task as the unit of analysis** (n=11), the paired L0v4−v6 mean effect is recall `+.0115`
(95% t interval `[+.0012,+.0217]`), precision `.0000` (`[-.0025,+.0025]`), and F1 `+.0055`
(`[+.0001,+.0108]`). Leave-one-task-out mean F1 stays positive throughout (`+.0034` to `+.0061`),
and a paired Wilcoxon test gives `p=.0156` for both recall and F1 (`p=.9531` for precision). Relative
to L0v3, mean precision improves `+.0835` (`[+.0709,+.0960]`, 11/11 tasks) and F1 `+.0236`
(`[+.0086,+.0386]`, 9/11 tasks). These are across-task uncertainty summaries of LLM-judged metrics,
not claims that individual pair judgments are independent.

In inventory terms, v6 has 33,123 clusters, L0v3 27,793, and strict L0v4 32,868. Independent strict
regrouping retained only **255 of 5,330** historical v6-source reductions (`4.8%`). That very small
retained subset nevertheless supplies the statistically stable recall gain above. The broad mechanism
is therefore selective recovery of a sparse set of true cross-source duplicates, not moderate trimming
of an otherwise valid aggressive merge policy.

The rejected merges are substantively diagnostic rather than lexical noise: Hilbert–Bernays
provability conditions had been merged with generic axiomatic derivation; grant “expected outcomes”
with project “impact and scope”; and citing the rule's name with citing a specific regulatory provision.
Each pair belongs near one another at R1/R2 but makes a different operational L0 judgment. Confirmed
score-2 controls remain genuinely substitutable (for example, correctly ordered equality chains;
mixed-expertise audience accessibility; internal consistency versus absence of contradictions).

### 2. The historical R2 dip combined a real abstraction boundary with a reconstruction-method artifact

During the fresh math rebuild, a second relation-contract mismatch was caught before application:
`STRICT_BUILD_PROTOCOL_R1.txt` required operationally interchangeable measurements, effectively
repeating L0, while the R1 arbiter and pipeline define direct facets of one narrow latent construct.
Three independent agents unanimously diagnosed the collapse and independently recovered the frozen
anchor labels by majority under the intended R1 definition. Twenty-eight wrong-protocol verifier
shards were quarantined. Arbiter, verifier, and confirm prompts now share one R1 boundary; the verify
protocol path/hash is frozen in the level manifest and validated again before partition application.

Matched net+Louvain on historical peer review declined R1 `.576` -> R2 `.241` -> R3 `.087`; the
old R3 rebound to `.691` appeared after switching to forced classify/derive. Thus the rebound is not
evidence that R3 is intrinsically easy. At the same time, independent judges disagree more around
focused-theme boundaries than exact criteria, so R2 remains a substantively contested abstraction.

The repaired pipeline removes the lexical-net fragmentation mechanism from R2: global blind LLM
clustering under the calibrated focused-operational-family definition, followed by independent blind
pair auditing. Rebased humor and peer-review R2 disagreement decisions are supported over historical
Sonnet 75.6% and 70.0%, respectively.

Matched evaluation is now complete for humor and peer (900 fresh LLM pairs/cell; identical
450-neighbor + 450-random sampling; chance correction). Chance-corrected recall is humor **R1 .745 ->
R2 .333 -> R3 .227** and peer **.646 -> .216 -> .215**. The historical R2→R3 rebound disappears in
both domains: humor continues declining, while peer is flat after R2. Thus “R2 is a trough followed by
easy R3” is rejected; the supported generalization is an upper-abstraction loss beginning at R2.
Caveat: R1 matched precision is only `.150/.138`, consistent with the overmerged L0/R1 lineage, so
the curve must be rerun after strict L0v4 descendants rebuild.

### 3. More top-level resolution helps some domains but is not universally superior

Without forcing approximately five categories, the new humor hierarchy yields 13 R3 categories and
peer review 14. Blind composed-hierarchy comparison supports the new hierarchy on 59.4% of humor
disagreements but only 52.5% for peer review. This suggests domain-dependent optimal top-level
granularity rather than one universal category count.

### 4. Candidate retrieval burden grows sharply with inventory size and lexical diffuseness

Full R1 retrievable nets range from 15,827 grant pairs to 103,081 code-review pairs. Held-out LLM-SAME
diagnostic coverage at cap 9,000 varies from `.556` (code review) to `.962` (grant), despite an
identical relation and generator. Inventory scale and representation diffuseness therefore determine
the cost of achieving comparable recall. These ceilings are routing diagnostics, not semantic scores.

Versioned sweeps for the six newer tasks give cap-9k/full diagnostic coverage: code `.556/1.000`,
press `.861/1.000`, patents `.747/.952`, grant `.962/.981`, notice `.829/.987`, legal `.736/.931`.
For code review, moving from 30k to 50k pairs captures no additional held-out SAME pair; the last three
appear only between 50k and the 103.5k full net. This exposes a steep, domain-dependent marginal-cost
curve that should be reported rather than hidden behind a universal candidate cap.

## Claims not yet licensed

- “The complete new hierarchy is overall better than v6.” Corrected L0v4 is now a modest balanced
  improvement across all 11 domains, but corrected R1–R3 descendants are still pending.
- “The R2 dip is gone.” Matched measurement removes the apparent R3 rebound; it does not remove the
  real upper-abstraction loss beginning at R2.
- “Thirteen to fourteen R3 categories is generally optimal.” Humor supports added resolution; peer is tied.
- “Codex is generally better than Sonnet.” Existing comparisons measure fit to a changed, frozen definition.
- “One reconstruction method works uniformly across domains.” Candidate ceilings and R3 comparisons reject this.

## Pending tests that can broaden or falsify the claims

1. Complete strict L0v4 and fresh LLM naming for the remaining four domains.
2. Rebuild R1–R3 from corrected, hash-frozen L0v4 parents; never mix historical descendants with a
   split parent.
3. Repeat matched R1–R3 evaluation on corrected descendants to separate abstraction loss from inherited
   L0/R1 overmerge.
4. Report macro effects, task heterogeneity, uncertainty, and leave-one-task-out stability.
5. Compare Codex/Sonnet only on blind disagreements under the same frozen relation definition.

## Reporting discipline

Final conclusions will be labeled as cross-task supported, domain-specific, mechanistic but diagnostic,
or provisional. Aggregate claims will include task counts, effect sizes, uncertainty, and explicit
exceptions rather than relying on a single representative domain.
