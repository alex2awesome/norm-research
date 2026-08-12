# Style Invitational — PAIRWISE FRONTIER-JUDGE PROBE

Date: 2026-08-11. Charge (user-approved; coordinator): SI is canonical humour-CURATION,
and its v2 bank **failed certification** — K=50 winner-vs-honorable-mention reads
**AUC .483 sign-corrected / .509 raw**, with **0 of 33 criteria** clearing |AUC−.5| ≥ .05
(`notes/2026-08-10__si_mature_bank_rebuild.md` §7c/§7e). That failure left two very
different diagnoses **confounded**:

  (i) **the criteria are wrong** — they do not name what separates a winner from an HM;
  (ii) **the judge cannot see it in absolute mode** — the editor's construct is inherently
       COMPARATIVE, and item-level absolute scoring cannot carry it.

An absolute-scoring instrument cannot tell these apart. A **pairwise** judge can.

Code + artifacts: `datasets/humor/style_invitational/pairwise/`.
Population: the clean 8,063-row SI population (`va_v2/population.csv.gz`, 316 weeks,
1,574 parse artifacts flagged and excluded). Bank: the v2 36-criterion rubric set.

---

## 1. VERDICT (holistic leg complete)

**THE FRONTIER JUDGE SEPARATES, decisively.** On **200 within-week, length-matched**
winner-vs-honorable-mention pairs, gpt-5.6-sol picks the editor's winner **81.0% of the
time** [Wilson 95% **.750, .858**], against a **57.8%** "pick the longer entry" baseline
on the same pairs.

This is coordinator fork **(b)**: the full pairwise instrument is warranted, and §5 specs
it before anything is built.

**The headline in one line:** the same corpus, the same clean population and the same
winner-vs-HM contrast that an absolute-scoring Gemma bank reads at **.483 (chance)** is
read at **.810** by a frontier judge asked comparatively. **The editor's cut is text-
recoverable. The v2 failure was not the construct.**

---

## 2. Design (fixed before any judge call) — `build_pairs.py`

1. **WITHIN-WEEK ONLY.** Both members of every pair come from the same `week_id`, so the
   contest prompt, topic, era and editor are identical by construction. This is the
   pairwise analogue of the programme's within-container readouts.
2. **Winner vs honorable mention.** `runnerup` rows are **excluded** — the cell's y pools
   winner+runnerup, but a probe should run at the sharpest available contrast first.
   Registered as a scope limit.
3. **The length confound is designed for, not discovered.** Winner median 104.5 chars vs
   HM 89. Two arms, both always reported: **MATCHED** (|log length ratio| ≤ .20, primary)
   and **FREE** (no caliper, secondary). The "pick the longer entry" baseline is computed
   on the same pairs and printed beside every judge number.
4. **Order balance + a position-bias arm.** Which entry appears first is set by a stable
   sha256; realised balance **.496**. A **SWAP** arm re-asks 60 matched pairs with the
   order reversed — the pairwise analogue of an anchor, and the failure mode most likely
   to fake a positive.
5. **Planted known-direction anchors**, corpus-native: a real entry vs a word-scrambled
   entry (ANCHOR_SCRAM) and a real entry vs one of the population's parse-artifact rows
   (ANCHOR_FRAGMENT).
6. **Tokens, not characters**; item view matched to the bank's; byline left in place
   exactly as the bank's item view leaves it.

**Realised set: 371 pairs over 285 weeks** — MATCHED 200, FREE 81, SWAP 60,
ANCHOR_SCRAM 15, ANCHOR_FRAGMENT 15. Candidate pool 6,279 within-week winner-HM pairs
across 281 weeks; at most one pair per week per arm.

**Judge: gpt-5.6-sol via `codex exec`, effort high, read-only sandbox outside the repo.**
Family recorded rather than chosen: the session's Claude subagent budget is exhausted.
371 pairs × 9 questions = 2,211 comparisons, batched 25 pairs per sealed call.

---

## 3. Holistic results

| arm | n | accuracy = pairwise AUC | Wilson 95% | longer-entry baseline | excess |
|---|---:|---:|---|---:|---:|
| **MATCHED (primary)** | 200 | **.810** | [.750, .858] | .578 | **+.233** |
| FREE | 81 | .741 | [.636, .824] | .556 | +.185 |
| SWAP (same 60 pairs, reversed) | 60 | .950 | [.863, .983] | .550 | +.400 |
| ANCHOR_FRAGMENT | 15 | .800 | [.548, .930] | .933 | — |
| ANCHOR_SCRAM | 15 | **.067** | [.012, .298] | .600 | — (see §4) |

**The length control is decisive and it goes the wrong way for a length story.** Inside
the MATCHED arm:

| | n | accuracy |
|---|---:|---:|
| winner is the LONGER entry | 112 | .786 |
| **winner is the SHORTER entry** | **81** | **.840** |

A length-driven judge is above chance when the winner is longer and **below** chance when
the winner is shorter. This judge is **better** when the winner is shorter. Length is not
carrying the result.

**Position bias is present but small and does not explain the effect.** Pooled side-A pick
rate **.531**. On the 60 pairs asked in both orders: original order **.817**, reversed
**.950**, **consistency .800** (share answering the same underlying entry both ways). The
order-averaged accuracy on those pairs is **.883**. So ~20% of pairs are position-labile,
the pooled estimate is not inflated by position (if anything the primary .810 is the more
conservative of the two orders), and the effect survives any reading.

---

## 4. THE SCRAMBLED ANCHOR IS INVALID BY CONSTRUCTION — a defect in MY probe, recorded

ANCHOR_SCRAM reads **.067**: the judge picked the word-scrambled string over the real
entry in 14 of 15 anchors. Inspected manually before drawing any conclusion (the V9 rule),
and the anchors are the problem, not the judge:

* SI entries are **short** (median 89 chars, often 4–10 words), so a scramble is often
  near-coherent: `AS01` offered `"Kevorkian's signature. Jack"` against
  `"What a dyslexic sees in his rearview mirror."` — the scramble reads as a plausible
  droodle answer.
* **This corpus runs rearrangement contests.** `AS02`'s own prompt is *"Take the name of a
  person or institution. Find within it a hidden message… you may not move letters
  around"*; `AS06`'s is a name-mating contest. Against such prompts a scrambled string is
  not merely plausible, it is **prompt-appropriate**.
* The holistic question asks *"which did the editor pick as WINNER?"* and states that both
  entries were published. Given that framing, "the strange one is the clever one" is a
  defensible inference — the judge is answering the question asked.

**So ANCHOR_SCRAM tests "which is real text?", not "which won", and it is not a valid
known-direction anchor for the holistic question.** It is retired for this question and
its .067 must never be quoted as a judge failure. It would be valid for a discrimination
question, and that is where a future probe should use it.

**The certification the probe does have** is ANCHOR_FRAGMENT at **.800** (a byline
fragment is genuinely not an entry under any reading), the position-bias arm, and the
length split — three independent checks, all passing. The anchor gate is met; the probe is
informative.

**Generalisable lesson:** on a corpus of very short items, word-scrambling does not
reliably produce a known-direction negative. Scramble anchors need an item long enough
that shuffling destroys sense — and a prompt family that does not reward apparent
nonsense.

---

## 5. WHAT THIS RESOLVES, AND WHAT IT DOES NOT

**Resolved.** The v2 certification failure is **not** evidence that the editor's cut is
unlearnable or that the winner/HM distinction is noise. The same contrast reads .483 under
absolute Gemma-4-31B scoring and **.810** under comparative frontier judging. Diagnosis
(i)-vs-(ii) is settled in favour of "the instrument's *mode* was wrong", at least in part.

**Now resolved too — and it is the second branch.** The criterion leg completed
(8 criteria × 200 matched pairs + anchors, 80 sealed calls, 95/95 jobs, **0 unanswered
pair slots**). **Not one criterion separates.**

| criterion | orient | acc | Wilson-lo | vs length baseline .578 | sign-corrected |
|---|---|---:|---:|---:|---:|
| a08 Comic leap takes more than one step | pos | **.575** | .506 | **−.003** | — |
| a32 Ending is the strongest beat | pos | .565 | .496 | −.013 | — |
| a11 Unlikely to be independently duplicated | pos | .550 | .481 | −.027 | — |
| a15 Punch word occupies the final position | pos | .525 | .456 | −.052 | — |
| a19 Every clause carries comic weight | pos | .510 | .441 | −.068 | — |
| a07 Joke is available from the prompt alone | neg | .495 | .426 | — | .505 (lo .436) |
| a01 Continues past its own punch | neg | .485 | .417 | — | .515 (lo .446) |
| a06 Could serve a different prompt unchanged | neg | .475 | .407 | — | .525 (lo .456) |

**0 of 8 clear the registered bar.** The best, a08, is the same criterion that was the v2
bank's strongest in absolute mode (.528 there, .575 here) — it is the only one whose
Wilson lower bound clears .5 at all, and its point estimate still sits *below* the
length baseline on the same pairs. Every negatively-oriented criterion is within noise of
chance under both readings.

**This is the pre-registered "only holistic carries it" branch, and it sharpens the
verdict of §1 rather than softening it.** Switching from absolute to comparative scoring
does **not** rescue the v2 criteria: the same frontier judge, on the same 200 pairs, in
the same call format, goes from **.810 when asked "which did the editor pick"** to
**≤.575 when asked about any named criterion**. So:

* the construct **is** text-recoverable (holistic .810) — the v2 certification failure was
  never evidence that winner-vs-HM is noise;
* but the failure is **not** attributable to scoring mode alone — **the v2 criteria do not
  name what the judge is using**. The gap between **.575 and .810** is a direct measure of
  how much of the editor's cut the current bank misses.

Both diagnoses from the charge were partly right, and the probe separates their shares:
mode mattered (absolute Gemma .483 → pairwise frontier .810 holistically), and content
mattered more (pairwise frontier on the named criteria: ≤.575).

*Caveat on the criterion anchors.* The per-criterion `anchor_acc` column mixes
ANCHOR_FRAGMENT with the ANCHOR_SCRAM pairs that §4 retires, and "which better exemplifies
<criterion>" has no known direction on a scrambled item anyway. Those numbers (.30–.97)
are diagnostics only and no criterion is certified or condemned by them.

---

## 6. SPEC — the full pairwise instrument (design note BEFORE building, per the charge)

Proposed, not improvised, and not yet built.

**6.1 Item scores from pairwise comparisons: Bradley-Terry.** Each within-week comparison
`i ≻ j` is a Bernoulli observation; fit
`P(i ≻ j) = σ(θ_i − θ_j)` by penalised MLE with an L2 prior on θ. Fit **within week** —
every comparison is within-week by construction, so θ is identified only up to a per-week
constant, which is exactly right: the readout the cell needs is *rank within contest*, and
a per-week intercept is a nuisance to be absorbed, not estimated. Report θ as the item
score and evaluate it with the programme's standard grouped-OOF AUC against `y_top_tier`,
so the pairwise instrument is directly comparable to VA_nl and T.

**6.2 Comparison budget and design.** Full within-week round-robin is O(n²) per week
(some weeks carry 40+ entries). Use a **sparse connected design**: per week, a random
regular graph of degree d = 4–6 over its entries, seeded by stable hash, which keeps the
comparison graph connected (a requirement for Bradley-Terry identifiability within the
week) at ~2–3 n comparisons instead of n²/2. Budget at d = 5: ≈ 20k comparisons over the
8,063-row population — batched 25 per call, ≈ 800 sealed calls.

**6.3 Per-criterion instrument — AND THE CRITERION LEG CHANGES WHAT GOES IN IT.** The
original plan was to reuse the graph for each surviving v2 criterion. **No v2 criterion
survived** (§5), so building per-criterion θ on the current bank would be fitting noise at
800 calls a criterion. The revised spec:
  * **Phase 1 (build now):** holistic θ only, on the sparse graph. That is the arm with
    demonstrated signal and it is what §6.5 evaluates.
  * **Phase 2 (mine first, then build):** run the closure programme's Track-A mining
    **in comparative form** — show a sealed fleet within-week *pairs* the holistic judge
    got right and wrong and ask what distinguishes them — and only then commit judge
    budget to per-criterion θ. Mining against an absolute-scoring residual is what
    produced the v2 bank; mining against a *pairwise* residual is the untried move, and
    the .575/.810 gap says there is a lot there to name.

**6.4 Mandatory controls, carried from this probe.**
* every comparison order randomised by stable hash, with a ≥10% swap arm and a reported
  consistency figure;
* the length baseline and the winner-longer/winner-shorter split reported for every
  criterion, not just pooled;
* ANCHOR_FRAGMENT-style planted pairs in every batch; **no scramble anchors** on this
  corpus (§4);
* **position-debiased fitting**: include a side term in the Bradley-Terry likelihood
  (`P = σ(θ_i − θ_j + γ·side)`) so the measured .53 side preference is absorbed rather
  than propagated into θ.

**6.5 The comparison that makes it worth building.** Fit Bradley-Terry θ on the pairwise
holistic judgements, then report grouped-OOF AUC of θ against `y_top_tier` beside
`VA_nl .6011` and `T .6241` on the same rows. If θ lands near the .81 pairwise accuracy
implies, this cell's whole ledger is re-based; if it lands near .60, the pairwise gain is
real but does not survive aggregation to item level, which is itself the finding.

---

## 7. Claim discipline

* The **.810** is a *within-week, length-matched, winner-vs-HM pairwise* accuracy from
  **gpt-5.6-sol**. It is not an item-level AUC and must never be compared directly to
  VA_nl .6011 or T .6241 without saying so — §6.5 is the comparison that would make them
  commensurable, and it has not been run.
* **Runners-up are excluded**; this is the winner-vs-HM contrast only.
* ANCHOR_SCRAM's .067 is a defect in the probe's own anchor design (§4), not a judge
  failure, and is never quoted as evidence about the judge.
* The judge is a single frontier family (gpt-5.6-sol). A second family has not been run;
  a cross-family replication is the obvious next control before the full instrument is
  built on this result.
* Position consistency is **.800**, not 1.0 — the pooled .810 is the conservative order;
  quote the consistency whenever the accuracy is quoted.
* **Never quote the .810 as evidence that the v2 criteria work.** They do not: the same
  judge on the same pairs reads ≤.575 on every named criterion (§5). The holistic number
  and the criterion numbers must always travel together.


---

## 8. PHASE 1 (approved 2026-08-11) — status, and the cross-family control that changes a claim

### 8.1 CROSS-FAMILY REPLICATION — the single-family caveat is now BOUNDED, and it bites

30 of the same 200 MATCHED winner-vs-HM pairs, **byte-identical prompt text and identical
A/B order assignment**, re-judged by **glm-5.2** (different family, different vendor):

| judge | accuracy on the same 30 pairs | anchors |
|---|---:|---:|
| **gpt-5.6-sol** (main wave) | **.933** | — |
| **glm-5.2** (replication) | **.533** | **6/6** |
| per-pair agreement between them | **.533** | — |

**The pairwise separation is FAMILY-SPECIFIC, not a general property of "a frontier
judge".** glm-5.2 passes the fragment anchors 6/6 — it is reading the entries perfectly
well — and is nonetheless **at chance** on winner-vs-HM, agreeing with sol no more often
than a coin. (Wilson on .533, n = 30, is [.361, .697]: it includes .5 and excludes .81.)

This materially rewrites §1's claim and the rewrite is the honest one:

* **NOT** "the editor's cut is text-recoverable by frontier judges";
* **BUT** "the editor's cut is text-recoverable **by gpt-5.6-sol**, at .810 on
  length-matched within-week pairs, and **glm-5.2 cannot do it at all**".

The construct sits **at the edge of current judge capability and is capability-graded** —
which also explains why Gemma-4-31B reads .483 in absolute mode without that being
evidence about the construct. Any instrument built on this result inherits a hard
dependency on one model family, and that dependency must travel with the number. A third
family (e.g. a Claude judge, unavailable this session) is the obvious next control.

### 8.2 Bradley-Terry instrument — built, running, not yet landed

Scope chosen so §6.5 is commensurable **by construction**: the graph covers the dense
arm's **eval ∪ test rows only** (810 + 797 = 1,607 items, 80 weeks, zero week overlap
between splits), because T exists only on held-out rows. Those are exactly the rows for
which the rebuild note reports same-rows VA_nl (eval .6165 / test .6042) and T
(eval .6241 / test .6237).

Graph: per-week circulant C_n(1, 2, ⌊n/2⌋), **mean degree 5.44**, **every week asserted
CONNECTED** (Bradley-Terry θ is identified only within a connected component), order
balance .495. **4,406 comparisons + 40 ANCHOR_FRAGMENT + 440 SWAP = 4,886.**

Fit (`fit_bt.py`, written and ready): penalised MLE of
`P(i≻j | i on side s) = σ(θ_i − θ_j + γ·s)`, with the **side term γ** absorbing the
measured position preference, **θ centred within week**, and a λ ∈ {0.3, 1, 3} sensitivity.

**Status: the judge leg is ~9% complete and still running** (425 / 4,886 comparisons at
the last checkpoint). Per-call latency at effort=high (~560 s) rather than parallelism is
the binding constraint. The leg is resume-by-output-file and the full resume state,
including exact commands and every registered decision, is written to
`datasets/humor/style_invitational/pairwise/phase1/RESUME.md`.

**§6.5 has therefore NOT been produced, and per the approval `.810` remains QUARANTINED
from the strict list until it is.**

### 8.3 What Phase 2 should inherit if θ lands

Phase 2 (mining against the pairwise residual) is on hold pending §6.5, correctly. Two
things from §8.1 should be folded into it when it is released: the mining fleet's
comparative judge must be the family that can actually see the construct, and the
capability-grading itself is now a finding worth a controlled arm (does separation rise
monotonically with judge capability, as the OSL executor-scaling line would predict?).
