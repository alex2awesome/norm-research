---
title: "What Can We Do to Improve Peer Review in NLP?"
authors: "Anna Rogers and Isabelle Augenstein"
year: 2020
journal: "Findings of EMNLP 2020"
source_type: "modern_academic"
era: "modern_post_2000"
url: "https://aclanthology.org/2020.findings-emnlp.112/"
domain: "nlp_machine_learning_peer_review"
captured: "2026-05-09"
---

# Rogers & Augenstein 2020 — Reviewer Heuristics in NLP (Anti-Patterns to Avoid)

Influential paper that catalogued the reviewer "anti-patterns" widely used (often unconsciously) in NLP peer review and proposed corrective practices. The list of heuristics was originally 8 and has since grown — incorporated into ARR Reviewer Guidelines.

## Reviewer Heuristics / Anti-Patterns to Avoid (per the paper and its later extensions)

A review should NOT recommend rejection on grounds of:

1. **"The paper is not SOTA"** — failure to set new state-of-the-art is not by itself a reason to reject; many useful contributions are not best-on-benchmark.
2. **"This is not novel / I have seen this idea before"** — without specific citation showing prior identical work, a vague feeling of familiarity is not a substantive critique.
3. **"The improvement is too small / not significant"** — small effects are not necessarily uninteresting; statistical significance and effect-size discussion are required of the reviewer.
4. **"Not enough experiments / I would have run X more experiments"** — reviewers should ask only for experiments **necessary** for the claim, not for additional experiments they personally find interesting.
5. **"This is not deep learning / not neural"** (or, conversely, "this is just neural networks") — methodological prejudice rather than substantive evaluation.
6. **"This is not interesting to me"** — reviewer's personal taste is not a discipline-relevant ground for rejection.
7. **"The paper is not well written"** — presentation defects, unless catastrophic, should be raised as fixable concerns, not rejection grounds.
8. **"Wrong venue / not for *ACL"** — fit complaints should be flagged to the area chair, not used as a default reject.

## Extended Anti-Pattern List (post-2020 ARR additions)

9. **"You did not cite my paper"** (coercive citation).
10. **"This requires expensive compute"** as a reject — penalises low-resource groups.
11. **"This is not in English"** as a critique (or expecting English-only references).
12. **"The dataset is too small"** without considering domain constraints.
13. **"You should have used model X"** — preference, not necessity.
14. **"Your method is not theoretically motivated"** when empirical contributions are valid.

## Constructive Reviewer Norms Proposed

The paper recommends positive reviewer behaviours:

- **Cite specific prior work** when claiming lack of novelty.
- **Distinguish "blocker" from "wishlist"** comments.
- **Consider resource constraints** of the authors.
- **Acknowledge contributions** before raising concerns.
- **Offer constructive alternatives** for any criticised choice.
- **Apply the same standards** across submissions in the batch.
- **Calibrate to the venue** (workshop / short / long paper).
- **Engage with the author response** during rebuttal, not stand on initial review.

## Operational Norms Adopted by ARR

- Reviewers must complete the **Responsible NLP Research Checklist** as part of their review.
- Reviewers must use the **standard scoring axes**: Soundness, Excitement, Reproducibility, Ethical Concerns, Overall Recommendation, Confidence.
- Reviewers must **provide actionable revision suggestions** for any below-threshold score.

Sources:
- [ACL Anthology: Rogers & Augenstein 2020](https://aclanthology.org/2020.findings-emnlp.112/)
- [The Gradient: How Can We Improve Peer Review in NLP?](https://thegradient.pub/how-can-we-improve-peer-review-in-nlp/)
- [ARR Reviewer Guidelines](http://aclrollingreview.org/reviewerguidelines)
- [Responsible NLP Research Checklist](http://aclrollingreview.org/responsibleNLPresearch/)
