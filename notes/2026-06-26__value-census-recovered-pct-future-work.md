# Future work: the supervised "recovered %" measure (value census, §12.3)

> **UPDATE 2026-06-26 (§12.3 reframe, later same day):** the value census is now **M_i-based and
> UNSUPERVISED** (per-metric, anchor-free — value = each criterion's contribution to recovering M_i,
> never the aggregate Y). `recovered %` is now `R_full / H(M_i)` = **fraction of M_i recovered**, not a
> Y-supervised quantity. So it is **no longer "supervised / deferred to future work"** — it is the
> canonical per-metric value census (`value_census.py` v(s)=I(M_i;σ), `run_value_census.py` per-metric,
> reads M_i from the checkpoint). The Y-based task-level run below is the SUPERSEDED earlier form; kept
> for the "is it a standard metric?" answer + the open questions, which still apply (now against M_i).

*Logged 2026-06-26 at the user's request — they find this interesting, want to think about it more, and
may use it later. It is NOT part of the current UNSUPERVISED metric-level α task.*

## What it is

**recovered % = R_full / H(Y) × 100**, computed in `methods/metric_implementer/experiments/run_value_census.py`
(`frac_label_recovered`), with the heavy lifting in `value_census.submodular_value` + `value_missing_mass`:

- **H(Y)** = binary entropy of the label Y (bits). E.g. peer-review accept/reject at base rate 0.73 → 0.84 bits.
- **R_full** = the *submodular greedy* total (bits): from nothing, repeatedly add the criterion with the
  largest **conditional** mutual info `I(Y; σ(s) | already-selected)` (Shannon CMI, 5-fold logistic OOF
  surrogate — so cross-validated, not overfit), stopping when the marginal gain < 0.005 bits. R_full = Σ
  of those marginal gains. Greedy over the **top-150 criteria by additive value** (sound because
  submodularity gives v(s|sel) ≤ v(s), so low-additive criteria can't be high-submodular).

Plain English: **the fraction of the label's uncertainty that the best combination of the criteria's
behaviors predicts.** Peer-review: 0.07 bits / 0.84 bits = 8%.

## It is SUPERVISED — separate from the unsupervised metric-level α task

- **α (behavior census, §12.1a) is UNSUPERVISED**: σ(p) = soft P(YES) vectors; species = behavioral
  equivalence; α = behavior-space dimensionality. No labels. This is what the per-R2-cluster metric α
  sweep measures.
- **recovered % / α_V (value census, §12.3) is SUPERVISED**: needs the label Y. So it is a *different,
  future-work track*, not part of the unsupervised metric α measurement the user is currently running.
  Run it when a supervised metric-evaluation question is on the table.

## Is it standard?

Yes, built from standard parts:
- `R_full = I(Y; greedy-selected set)` = **submodular mutual-information feature selection** (Krause &
  Guestrin near-optimal greedy info-gain, (1−1/e) guarantee).
- `I(Y;X) / H(Y)` = the **uncertainty coefficient** `U(Y|X)` — a standard normalized-mutual-information
  variant, i.e. an **information-theoretic R²** ("fraction of entropy explained").
- So recovered % = the uncertainty coefficient of a submodular-greedy-selected criterion set. Both
  ingredients standard; the combination is clean and defensible.

## Task-level empirical result (2026-06-26, context only)

Ran the value census on all 8 task-level behavior checkpoints. recovered % uniformly weak:

| task | recovered % | verdict |
|---|---|---|
| math | 4% | FLAT-LOW |
| notice-and-comment | 5% | FLAT-LOW |
| patents | 7% | FLAT-LOW |
| creative-writing | 7% | FLAT-LOW |
| peer-review | 8% | FLAT-LOW |
| news-homepages | 9% | FLAT-LOW |
| law | 15% | TRACKS* |
| humor | 19% | TRACKS* |

(*barely above the 15% FLAT-LOW threshold.) α_V ≈ α everywhere → the breadth gap is uninformative at the
task level because the signal itself is weak (proposer criteria don't carve the task objectives). This is
the TASK level; the METRIC level (tighter, on-target criteria) should recover a much higher fraction —
worth revisiting once a supervised per-metric question arises.

## Open questions to think about

- **Label choice:** Y = accept/reject (external objective, but off-target for review-quality criteria, cf.
  the FLAT-LOW finding) vs M_ω = the recovered metric (on-target but partly circular — criteria predict a
  metric built from criteria). GV1 in §12.3 flags both framings.
- **additive vs submodular gap** = the criterion space's redundancy (the additive Σ I(Y;σ) over-counts;
  the gap quantifies overlap). Already reported; could be a standalone "redundancy" diagnostic.
- **Greedy/top-K cap & eps-stop** make R_full a lower bound on joint predictability — when does the full
  criterion set (vs greedy-150) materially change recovered %?
- **Per-metric recovered %** (supervised) paired with per-metric α (unsupervised): a metric could be
  low-α (coverable) yet high-recovered% (the few criteria capture the objective) — the ideal case.

## Status

Built + tested (7 unit tests; `value_census.py`, `run_value_census.py`, `manifest.load_corpus_labels`).
Deferred to future work per user 2026-06-26: "interesting measure, not relevant to the current
unsupervised metric-level task; want to think about it more." Codex audit of the implementation was
queued (task #71) — its findings should be folded in before any future supervised use.
