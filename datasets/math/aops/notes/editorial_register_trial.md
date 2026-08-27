# Editorial-register trial: wiki (y=1) vs forum (y=0) classifier

2026-06-12. Script: `scripts/editorial_register_trial.py`; OOF scores:
`editorial_register_oof.parquet`. Design: problems present in both corpora,
GroupKFold(5) by problem, near-dup (sim>0.9 transcribed) handling in two runs.

## AUC (out-of-fold, group-split by problem)

| representation | A: near-dups excluded | B: included |
|---|---|---|
| word TF-IDF + LR | 0.911 | 0.909 |
| char 3-5gram + LR | 0.919 | 0.917 |
| combined | **0.920** | 0.918 |
| structural only (len, display math, paras, enum) | 0.630 | 0.628 |

Editorial register is highly learnable and overwhelmingly *linguistic*
(combined − structural ≈ 0.29). Near-dup contamination doesn't drive it.

## What the register consists of (top word features)

- **→ editorial**: `boxed`/`textbf` answer-format convention (the dominant
  block), didactic connectives (*therefore, because, since*), expository
  *we*, `align` environments, "in solution"/"this solution" cross-references.
- **→ forum**: first-person (*my, me, think, did, got*), informal evaluatives
  (*just, easy, nice, clearly*), competition-proof jargon (*claim,
  blacksquare, iff, implies*), quote blocks.

CAVEATS: (1) the `\boxed{\textbf{(A)}}` answer convention is a formatting
norm, not prose — part of the 0.92 is convention-detection; (2) one username
leak survived cleaning (`mathboy282`, wiki attribution lines) and `com` (URLs).

## Run 2 (2026-06-12): attribution-stripped rerun — finding ROBUST

`_strip_attributions()` added (Solution-by/~username/user-page links).
Combined AUC 0.920 → **0.916** (excl near-dups; 0.914 incl) — attribution
leakage was worth ~0.004. Skew unchanged (3.4% of forum posts above wiki
median; 16.7% > 0.5). Thanks follow-on still null everywhere (within-problem
wilcoxon p=0.55–0.92, incl. verified-correct). Residual leak: one prolific
contributor's signature variant (`vladimir shelomovskii`/`vvsss`, coef ~3.2)
survived the regex — small vs `boxed`-convention coefs (~12–15); not worth a
third run. The didactic-vs-personal register axis stands: struct-only 0.63 →
0.92 requires the language itself.

## Skew: do forum answers "look editorial"? Mostly no.

Forum OOF P(editorial) deciles: 0.001 / 0.041 / 0.070 / 0.102 / 0.142 /
0.191 / 0.257 / 0.340 / 0.449 / 0.603 / 0.978.
Wiki median = 0.782.

- forum posts above wiki median: **3.2%**
- forum posts above 0.5: **16.2%**
- forum posts above 0.9: **0.5%**

The two registers barely overlap; a thin (~3%) editorial-looking tail of
forum posts exists (candidates: transcription sources, editorial-minded
posters).

## Follow-on: editorial-likeness does NOT earn thanks

OOF P(editorial) vs de-confounded thanks (`thanks_resid`):

| subset | pooled ρ | within-problem mean ρ | wilcoxon p |
|---|---|---|---|
| all forum (n=32,815) | +0.028 | +0.009 | 0.48 |
| excl near-dups | +0.027 | +0.004 | 0.80 |
| verified-correct (n=7,854) | +0.032 | +0.004 | 0.84 |

Same verdict as raw similarity: editorial register is a *correctness-adjacent
style*, not what the community rewards. Combined with the de-confounding
result (length AUC 0.594 is the only live thanks signal), the picture is:
thanks ≈ effort/length, not polish, not correctness.
