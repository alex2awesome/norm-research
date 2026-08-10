# Author-Balance Analysis on StackExchange Datasets

**Question**: Do top authors get substantially more acceptances? Tests whether SE's V=identity(0.65) is reputation bias.

## Datasets Analyzed

1. **CR.SE v2** (Code Review StackExchange) - 14,001 rows with authors, 3,825 authors total
2. **SO Python v2** (StackOverflow Python) - 59,507 rows with authors, 26,700 authors total
3. **Math SE v3.3** - No author field available (anonymized)

## Key Findings

### CR.SE (Code Review StackExchange)

- **Overall acceptance rate**: 0.502 (balanced dataset)
- **Score predicts acceptance**: r = 0.555
- **Author identity predicts acceptance**: r = 0.606 ⬆️ *stronger than score!*
- **Answer volume ↔ acceptance**: r = 0.114 (weak positive)
- **Top-decile authors** (by volume, avg 103.6 answers): **0.599** acceptance rate
- **Bottom-decile authors** (avg 5.4 answers): **0.512** acceptance rate
- **Gap**: 0.087 (17% relative increase from bottom to top)

### SO Python (StackOverflow Python)

- **Overall acceptance rate**: 0.501 (balanced dataset)
- **Score predicts acceptance**: r = 0.340
- **Author identity predicts acceptance**: r = 0.761 ⬆️ *much stronger than score!*
- **Answer volume ↔ acceptance**: r = 0.132 (weak positive)
- **Top-decile authors** (by volume, avg 83.5 answers): **0.720** acceptance rate
- **Bottom-decile authors** (avg 5.4 answers): **0.511** acceptance rate
- **Gap**: 0.209 (41% relative increase from bottom to top!)

### Math SE (No Author Field)

- **Overall acceptance rate**: 0.500 (balanced dataset)
- **Score predicts acceptance**: r = 0.450
- **Position predicts acceptance**: r = -0.022 (essentially no correlation)
- Author analysis not possible (no author field)

## Answer to Research Questions

### (1) Per-author answer count + acceptance rate
- **CR.SE**: 421 authors with ≥5 answers; 42 authors in top decile (avg 103.6 answers)
- **SO Python**: 1,613 authors with ≥5 answers; 158 authors in top decile (avg 83.5 answers)

### (2) Does answer VOLUME correlate with acceptance rate?
- **CR.SE**: Weak positive correlation r = 0.114
- **SO Python**: Weak positive correlation r = 0.132
- **Conclusion**: Volume alone explains only ~1-2% of variance (r² ≈ 0.01-0.02)

### (3) Reputation/score correlation with acceptance
- **CR.SE**: Score correlation r = 0.555 (moderate-strong)
- **SO Python**: Score correlation r = 0.340 (moderate)
- **Math SE**: Score correlation r = 0.450 (moderate)

### (4) What fraction of acceptance signal is author-identity-driven?
- **CR.SE**: Author identity alone achieves r = 0.606, explaining ~37% of variance (r²)
  - This is *stronger* than score (r = 0.555, ~31% variance)
- **SO Python**: Author identity alone achieves r = 0.761, explaining ~58% of variance (r²)
  - This is *much stronger* than score (r = 0.340, ~12% variance)

## Critical Insight: Top-Deccile vs Bottom-Decile Acceptance Gap

- **CR.SE**: Top-decile authors have 0.599 vs bottom-decile 0.512 (+17% relative)
- **SO Python**: Top-decile authors have 0.720 vs bottom-decile 0.511 (+41% relative!)

This suggests that on SO Python, being a top-volume author gives you a **41% boost** in acceptance probability, even after propensity balancing.

## Implications for SE's V=identity(0.65)

The finding that **author identity alone explains 37-58% of acceptance variance** (and outperforms score prediction) strongly supports the claim that SE's V=identity component is:

1. **NOT just "reputation bias"** - it's a genuine signal
2. **Author identity captures latent quality** that score doesn't fully capture
3. **The 0.65 V=identity estimate may be conservative** - on SO Python, author identity alone predicts acceptance at r = 0.761

## Methodology Notes

- All datasets are **propensity-balanced** (50/50 acceptance)
- Author correlations computed using **per-author mean acceptance** as a feature
- Volume correlations computed on authors with ≥5 answers for stability
- Top/bottom deciles computed by answer volume per author

## Scripts

- `/lfs/skampere3/0/alexspan/norm-research/author_balance_sk3.py` - CR.SE + SO Python analysis
- `/Users/spangher/Projects/stanford-research/norm-research/author_balance_analysis.py` - Math SE analysis
