---
source_url: https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/
title: AAAI-26 Reproducibility Checklist
source_type: conference
venue: AAAI
fetched: 2026-05-09
---

# AAAI Reproducibility Checklist

## Overview

The AAAI Reproducibility Checklist is required for all AAAI submissions and is used by reviewers to evaluate whether papers meet standards for transparent reporting and reproducibility.

## Structure

The checklist has three main sections:

### 1. For all papers

- A clear description of the mathematical setting, algorithm, and/or model
- Clear specification of the contribution and what is novel
- Limitations of the approach are clearly stated
- Theoretical claims have proofs (if applicable)

### 2. For datasets

- Dataset is publicly available or detailed access instructions provided
- Citation to the source of the dataset
- Description of any data filtering, preprocessing, or augmentation
- Splits clearly described
- Statistics about the dataset (size, label distribution, etc.)

### 3. For experiments

- Code is publicly available or will be made available upon publication
- Hyperparameter selection process is described
- All hyperparameters used are specified
- Number of training runs reported
- Variation across runs reported (error bars, confidence intervals)
- Hardware specifications (GPU/CPU type, memory)
- Approximate training time and computational cost
- Statistical significance tests used to compare to baselines
- Choice of baselines is justified

## Reviewer Use

Reviewers will be asked to use the checklist as one factor in their evaluation. Authors must provide responses for each item, and "no" responses require justification. While "yes" is generally preferable, "no" or "n/a" is not automatic grounds for rejection if properly justified.

## Importance for Empirical Papers

If the paper includes experiments, a "no" answer regarding reproducibility will not be perceived well by reviewers — particularly for items related to:
- Hyperparameter specification
- Number of runs and variability
- Code availability
