---
source_url: https://reforms.cs.princeton.edu/
title: REFORMS - Reporting Standards for Machine-Learning-Based Science
source_type: professional_org
venue: Princeton / REFORMS Consortium
fetched: 2026-05-09
---

# REFORMS: Consensus-based Recommendations for ML-based Science

## Overview

REFORMS is a checklist for recommendations in machine-learning-based science consisting of **32 questions and a paired set of guidelines**. REFORMS was developed on the basis of a consensus of 19 researchers across computer science, data science, mathematics, social sciences, and biomedical sciences.

## Purpose

Machine learning methods are proliferating in scientific research, but the adoption of these methods has been accompanied by failures of validity, reproducibility, and generalizability. These failures can hinder scientific progress, lead to false consensus around invalid claims, and undermine the credibility of ML-based science.

## Eight Modules

The REFORMS checklist consists of 8 modules (categories):

1. **Study goals** — What questions does the study aim to answer? Predictions, interventions, hypotheses?
2. **Computational reproducibility** — Code, data, environment, random seeds
3. **Data quality** — Provenance, preprocessing, missing data
4. **Data preprocessing** — Feature engineering, normalization, transformations
5. **Modeling** — Model class, hyperparameters, training procedure
6. **Data leakage** — Train-test contamination, target leakage, temporal leakage
7. **Metrics and uncertainty quantification** — Appropriate metrics, confidence intervals
8. **Generalizability and limitations** — Population, time, domain shifts

## Common Pitfalls Addressed

- Train/test contamination
- Inappropriate baselines
- Cherry-picked metrics
- Missing uncertainty quantification
- Overclaiming generalization
- Hidden hyperparameter tuning
- Selective subgroup reporting
- Use of leakage-prone features

## Applications

REFORMS can serve as a resource for:
- **Researchers** when designing and implementing a study
- **Referees** when reviewing papers
- **Journals** when enforcing standards for transparency and reproducibility

## Reviewer Use

When reviewing ML-based science papers, reviewers should systematically check the REFORMS items as part of methodological evaluation, particularly attending to data leakage and generalizability claims.
