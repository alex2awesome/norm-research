---
source_url: https://www.kdnuggets.com/2020/04/peer-reviewing-data-science-projects.html
title: Peer Reviewing Data Science Projects
source_type: blog
venue: KDnuggets
published: 2020-04
fetched: 2026-05-09
---

# Peer Reviewing Data Science Projects (KDnuggets)

A two-phase peer-review framework for industry data science.

## Two Review Phases

1. **Research-phase review** — methodology selection and approach viability.
2. **Model-development review** — implementation and model performance.

## Research-Phase Structure

**Preparation**
- Reviewed scientist prepares: scope, KPIs, assumptions, data exploration, possible approaches, recommendation, contingency plans.
- Meeting ≥60 min.
- Reviewer studies the checklist beforehand.

**Meeting flow**
1. Presentation by reviewed scientist.
2. General feedback from reviewer.
3. Checklist examination.
4. Approval / rejection decision.
5. Action items.

## Research-Phase Checklist Categories

- Data properties (sampling, bias, representativeness).
- Approach assumptions (validity / edge cases).
- Past experience (relevant precedents).
- Objective alignment (loss functions vs. KPIs).
- Implementation (tools, support).
- Scaling (computational requirements).
- Composability (modularity).
- Information requirements (data sufficiency).
- Domain adaptation (cold-start).
- Noise / bias resilience.

## Model-Development Phase

Separate checklist: data assumptions, preprocessing, leakage detection, causality, evaluation metrics, overfitting, runtime, common implementation errors.

## Core Motivation

"Approach failures…are *very* costly to make" when discovered late. Early peer scrutiny prevents production deployment of flawed models.
