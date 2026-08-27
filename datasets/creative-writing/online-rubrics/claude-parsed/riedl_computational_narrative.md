---
source_url: https://mark-riedl.medium.com/computational-narrative-intelligence-past-present-and-future-99e58cf25ffa
title: Mark Riedl - Computational Narrative Intelligence (Evaluation Criteria)
source_type: computational_creativity
fetched: 2026-05-09
---

# Mark Riedl — Computational Narrative Intelligence (Story Generation Criteria)

Georgia Tech researcher; long-running program in computational narrative. Evaluation criteria for AI-generated stories that double as criteria for human-authored stories.

## Two Foundational Evaluation Dimensions

### 1. Plot Coherence
"The perception by the audience that the main events follow logically from one another." Each event must be motivated by what came before; nothing should appear arbitrary.

### 2. Character Believability
"The perception by the audience that the actions performed by characters do not negatively impact the audience's suspension of disbelief. Specifically, characters must be perceived by the audience to be intentional agents."
- Characters must have **goals**
- Their actions must serve those goals
- Their goals must be intelligible given who they are

## The Intent-Based Model (IPOCL)

Riedl's planner adds, to traditional plot causality, the requirement that **every character action be explained by an intention** — an attributable goal that the character is pursuing. This is a craft criterion: a story fails when a character does something for which we cannot construct an intent.

## Additional Implicit Quality Dimensions

### Affective Response
Stories should "invoke emotional responses" — suspense, surprise, sympathy, fear. Surface features have low correlation with emotional effect; affect is hard to predict from text alone.

### Commonsense Plausibility
Stories must satisfy implicit social norms — characters behave in socially intelligible ways unless their deviation is the point.

### Spatiotemporal Continuity
Characters cannot be in two places. Time cannot run backward without notice. Continuity errors break the world.

### Dramatic Structure
Generated stories should exhibit recognizable dramatic shape — rising action, climax, resolution — without which they read as event chronicles.

### Novelty / Tellability
A story should be worth telling — should contain something unexpected, surprising, or non-trivial. Routine sequences are not stories.

## Evaluation Methods (in computational creativity)

- **Human ratings** on dimensions above
- **Intentional reasoning trace** — can we recover plausible character intentions from each action?
- **Plot-graph distance** — how far does the generated structure deviate from canonical templates?
- **Surprise vs. suspense metrics** — calibrated information theory measures

## Implications for Human Writers

A short story generated **or** written can be evaluated by:
- Coherent causal chain among events
- Every character action attributable to a goal
- Spatiotemporal consistency
- Dramatic shape (not just sequence)
- Affect actually produced in readers
- Tellability — worth telling, not routine
