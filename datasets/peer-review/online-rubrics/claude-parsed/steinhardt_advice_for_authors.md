---
source_url: https://jsteinhardt.stat.berkeley.edu/blog/advice-for-authors
title: Advice for Authors (from ICML reviewing patterns)
source_type: researcher_blog
author: Jacob Steinhardt
venue: jsteinhardt.stat.berkeley.edu
fetched: 2026-05-09
---

# Advice for Authors (Jacob Steinhardt — distilled from ICML reviews)

A senior ML reviewer / area chair's *running* feedback file, distilled from feedback he kept giving while reviewing ICML papers.

## General Writing Principles

- "Be precise" and "be concise" are foundational.
- Avoid complex sentence structure.
- Maintain consistent phrasing in technical writing — varying word choice for entertainment value confuses readers.

## Abstract Structure

For unfamiliar ideas, open with universally accepted context, then present a surprising insight that builds logically from that foundation. Don't open with "coverage of the space is important" without giving the reader prior context.

## Introduction Guidance

- Avoid vague hedging ("increasingly important") — frequent but imprecise.
- Always provide context before introducing specialized concepts.
- Spell out the contribution's significance in the introduction. Many reviewers stop reading partway through Section 1.

## Conclusion Strategy

Reserve the conclusion for insights readers couldn't appreciate without prior content. Include open research questions here. For theory papers, conclusions are optional unless reviewers expect them.

## Technical Recommendations

- Use `\citep{}` and `\citet{}` for proper citation formatting.
- Avoid the `fullpage` package — overrides conference style files.
- Choose carefully between displayed and inline equations.
- Limit theorems to roughly one per paper.
- Encapsulate non-trivial arguments as Propositions or Theorems.

## Implication for Reviewers

These items reflect what Steinhardt-as-reviewer keeps flagging — i.e., they form an implicit reviewer rubric for ML conference papers.
