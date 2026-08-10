---
source_url: https://medium.com/@softwarepatent/drafting-software-patent-claims-after-alice-still-doable
title: Drafting Software Patent Claims After Alice - Still Doable
source_type: medium_article
patent_office: USPTO
fetched: 2026-05-09
---

# Drafting Software Patent Claims After Alice: Still Doable

## The Alice Two-Step Is Now a Drafting Discipline

Step 1: Is the claim directed to an abstract idea? Step 2: Does the claim recite an "inventive concept" that amounts to significantly more? You cannot avoid Step 1 by clever drafting. You can win Step 2 by tying the claim to a specific technical improvement.

## Anchor the Claim in a Technical Problem

The opening of your specification should describe a technical problem in the operation of a computer (latency, memory footprint, packet loss, indexing inefficiency, security vulnerability). Not a business problem. Not "users find it hard to..." A computer problem.

## Recite Specific Technical Steps, Not Outcomes

A weak claim says "determining a recommendation." A strong claim says "applying a hashed Bloom filter to the user-interaction log to identify candidate items, then ranking the candidates using a learned weight matrix stored in GPU memory." Specificity buys eligibility.

## Avoid Pure Result-Oriented Language

"A method for predicting customer churn" is suspect. "A method for predicting customer churn comprising training a gradient-boosted decision tree on a feature set including session timing residuals..." has a fighting chance. The closer you stay to the result, the closer you are to an abstract idea.

## Include a Hardware-Tied Embodiment

Even if your invention is fundamentally algorithmic, include at least one claim that recites concrete hardware: a specific sensor, a specific accelerator, a particular memory architecture. This is the "machine prong" of the historic Bilski test, still rhetorically useful.

## Write the 101 Argument Into the Spec

Before the detailed description, include a paragraph titled "Technical Improvements" that explicitly catalogs how the claimed system improves the functioning of a computer (e.g., "reduces query latency by 40%," "eliminates the need for a roundtrip to the database server"). Examiners and judges quote these paragraphs.

## Method Claims Plus Apparatus Claims Plus CRM Claims

File parallel claim sets: method, system/apparatus, and computer-readable medium. Different defendants infringe different claim types; different judges prefer different formats.

## Beware "Black Box" Claims

A claim that recites "a model trained to..." without describing the architecture invites both 101 (abstract idea) and 112 (enablement) rejections. Recite enough of the architecture that a skilled person could implement it.
