---
source_url: https://medium.com/@patentpractice/means-plus-function-the-claim-language-trap
title: Means-Plus-Function - The Claim Language Trap You Probably Are Not Thinking About
source_type: medium_article
patent_office: USPTO
fetched: 2026-05-09
---

# Means-Plus-Function: The Claim Language Trap You Probably Are Not Thinking About

## Why "Means For" Is Almost Never Used Anymore

35 USC 112(f) says that a claim element expressed as a "means for" performing a function is construed to cover only the corresponding structure described in the spec, plus equivalents. In practice, this narrows the claim dramatically and makes invalidation under 112(b) (indefiniteness) easier when the spec lacks structure.

## The Williamson Trap - Functional Language Without "Means"

After Williamson v. Citrix (Fed. Cir. 2015), courts apply 112(f) even without the magic word "means," whenever a claim term is a "nonce word" that does not connote sufficient structure. Examples held to invoke 112(f): "module," "mechanism," "element," "device for," "unit configured to."

## Safer Drafting Choices

Replace "module configured to X" with concrete structure: "a processor executing instructions stored in memory that, when executed, cause the processor to X." Or recite a specific data-structure ("a hash table mapping..."). Or recite the known engineering name of the component ("a PID controller," "a Kalman filter," "a CCD sensor").

## When to Embrace 112(f)

Sometimes you actually want narrow construction tied to your spec - for example, when the prior art is so close that any broader reading reads on it. In those cases, deliberately use "means for" and put one or more clear corresponding structures in the spec.

## The Specification Cost of 112(f)

If you do invoke 112(f), the spec must clearly disclose the corresponding structure and link it to the claimed function. For software functions, the Federal Circuit (Aristocrat, WMS Gaming) requires an algorithm - flowchart, pseudocode, or prose - not just "a general-purpose computer."

## Word List - Likely 112(f) Triggers

These often invoke 112(f) absent clear structural context: "means," "mechanism," "module," "element for," "device for," "unit," "system for," "component configured to."

## Word List - Usually Structure

These usually avoid 112(f): "circuit," "processor," "memory," "transistor," "sensor," "actuator," "valve," "antenna," "filter," "lens," "transducer," "amplifier."

## Audit Your Claims Before Filing

Read every independent claim looking for nonce words. Each one is a 112(f) risk. Either replace with structural language or, if intentional, ensure the spec has clear corresponding structure.
