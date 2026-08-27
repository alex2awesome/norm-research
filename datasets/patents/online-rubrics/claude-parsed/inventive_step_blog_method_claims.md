---
source_url: https://inventivestep.net/method-claims-best-practices
title: Inventive Step Blog - Best Practices for Method Claims
source_type: practitioner_blog
patent_office: USPTO
fetched: 2026-05-09
---

# Inventive Step Blog: Best Practices for Method Claims

## What Is a Method Claim?

A method (or process) claim recites a series of steps. Method claims are useful for inventions that are sequences of operations: software algorithms, manufacturing processes, treatment regimens, communication protocols, control sequences.

## Single-Actor Architecture

Each step should be performed by a single entity, and ideally all steps should be performed by the same entity. Divided infringement (Akamai) is a real risk; a method claim where steps 1-3 are performed by the user and steps 4-5 are performed by the server may not have a direct infringer.

## Active Voice Verbs

Each step should use an active verb: "applying," "determining," "transmitting," "processing." Passive constructions ("the data is received") create ambiguity about who performs the step.

## Order of Steps

By default, the order in which steps are recited in a method claim does not require a specific order of performance. To require an order, use language like "after the receiving step" or "in response to the determining step."

## Step Granularity

Each step should be granular enough to be a meaningful technical operation, but not so granular that the claim becomes a rigid recitation of one specific implementation. "Applying a hash function to the input" is good; "computing 256 SHA-256 rounds" is too narrow.

## Step Coverage

Every step necessary for the inventive concept should be in the claim. Implicit steps (e.g., "of course you have to power on the device first") should not be in the claim, but everything that distinguishes the invention should be.

## Beauregard Claims (Computer-Readable Medium)

For software inventions, file a parallel CRM claim: "A non-transitory computer-readable medium storing instructions that, when executed by a processor, cause the processor to: [recite the same method steps]." This reaches different infringers.

## System Claims as Backup

For each method claim, draft a parallel system claim: "A system comprising: a processor; and a memory storing instructions that, when executed, cause the processor to: [steps]." System claims survive different validity attacks and reach different defendants.

## Avoid Mental Steps

Method steps that can be performed entirely in the human mind raise 101 (abstract idea) and inducement-of-infringement issues. Anchor steps to concrete operations on data: "storing in memory," "transmitting over a network," "displaying on a screen."

## Result Limitations vs. Action Limitations

A method step that recites a result ("determining a recommendation") is broader than one that recites the action ("applying a learned ranking function to a feature vector to determine a recommendation"). The action recitation is more enabled and more 101-eligible; the result recitation is broader but more vulnerable.

## Number of Steps

Most well-drafted method claims have 4-7 steps. Fewer than 4 may be too abstract; more than 8 invites obviousness rejections (more steps = more opportunities for the prior art to disclose them).
