# Detecting Reference Errors in Scientific Literature with Large Language Models

SOURCE_URL: https://arxiv.org/abs/2411.06101
DOMAIN: academic

## What this source is

An arXiv paper that (a) restates the standard citation-error vs. quotation-error distinction, (b) operationalizes quotation-error detection as a clean 3-way substantiation-level classification task, (c) builds a small but carefully-sourced dataset of real statement/reference pairs known to have errors (drawn from prior audits, post-publication peer review comments, and published corrections), and (d) tests LLMs as automated judges of whether a reference substantiates a statement, varying how much of the reference the model is allowed to see. This is the most directly usable source for prompting/labeling an LLM-based claim-matching judge, since it gives an explicit label schema plus a description of the judge framing used.

## Citation error vs. quotation error (restated definition)

- **Citation errors**: "typographical errors in referencing, such as incorrect reference information (e.g., incorrect authors, title, journal, or year)" — bibliographic/formatting mistakes.
- **Quotation errors**: "the situation where a reference fails to support the statement for which it is cited" — a content/support failure. This paper focuses exclusively on quotation errors, which is the correct scope for a claim-matching metric.

## The 3-way substantiation taxonomy

This is the core operational contribution — a claim/citation-pair label set with concise, quotable definitions:

- **Fully substantiated**: "The reference article fully substantiates the relevant part of the statement."
- **Partially substantiated**: "According to the reference article, there is a minor error in the statement, but the error does not invalidate the purpose [of the citation]." — the cited source is basically right but doesn't perfectly match every detail of the claim (this maps closely onto the "minor error" tier in the medical quotation-accuracy literature).
- **Unsubstantiated**: "The reference part does not substantiate any part of the statement. This could be because the statement is contradictory to, unrelated to, or simply missing from the reference article." — notably, this single label deliberately merges three distinct underlying causes (contradiction, irrelevance, and simple absence of evidence) rather than splitting them, which is a simpler but coarser choice than the 8-way biomedical NLP scheme (see companion note on PMC11231046) that keeps CONTRADICT, IRRELEVANT, and NOT_SUBSTANTIATE separate. For a claim-matching metric bank, this is worth flagging as a design choice/tradeoff: 3-way is easier to get high agreement on and easier to prompt an LLM for, but loses the ability to distinguish "the source disagrees with you" from "the source just never talks about this."

## Task framing / detection procedure

The prediction task is defined simply: **given a statement and its cited reference, predict whether the pair is fully / partially / unsubstantiated.** The paper tests this task under three levels of reference visibility, which is directly useful as an ablation design for any claim-matching evaluation:
1. Reference **title only**
2. Reference **title + abstract**
3. Reference **title + abstract + excerpts** (i.e., actual passages/snippets from the body of the cited work)

Performance is compared across these three conditions — this operationalizes "how much of the source do you need to read to correctly judge support," which is a natural robustness/ablation axis to replicate for any automated claim-matching pipeline (e.g., does judgment accuracy on the SUPPORT/PARTIAL/NOT-SUPPORT task saturate once excerpts are provided, or does it need full text?).

## LLM-judge prompt framing

The model is instructed to act as **"an experienced scientific writer and editor"** evaluating whether the reference content supports the claim, and to output a **structured JSON** object containing (a) the predicted label among the three substantiation categories and (b) a concise explanation justifying the label. This "expert persona + forced structured output + required justification" pattern is a reusable prompt template for any LLM-as-judge claim-matching implementation — the justification requirement in particular forces the model to point to the specific textual basis (or absence) for its verdict rather than emitting a bare label.

## Dataset construction (source of known-error examples)

250 statement–reference pairs were assembled from real-world sources of *known* citation problems, spanning seven scientific domains:
- **65.2%** from prior citation-verification/quotation-accuracy audit studies (i.e., reusing findings from manual audits like the medical quotation-accuracy literature)
- **32.0%** from **PubPeer** comments (post-publication peer review flagging specific citation/quotation problems)
- **2.8%** from **corrections/errata** published in PubMed

This is a useful sourcing strategy in its own right for building a claim-matching benchmark: rather than relying solely on newly-commissioned manual annotation, harvest pre-existing, already-adjudicated error reports from post-publication review platforms (PubPeer) and formal correction/errata records, which come with a built-in "someone already litigated this and it was accepted as an error" quality bar.

## Practical takeaways for a claim-matching metric bank

1. A minimal, prompt-friendly label set for an LLM judge: **fully substantiated / partially substantiated / unsubstantiated**, with unsubstantiated explicitly covering contradiction, irrelevance, *and* absence-of-evidence as a single merged "fail" bucket (accept this as a simplification, or split it out if the metric bank needs the finer 8-way distinctions).
2. Structure the judge prompt as: expert persona ("experienced scientific writer/editor") + the statement + the reference material + a forced-JSON output containing **label + justification**, not just a bare label — the justification is what makes the judgment auditable.
3. Ablate over **how much of the cited source the judge sees** (title-only vs. title+abstract vs. title+abstract+excerpts) as a standard robustness check — this reveals whether a metric is actually reading the evidence or pattern-matching on topical similarity between title and claim.
4. Consider **PubPeer comments and journal corrections/errata** as a real-world sourcing channel for known-bad citation examples when constructing calibration/anchor sets, rather than relying only on freshly commissioned human annotation.
