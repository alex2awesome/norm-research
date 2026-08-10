"""Prompts for the claim-verification framework (patents localize-then-verify transported to
journalism/PR). All judge calls are guided-JSON; every verdict must quote a VERBATIM span
(patents discipline: ~99% verbatim grounding was the integrity check)."""

CLAIM_EXTRACT = """You are analyzing a {doc_kind}. Extract the ATOMIC CLAIMS asserted in the \
HEADLINE and OPENING (first ~2 paragraphs) below. An atomic claim is a single checkable \
assertion of fact (an event happened, a number/quantity, a comparison/superlative, an \
attribution of action/intent to a named actor). Do NOT extract opinions, style, or boilerplate.

TEXT:
{head}

Return JSON: {{"claims": [{{"claim": "<one-sentence restatement of the atomic claim>", \
"kind": "<event|quantity|comparative|attribution|forecast>"}}]}}
At most {max_claims} claims, most central first. Return ONLY the JSON."""

LOCALIZE_VERIFY = """You are verifying whether a claim is SUPPORTED by evidence in the body of a {doc_kind}.

CLAIM: {claim}

BODY PASSAGES (numbered):
{passages}

Step 1 - LOCALIZE: find the passage(s) that bear on this claim, if any.
Step 2 - VERIFY: does the evidence SUPPORT the claim?
  FULL    = a passage substantiates the claim with specifics (data, named source, document, direct fact)
  PARTIAL = a passage repeats or weakly supports the claim without independent specifics
  NONE    = no passage supports it (claim is asserted only in the headline/opening)

Step 3 - EVIDENCE TYPE (if FULL/PARTIAL): one of
  data_statistic | named_source_quote | document_official | expert_attribution | assertion_only

Return JSON: {{"verdict": "FULL|PARTIAL|NONE", "passage_idx": <int or null>, \
"span": "<VERBATIM quote (<=200 chars) from the passage that supports, or empty>", \
"evidence_type": "<type or null>", "reason": "<one sentence>"}}
The span MUST be copied verbatim from a passage. Return ONLY the JSON."""

SOURCE_CENSUS = """Count the SOURCES cited in this {doc_kind}. A source is an entity the text \
attributes information to.

TEXT:
{body}

Return JSON with counts:
{{"named_person_sources": <people quoted/cited BY NAME with title/affiliation>,
"anonymous_sources": <"sources familiar", "officials said" without names>,
"document_sources": <filings, reports, studies, court documents, datasets cited>,
"organization_sources": <companies/agencies cited as asserting something>,
"self_reference_only": <true if the only source is the document's own author/issuer>}}
Return ONLY the JSON."""
