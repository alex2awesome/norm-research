# Our Process (FactCheck.org)
SOURCE_URL: https://www.factcheck.org/our-process/
DOMAIN: journalism

FactCheck.org (a project of the Annenberg Public Policy Center) publishes a standing methodology page describing exactly how it selects, researches, verifies, edits, and — when necessary — corrects its fact-checks. It is one of the more granular publicly documented pipelines for how a newsroom actually matches a political claim against primary-source evidence.

## Selection: what gets checked

- Staff systematically review transcripts and video of political speech (Sunday shows, C-SPAN, campaign ads, presidential remarks, congressional statements) looking for **statements presented as fact** — the trigger for further work is a factual assertion, not an opinion.
- Once a statement is flagged as *possibly* inaccurate or misleading, the outlet attempts to engage directly with the person or organization who made the claim, giving them the opportunity to respond or provide support before publication.
- Coverage focus shifts by election cycle: in presidential election years, primary focus is on presidential candidates; in midterm years, on top Senate races; in off-election years, on the president, administration officials, and congressional/party leaders.
- The outlet explicitly aims to devote **"an equal amount of time reviewing claims by Republicans and Democrats"**, including by comparing statements made in equivalent venues (e.g., comparable interview settings) to avoid a structural imbalance in scrutiny.

## Burden of proof

The methodology states directly: **"The burden is on the person or organization making the claim to provide the evidence to support it."** Practically, this means:
1. FactCheck.org first asks the claimant (or their staff/office) for their sourcing.
2. If the claimant provides supporting material, FactCheck.org checks whether that material actually supports the claim as stated (rather than merely being related to the topic) — the methodology notes they will judge a claim as adequately supported when the evidence the claimant provides holds up, and will not conduct further independent research in that case.
3. **If the supporting material does not actually support the claim, or if no evidence is provided at all, FactCheck.org conducts its own independent research**, and in that independent research phase it relies on primary sources first.

This two-track structure (claimant-supplied evidence checked for actual relevance/support, falling back to independent primary-source research only if that check fails) is a distinct and reusable design for a claim-matching pipeline: it separates "does the cited evidence actually support the claim" (a relevance/entailment check) from "what does independent primary evidence show" (an independent verification check), and only invokes the second when the first fails.

## Primary source hierarchy

FactCheck.org explicitly lists the categories of primary source it relies on, in preference to secondary reporting or press releases:
- The Library of Congress, for congressional testimony records.
- The House Clerk and Senate Secretary's offices, for official roll-call vote records.
- The Bureau of Labor Statistics, for employment and price data.
- The SEC, IRS, Bureau of Economic Analysis, and Energy Information Administration, for financial and economic data specific to their domains.
- Nonpartisan government analytic bodies — the Congressional Budget Office (CBO), the Government Accountability Office (GAO), and the Congressional Research Service (CRS) — treated as authoritative, nonpartisan intermediaries between raw data and a checkable conclusion.
- "Respected and trustworthy outside experts" (the Kaiser Family Foundation is cited as an example in the health-policy domain) are consulted when the topic requires subject-matter expertise the staff does not have in-house.

This amounts to a domain-indexed source hierarchy: for any given claim's subject area (economic, health, legislative), there is a designated class of primary-source-of-record that should be checked before any secondary account is relied upon.

## Distinguishing factual claims from opinion / non-checkable statements

The methodology reiterates that the target is statements "based on facts" — the outlet's stated purpose is to **"focus on claims that are false or misleading,"** implying an initial screen that filters out claims for which the claimant's own supplied evidence turns out to be adequate (these are simply not written up as debunkings, since there is nothing to correct).

## Editorial / quality-control layers

Every published item goes through four sequential review layers before publication:
1. Line editing
2. Copy editing
3. Fact-checking (an independent check of the piece's own claims and citations, separate from the original researching reporter)
4. Director-level review

All sources used are hyperlinked directly in the published piece, so that a reader can independently follow the same evidentiary chain the reporter did — functionally the same "replicability" requirement found in the IFCN Code of Principles.

## Corrections

**"If any new information comes to light after we publish a story that materially changes that story,"** FactCheck.org will clarify, correct, or update it, accompanied by an explanatory note stating why the change was made and the date it occurred — corrections are appended and dated rather than silently edited in place.

## Relevance to claim-matching

FactCheck.org's process supplies a concrete two-stage claim-verification algorithm that generalizes well to an automated claim-matching setting: **Stage A — does the evidence the claimant themselves cites actually entail/support the specific claim as stated (not just relate to the same topic)?** If yes, the claim is adequately supported and no further check is needed. **Stage B — if Stage A fails (irrelevant, insufficient, or absent evidence), consult a domain-appropriate primary-source-of-record** (the specific agency/database that is authoritative for that subject area) and use that as the independent ground truth. This gives a reusable decision rule for when self-supplied evidence should be trusted at face value versus when independent primary verification is required.
