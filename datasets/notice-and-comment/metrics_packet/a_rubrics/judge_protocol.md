# A-rubric judge protocol

Each (comment, rubric) pair is scored independently by an LLM judge with a single-token
answer. Campaign judge: Gemma-4-31B (offline batch vLLM, greedy, max_tokens≈5); any
capable instruction model can substitute, but scores are judge-dependent — do not mix
judges within one analysis.

## System prompt (verbatim from the campaign scorer)

```
You are an expert regulatory analyst reviewing PUBLIC COMMENTS submitted on proposed
federal rules. You are given one public comment and ONE quality criterion. Decide how
strongly the comment, on its own evidence, satisfies that criterion. Answer with
EXACTLY ONE token:
  1.0 = clearly satisfies the criterion
  0.5 = partially / weakly / borderline
  0.0 = fails / cuts against the criterion
  NA = the comment gives no evidence bearing on this criterion
Judge the comment's quality as a piece of regulatory input, not whether you agree with
its position. Output only the token.
```

The user turn supplies the criterion (`name: description` from `rubrics.jsonl`) and the
comment text. Parse leniently: "na"/"n/a" → NA (missing), then match 0.5 / 1.0 / 0.0.

## Anchor comments (always include in every scoring batch)

Three synthetic comments ride along as blinded known-quality anchors; their mean scores
across the bank are the judge-sanity check. Campaign reference values (Gemma-4-31B,
pre-GEPA bank): strong ≈ .87, mid ≈ .52, weak ≈ .03, overall NA rate ≈ .63. If your
run's anchors don't separate cleanly in that order, the judge or parsing is broken —
also check the full score distribution (a run collapsing to all-one-value is a known
guided-decoding failure mode).

**strong** — trade-association comment on 40 CFR Part 60: challenges the emissions
baseline with the agency's own CEMS data, quantifies the cost-benefit change, proposes
a specific phased alternative with modeling, attaches proposed regulatory text, flags an
E.O. 12866 RIA gap with a sensitivity analysis.

**mid** — rural nurse: relevant first-hand experience, supports the goal, raises a real
feasibility concern, asks for more time or an exception; no citations, no data, no
specific text.

**weak** — all-caps rant: pure opposition, no engagement with the rule, insults, no
evidence.

(Full anchor texts: `v4/score_va_gemma_nc.py::ANCHORS` in the source repo.)
