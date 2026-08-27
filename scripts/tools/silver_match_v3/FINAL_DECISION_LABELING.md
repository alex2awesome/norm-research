# Blind final-decision labeling

Label only the `.blind.jsonl` packet supplied to you. Do not inspect its
`.key.jsonl`, final system outputs, or prior model decisions.

For each human statement and evidence passage, inspect the referenced complete
task bank and return exactly one decision:

- `MATCH`: one current-bank metric is explicitly invoked; set `metric_id` to
  that exact bank ID.
- `MATCH_FAMILY_ONLY`: an explicit criterion is present but the evidence does
  not distinguish a single sibling leaf; keep `metric_id` null.
- `NO_EXPLICIT_CRITERION`: the passage does not state or invoke an evaluative
  criterion.
- `CONTEXT_NEEDED`: the criterion cannot be identified without unavailable
  context.
- `GENERIC_VERDICT`: the passage gives only a holistic verdict without a
  bank-level criterion.
- `NO_CANDIDATE_FITS`: a specific criterion is explicit, but none of the full
  current-bank metrics is an exact fit.
- `NOISE`: the extraction is garbled or not faithfully grounded.

Also fill `confidence` (`high`, `medium`, or `low`) and a concise evidence-based
`reason`. Do not force a nearby metric. A thematic or family resemblance is not
an exact `MATCH`.
