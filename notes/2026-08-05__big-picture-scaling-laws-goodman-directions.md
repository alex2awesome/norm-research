# Big picture: what the scaling laws are for + directions from the Goodman conversation
(2026-08-05; source = user's notes from conversation with Noah Goodman, gaps filled from
program assets. Doc-of-record for the two deep-dive directions.)

## The thesis the scaling laws serve

Articulation is a channel with measurable per-criterion capacity. Scaling laws convert "this
criterion is tacit" from anecdote into a falsifiable curve-shape claim, because the same
observed articulation gap has three different causes with three different curve signatures,
separable ONLY by scaling:

| cause | prediction | our signature | regime |
|---|---|---|---|
| H-reader: executor can't inductively use the norm | gap shrinks with z | rising curves; unit floors | RISING |
| H-window: content already in weights at high z | articulation adds ~0 or subtracts | β<0 ("articulation subtracts"); executor-relative unit windows; REACHES steep-slope | REACHES |
| H-message: info not in words/examples at all | flat in z AND frontier non-convergence given full dossier | frontier κ≈0 voice/persona candidates; BOUNDED beyond-text loading (within-domain p=.049/.004) | BOUNDED |

H-window is the bridge to Goodman's in-weights vs in-context memory discussion: same content,
different storage; the unit-value window is the in-context/in-weights boundary sliding with
scale. Curve asymptote L answers "how much is there"; regime answers "what kind of thing is
it" (Goodman's #3). Backtests (median err .02-.06, tails disclosed) make the forecast claim
honest: rising → free improvement with scale; reaches → done; bounded → needs a different
channel.

## Direction 1 — inside prompt evolution: modes of articulation (exemplars vs statements)

Known (change-types, HB174/175): decompression writes CONCEPTS (def 51%/mech 49%); GEPA writes
MEASUREMENT (anchors 75-77%); M_ω writes PROCEDURE (47%). Open: which MODE carries which
criterion; where EXEMPLARS sit (GEPA already optimizes over "can't state it, can show it").

- **1a. Exemplar-prefix vs unit-prefix curves** (headline experiment; no humans): identical
  prefix-in-k protocol, arms = k articulated units (have) vs k labeled decision exemplars
  (existing silver/bank labels), token-matched; ICL literature predicts exemplar rollover at
  ~5-10 vs unit curve rising to k=40 (hotpot); asymptote comparison = stated-vs-shown channel
  capacity. Also supplies the ICL-scaling-laws related-work hook (queued; doc frozen).
- **1b. Example-blind GEPA**: reflection sees scores-only vs trace-full → how much of prompt
  optimization is example-mediated.
- **1c. Mode × metric grid**: extend z×a arms (name/def/expl/dossier + placebos — machinery
  exists) with exemplars and exemplars+definition arms; cross with regime labels. Target
  readout: BOUNDED responds to exemplars where definitions fail? REACHES responds to nothing
  (internalized)?
- **1d. Archival definition-count law**: regress gains on concept/measurement/procedure content
  counts across the 487 evolution steps + evolved prompts (ICL-analog observational law).
- **1e. Nonce/synthetic constructs** (probe-scale, NOT a new paper): planted-metric +
  nonce-definition + stipulation machinery exists; invented constructs make H-window impossible
  by construction → clean H-message isolation. ~1 week, appendix.

## Direction 2 — inside OSL: what new info bigger models use

- **2a. Type-conditioned unit-value curves**: unit-type labels × per-unit marginal delta ×
  executor ladder (hover 4B/8B/32B data exists) → which CONTENT KINDS have windows.
- **2b. De-censor RISING tail**: today's refit decomposition (osl_deep_20260805): 58%
  near-saturated / 27% deep-censored (~235 metrics); one stronger rung on just those settles
  bounded-late vs still-rising.
- **2c. Per-metric gap-source verdict table**: unify the three signatures (piecemeal today)
  into one H-reader/H-window/H-message verdict per metric. Likely the paper's most quotable
  object.
- 2d (flagged, later): in-weights point — small LoRA-on-exemplars probe, one metric family;
  full fine-tune scaling = future work.

## Parked (avoid-humans constraint honored)

Stated-vs-revealed / individual-vs-nominated-by-group / deliberation-vs-vote: doable as
LLM-panel deliberate-then-consensus vs aggregate-of-votes on existing labeled items
(frozen-crowd machinery generalizes), but drifts toward the VAT paper's territory —
third priority, only on user request.

## Priority recommendation

1a (exemplar-vs-unit curves) first — biggest thesis payoff per GPU-hour, machinery ~90% built;
then 2c (verdict table — pure analysis); then 1c (mode grid, reuses z×a); 1b/1d cheap adjuncts;
1e/2b when capacity is idle.
