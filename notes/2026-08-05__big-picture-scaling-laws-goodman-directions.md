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

## Addendum (2026-08-05, later): Collins mapping, sources-of-tacitness ledger, 1e design,
## decompression-confound ruling

### H-triple ↔ Collins (unity for the paper)

Collins' three tacit-knowledge types map one-to-one onto the curve signatures, and onto the
intro's existing three-criteria structure (main.tex ~line 244):

| Collins type | reason for resistance | our hypothesis | curve signature | intro slot |
|---|---|---|---|---|
| Relational (RTK, weak) | contingent — could be told, isn't/needn't be | **H-window** | articulation adds ~0 or subtracts at high z (β<0); unit-value windows | (a) "already understood" |
| Somatic (STK, medium) | limits of body/brain — telling ≠ performing | **H-reader** | gap shrinks with z; RISING | (b) "will one day be understood" |
| Collective (CTK, strong) | located in society; irreducibly social | **H-message** | flat in z + frontier κ≈0 under full dossier; BOUNDED beyond-text | (c) "will likely not be understood" |

The paper's unity sentence: the three regimes are Collins' three types made measurable — scaling
is the instrument that separates them, because each type predicts a different curve shape.

### Are these the only sources? The full ledger (4 real + artifact modes)

1. **H-writer** (new name for an existing pipeline piece): the right string exists but has not
   been FOUND — mining incompleteness. Collins criterion (1) "a different, better string."
   Already priced by the missing-mass/EVT ε in the certificates. Distinct from H-message
   (unfound ≠ unfindable).
2. **H-reader** (STK analog) — capability floor.
3. **H-window** (RTK analog) — already internalized at scale.
4. **H-message** (CTK analog) — truth conditions outside the text.
5. **Artifact modes that masquerade as tacitness and must be cleared first**: (i) probe-support
   mismatch (cluster-A lesson: brand-identity yes .06→.24 on 12× longer probes); (ii)
   instrument/judge noise (eval-noise ledger); (iii) in-context bandwidth (ICL rollover —
   content usable in weights but the context channel saturates; RTK-adjacent in Collins'
   terms, probed directly by 1a's exemplar-vs-unit rollover comparison).

### 1e (synthetic/nonce constructs) — resolved design

User concern is right: standard reconstruction can't run (decoder can't name a novel concept).
Fix: synthetic constructs get CODE-CHECKABLE truth (planted-metric machinery) — compose
criteria from measurable text properties at graded complexity (1..k clauses, thresholds,
exceptions). Readout = executor agreement with programmatic truth; no decoder, no MCQ needed
(MCQ-with-definitions would work but is strictly noisier). Arms: nonce-name / definition /
definition+exemplars / exemplars-only, × executor ladder. What it buys (and only this): a
**calibration frontier** — per-mode transmission capacity when nothing is tacit (H-window
impossible by construction, message complete by construction, H-writer eliminated). Real
metrics are then read AGAINST the matched-complexity frontier: transmission shortfall below it
= genuine residual, not reading failure. Run cheap, interpret comparatively; if the frontier is
flat/boring, drop it — agreed it is a side experiment.

### Decompression-confound ruling (user question)

Correction to the recollection: the decompression corpus was NOT GEPA-targeted — its steps are
the tacit line's authored articulation rungs (name→definition→explanation, generated by
instructing a model to define/explain). So its 100% concept-content is a MANIPULATION CHECK by
construction (recorded: HB174 "A rows ≈ manipulation check"; H4 framing correction owed in
print). The user's underlying instinct is correct and the defensible claim is the asymmetry's
OTHER half: optimizers write ~0–6% concept content even when free to — including (H2-on-nc
test) zero concept edits from low-fidelity starts (r0=.02–.4) where definitions would plausibly
help. Selection concern (decompression metrics are the decompressible ones) is real but
secondary: the optimization-side zeros hold across all corpora including the same domains.
1b (example-blind GEPA) and 1c (mode grid) are the direct closures: they put concept-writing
INSIDE the optimizer's reachable set and measure whether it ever pays.

### Launch state

Appendix C (z external validation, r=.87/ρ=.90, gemma2-27b composite-collapse diagnosis) is IN
THE PAPER (submodule 5a712a2). Main-body mention = USER's to write (reminder recorded).
Experiment battery 1a–1d + 2a–2c specs above; implementation begins as the sk1 robustness lanes
clear (hvcert rescore + 2 hotpot seeds + ifbench/aime chain still on GPUs 0–3). 1a runs in the
metric-recovery frame via zxa-style freeze arms (exemplars-in-rubric = zero runner changes;
exemplar labels from frontier-consensus verdicts, reconstruction-only rule preserved); sk3 box
per metric_implementer rule, GPUs 3/6/7 only.

### FRAMING DECISION (user, 2026-08-05) — SUPERSEDES the RTK/STK/CTK mapping above

Paper #2 does NOT probe types of tacit knowledge — that claim is retracted from the framing.
The user's ruling, verbatim in substance: the paper probes the BLOCKERS TO EXPLICIT knowledge —
Collins' criteria/remedies (better string / coded string / better listener) — and that framing
is preserved. The RTK/somatic/collective mapping is NOT bought and must not structure the paper.

What this changes:
- The H-triple (reader/window/message) is DEMOTED from "Collins' types made measurable" to
  LOCAL FAILURE EXPLANATIONS: candidate mechanisms invoked when a specific remedy
  underperforms (why more string stops paying → window/reader; why scaling doesn't close a
  gap → message-as-explanation). They are hypotheses about mechanism, never a taxonomy of the
  tacit, and never certified as "this metric's knowledge is collective/somatic/etc."
- The diagnosis matrix (remedies × failure explanations) survives, read column-first: each
  instrument administers one remedy; H-labels annotate failures descriptively.
- This is also the defensible claim ordering: certifying a TYPE of tacitness requires ruling
  out every rival explanation (incl. H-writer/search incompleteness + artifact modes) — hard;
  "this remedy fails here, best-supported local explanation is X" is what the data supports.

What this does NOT change: the experiment battery. 1a–1d probe the internal structure of the
"better string" remedy (WHICH KIND of string: definitions vs exemplars vs anchors vs
segmentation); 2a–2c probe the "better listener" remedy. 1e calibrates the string channel.
All interpretation text for these experiments should be written in remedy language.

### Battery launch ledger (2026-08-05 evening, user green-lit Directions 1+2 in full)

- **1a LAUNCHED**: `prefix_modes_1a.py` + `modes1a_lane_sk3.sh` — hotpot, ONE session
  (sk3 GPU6, port 8214, Qwen3-8B by HF id): seed + unit-prefix (k=1..20,24,28,32,36,40; frozen
  pool delta_8b desc, md5 27966bce verified identical sk1/sk2/sk3) + exemplar-prefix
  (k=1..12,14,16,20; train-set Q→A pairs in fixed train order appended to final_answer.predict),
  5 passes each, added-chars recorded per point for token-matched analysis. Same-session by
  design (HB192: cross-session curves never share a panel). ~61.5k scored items ≈ 1 day.
  Results: runs/prefix_modes1a_hotpot.json on sk3. Monitor armed ("MODES1A COMPLETE").
  Launch notes: sk1 GPU4 attempt aborted safely by busy-guard (eshaanb sglang took GPUs 4-7);
  sk1 now full (our 0-3 + theirs 4-7). sk3 paperexact_arms.py synced from sk2 canonical.
- **Queue** (start as capacity/wakes allow): 2c per-metric remedy-failure verdict table (pure
  CPU, next), 1d archival content-count law (CPU), 1c exemplar arms added to z×a freezes (sk3),
  1b example-blind GEPA (one GPU), 2b de-censor rung (~235 metrics, one strong executor),
  1e calibration frontier (planted machinery), 2d LoRA in-weights probe (LAST, needs training).
- Still in flight from the robustness batch: hvcert10110 rescore (sk1 GPU0 server / sk2
  client), hotpot seeds s1/s2 (sk1 GPU1/2), ifbench→aime chain (sk1 GPU3), hover 5-pass
  prefix (sk2, queued behind l1ly).
