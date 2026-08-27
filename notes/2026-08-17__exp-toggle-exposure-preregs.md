# Two preregs: EXP-TOGGLE-1 (deliberation x message form) and EXP-EXPOSURE-1 (crossing vs corpus exposure)

Status: PREREGISTERED 2026-08-17, user-approved sequence ("i like this sequence"). No
confirmatory calls made as of freeze.

===============================================================================
## EXP-TOGGLE-1 — is deliberation a form of articulation? (receiver-side primary)

Instrument: Qwen3 hybrid family on sk3 (1.7B/4B/8B/14B/32B, shared cache), vLLM
`chat_template_kwargs={"enable_thinking": bool}` (verified live 2026-08-17: think-on emits
<think> trace, think-off answers directly; same weights).

Battery: the FROZEN zxa slate and arm prompts verbatim (name / definition / explanation /
dossier + dossier_mismatched + definition_padded), humor + cw + math + peer_review panels'
probe sets. Plus the 240-item anchor battery run on every (rung x mode) — Qwen3 has no battery
z; both-mode z is required for H0.

Design: 5 rungs x 2 modes x all slate arm-entries x 300 probes; temperature 0; think-on budget
cap 2048 tokens with unclosed-think rows excluded and their rate REPORTED (stall rate is an
outcome per the seam-pilot precedent, never silent missingness). Scoring: LOFO family-balanced
panel consensus PRIMARY (key policy), dossier-LOO key as continuity sensitivity.

Hypotheses (competing, all preregistered):
- H0 (null; "just test-time scaling"): thinking = pure capability shift. Test: the think-on
  message-form winner distribution and per-arm recoveries at rung r match the think-off ladder
  evaluated at z(r, think-on) — i.e., predictable from the battery-z shift alone. If H0 holds,
  report in one sentence.
- H1 (substitution / self-articulation): think-on shifts winners toward SHORTER forms; the
  name-arm crossing z* drops under thinking (beta_think > 0 in z units).
- H2 (articulability-selectivity): think-on gains track articulability — concentrated on
  reaches/rising constructs, ~absent on the bounded/tacit cluster.
- H2' (compliance-relief): gains concentrate on the LONG arms at small rungs (dossier collapse
  reversal), not the short arms. H1 vs H2' = which arm benefits.
- MECHANISM READOUT (toggle-insensitivity x crossing status): per construct, sensitivity =
  recovery(on) - recovery(off) at each rung. Prediction: constructs PAST their crossing z*
  (name-indexed at that rung) are toggle-INSENSITIVE (|delta| < .03); constructs BEFORE
  crossing are toggle-SENSITIVE (delta > 0). Report the 2x2 (crossed x sensitive) with
  metric-level bootstrap.
Secondary (declared): label 100 sampled think-on traces with the frozen change-type codebook
(concept vs measurement content, kappa=.93 grain) — does deliberation WRITE definitions/
explanations of the criterion? Secondary leg (transmitter-side, small): Qwen3-32B authors
definition+explanation arms with thinking on vs off (length-matched, 24 constructs); evaluated
at think-OFF receivers only. Expected null disclosed (messages already near authoring ceiling).

Gates: planted arms must be toggle-well-behaved (think-off planted execution within .05 of the
prior no-thinking Qwen3-free ladder pattern); unclosed-think rate reported per cell; any cell
with >40% unclosed is excluded and counted. No optional stopping.

===============================================================================
## EXP-EXPOSURE-1 — does name-indexing track pretraining exposure? (analysis-only, zero GPU)

Hypothesis (from the fact-frequency scaling literature): a criterion's crossing z* (the
capability at which its NAME alone suffices) decreases with the concept's pretraining exposure.
Corpus proxy: Dolma via the public infini-gram API (declared: we cannot query the actual Qwen/
Llama corpora; Dolma is a declared open-corpus proxy, limitation stated).

Exposure estimator (frozen before any query): for each construct name, content terms = name
tokens minus stopwords/punctuation; exposure = mean log10(count+1) over the content UNIGRAMS
and all adjacent content BIGRAMS, counts from infini-gram COUNT on Dolma-v1.7. Sensitivity
estimators (reported alongside, not headline): min instead of mean; definition-text terms
instead of name terms.

Sample: all per-metric rows with a defined crossing in the z x a fit (humor + news; family =
llama and qwen25 separately — family-relative throughout). Primary readout: Spearman rho
(exposure, z*_name) within family, censored rows handled by rank with censoring (report both
excluded and bound-imputed variants). Secondary: dialect prediction — constructs whose
crossing differs most across families should sit at intermediate exposure. Caveat carried: the
current crossings are dossier-key fits; symmetric-key refresh pending (footnoted in the paper);
this analysis reruns mechanically when the refreshed fits land.
Decision framing: supportive if rho <= -.35 with bootstrap CI < 0 in BOTH families;
the interesting object either way is the residual (concepts crossing earlier/later than
exposure predicts).

===============================================================================
## EXP-EXPOSURE-1 — ADDENDUM A (generality; 2026-08-17, before any addendum query)
User question: is the exposure-null specific to the isomorphism (slate-crossing) sample or
general to all rising concepts? Extension, same frozen estimator and pipeline/cache:
- SAMPLE: all 1,270 fitted bank metrics (8 domains), non-planted.
- READOUT: per-metric acquisition onset under the bank rubric = fitted logistic midpoint z0
  from laws_<task>.json where defined; else the interpolated z at which the pooled recovery
  curve first reaches .65 (declared fallback; which was used is recorded per metric). Verdict
  class as a secondary grouping (exposure by rising/reaches/bounded).
- ANALYSIS: within-domain Spearman rho(exposure, onset) pooled by Fisher-z across domains
  (never pooled raw across domains — composition); bootstrap CIs; residual top-10s.
- CAVEATS DECLARED: pooled-panel curves (not family-split — weaker than the slate analysis on
  the family-relativity axis; descriptive generality probe, not a headline); same Dolma-proxy
  and estimator limitations as the parent.
Same decision framing as parent, evaluated in-session.
