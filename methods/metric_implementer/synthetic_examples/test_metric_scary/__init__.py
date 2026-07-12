"""`is_scary` — a planted creative-writing metric for end-to-end pipeline calibration.

The metric asks: does a short story read as scary? The signal is PLANTED through a small,
fixed lexicon of scary "cue categories" (sound, presence, dread, body, danger). A story is
scary iff it contains markers from >=1 category; non-scary stories carry length-matched calm
markers instead. Because the signal is known, this is a known-answer world: a judge that
names the cue categories must recover the planted label, and the metric-implementer's
unsupervised recovery measures (reconstruction, counterfactual, reliability) have a ground
truth to be checked against.

See `cues`/`scary_metric` for the lexicon + seed, `scary_judge` for the deterministic planted
judge (zero LLM spend), `build_dataset` for the confound-controlled corpus, and
`run_optimizers` to drive the GEPA loop over it.
"""
