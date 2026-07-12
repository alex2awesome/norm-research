# `is_scary` — a planted creative-writing testbed

A **known-answer** world for exercising the whole metric-implementer stack (optimize → score →
recover) with **zero LLM spend**. The metric asks: *does this short story read as scary?* The
scary signal is **planted** through a small fixed cue lexicon, so the ground-truth label is
known and the unsupervised recovery measures (reconstruction, counterfactual validity,
reliability, oracle agreement) have something to be checked against.

It serves three purposes:

1. **End-to-end plumbing test** — a deterministic judge runs the GEPA loop locally, so the
   pipeline can be unit-tested without a GPU, network, or API key.
2. **Articulability calibration** — the judge can only apply cue categories the rubric *names*,
   so a rubric that articulates more of the metric recovers more of the planted signal. This is
   a controlled instance of the flagship question (*how much of a metric can be put into words*)
   with a known answer.
3. **Confound-controlled data** — setting and character count are held in equal proportion
   across the label, so the metric cannot be recovered by a shortcut.

## The planted design

A scary story contains markers from one or more of five **cue categories** — `DREAD`, `SOUND`,
`PRESENCE`, `BODY`, `DANGER` (`cues.MARKERS`). A non-scary story carries length-matched **calm**
markers instead (`cues.CALM_MARKERS`). Ground truth: `is_scary = 1` iff any scary category fires.

Two disjoint vocabularies separate *text content* from *rubric language*:

| side | vocab | role |
|---|---|---|
| text | `cues.MARKERS` | scene fragments planted in stories; a category **fires** iff a marker appears |
| rubric | `cues.KEYWORDS` | abstract words a rubric must mention to **cover** a category |

The planted judge scores an item by the rule in `cues.planted_score`: **a category counts iff it
both fires in the text and is covered by the rubric.** So the crude seed (names only `DREAD`)
recovers the signal coarsely; a rubric enumerating all five cues recovers the full label. That
gap is the headroom the optimizer climbs.

## Confound control

The corpus is built as **matched pairs** (`build_dataset.py`): for each `(setting, n_characters)`
cell, equal numbers of scary and non-scary stories share the *same* setting, character count, and
character names — the two members of a pair differ **only** in scary vs calm markers. So setting
and #characters are exactly balanced across the label (verified in tests), and length is matched
because scary/calm fragments are length-similar.

For 1000 examples: 500 matched pairs, 5 settings × 4 character-counts = 20 cells × 25 pairs.
Every setting is 100/100 across labels; every character-count is 125/125.

## Files

| file | what |
|---|---|
| `cues.py` | the planted cue lexicon + `fires` / `coverage` / `planted_score` — single source of truth shared by builder and judge |
| `scary_metric.py` | the `is_scary` seed prompt (crude, names one cue), canonical description, and fully-articulated reference rubric (the ceiling) |
| `build_dataset.py` | confound-controlled corpus generator → `data/scary_pool.jsonl` + `data/scary_labels.csv` |
| `scary_judge.py` | `ScaryFakeBackend` — deterministic, rubric-sensitive planted judge serving every role (judge/reviser/reconstructor/grader/generator/oracle); `scary_roles()` |
| `run_optimizers.py` | drives the GEPA loop over the pool; prints the optimized prompt + floor/gepa/ceiling comparison → `runs_out/is_scary_result.json` |

## Run

```bash
# (re)generate the 1000-example corpus
python -m methods.metric_implementer.synthetic_examples.test_metric_scary.build_dataset

# run prompt optimization offline (deterministic planted judge, $0)
python -m methods.metric_implementer.synthetic_examples.test_metric_scary.run_optimizers

# unit tests
python -m pytest methods/metric_implementer/tests/test_scary_synthetic.py -q
```

Representative offline result (planted judge, 200 items, 4 rounds): the optimizer climbs by
*articulating* cues — `INIT → ANCHOR → EDGE → DECOMPOSE` — from the 1-cue seed (fidelity ≈ 0.71)
to a 4/5-cue rubric (≈ 0.85), near the fully-articulated ceiling (≈ 0.86).

## Offline vs. real judge

The default judge is deterministic and rubric-sensitive — ideal for tests and calibration, but a
*simulation* of an LLM judge. `run_optimizers --real --judge-model <slug>` swaps in a live
backend (sk3 offline-vLLM / OpenRouter) for the genuine article; it is not run by the tests.

**On "all the prompt optimizations":** only GEPA (the native `optimizer.improve` loop, which
explores the full operator set) is wired today. The external Phase-1 adapters (EvoPrompt,
ProTeGi, Auto-Rubric, OPRO/APE, a weak/random baseline) are deliberately *not* faked here —
under a deterministic judge every search strategy over the same deterministic operator collapses
to one trajectory, so a meaningful multi-optimizer comparison needs the real-LLM path.
`run_optimizers.OPTIMIZERS` is the extension point.
