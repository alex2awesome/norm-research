# Creature-dossier scenario

A self-contained, end-to-end test of metrics-tree infilling on **text** with embedded,
discoverable structure. It is a deliberate miniature of the research problem: a community's
**published rubric** versus the **tacit norms** it actually applies.

## The world

Short field-guide entries for invented creatures (so the only signal about a creature is in
its prose — a reader cannot shortcut via prior knowledge). Each creature has attributes, each
realized with several interchangeable phrasings:

| Attribute | Values | Role |
|---|---|---|
| habitat | grove / marsh / cavern | **known** — context (defines regions) |
| size | tiny / hulking | **known** — Code criterion |
| feeding | grazer / hunter | **known** — Code criterion |
| pelt | furred / scaled | **known** — Code criterion |
| **glow** | luminous / dim | **hidden tacit norm** (cavern) |
| **song** | melodious / harsh | **hidden tacit norm** (marsh) |
| color | azure / ochre | **decoy** — no effect on the verdict |
| limbs | four / six | **decoy** — no effect on the verdict |

Habitat is sampled unequally: **grove 45%, marsh 35%, cavern 20%.**

## The label (the elders' verdict — a *norm*, not a fact)

`judgement = 1` iff the village keeps the creature as a companion:

- **grove (Code works):** kept if it is small **and** gentle **and** soft — a *combination*
  of published criteria, each individually modest.
- **marsh (Code silent):** kept iff **melodious** — the broad tacit **song** norm.
- **cavern (Code silent):** kept iff **luminous** — the narrow tacit **glow** norm.
- color / limbs never matter.

So the published Companion Code (habitat, size, feeding, pelt) explains the grove but is
useless in the marsh and cavern, where the elders fall back on unstated aesthetic preferences.

## What the loop should discover

| | true region | should be discovered as | measured coverage* |
|---|---|---|---|
| **song** (broad) | marsh ≈ 34% | a tacit norm governing the larger marsh | ~0.33 |
| **glow** (narrow) | cavern ≈ 20% | a tacit norm governing the smaller cavern | ~0.23 |
| color, limbs | — | **never** (decoys) | — |

\* *coverage* = fraction of the population in leaves where the discovered feature is *active*
(non-negligible standardized coefficient) in the final tree. It is the method's measured
**generality**: song's coverage exceeds glow's, recovering "song is the broader norm" from
data — depth/generality is measured, not assigned.

## Files

| File | Role |
|---|---|
| `world.py` | single source of truth: attributes, phrasings, label rule, `detect()` |
| `generate.py` | builds `corpus.csv` + `answer_key.csv` (seeded, parameterized) |
| `corpus.csv` | committed corpus (2400 items): `id, text, judgement` |
| `answer_key.csv` | per-item ground-truth attributes + region + p(kept) |
| `metrics.py` | the published Companion Code (known code-metrics) |
| `oracle.py` | deterministic offline proposer + judge (CI; no LLM) |
| `test_discovery.py` | asserts both norms found, decoys rejected, coverage ranks song > glow |

## Running

```bash
# offline (deterministic oracle, no LLM, no keys) — the CI path
python -m pytest methods/metrics_tree_infilling/tests/test_scenario/

# regenerate the corpus (e.g. different size/seed)
PYTHONPATH=methods python -m metrics_tree_infilling.tests.test_scenario.generate --n 2400 --seed 7

# LIVE: real LLM proposer + judge actually read the prose and articulate the norms
PYTHONPATH=methods python -m metrics_tree_infilling.tests.test_scenario.test_discovery --live
```

## Offline oracle vs live LLM — what each proves

The loop needs an LLM in two places: the **proposer** (articulate the missing property from
the WRONG-vs-RIGHT contrast) and the **judge** (score a proposed rubric over the corpus). The
offline run replaces both with deterministic stand-ins:

- **`oracle_proposer`** reads the example texts and returns the non-Code attribute that best
  separates kept from not-kept (skipping the known attributes, just as a real LLM is told to
  avoid criteria "already covered"). It chooses among the two tacit norms **and the two
  decoys** by genuine label-separation — so it can be fooled, and the test checks it isn't.
- **`oracle_judge_scorer`** detects a feature's target attribute via `world.detect`.

The offline run therefore proves the **algorithm** (gap → contrast → propose → materialize →
reinsert → guards → measured coverage) end-to-end, deterministically and for free. It does
**not** prove that a *real* model can articulate "it is bioluminescent" from "its hide casts a
soft glow"; only the `--live` run tests that.
