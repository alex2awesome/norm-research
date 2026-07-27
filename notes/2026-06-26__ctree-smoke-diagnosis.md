# ctree live-smoke diagnosis — the broken seam is the PROPOSER, not the engine (2026-06-26)

The 500-item live smoke (`smoke_infill_live.py`, glm-5.2 proposer+judge) failed to rediscover
the planted tacit norms (song/glow), proposing a redundant feature instead. The
`2026-06-26__ctree-location-and-live-path.md` note *guessed* "needs the full 2400-item corpus."
Two free + one cheap experiment refute that and pin the cause on the **LLM proposer**.

## 1. Oracle 2×2 bisect — engine, cfg, and corpus size are all EXONERATED

`methods/metrics_tree_infilling/tests/test_scenario/diag_bisect.py` swaps the oracle (no-LLM)
proposer+judge into a 2×2 of {n=500, 2400} × {smoke_cfg, test_cfg}. **All four conditions
rediscover both song and glow:**

| cond | n | cfg | song | glow | rounds | final_gaps |
|---|---|---|---|---|---|---|
| A | 500 | smoke (the exact failing cfg) | ✓ | ✓ | 2 | 0 |
| B | 500 | test | ✓ | ✓ | 3 | 0 |
| C | 2400 | smoke | ✓ | ✓ | 2 | 0 |
| D | 2400 | test | ✓ | ✓ | 3 | 0 |

→ **A passes**: the *exact* smoke cfg + n=500 works with the oracle. So corpus size and cfg
tightness (`max_outer_rounds=2, max_depth=4, min_node_size=30`) are **not** the problem. The
failure is the **LLM seam**.

## 2. Proposer probe — the proposer ignores "don't re-derive known criteria"

`diag_proposer_probe.py` fits the tree with the **free oracle judge**, flags the gap node, builds
the residualized contrast, and calls a **live Gemma-4 proposer** (OpenRouter) on the *same* prompt
the oracle answers. n=500, one gap node (n=188, depth 1):

- **Oracle proposer** (correct): `glow_luminous` — "whether the creature's hide casts a soft glow." ✓
- **Live Gemma-4 proposer**: `Dietary Habit` — "herbivore (grazes) vs carnivore (hunts)."

`feeding`/`grazer` is a **published Code criterion** (`metrics.py`: "Gentle feeder…"). The
proposer prompt explicitly says *"KNOWN CRITERIA (do not re-derive these or anything they already
cover)."* The oracle is hard-coded to skip `world.KNOWN_ATTRS`; **Gemma-4 ignored the instruction**
and reached for a salient lexical shortcut ("grazes" / "tears into" / "stalks") that partially
correlates in the muddy mixed gap node, instead of the residualized tacit norm. glm-5.2 (the
smoke's model) is weaker than Gemma-4, so it failed at least as hard.

## 3. Conclusion + plan implication

**The broken seam is the proposer (instruction-following on "not already covered"), not the judge,
not the engine, not n.** The smoke proposed a redundant known criterion → dropped by the redundancy
guard → no tacit norm discovered.

**This means the GEPA+Gemma distillation pathway (`run_distillation.py`, tasks a/b/d) is pointed at
the WRONG seam for fixing discovery quality.** GEPA optimizes the **judge** rubric prompt
(materialization fidelity); the proposer is a separate, broken component. Distillation remains
valid for its **cost/scale** motivation (cheap judge materialization off OpenRouter/sk3 instead of
z.ai), but it will **not** make the toy rediscover song/glow.

## 4. Judge half — PERFECT (exonerated)

`diag_judge_probe.py`: gave the live Gemma-4 judge the *correct* glow/song rubrics over a 100-item
sample and compared to `world.detect` truth:

| feature | acc | appl | score dist (0 / 0.5 / 1) | reliability | truth=pos mean | truth=neg mean |
|---|---|---|---|---|---|---|
| glow | **1.000** | 1.00 | 0.42 / **0.00** / 0.58 | **1.000** | 1.00 | 0.00 |
| song | **1.000** | 1.00 | 0.48 / **0.00** / 0.52 | **1.000** | 1.00 | 0.00 |

The judge is perfect on a correct feature: 100% accuracy, perfect test-retest, cleanly bimodal
(no all-0.5 collapse). **The judge is not the problem.** → A proposer fix alone suffices; GEPA
judge-optimization (the distillation pathway) is NOT needed for discovery quality — only for
cost/scale. And a proposer fix will validate cleanly (the judge scores glow/song perfectly).

## 5. Fixes for discovery quality (proposer side)

1. **Harden the proposer prompt** (`feature_gen._PROMPT`): enumerate the known criteria *by name*
   in a "MUST NOT be your answer" block, not just as a prose list; require the answer to be a
   property none of them capture.
2. **Pre-materialization redundancy guard**: reject a proposal that duplicates a known criterion
   *before* scoring it, and re-propose. The existing `redundancy_check` only fires after
   materialization (vs the X columns); the next-steps plan's "adversarial proposer test" already
   flags that **redundancy currently does no work in any test** — this is exactly that gap.
3. **k-candidate proposals** (next-steps Phase 4): sample k, keep best closure — a wrong first
   proposal is then recoverable.
4. **Cleaner contrasts**: the single n=188 gap node mixed marsh+cavern+grove; deeper/per-region
   contrasts (the depth_dial pooling, or more rounds) would reduce the muddy-correlation shortcuts.
   (Secondary — the proposer ignores the known-criteria constraint even on a clean contrast.)

## 6. Fix applied + capstone validated (2026-06-26)

Fixed the proposer seam in `feature_gen.py`:
- `_PROMPT`: prominent **FORBIDDEN** known-criteria block + requests up to
  `cfg.proposer_k_candidates` (default 4) distinct candidates, each not a restatement of a known
  criterion.
- `propose_feature`: parses single-object *or* `{"candidates":[...]}`, returns the first candidate
  not token-redundant with a known criterion (coarse lexical backstop).
- Side fix: `oracle.py` + `test_latent_split._oracle_proposer` parsed the prompt by the literal
  delimiter `"Identify exactly ONE"`; the prompt edit broke them. Made block parsing robust (each
  example block runs to the next blank line). Latent coupling, now removed.

**Capstone** (`smoke_infill_gemma.py`): full live loop, Gemma-4 proposer+judge via OpenRouter,
n=500, smoke_cfg. **Passed end-to-end (321s):** R1 KEEP `Luminosity` (glow, depth 3, rel 1.0,
coverage 0.193 ≈ cavern 20%); R2 KEEP `Vocal Melodiousness` (song, depth 0, rel 1.0, coverage
0.807); R3 final_gaps=0. song=True, glow=True, no decoys. Live coverage (0.193 / 0.807) **matches
the oracle** diag_bisect condition A exactly. First time the live path works on the toy. 10/10
offline tests green.

## 7. Generalization boundary (do NOT overclaim)

This validates the **pipeline + the proposer fix on the EASY toy only.** It does NOT establish
generalization:
- glow/song are near-explicit **lexical cues** ("bioluminescent shine") — a reading task, not a
  judgment task. The 100% judge accuracy / rel=1.0 is an artifact of easy cues, not a property of
  the judge on abstract real rubrics.
- The **lexical redundancy gate is weak** — it scored "diet↔feeding" at 0.2 overlap and would MISS
  that semantic redundancy; the toy was saved by the *prompt*, not the gate. Real tasks need
  embedding/LLM redundancy.
- FORBIDDEN-block efficacy untested at ~40 known criteria (toy had 4).
- Untested on real tasks; single seed; the judge-collapse failure mode (guided-JSON → all-0.5,
  valid schema / zero variance) is unscreened on abstract rubrics.

**The only thing that would test generalization: run on a real task** (peer-review / press-release)
with abstract rubrics, multiple seeds, judge-distribution check, upgraded redundancy. Until then the
honest claim is: *pipeline composes on easy cues; real-task generalization untested.*
