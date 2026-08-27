# V_new compilation pilot — three terminal cells

Date: 2026-08-11. Agent: claude-so-votes-audit-fable (continued assignment).
Status: RUNNING — triage complete, compile fleet in flight; results fill in.
Charge: the grid's V_new column has never been real because no stage compiles
discovered criteria into deterministic code. Pilot before any scale-up
(validate-before-scaling) on three TERMINAL cells: jokes_community (124-feature
terminal bank; priority target "Read-aloud cadence", judged alone-AUC .682
campaign / .671 fit-mine here), press_verdict (r2 terminal), nc_responded (r5
terminal).

Terms unpacked: **V_new** = V (hand-coded surface features) + CERTIFIED
compiled columns (deterministic Python translations of judged criteria; no
judge calls at runtime). **terminal bank** = each closure campaign's final
judged criterion set (base A bank + A-routed non-collapsed mined criteria;
incumbent phrasings/columns). **certification** = held-out (MONITOR rows)
Spearman ρ(compiled, judged parent) + alone-AUC comparison; certified = ρ ≥
.30 AND modal share ≤ .98 (declared before any compile ran). **fit_block** =
the closure campaigns' frozen Layer-1 estimator (md5-identical
`closure_core.py` across cells: HistGB grid {15,31}, lr .06, seeds 0/1/2,
group-OOF; linear leg beside it).

## 0. Method (seam-program discipline, reused not rebuilt)

- Machinery mirrored from `methods/metric_seam/battery/agentic_run.py` +
  `cert_agentic_r2.py` (R12 AGENTIC-COMPILE conventions): programs are pure
  functions of the item text; iteration/authoring never touches held-out; the
  h0-style reference here is the judged parent column itself.
- **Label-blindness (stricter than the seam harness):** compilers received
  ONLY the criterion name (+ the judge instruction where the campaign stored
  one — nc) and 8 UNLABELED fit-split sample texts. No y, no judged scores,
  no AUCs. Certification is the first time a compiled column meets data.
- Compiler/triage fleet: **codex `gpt-5.6-luna`** (`codex exec`, reasoning
  effort high, read-only sandbox, scratch wd outside the repo — the
  run_fleet.py codex-leg pattern). Family recorded in every artifact.
- Driver: `methods/taste_decomposition/vnew_pilot/pilot.py`
  (bank / prompts / codex / certify / refit; one cell per process; CPU only).
- Banks assembled per cell from the campaigns' own artifacts, ids-carried:
  jokes 47 base + 50 mined-A (+27 V = the documented 124 terminal features);
  press 40 base + 27 mined-A; nc 198 base + 68 mined-A. Incumbent judged
  columns (pre-GEPA phrasing swaps) are used for certification AND for the
  VA refit legs, so every comparison in §3 is internally consistent; where
  terminal-quoted campaign numbers differ (GEPA-swapped columns), that is a
  labelled difference, not an error.
- nc holdout note: MONITOR here = the campaign's monitor_full (1,892 rows,
  docket-disjoint from fit_mine); the 377-row decision monitor is too small
  for stack readouts. Declared deviation.

## 1. Codability triage (gpt-5.6-luna; rubric: deterministic stdlib Python of
the text alone — regex/counting/syllable/stress heuristics/word lists OK; no
model calls, no external data, no runtime semantics)

| family | jokes (codable/n) | press (codable/n) | nc (codable/n) |
|---|---|---|---|
| phonetic_prosodic | **5/5** | — | — |
| lexical_surface | 4/4 | 2/2 | 1/2 |
| structural_format | 6/9 | 10/14 | 14/25 |
| register_pragmatic | 1/5 | 10/14 | 5/7 |
| semantic_content | 4/55 | 15/18 | 0/194 |
| relational_context | 0/10 | 2/15 | 1/36 |
| affective_subjective | 0/9 | 0/4 | 0/2 |
| **total** | **20/97 (21%)** | **39/67 (58%)** | **21/266 (8%)** |

First read of the profile (triage-level; certification will discipline it):
- **Prosody is the codable island in humor** — all five phonetic/prosodic
  criteria (incl. Read-aloud cadence) marked codable, exactly the charge's
  hypothesis (syllable/stress/rhythm features).
- **The mined semantic mass is uncodable** — nc's 194 semantic criteria: 0
  codable. What the closure campaigns discovered is overwhelmingly knowledge
  that does NOT survive translation to deterministic code.
- press's high rate (58%) reflects its bank's press-release *form* criteria
  (structure/register/boilerplate); luna was generous on semantic_content
  (15/18) — the certification gate is the arbiter, not the triage.

## 2. Compile + certification (held-out MONITOR; ρ floor .30 + collapse gate)

15 compile batches returned; 2 of 15 modules had syntax errors and were
dropped whole (a compile-infrastructure failure mode worth recording: 21 of
80 triaged-codable criteria were lost to two unparseable modules — jokes
batch 03, press batch 01). Of the 72 loaded functions:

| cell | loaded | certified | median ρ(monitor) | certified examples (ρ / judged-AUC / compiled-AUC) |
|---|---|---|---|---|
| jokes_community | 18/20 | **4** | .064 | r1:A03 .647/.559/.479; r2:A07 .610/.565/.550 |
| press_verdict | 33/39 | **13** | .279 | base:24 .598/.452/.438; base:35 .552/.545/.516 |
| nc_responded | 21/21 | **11** | (nan-mixed) | base:174 .703/.551/.507; base:63 .485/.487/.464 |

**The headline negative result — the priority target fails certification.**
"Read-aloud cadence" (judged alone-AUC .687 on monitor — the campaign's
strongest single mined criterion) compiles to a function with ρ = **.098**
against its judged parent and alone-AUC .548. The compiled prosody features
(syllable/stress/rhythm heuristics) carry *some* signal but track a different
construct than what the Gemma judge scores under that name: the judged
criterion is not mechanical prosody. Under the charge's framing, prosody is
"partially codable" only in the weak sense — a code approximation exists but
it is NOT the judged channel (single-shot; a seam-style train-split iterated
refinement is the named follow-up, not run in this pilot).

**Second-stage codability profile (certification-disciplined):** of 430
terminal-bank criteria, 80 triaged codable (19%), 72 compiled cleanly, **28
certified (6.5% of the banks)** — and the certified set is concentrated in
structural/register/boilerplate criteria whose judged parents are themselves
weak (judged alone-AUCs .44–.57). Discovered knowledge that MATTERS
(semantic, relational, the strong mined criteria) does not survive
translation to deterministic code; what survives is mostly the part V
already approximates.

## 3. V_new stack refits (frozen fit_block; MONITOR nl / fit-mine OOF nl)

| cell | V | V_new | Δ(V_new−V) | VA | V_new+A | Δ(+A leg) |
|---|---|---|---|---|---|---|
| jokes_community (4 cols) | .6195 / .6375 | .6262 / .6372 | +.0067 / −.0003 | .7501 / .7566 | .7488 / .7556 | −.0013 / −.0010 |
| press_verdict (13 cols) | .7074 / .6656 | .7136 / .6702 | +.0062 / +.0046 | .7447 / .7380 | .7318 / .7392 | −.0129 / +.0012 |
| nc_responded (11 cols) | .7065 / .7080 | **.7455** / .7219 | **+.0390 / +.0139** | .7875 / .7429 | **.8087** / .7495 | **+.0212 / +.0066** |

(each pair = MONITOR nl / OOF fit-mine nl; ids-carried OOF in `vnew_out/oof_<cell>.npz`)

## 4. Pilot verdict

1. **The V_new column can be made real, but only for the shallow tail.** 28 of
   430 terminal criteria (6.5%) certify; the strong mined criteria — the ones
   that made the closure campaigns interesting — do not compile (read-aloud
   cadence ρ .098; nc's 194 semantic criteria triage 0% codable).
2. **Where it works, it works as V-enrichment, not A-replacement.** On top of
   the full judged bank, compiled columns add ≈0 on jokes/press (−.001/−.013
   MON, ~0 OOF) — faithful-but-weak copies of channels A already carries. On
   nc_responded they add real signal (+.021 MON / +.007 OOF over VA;
   +.039/+.014 over V): boilerplate/salutation/structure criteria compile
   well and the judge measures them noisily, so the code version is BETTER
   than its parent there.
3. **Scale-up: SELECTIVELY warranted.** Worth running as a cheap CPU stage on
   cells with structural/register-heavy banks and thin V (the nc pattern);
   NOT warranted as a general pipeline stage for semantic banks (jokes
   pattern), and no substitute for the judge on the criteria that carry the
   closure gains. Compile-infrastructure note for any scale-up: 2/15 modules
   lost to syntax (21 criteria) — add a parse-retry loop.
4. **Codability profile (the deliverable):** codable ≈ phonetic/lexical/
   structural/register families (5/5, 7/8, 30/48, 16/26 triaged; certification
   halves each); uncodable ≈ semantic/relational/affective (19/267, 3/61,
   0/15). Discovered knowledge that matters is judge-bound: mined criteria
   survive translation to code at a LOWER rate than base-bank criteria on
   the same families.

### 4b. Increment bootstraps (group-level, MONITOR) + length screen

| cell | V_new−V [95% CI] | (V_new+A)−VA [95% CI] |
|---|---|---|
| jokes | +.0067 [+.0017, +.0152] | −.0014 [−.0030, +.0018] |
| press | +.0062 [−.0236, +.0794] | −.0129 [−.1610, +.0059] |
| nc | **+.0389 [+.0040, +.0772]** | **+.0212 [+.0016, +.0421]** |

Only nc_responded's increments exclude 0 at the group level (press is
noise-dominated at 20 monitor groups). **Length-nuisance flag (coordinator
screen): the top certified compiled columns are strongly length-correlated**
— nc base:174 ρ(char_len) .62, base:149 .72, base:175 .53; press base:24
.66, base:35 .60; jokes r1:A03 −.73, r5:A02 +.71. The nc gain in particular
may be partly a re-packaged length/verbosity channel (length is a known
strong N&C channel, alone-AUC .62). Before any registry quote of the nc
V_new gain, run the Layer-2 length-stratified readout on the V_new stack —
FLAGGED, not run here (context-bounded).

Follow-up (named, not run): seam-style train-split agentic iteration on
read-aloud cadence — single-shot failed; the R12 harness pattern
(`agentic_run.py`) is the tool if anyone wants to know whether iteration
closes the ρ gap.

## 4. Artifacts

- Driver + outputs: `methods/taste_decomposition/vnew_pilot/`
  (`pilot.py`; `vnew_out/bank_<cell>.json`, `judged_<cell>.npz`,
  `triage_<cell>.json`, `codability_profile.json`, `cert_<cell>.json`,
  `compiled_<cell>.npz`, `results_<cell>.json`, `oof_<cell>.npz`).
- Scratch (prompts/raw codex): `/tmp/vnew_pilot_scratch/<cell>/`.
- No GPU used; all CPU. No campaign artifact modified.
