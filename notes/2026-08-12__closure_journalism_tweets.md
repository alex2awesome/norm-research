# Layer-3 articulation closure — journalism tweets (V9). APPENDIX CELL.

Cell build note: `notes/2026-08-08__v9_journalism_community_build.md`.
Layer-1 ledger: `methods/taste_decomposition/results/journalism_tweets_ledger.json`.
Prereg: `notes/2026-08-05__layer3-closure-prereg.md` + FREEZE DECLARATION + ADDENDA.
Campaign dir: `methods/taste_decomposition/closure/journalism_tweets/`.
Agent: `claude-v9-journalism-tweets` (journalism discovery lane, cell 3 of 3).

**SCOPE IS BOUNDED BY RULING** (user, via coordinator 2026-08-12): tweets is the
journalism column's **appendix** cell. This campaign runs **round 0 and the gate
readout only**. Rounds do not follow automatically even on a gate pass — appendix
cells do not earn fleet spend without an explicit go, because the sealed fleet's one
blind look is the scarce resource.

## Round 0

### Hard gates

**Dense join gate — PASS.** Preds carry no row_id (order join, the registry landmine);
proven element-wise on the `(judgement, group)` sequence for all three seeds.

| leg | n | AUC seed 42 | shuffled counterfactual |
|---|---|---|---|
| eval | 3,114 | .6323 | .5064 |
| test | 3,112 | .6457 | .5010 |

**Splits — PASS.** Stable-hash `sha256("journalism-tweets-closure-v1|" + outlet\|day)
< .20`, MONITOR strictly inside dense-held-out, group-disjointness asserted.

| | value |
|---|---|
| MONITOR | **1,258 rows / 32 groups** |
| FIT+MINE | 29,871 rows |
| mining slice M | 4,968 rows |
| pos rate MONITOR / FIT+MINE | **.5000** / .49998 |

The MONITOR positive rate lands at exactly .5000 because this cell's y is a
**within-group median split** — the same structural property that puts group identity
alone at .5000 (Layer-1) and makes pooled ≈ within-group by construction. So unlike BBC
most-read (group identity .5814, within-day primary), the **pooled MONITOR tier is the
honest primary here**.

**The number to watch is 32 groups, not 1,258 rows.** Every CI in this campaign
resamples groups, so 32 is the effective sample size for the bootstrap — fewer than
BBC's 88 or homepage's 72 MONITOR groups, despite the larger row count. Whether the
residual is resolvable is therefore genuinely open, and a resolution-bound terminal is
an acceptable and expected outcome.

**Item view.** V, A and the dense arm all read the byte-identical string
`"HEADLINE: " + anchor headline`, so no view asymmetry is possible (the SO lesson);
headlines are short and nothing truncates.

### Anchor

| quantity | MONITOR (n = 1,258 / 32 groups) |
|---|---|
| VA_lin | .5825 |
| VA_nl per seed | .6176 / .6160 / .6169 |
| **VA_nl** | **.6168** |
| T per dense seed | .6587 / .6527 / .6612 |
| **T** | **.6575** |
| **Δ₀** | **+.0407** |

sklearn 1.8.0 here and in the Layer-1 ledger — **no version drift on this cell**.
Item view byte-identical across arms; max 193 tokens, nothing truncates.

### VERDICT — TERMINAL AT ROUND 0, resolution-bound. **And it is a DIFFERENT terminal from homepage's.**

The two pre-round checks **disagree**, and the disagreement is the finding:

| check | result |
|---|---|
| gate (Δ₀ vs .02) | **PASS** — Δ₀ = +.0407, twice the threshold |
| **residual resolvability** (is Δ_beyond ≠ 0?) | **HOLDS** — T − VA_nl = **+.0445**, 95% CI **[+.0133, +.0791]**, P(>0) = **.9975**, does **not** span zero |
| **ε-resolvability** (can a round's .005 gain be seen?) | **FAILS** — paired SD **.00804** = 1.6× ε |

So: **this cell has a real, statistically resolvable articulation residual — and the
closure instrument cannot track progress against it.** A round's gain of ε would be
indistinguishable from noise, so the saturation rule (two consecutive sub-ε rounds)
would fire essentially at random and the resulting curve would not mean anything.

This must not be collapsed with the homepage terminal. They are opposite failures:

| | homepage curation | journalism tweets |
|---|---|---|
| Δ₀ | +.0035 (below gate) | **+.0407** (passes gate) |
| residual CI | [−.0429, +.0642] — **spans zero** | [+.0133, +.0791] — **excludes zero** |
| reading | **no residual is measurable** | **a real residual exists** |
| why it stops | nothing to close | closure progress is untrackable |

The correct wording for tweets is therefore: *"a real residual of roughly +.04 that
this closure design cannot resolve round-over-round"* — **not** "closed at this cell's
resolution" (that is homepage's phrase) and emphatically **not** "saturated".

**What would unblock it, recorded rather than acted on.** The binding constraint is
**32 MONITOR groups**, not the 1,258 rows — every CI resamples groups. Cross-fitting,
the freeze's stated remedy, shrinks the fit-seed component but not the group-sampling
component, so it is unlikely to be sufficient on its own. Enlarging MONITOR (this cell
has ~124 held-out groups, of which 32 are in MONITOR) would help, but that is a split
redesign and would require a prereg amendment — **not** a post-hoc adjustment made
after seeing this result. Parked for the coordinator.

**Swap baseline:** C₊ **.7425** / C₋ **.3770** (dense concordance .6654, 395,641 pairs).

Decision rule encoded in the script:

| condition | verdict |
|---|---|
| Δ₀ ≤ .02 | STOP AT ROUND 0 (terminal, gate) |
| Δ₀ > .02 but ε-check fails **or** residual CI spans zero | STOP AT ROUND 0 (terminal, **resolution-bound**) |
| Δ₀ > .02 and resolvable | **report to coordinator before any rounds** (appendix cells do not auto-earn fleet spend) |

A resolution-bound stop is reported as "closed at this cell's resolution", **never**
"saturated" — the press_verdict wording, and the same distinction homepage curation
turned on.

---

# LANE HAND-OFF / RESUMPTION STATE

Written deliberately so this lane can be resumed cold. Current as of 2026-08-12.

## Where each cell stands

| cell | status | next action |
|---|---|---|
| **BBC most-read** | round 0 COMPLETE, gate PASS (Δ₀ +.0749), ε-resolvable (.00252). Round-1 stage-1 slice BUILT and on disk. **Fleet HELD.** | run the sealed fleet the moment a second family is available |
| **homepage curation** | **TERMINAL at round 0** (noise-floor). Logged. | none — closed |
| **journalism tweets** | round 0 running | read `round0_results.json`, apply the decision table above, report |

## The one blocker, and exactly what unblocks it

BBC round 1's sealed fleet needs **≥ 2 proposer families** (the freeze's recorded
floor). Right now:

- **Claude legs**: unavailable — subagent cap exhausted (500/500), not raisable in this
  session. A fresh session restores these.
- **GLM-5.2 (both keys)**: HTTP 429, error code **1302** ("Rate limit reached for
  requests"), persisting across both keys and across pauses. Shared quota — other agents
  draw on the same two keys. Re-probe **lazily at stage boundaries only**, never a poll
  loop.
- **codex gpt-5.6-luna**: WORKING (`CODEX_SMOKE_OK`).

So the unblock is: *either* a fresh session (Claude legs return) *or* GLM clearing.
Then seal and run at P=8 / 2 families on the **byte-identical slice already on disk** —
`closure/bbc_mostread/slice_r1.json` and `slice_r1_cards.txt`. The slice is
deterministic; do **not** rebuild it, and do not let a proposer see it twice.

Cheap GLM probe:

```python
import json,urllib.request,pathlib
key=pathlib.Path("~/.z-ai-api-key.txt").expanduser().read_text().strip()
body={"model":"glm-5.2","max_tokens":16,"messages":[{"role":"user","content":"OK"}]}
req=urllib.request.Request("https://api.z.ai/api/anthropic/v1/messages",
  data=json.dumps(body).encode(),
  headers={"content-type":"application/json","x-api-key":key,
           "anthropic-version":"2023-06-01"})
print(json.loads(urllib.request.urlopen(req,timeout=60).read()).get("usage"))
```

## Round-1 proposer framing — a labelled augmentation, already decided

Of the 60 highest-disagreement rows, the **bank scores higher than dense on 43**, dense
higher on 17. The Track-A prompt therefore carries a **neutral both-directions
direction note**, explicitly labelled as an augmentation: ask both why the articulated
bank may be *over*-predicting these items and what the dense arm may be seeing that the
bank misses. The prereg instruction is **not** rewritten mid-campaign (coordinator
ruling); the direction note sits beside it.

## What is reusable, and what had to be written

Reused unchanged: `scaleupC_layer1` (bank loader, dense_T, group bootstraps),
`layer1_gemma_cells` (frozen grid, folds, OOF estimators),
`datasets/patents/build_dense_standard_claimfell.py` (split bucketer),
`datasets/va_gemma_banks/score_{va_gemma,scaleupC}_banks.py` (Gemma scoring, anchors,
battery), and the closure protocol modules under `closure/` (`run_bmerge_judges.py`,
`missing_mass.py`, `swap_analysis.py`).

Written for this lane, and shared across its three cells:
`closure/bbc_mostread/round0_bbc.py` exports `paired_seed_noise`, `fit_va` and
`swap_pair`; `closure/homepage_curation/round0_homepage.py` exports `group_boot_ci`.
The other two round-0 drivers import from those rather than re-implementing, so a fix
to the ε check or the swap algebra propagates to all three.

## Standing cautions this lane established

1. **Run the ε-resolvability check before the curve, always.** It cost minutes and
   saved homepage curation an entire campaign that could not have produced an
   interpretable number.
2. **Dense preds in this program carry no row_id.** Every cell's join is by order and
   must be proven element-wise with a shuffled counterfactual.
3. **Never `pgrep -f` on a string that appears in the launching shell's own command
   line.** The V9 build deadlocked exactly that way (the heredoc writing the script
   contained the script's name). Chains here wait on a PID via `kill -0`.
4. **Anchor rows must match the provenance the system prompt asserts** — the BBC bank's
   battery read .481 with mixed-outlet anchors and .602 with BBC-only anchors, same
   bank, same judge.
5. `setsid --fork` + assert `ppid == 1` for every detached job; verify after the
   launching shell exits, not before.

## Compute

No GPU held by this lane. All round-0 work is CPU. CW has priority on free cards; when
BBC round 1 scores, claim **one** card via the ledger and rely on the shard checkpoint
in `score_bank` to yield and re-claim.
