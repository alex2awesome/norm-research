# AoPS curation closure — sealed-agent dispatch card (one per round)

Cell note: `notes/2026-08-09__closure_aops.md`. Prereg:
`notes/2026-08-05__layer3-closure-prereg.md` + FREEZE DECLARATION + addenda 1–4.
Lane B, GPU 6. Everything here is prompt-file-in / JSON-file-out; no agent is told
anything that is not in its file.

Scratch root:
`/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/aops_curation`

## Order of operations per round

```
stage1_slice -> harness build -> [6 sealed Claude proposers | codex leg | glm leg]
  -> harness collect -> species
  -> species_merge build --track B -> 2 SEALED JUDGES -> species_merge apply --track B
  -> species_merge build --track A -> 2 SEALED JUDGES -> species_merge apply --track A
  -> audit build -> AUDITOR -> audit finalize
  -> arbiter build -> ARBITER -> arbiter apply -> audit finalize
  -> launch_score.sh (sk3 GPU 6, offline batch, 8192 ctx) -> readout
```

**BOTH tracks are blind-merged, BEFORE the audit** (coordinator brief 2026-08-09). The
sibling math.SE campaigns merged Track B only, which left the A-side mass inflated by the
same f₁ mechanism the B-side merge exists to remove.

**Every wait on an agent's output uses `waitfile.py`** — parseability + byte-stability,
never mere existence (coordinator ruling 2026-08-09, from the jokes half-written-verdicts
race).

## 1. Proposer legs — 6 sealed Claude subagents per round

One subagent per (model, track) slot, so a Track-A proposer never sees the Track-B
instruction. Opus ×2 salts + Sonnet ×1, matching `harness_maps.PROPOSERS`.

| slot | prompt | output |
|---|---|---|
| opus / A | `<scratch>/aops_curation_r<R>/prompt_A_claude_opus.txt` | `out_A_claude_opus.txt` |
| opus / B | `prompt_B_claude_opus.txt` | `out_B_claude_opus.txt` |
| sonnet / A | `prompt_A_claude_sonnet.txt` | `out_A_claude_sonnet.txt` |
| sonnet / B | `prompt_B_claude_sonnet.txt` | `out_B_claude_sonnet.txt` |
| opus_b / A | `prompt_A_claude_opus_b.txt` | `out_A_claude_opus_b.txt` |
| opus_b / B | `prompt_B_claude_opus_b.txt` | `out_B_claude_opus_b.txt` |

Seal wording, verbatim:

> You are a SEALED PROPOSER. Read exactly one file: `<path>`. Follow its instructions
> exactly and write your answer — the single JSON object it asks for, nothing else — to
> `<outpath>`. Do not read any other file in that directory, do not read the repository,
> do not look for labels, outcomes, votes, scores, or any other model's output. Reply
> with only the number of items you emitted.

Dispatch in two waves of 3 (pacing rule ≤3–4 concurrent). The other ten slots run
headlessly:

```bash
python3 run_fleet.py codex --tags aops_curation_r<R> --tracks A,B
python3 run_fleet.py glm   --tags aops_curation_r<R> --tracks A,B
```

Smoke-test both non-Claude legs immediately before dispatch and record the result; the
GLM leg runs ~250 s per call at this prompt length (~38 K input tokens, ~20 K output),
so budget ~20 min for its four calls.

## 2. Blind species-merge judges — 2 SEALED judges per track per round

> You are a SEALED BLIND JUDGE. Read `<tag>_bmerge<T>_packet.json`, decide SAME or
> DIFFERENT for every pair, and write the single JSON object it specifies to
> `<tag>_bmerge<T>_judge{A,B}.json`. You are not told who wrote any item; do not try to
> find out, and do not read any other file.

Strict rule: a merge edge exists only when BOTH judges say SAME. Two planted anchors
(one SAME, one DIFFERENT) per packet must pass for both judges.

## 3. Blind routing auditor — 1 FRESH Sonnet-class subagent per round

A different instance every round, so no auditor sees the same planted probe pair twice.
`audit.probes_for` draws 2 of 4 AoPS-matched pairs by chaining off each round's REALIZED
draw (verified disjoint r1↔r2, r2↔r3, r3↔r4, r4↔r5).

> You are a blind auditor. Read `<tag>_audit_prompt.txt`, follow it exactly, and write
> the single JSON object it specifies to `<tag>_audit_verdicts.json`. You are not told
> who wrote any criterion; do not try to find out, and do not read any other file.

## 4. Arbiter — frontier model, provenance VISIBLE by design

Runs only when `audit.py finalize` reports disputes.

> You are the ARBITER. Read `<tag>_arbiter_prompt.txt` and write the single JSON object
> it specifies to `<tag>_arbiter_raw.json`. You may see which track proposed what — that
> is intentional, you are adjudicating between them.

Then `python3 arbiter.py apply --raw <tag>_arbiter_raw.json` and re-run
`audit.py finalize`.

## Cell-specific things an agent on this campaign must not forget

* **The item view is the bank's, not a re-truncation.** `cells.item_view()` is the single
  definition; `aops_curation_population.csv` carries it in `text` and
  `score_gemma_maps.py` consumes it unchanged. A whole-view HEAD/TAIL cut would show
  mined criteria a different document from the A bank's.
* **8192 context** on every Gemma pass (`launch_score.sh` sets it).
* **No reference-comparison criteria.** The judge never sees the editorial solution; a
  criterion of the form "resembles the standard treatment" is unscorable here and is
  discarded. Stated in both proposer prompts.
* **The collapse gate is enforced inside `clean_fit`** — it fires on every refit, and
  each `fit_block` result carries a `collapse_gate` record.
* **Matched sampling is NOT armed** on this cell (observed position family reads .633 <
  .65). Decile stratification is the estimator of record; the matched row is a declared
  sensitivity.
* **One dense seed.** T and the "seed ensemble" figure are the same number, so Δ values
  from the curve, the discount tables and the matched readout are all on one convention
  and may be differenced — the math.SE caveat does not apply.
* **Every row is dense-held-out.** HONEST = the full population = the master ledger's E
  rows; M = FIT+MINE.
