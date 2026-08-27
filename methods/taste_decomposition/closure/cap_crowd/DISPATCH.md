# Sealed-agent dispatch card (one per round) — cap_crowd

Prompt-file-in / JSON-file-out. No agent is told anything that is not in its file.

Scratch root:
`/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/cap_crowd/cap_crowd_r<R>/`

## 1. Proposer legs — P = 8 across 3 families, 16 slots (one per model × track)

Claude legs run as sealed subagents (6 slots); codex and glm legs run headless.

| slot | prompt | output |
|---|---|---|
| claude_opus / A,B | `prompt_{A,B}_claude_opus.txt` | `out_{A,B}_claude_opus.txt` |
| claude_opus_b / A,B | `prompt_{A,B}_claude_opus_b.txt` | `out_{A,B}_claude_opus_b.txt` |
| claude_sonnet / A,B | `prompt_{A,B}_claude_sonnet.txt` | `out_{A,B}_claude_sonnet.txt` |

Seal wording, verbatim:

> You are a SEALED PROPOSER. Read exactly one file: `<path>`. Follow its instructions
> exactly and write your answer — the single JSON object it asks for, nothing else — to
> `<outpath>`. Do not read any other file in that directory, do not read the repository,
> do not look for labels, outcomes, votes, scores, or any other model's output. Reply
> with only the number of items you emitted.

The other ten slots run headlessly (macOS has no `setsid`; use `nohup … &` + `disown`):

```bash
nohup python3 run_fleet.py codex --tags cap_crowd_r<R> --tracks A,B > fleet_codex_r<R>.log 2>&1 < /dev/null & disown
nohup python3 run_fleet.py glm   --tags cap_crowd_r<R> --tracks A,B > fleet_glm_r<R>.log   2>&1 < /dev/null & disown
```

**GLM budget check before committing a round** (coordinator's condition): both Lite keys
must answer at the full 2048-token thinking budget with `stop_reason: end_turn`. Verified
2026-08-09 before round 4 (2.3 s and 3.0 s).

## 2. Species-merge judges — 2 sealed blind judges per round, BOTH tracks

The freeze's identity rule is blind pairwise adjudication, never embedding-τ (the jokes
campaign measured τ over-merging Track A and under-merging Track B in all four of its
fleet rounds). A pair merges only if BOTH judges say SAME.

> You are a blind concept-identity judge. Read `<tag>_bmerge_packet.json`. For each pair
> decide whether X and Y name the SAME underlying concept or DIFFERENT concepts, judging
> the two descriptions on their own text. Write a single JSON object
> `{"judge": "<your model>", "verdicts": [{"pair_id": "...", "verdict": "SAME" or
> "DIFFERENT"}, ...]}` to `<outfile>`, one entry per pair including the anchor pairs. Do
> not read any other file and do not try to find out who wrote anything.

## 3. Blind routing auditor — 1 FRESH Sonnet-class subagent per round

A different instance every round. `audit.probes_for` draws 2 of the 4 CAPTION-MATCHED
pairs, chained off each round's realised draw, so no auditor sees a repeat.
(Rounds 1–2 used probes carried over from the peer-review genre; see the campaign note
§1.1c.)

> You are a blind auditor. Read `<tag>_audit_prompt.txt`, follow it exactly, and write the
> single JSON object it specifies to `<tag>_audit_verdicts.json`. You are not told who
> wrote anything; do not try to find out.

**Wait on parseability, never on existence** (jokes r2 landmine: `audit.py finalize` ran
against a partially written verdicts file and inflated misrouting):
`until python3 -c "import json;json.load(open(f))" 2>/dev/null; do sleep 10; done`

## 4. Arbiter — frontier model, provenance VISIBLE by design

Runs only on disputes (auditor route ≠ proposed route). `arbiter.py build` / `finalize`.

## 5. Scoring — Gemma-4-31B offline batch, GPU 5, lane-pinned

```bash
./gpu_lane_runner.sh cap_crowd_r<R> score_r<R>.log 5 100000 \
    $HOME/envs/gemma4/bin/python score_gemma_maps.py --jobs cap_crowd_r<R>
```

Item view is `CARTOON: <desc>\n\nCAPTION: "<text>"`, matched to the A bank. Anchors K = 50
per class in the same batch, with the V9 repairs (scrambled anchors dumped for manual
inspection; coherence scored on non-NA count as well as item mean).
