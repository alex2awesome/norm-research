# Sealed-agent dispatch card (one per round) — jokes_community

The agent roles the frozen protocol requires, and the exact seal each one carries.
Everything here is prompt-file-in / JSON-file-out; no agent is told anything that is not
in its file.

Scratch root:
`/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/jokes_community/jokes_community_r<R>/`

## 1. Proposer legs (6 sealed Claude subagents per round)

One subagent per (model, track) slot, so a Track-A proposer never sees the Track-B
instruction and vice versa. Each is told **only**: read exactly this one file, follow it,
write exactly this one file, reply with the count.

| slot | prompt | output |
|---|---|---|
| opus / A | `prompt_A_claude_opus.txt` | `out_A_claude_opus.txt` |
| opus / B | `prompt_B_claude_opus.txt` | `out_B_claude_opus.txt` |
| opus_b / A | `prompt_A_claude_opus_b.txt` | `out_A_claude_opus_b.txt` |
| opus_b / B | `prompt_B_claude_opus_b.txt` | `out_B_claude_opus_b.txt` |
| sonnet / A | `prompt_A_claude_sonnet.txt` | `out_A_claude_sonnet.txt` |
| sonnet / B | `prompt_B_claude_sonnet.txt` | `out_B_claude_sonnet.txt` |

Seal wording to use verbatim:

> You are a SEALED PROPOSER. Read exactly one file: `<path>`. Follow its instructions
> exactly and write your answer — the single JSON object it asks for, nothing else — to
> `<outpath>`. Do not read any other file in that directory, do not read the repository,
> do not look for labels, outcomes, votes, scores, or any other model's output. Reply
> with only the number of items you emitted.

The other ten slots run headlessly (macOS has no `setsid`; use `nohup … &` + `disown`):

```bash
nohup python3 run_fleet.py codex --tags jokes_community_r<R> --tracks A,B > fleet_codex_r<R>.log 2>&1 < /dev/null & disown
nohup python3 run_fleet.py glm   --tags jokes_community_r<R> --tracks A,B > fleet_glm_r<R>.log   2>&1 < /dev/null & disown
```

## 2. Species-merge judges (2 sealed blind judges per round, BOTH tracks)

The freeze's identity rule is blind pairwise adjudication, not embedding-τ. Two
independent sealed judges read the same blind packet; a pair merges only if BOTH say
SAME (strict).

> You are a blind concept-identity judge. Read `<tag>_bmerge_packet.json`. For each pair
> decide whether X and Y name the SAME underlying concept or DIFFERENT concepts, judging
> the two descriptions on their own text. Write a single JSON object
> `{"judge": "<your model>", "verdicts": [{"pair_id": "...", "verdict": "SAME" or
> "DIFFERENT"}, ...]}` to `<outfile>`, one entry per pair including the anchor pairs. Do
> not read any other file and do not try to find out who wrote anything.

## 3. Blind routing auditor (1 FRESH Sonnet-class subagent per round)

A different instance every round — the freeze requires a fresh auditor so no auditor sees
the same planted probe pair twice (`audit.probes_for` draws 2 of the 4 pairs, chained off
each round's realised draw).

> You are a blind auditor. Read `<tag>_audit_prompt.txt`, follow it exactly, and write the
> single JSON object it specifies to `<tag>_audit_verdicts.json`. You are not told who
> wrote any criterion; do not try to find out, and do not read any other file.

## 4. Arbiter (frontier model, provenance VISIBLE by design)

Runs only when `audit.py finalize` reports disputes.

> You are the ARBITER. Read `<tag>_arbiter_prompt.txt` and write the single JSON object it
> specifies to `<tag>_arbiter_raw.json`. You may see which track proposed what — that is
> intentional, you are adjudicating between them.

Then `python3 arbiter.py apply --raw <tag>_arbiter_raw.json --default-round <R>` and
re-run `audit.py finalize`.

## LANDMINE: wait on PARSEABILITY, not existence

A subagent's output file exists from the first byte it writes. `until [ -f <file> ]` fires
mid-write, and a `json.load` of the partial file silently loses entries — on r2 this made
`audit.py finalize` record one criterion as having no verdict and default its route.
Always wait like this instead:

```bash
until [ -f "$F" ] && python3 -c "import json;json.load(open('$F'))" 2>/dev/null; do sleep 15; done
```

## Order of operations per round

```
stage1_slice -> harness build -> [6 sealed proposers | codex leg | glm leg]
  -> harness collect -> species -> species_merge build -> 2 MERGE JUDGES
  -> species_merge apply (strict) -> audit build -> AUDITOR -> audit finalize
  -> arbiter build -> ARBITER -> arbiter apply -> audit finalize
  -> launch_score.sh (sk3 GPU 5, offline batch) -> readout
```
