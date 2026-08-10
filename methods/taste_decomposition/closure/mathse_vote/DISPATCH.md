# Sealed-agent dispatch card (one per round)

The three agent roles the frozen protocol requires, and the exact seal each one carries.
Everything here is prompt-file-in / JSON-file-out; no agent is told anything that is not
in its file.

## 1. Proposer legs (4 sealed Claude subagents per round)

One subagent per (model, track) slot, so a Track-A proposer never sees the Track-B
instruction and vice versa. Each is told **only**: read exactly this one file, follow it,
write exactly this one file, reply with the count.

| slot | prompt | output |
|---|---|---|
| opus / A | `.../scratchpad/mathse_vote/mathse_vote_r<R>/prompt_A_claude_opus.txt` | `out_A_claude_opus.txt` |
| opus / B | `prompt_B_claude_opus.txt` | `out_B_claude_opus.txt` |
| sonnet / A | `prompt_A_claude_sonnet.txt` | `out_A_claude_sonnet.txt` |
| sonnet / B | `prompt_B_claude_sonnet.txt` | `out_B_claude_sonnet.txt` |

Seal wording to use verbatim:

> You are a SEALED PROPOSER. Read exactly one file: `<path>`. Follow its instructions
> exactly and write your answer — the single JSON object it asks for, nothing else — to
> `<outpath>`. Do not read any other file in that directory, do not read the repository,
> do not look for labels, outcomes, votes, scores, or any other model's output. Reply with
> only the number of items you emitted.

The other four slots run headlessly:

```bash
python3 run_fleet.py codex --tags mathse_vote_r<R> --tracks A,B
python3 run_fleet.py glm   --tags mathse_vote_r<R> --tracks A,B
```

## 2. Blind routing auditor (1 FRESH Sonnet-class subagent per round)

A different instance every round — the freeze requires a fresh auditor so no auditor sees
the same planted probe pair twice (`audit.probes_for` draws 2 of 4 pairs by stable sha256
of the round tag).

> You are a blind auditor. Read `<tag>_audit_prompt.txt`, follow it exactly, and write the
> single JSON object it specifies to `<tag>_audit_verdicts.json`. You are not told who
> wrote any criterion; do not try to find out, and do not read any other file.

## 3. Arbiter (frontier model, provenance VISIBLE by design)

Runs only when `audit.py finalize` reports disputes.

> You are the ARBITER. Read `<tag>_arbiter_prompt.txt` and write the single JSON object it
> specifies to `<tag>_arbiter_raw.json`. You may see which track proposed what — that is
> intentional, you are adjudicating between them.

Then `python3 arbiter.py apply --raw <tag>_arbiter_raw.json` and re-run
`audit.py finalize`.

## Order of operations per round

```
stage1_slice -> harness build -> [4 sealed proposers | codex leg | glm leg]
  -> harness collect -> species -> audit build -> AUDITOR -> audit finalize
  -> arbiter build -> ARBITER -> arbiter apply -> audit finalize
  -> score_gemma_maps (sk3 GPU, offline batch) -> readout
```
