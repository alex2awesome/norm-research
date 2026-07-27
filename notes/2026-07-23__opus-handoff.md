# OPUS HANDOFF — metric-lexicon paper, closing cycles (2026-07-23)

ROLE: keep the machines moving, execute pre-staged steps, record results. Do NOT design
new experiments, refreeze/modify preregistrations, reinterpret nulls, or pool model
families. All analysis code is ALREADY WRITTEN — run it, don't rewrite it.
SP = /private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research/6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad
REPO = /Users/spangher/Projects/stanford-research/norm-research  (run python from here)

## STATE (everything done unless listed below)
Done: PREREG-3 five families (GLM/GPT/Claude/Gemma4 collapse+news-null; Llama-8B reversed
= emergence finding); PREREG-10 register table + LLM=official-class; PREREG-11 supported
3/3 capable families (GPT/Claude/Gemma) + Llama-8B noise-floor; lay corpus final;
community-rule class; construct matching; all ledgered through entry 2026-07-23g.

## PENDING QUEUE (in order)

1. **GLM scoring lane finishes** (bg task bxry9e287 = p11_glm_runner.py; watcher b9gcbii2a
   announces "ALL THREE P11 LANES COMPLETE").
   - Check: `ls $SP/p11_glm_out_*.jsonl | wc -l` → need 140.
   - If the runner died short of 140: relaunch (resume-safe):
     `cd REPO && python3 $SP/p11_glm_runner.py` (background, then wait).
   - When 140: `cd REPO && python3 -m methods.codability.lexicon.prereg11_analysis`
     (re-runs gpt56+claude identically — same frozen code/seed, numbers must match the
     ledgered ones; the NEW number is the glm row). Append the glm row to ledger entry
     format of 2026-07-23g and note "5-family scoring table complete".

2. **Decode Claude-name register row**: `python3 $SP/p10c_decode.py`
   (4 chunks already judged; script gates anchors >=8/10 and updates
   outputs/lexicon/prereg10_results_20260723.json). Ledger one line: llm_claude
   formality/latinate vs the other classes (expectation from other LLM rows: near or above
   official 4.24/.743 — report whatever it says, no interpretation beyond the comparison).

3. **PREREG-10 companions (descriptive)**: `python3 $SP/p10_companions.py`
   Report (a) subscriber-gradient rho and (b) reddit-vs-non split AS DESCRIPTIVE ONLY in
   the ledger (registry marks these non-confirmatory).

4. **OPTIONAL, only if user asks**: register-judge the gemma4/llama8b names to extend the
   class table to 8 rows. Procedure: mirror what was done for Claude names — build chunks
   from $SP/openweights/{gemma4,llama8b}_naming_out.jsonl exactly as $SP/p10c chunks were
   built (see the builder block in the session transcript, or adapt p10c_key.json format:
   160 terms/chunk + 10 etymology anchors from the standard list), then launch one Sonnet
   general-purpose agent per chunk with the prompt: "Follow the instructions in
   $SP/p10_judge_instructions.txt exactly. INPUT = <chunk> OUTPUT = <out>", then decode
   with a copy of p10c_decode.py pointed at the new key/out files.

## STANDING RULES (do not violate)
- Families reported separately, never pooled. One analysis run per frozen test — the runs
  above are the sanctioned ones.
- Anchor gates as coded in the scripts (never lower a threshold to pass a chunk).
- Append+dedup, never delete data; quarantine, don't overwrite.
- Kill by specific PID only; kill wrapper shells before pythons.
- sk3: nothing further needed; GPUs 0/1 jobs are DONE (verify no ow_runner PIDs remain:
  `ssh sk3 "pgrep -f ow_runner"` — if any remain after outputs are complete, kill those
  specific PIDs).
- Every result: one ledger entry in notes/2026-07-06__hierarchy-reconstruction-ledger.md
  (entry ids continue 2026-07-23h, i, ...) + update memory hook in
  project_metric_lexicon_paper_plan.md ONLY at natural consolidation points.
- Spell out experiment names in every user-facing message (standing feedback rule).

## AFTER THE QUEUE: drafting support only
When 1-3 are done, the experimental surface is 100% complete. The next work is W8 drafting
(user-led). Do not start new collection. If the user asks "what's left": the answer is
"nothing experimental; GLM row done/pending, drafting is next; optional items: 8-row
register table, W5d reconstruction cross-coding audit (unstarted, read-only, needs design
sign-off)."
