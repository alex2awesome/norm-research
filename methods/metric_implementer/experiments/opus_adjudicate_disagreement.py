#!/usr/bin/env python3
# pylint: disable=line-too-long
"""
================================================================================
OPUS ADJUDICATOR — agent brief for re-judging the Opus/GLM-5.2 DISAGREEMENT pairs
(peer-review rubric statements)
================================================================================

>>> WHY THIS EXISTS (read this first) <<<

Two judges already labeled 3000 peer-review rubric pairs "same/different":
  - GLM-5.2  (calls ~19% of pairs SAME — conservative)
  - Opus     (calls ~27% of pairs SAME — more liberal)
They agreed on 84% (binary). On the 484 they DISAGREED, we need to know who is
right, because every clustering method is being scored against one or both of them.

You (Opus) already produced one of these two label sets — BUT the original Opus run
was a batch agent pass, and spot-checking suggests some of its SAME calls at LOW
similarity are clearly wrong (e.g. "clinical trial should be registered" vs "use
'Father' for church fathers" labeled SAME). So this task is NOT "defend your old
labels." It is a FRESH, INDEPENDENT, CAREFUL re-judgment of ONLY the 484 disputed
pairs, done as if you have never seen them. The goal is to find out whether careful
Opus agrees with the earlier Opus labels, or with GLM-5.2, or charts its own course.

>>> EXACTLY WHAT TO DO (step by step) <<<

1. READ the input file (the 484 disagreement pairs):
       outputs/analyses/arbiter_disagreements.json
   It is a JSON array; each element has:
       pid        int    -- pair id (use as the key in your output)
       text_a     str    -- statement A   (judge THIS, and only this)
       text_b     str    -- statement B   (judge THIS, and only this)
       sim        float  -- embedding cosine similarity. *** IGNORE WHEN JUDGING. ***
       opus_label int    -- the EARLIER Opus verdict. *** DO NOT LOOK AT THIS. *** It
                           would anchor you. You are re-judging from scratch.
       glm52_label int   -- the GLM-5.2 verdict. *** DO NOT LOOK AT THIS EITHER. ***
       direction  str    -- which way the disagreement went. *** IGNORE. ***
       glm52, opus ...   -- ignore these internal fields.
   When you read a pair to judge it, look ONLY at text_a and text_b.

2. RESUME if partly done: if outputs/analyses/disagreement_opus_verdicts.jsonl
   already exists, read the pids already present and skip them. Append; never
   rewrite or duplicate a pid. Crash-safe.

3. JUDGE each not-yet-done pair using the EXACT prompt below (JUDGE_PROMPT) — the
   SAME prompt both arbiters used, so the only variable is careful judgment.
   Label scale:
       2 = SAME criterion. One statement is merely a REPHRASING of the other — the
           identical evaluative judgment on the same aspect, differently worded.
           e.g. "The literature review should be thorough and recent"
                vs "The work should comprehensively cover up-to-date relevant literature" -> 2
       1 = RELATED but genuinely DIFFERENT / borderline. Same general area, distinct
           sub-aspects, or genuinely unclear.
           e.g. "Citations should be complete" vs "References should follow journal style" -> 1
       0 = DIFFERENT criteria. Distinct aspects of the work, even if topically
           related or commonly co-assessed.
           e.g. "The statistical analysis should be appropriate"
                vs "The abstract should be an accurate summary" -> 0

4. *** READ THE ACTUAL WORDS. *** The single most important instruction in this
   brief. Do NOT skim. Do NOT pattern-match on a shared keyword. Many pairs share a
   topic word ("outcome", "data", "reference", "report") but evaluate entirely
   different aspects — shared vocabulary is NOT sameness. Before assigning 2, state
   (to yourself) the exact single judgment both statements make; if they make two
   different judgments, it is 0 or 1.

5. Work in modest batches (e.g. 20-40 pairs) but apply independent judgment to every
   pair. Do not let earlier pairs in a batch anchor later ones, and do not "go easy"
   or "go strict" to compensate for anything.

6. WRITE your verdicts to EXACTLY this path (downstream code reads here):
       outputs/analyses/disagreement_opus_verdicts.jsonl
   Format: JSONL, one object per line, EXACTLY:
       {"pid": <int>, "label": <2|1|0>}
   Append + flush after each batch so progress is never lost. One line per disputed
   pid. Do NOT include text, sim, or prior labels in the output.

>>> WHEN YOU ARE DONE <<<

Report: how many of the 484 you labeled, and the distribution (# of 2 / 1 / 0).
Then, ONLY if you are curious, you may compare to the prior labels — but your output
file must be the blind judgments you made BEFORE that comparison. Do not edit
disagreement_opus_verdicts.jsonl after comparing.

>>> WHAT DOWNSTREAM WILL COMPUTE (for reference; you do NOT need to) <<<

For each disputed pair we will compare your fresh label to (a) the earlier Opus
label and (b) the GLM-5.2 label. If careful-Opus matches GLM-5.2 far more than it
matches earlier-Opus, that is strong evidence the original Opus pass was sloppy and
GLM-5.2 is the trustworthy arbiter. If careful-Opus matches earlier-Opus, then
Opus's more-liberal standard is a genuine (if different) criterion and the truth is
genuinely ambiguous.

================================================================================
"""

from __future__ import annotations
import json, os, sys

IN_FILE = sys.argv[1] if len(sys.argv) > 1 else "outputs/analyses/arbiter_disagreements.json"
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "outputs/analyses/disagreement_opus_verdicts.jsonl"   # <<< agent writes HERE

# ---- EXACT judge prompt both arbiters used (reuse verbatim; only the judge differs) ----
# {listing} is the numbered list of pairs, formatted as:  [pid] A: "..." | B: "..."
JUDGE_PROMPT = (
    "You are an expert peer-review rubric judge. For each pair, decide if they express the "
    "SAME evaluation criterion (one is merely a rephrasing of the other — identical judgment in "
    "different words) or DIFFERENT criteria (they evaluate distinct aspects, even if topically related).\n\n"
    "Label: 2 = SAME criterion (rephrased), 1 = related but genuinely different / borderline, "
    "0 = DIFFERENT criteria.\n\n"
    "Judge each pair independently and conservatively (only 2 if truly the same judgment). "
    "Read the actual words — a shared topic word does not make two criteria the same. "
    'Return ONLY a JSON array, one entry per pair in order: [{"pid":0,"label":2},...]\n\n'
    "Pairs:\n{listing}"
)


def print_brief():
    print("=" * 80)
    print("ADJUDICATION SPEC SUMMARY")
    print("=" * 80)
    print(f"INPUT  (read):      {os.path.abspath(IN_FILE)}")
    print(f"OUTPUT (write):     {os.path.abspath(OUT_FILE)}")
    print(f"OUTPUT line format: {{\"pid\": <int>, \"label\": <2|1|0>}}")
    if os.path.exists(IN_FILE):
        n = len(json.load(open(IN_FILE)))
        done = sum(1 for _ in open(OUT_FILE)) if os.path.exists(OUT_FILE) else 0
        print(f"INPUT pairs:        {n}  (already done: {done})")
    else:
        print("INPUT pairs:        (file missing)")
    print("\nEXACT JUDGE PROMPT (use verbatim; same as both arbiters):\n")
    print(JUDGE_PROMPT.replace("{listing}", "[pid] A: \"<text_a>\" | B: \"<text_b>\"\n..."))
    print("\n>>> REMINDER: judge ONLY from text_a/text_b. Never look at opus_label/glm52_label/sim. <<<")


def build_listing(max_pairs=None):
    """Emit a plain listing the agent can paste into its judgment (texts only, no labels)."""
    d = json.load(open(IN_FILE))
    if max_pairs:
        d = d[:max_pairs]
    lines = []
    for x in d:
        lines.append(f"[{x['pid']}] A: \"{x['text_a']}\" | B: \"{x['text_b']}\"")
    return "\n".join(lines)


def main():
    print_brief()


if __name__ == "__main__":
    main()
