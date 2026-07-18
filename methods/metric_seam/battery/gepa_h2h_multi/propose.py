"""GEPA-H2H reflective proposer (seam note Sec 9, arm G) -- GLM-5.2 via zai_anthropic.

Reads state.json's round-<round> history entry per criterion (rho_dev trend + the 10 worst
rank-residual dev items, produced by ingest_round.py) and asks GLM-5.2 ONCE per criterion to
rewrite the scoring-prompt BODY. Never shows GLM a raw judge score -- only aggregate rho
history and "scored too HIGH/LOW relative to peers" rank language, per the reconstruction-only
constraint (own-verdict reconstruction, not label-fitting).

BE SPARING: exactly one API call per pending criterion (a single automatic retry fires only
if GLM's reply fails to parse / drops the <<<DOCUMENT>>> marker -- rare, defensive, not part of
the nominal budget). With 12 criteria x <=4 rounds that's <=48 calls total, worst case.

Usage:
  python3 propose.py <round> --dry-run   # print the exact feedback prompts, no API call, no
                                          # state.json write -- use this first every round.
  python3 propose.py <round>             # real GLM-5.2 calls; advances state.json's
                                          # round/prompt for every criterion it succeeds on.

KEY_PATH: see common.KEY_PATHS (checked in order; first that exists on disk wins). Fill/
reorder there if your local z.ai key file lives somewhere else. No key is read in --dry-run.
"""
import json, os, re, sys, urllib.request
from pathlib import Path

from common import CRITERIA, DOC_MARKER, GLM_MODEL, KEY_PATHS, TASK_DOMAIN, ZAI_URL
from common import crit_key, load_state, save_state

N_WORST_SHOWN = 10
MAX_TOKENS = 2000
TEMPERATURE = 0.7

SYSTEM_T = """You are a prompt-optimization engine (GEPA) for a single-call LLM document-\
scoring prompt. Domain: {domain}. Criterion being scored: {criterion_name} -- \
{criterion_description}

You revise the prompt's scoring guidance to fix concrete RELATIVE ranking errors -- \
documents that were scored too high or too low relative to their peers -- while keeping \
what already works. You do NOT see the underlying judge's actual scores, only which \
documents were mis-ranked relative to each other.

Respond with ONLY a JSON object: {{"prompt": "<revised prompt body text>"}}.
The literal token {marker} MUST appear EXACTLY ONCE in your revised text -- it marks where \
the document gets inserted. Do not remove, rename, or duplicate it.
Do NOT add any reply-format / output-format instructions (e.g. do not mention "SCORE:" or \
any 0-10 reply format) -- the harness appends that separately every round; focus only on \
the scoring guidance itself."""

USER_T = """CURRENT PROMPT (the model sees this with {marker} replaced by one document; a \
fixed reply-format instruction is appended separately and is not shown here):

{current_prompt}

DEV-SET ROUND HISTORY (Spearman rho vs this criterion's own judge verdict, TRAIN dev set, \
same {n_dev} items every round -- never test data):
{history_lines}

RANK-RESIDUAL FEEDBACK for round {round} (the {n_worst} dev documents whose ranking, \
relative to their peers, most disagreed with the judge's ranking -- you are NOT shown the \
actual judge scores, only that these specific documents were mis-ranked):
{worst_block}

Rewrite the scoring guidance so documents like these get ranked correctly relative to their \
peers next round. Keep the {marker} marker exactly once. Output ONLY the JSON object \
described in the system prompt."""


def read_key():
    for p in KEY_PATHS:
        pp = Path(os.path.expanduser(p))
        if pp.exists():
            return pp.read_text().strip()
    raise FileNotFoundError(f"no z.ai key found in {KEY_PATHS} -- fill in common.KEY_PATHS")


def glm_call(prompt, system, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, retries=3):
    key = read_key()
    body = {"model": GLM_MODEL, "max_tokens": max_tokens, "temperature": temperature,
            "system": system, "messages": [{"role": "user", "content": prompt}]}
    last = ""
    for attempt in range(retries):
        req = urllib.request.Request(
            ZAI_URL, data=json.dumps(body).encode(), method="POST",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=150) as r:
                obj = json.loads(r.read().decode())
            if obj.get("type") == "error" or "error" in obj:
                raise RuntimeError(str(obj.get("error") or obj))
            c = obj.get("content") or []
            last = c[0]["text"] if c and isinstance(c[0], dict) else ""
            return last
        except Exception as e:
            print(f"  glm retry {attempt}: {str(e)[:150]}", flush=True)
            import time; time.sleep(2.0 * (attempt + 1))
    return last


def parse_prompt_json(raw):
    """-> revised body text, or None if unparseable / missing the DOC_MARKER."""
    s = (raw or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    i = s.find("{")
    if i < 0:
        return None
    try:
        obj = json.loads(s[i:])
    except Exception:
        return None
    body = obj.get("prompt") if isinstance(obj, dict) else None
    if not isinstance(body, str) or DOC_MARKER not in body:
        return None
    return body.strip()


def build_feedback(c, rd):
    entry = next((h for h in c["history"] if h["round"] == rd), None)
    if entry is None:
        return None
    history_lines = "\n".join(f"round {h['round']}: rho={h['rho_dev']} (n={h['n']})"
                              for h in c["history"])
    if entry.get("rho_dev") is None:
        history_lines += ("\nWARNING: the round-{} prompt produced the SAME score for "
                          "(nearly) all documents — zero variance, rho undefined. The rubric "
                          "collapsed to a constant verdict; the revision MUST spread documents "
                          "across the 0-10 range (treat it as a graded quality, not a "
                          "pass/fail gate).".format(rd))
    worst = entry["worst_items"][:N_WORST_SHOWN]
    worst_block = "\n".join(
        f'- [{w["direction"]}] "{w["snippet"]}"' for w in worst) or "(none)"
    system = SYSTEM_T.format(domain=TASK_DOMAIN[c["task"]], criterion_name=c["criterion_name"],
                             criterion_description=c["criterion_description"], marker=DOC_MARKER)
    user = USER_T.format(marker=DOC_MARKER, current_prompt=c["prompt"],
                         n_dev=len(c["dev_ids"]), history_lines=history_lines, round=rd,
                         n_worst=len(worst), worst_block=worst_block)
    return system, user


def main():
    if len(sys.argv) < 2:
        print("usage: python3 propose.py <round> [--dry-run]"); sys.exit(1)
    rd = int(sys.argv[1])
    dry = "--dry-run" in sys.argv[2:]
    state = load_state()

    n_calls = 0
    for task, aid in CRITERIA:
        key = crit_key(task, aid)
        c = state["criteria"][key]
        if c["round"] != rd:
            print(f"{key}: SKIP (state round={c['round']} != {rd})")
            continue
        if c["proposed_through"] >= rd:
            print(f"{key}: SKIP (already proposed through round {rd})")
            continue
        fb = build_feedback(c, rd)
        if fb is None:
            print(f"{key}: SKIP (no history entry for round {rd} -- run ingest_round.py "
                  f"{rd} first)")
            continue
        system, user = fb
        if dry:
            print(f"\n===== {key} (round {rd}) DRY-RUN feedback =====")
            print("--- system ---"); print(system)
            print("--- user ---"); print(user)
            continue

        raw = glm_call(user, system)
        n_calls += 1
        new_body = parse_prompt_json(raw)
        if new_body is None:
            print(f"{key}: PROPOSE_FAIL (unparseable / missing marker); keeping current "
                  f"prompt unchanged for round {rd + 1}. raw head: {raw[:200]!r}")
            new_body = c["prompt"]
        else:
            print(f"{key}: proposed new prompt for round {rd + 1} "
                  f"({len(new_body)} chars, was {len(c['prompt'])})")
        c["prompt"] = new_body
        c["round"] = rd + 1
        c["proposed_through"] = rd
        save_state(state)  # incremental save: safe to resume after a mid-loop failure

    if dry:
        print(f"\n(dry-run: 0 API calls made, state.json untouched)")
    else:
        print(f"\n-> {n_calls} GLM-5.2 call(s) this invocation; state.json updated")


if __name__ == "__main__":
    main()
