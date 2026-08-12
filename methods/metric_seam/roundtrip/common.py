"""Shared pieces of the code round-trip pipeline (2026-08-12).

Design (frozen): for every seam-survey metric with a code channel, the reconstruction run
(methods/metric_seam/reconstruction/run_recon.py) stored the decoder's prose reconstruction
m_hat of that channel plus the exact 40-item eval split. The round-trip experiment compiles
m_hat BACK to code (blind; N independent trials; optional feedback-optimization rounds) and
scores R_home = spearman(recompiled scores, original channel scores) on the SAME eval items.
The prompt arm's home round trip is the stored channel="judge" R. Eval items are FROZEN:
no compile trial or optimization round ever sees them.
"""
import json
import math
import pathlib
import statistics as st

REPO = pathlib.Path(__file__).resolve().parents[3]
PILOT = REPO / "outputs/metric_seam_pilot"
RECON = PILOT / "reconstruction"
WORK = PILOT / "roundtrip"           # all pipeline outputs live here
INSTR = pathlib.Path(__file__).parent / "RT_INSTRUCTIONS.md"
CODEX = ("/Users/spangher/.claude/plugins/cache/openai-codex/codex/1.0.5/"
         "scripts/codex-companion.mjs")
N_CHUNKS = 6
SEED = 3                              # job-shuffle seed (frozen; matches the 2026-08-12 packs)

PLANTED = [
    ("CAL1", "Score the document by its use of numeric digits. Award 10 if the document "
             "contains three or more digit characters (0-9), 5 if it contains one or two "
             "digits, and 0 if it contains none."),
    ("CAL2", "Grade on question usage: 10 if the document contains at least one question "
             "mark character, otherwise 0."),
    ("CAL3", "Score document length: 10 if the document is longer than 150 words, 5 if "
             "between 50 and 150 words, 0 if under 50 words."),
    ("CAL4", "Award 10 if the document contains at least one line that begins with the "
             "character '#', otherwise award 0."),
    ("CAL5", "Score by uppercase emphasis: count words written fully in capital letters "
             "with length >= 3; award 10 if five or more, 5 if one to four, 0 if none."),
    ("CAL6", "Grade 10 if the word 'the' appears more than ten times in the document, "
             "5 if it appears three to ten times, 0 otherwise."),
]


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def load_items(task):
    p = (PILOT / "v1/items_v1.json") if task == "press_releases" else (
        PILOT / "tasks" / task / "items.json")
    return {x["datapoint_id"]: x["ctext"] for x in json.load(open(p))}


def load_functions(pattern):
    """exec every matching chunk file in a restricted namespace; return {job_id: fn}."""
    import collections
    import re
    import string
    funcs = {}
    for f in sorted(WORK.glob(pattern)):
        ns = {"re": re, "math": math, "statistics": st, "string": string,
              "collections": collections, "__builtins__": __builtins__}
        try:
            exec(open(f).read(), ns)
        except Exception as e:
            print(f"LOAD FAIL {f.name}: {e}")
            continue
        for k, v in ns.items():
            if k.startswith("score__"):
                funcs[k[len("score__"):]] = v
    return funcs


def channel_scores_full(job):
    """The original code channel's scores on ALL its items (train pool = non-eval)."""
    import random
    task, aid, channel = job["task"], job["aspect"], job["channel"]
    flavor = channel.replace("code_", "")
    maps = []
    if task == "press_releases":
        for f in ("v1/code_scores_v1.json", "v2/code_scores_v2.json", "v3/code_scores_v3.json"):
            p = PILOT / f
            if p.exists():
                maps.append(json.load(open(p)))
    else:
        p = PILOT / "tasks" / task / "code_scores.json"
        if p.exists():
            maps.append(json.load(open(p)))
    col = {}
    for m in maps:
        col.update({d: v for d, v in m.get(f"{aid}_{flavor}", {}).items() if v is not None})
    det = json.load(open(RECON / "detail" / f"{task}__{aid}__{channel}.json"))
    eval_ids = set(det)
    return ({d: v for d, v in col.items() if d not in eval_ids},
            {d: det[d][1] for d in det if det[d][1] is not None})
