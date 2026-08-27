"""Reconstruction-objective (R) redo of the seam survey — GLM-5.2 recoverer, runs on sk3.

The theory-native objective (PO-stack, canonical no-anchor recovery): for each measured
metric CHANNEL — the Gemma judge channel, the best description-compiled code program, and
(press releases) the evolved hybrid — ask whether a strong recoverer can reconstruct the
scoring rule from the channel's own (document, score) pattern, then re-execute the
recovered rule and measure held-out rank fidelity to the channel:

    R = spearman( execute(m_hat, x) , channel_score(x) )   on 40 held-out items

Channels per aspect:
  judge          2-pass Gemma mean            (recovery is BLINDED: no metric name shown)
  judge_truedesc no recovery — execute the TRUE aspect description with GLM; this is the
                 GLM-as-executor reference that separates "recovery failed" from
                 "GLM cannot reproduce this channel at all"
  judge_null     recovery on PERMUTED scores (identifiability control, 5 aspects/task)
  code           best flavor by full-sample spearman vs judge
  hybrid         programs_v2 / v1 hybrid scores (press_releases only)

Priority order (survives quota exhaustion): PR hybrid aspects -> PR rest -> math ->
patents -> code_review(comments). Fully resumable: (task, aspect, channel) keys already in
recon_results.jsonl are skipped. Per-job exec detail saved under reconstruction/detail/.

Usage (sk3):  HOME=/lfs/skampere3/0/alexspan nohup python3.11 run_recon.py >> recon.log 2>&1 &
"""
import json, math, pathlib, random, re, statistics as st, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

R = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
OUT = R / "outputs/metric_seam_pilot"
REC = OUT / "reconstruction"
DETAIL = REC / "detail"
RESULTS = REC / "recon_results.jsonl"
RUNS = R / "runs/validity_full/v2"

MODEL = "glm-5.2"
URL = "https://api.z.ai/api/anthropic/v1/messages"
N_EVAL, N_EX, WORKERS = 40, 24, 3   # 3 = gentler on z.ai (was 6; 529 overloads at 6)
EX_HEAD, EX_TAIL = 900, 300          # exemplar excerpt in the recovery prompt
DOC_HEAD, DOC_TAIL = 3000, 800       # document excerpt in the execution prompt


# ------------------------------------------------------------------ GLM client
def _key():
    import os
    env = os.environ.get("ZAI_KEY_FILE")
    if env:
        p = pathlib.Path(os.path.expanduser(env))
        return p.read_text().strip() if p.exists() else env.strip()
    for cand in ("~/.z-ai-api-key-alexander-spangher.txt", "~/.z-ai-api-key-spangher.txt",
                 "~/.z-ai-api-key.txt"):
        p = pathlib.Path(cand).expanduser()
        if p.exists():
            return p.read_text().strip()
    raise FileNotFoundError("no z.ai key")


KEY = _key()


def glm(prompt, max_tokens=1024, temperature=0.0, retries=8):
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens,
                       "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                URL, data=body, method="POST",
                headers={"x-api-key": KEY, "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                obj = json.loads(r.read().decode())
            if obj.get("type") == "error" or "error" in obj:
                raise RuntimeError(str(obj.get("error") or obj)[:200])
            content = obj.get("content") or []
            return "".join(c.get("text", "") for c in content if isinstance(c, dict))
        except Exception as e:
            if attempt == retries - 1:
                raise
            # 529/503/500 = z.ai upstream overload (transient): exponential backoff
            # capped at 60s so a saturated API is ridden out, not abandoned mid-job.
            msg = str(e)
            server_busy = any(c in msg for c in ("529", "503", "500", "502",
                                                 "Overloaded", "overload"))
            base = min(60.0, 4.0 * (2 ** attempt)) if server_busy else 3.0 * (attempt + 1)
            time.sleep(base + random.random() * 2)
    return ""


# ------------------------------------------------------------------ stats
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


# ------------------------------------------------------------------ data loading
def load_judge(path):
    p1, p2 = {}, {}
    for line in open(path):
        r = json.loads(line)
        if r.get("aspect_id") == "scope" or not isinstance(r.get("score"), int):
            continue
        d = p1 if r["channel"] == "pass1" else p2 if r["channel"] == "pass2" else None
        if d is not None:
            d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    out = {}
    for aid in set(p1) | set(p2):
        for dp in set(p1.get(aid, {})) | set(p2.get(aid, {})):
            vals = [m[aid][dp] for m in (p1, p2) if dp in m.get(aid, {})]
            out.setdefault(aid, {})[dp] = sum(vals) / len(vals) / 10.0
    return out


def load_items(path):
    return {x["datapoint_id"]: x["ctext"] for x in json.load(open(path))}


def best_code(aid, judge_a, code_maps):
    best = (None, float("-inf"), None)
    for cm in code_maps:
        for fl in ("v0_keyword", "v1_structure", "v2_holistic"):
            col = cm.get(f"{aid}_{fl}")
            if not col:
                continue
            sel = [d for d in judge_a if col.get(d) is not None]
            if len(sel) < 60:
                continue
            r = spearman([col[d] for d in sel], [judge_a[d] for d in sel])
            if r == r and r > best[1]:
                best = (fl, r, {d: col[d] for d in sel})
    return best[0], best[2]


def jload(path):
    p = pathlib.Path(path)
    return json.load(open(p)) if p.exists() else {}


# ------------------------------------------------------------------ prompts
RECOVER = """You are a measurement scientist reverse-engineering an unknown document-scoring rule.

Below are {n} documents (excerpted; '[...]' = elided middle) together with the score an \
unknown evaluation rule assigned each one (0-10 scale, higher = better satisfies the rule).

{exemplars}

Study which document properties the scores track. Then state your single best hypothesis \
of the scoring rule as a precise, self-contained grading instruction that another grader \
could apply to a brand-new document with no other context. Do not reference these examples. \
Reply with ONLY the instruction, at most 120 words."""

EXECUTE = """You are grading ONE document against ONE criterion.

Criterion: {rule}

Document ('[...]' = elided middle):
<document>
{text}
</document>

Grade 0-10 (0 = criterion badly violated or absent, 10 = fully exemplified). \
Reply with exactly one line:
SCORE: <integer 0-10>"""

_SCORE = re.compile(r"SCORE:\s*(10|[0-9])\b")
_ANYNUM = re.compile(r"\b(10|[0-9])\b")


def parse_score(txt):
    m = list(_SCORE.finditer(txt or ""))
    if m:
        return int(m[-1].group(1))
    m = list(_ANYNUM.finditer((txt or "").strip()[-40:]))
    return int(m[-1].group(1)) if m else None


def excerpt(t, head, tail):
    return t if len(t) <= head + tail + 200 else t[:head] + "\n[...]\n" + t[-tail:]


def scale_scores(s):
    lo, hi = min(s.values()), max(s.values())
    if hi - lo < 1e-9:
        return None
    return {d: round(10 * (v - lo) / (hi - lo), 1) for d, v in s.items()}


# ------------------------------------------------------------------ one job
def run_job(task, aid, name, channel, scores, items, rule=None, seed_extra=""):
    ids = sorted(d for d in scores if d in items and scores[d] is not None)
    if len(ids) < 80:
        return {"skip": f"n={len(ids)}"}
    rng_eval = random.Random(f"{task}:{aid}")            # SAME eval split across channels
    eval_ids = sorted(rng_eval.sample(ids, N_EVAL))
    pool = [d for d in ids if d not in set(eval_ids)]
    pres = scale_scores({d: scores[d] for d in pool})
    if pres is None:
        return {"skip": "constant-scores"}

    if rule is None:                                      # recovery (blinded)
        ordered = sorted(pool, key=lambda d: scores[d])
        k = len(ordered)
        picks = (ordered[:N_EX // 3] + ordered[k // 2 - N_EX // 6: k // 2 + N_EX // 6]
                 + ordered[-N_EX // 3:])
        rng_sh = random.Random(f"shuffle:{task}:{aid}:{channel}{seed_extra}")
        rng_sh.shuffle(picks)
        ex = "\n\n".join(
            f"=== DOCUMENT {i+1} (score: {pres[d]}) ===\n{excerpt(items[d], EX_HEAD, EX_TAIL)}"
            for i, d in enumerate(picks))
        rule = glm(RECOVER.format(n=len(picks), exemplars=ex),
                   max_tokens=2000, temperature=0.4).strip()
        if not rule:
            return {"skip": "empty-recovery"}

    def one(d):
        txt = glm(EXECUTE.format(rule=rule, text=excerpt(items[d], DOC_HEAD, DOC_TAIL)))
        s = parse_score(txt)
        if s is None:
            s = parse_score(glm(EXECUTE.format(
                rule=rule, text=excerpt(items[d], DOC_HEAD, DOC_TAIL)), temperature=0.2))
        return d, s

    with ThreadPoolExecutor(max_workers=WORKERS) as tp:
        got = dict(tp.map(one, eval_ids))
    pairs = [(got[d], scores[d]) for d in eval_ids if got.get(d) is not None]
    if len(pairs) < 25:
        return {"skip": f"parsed={len(pairs)}", "m_hat": rule}
    gs = [p[0] for p in pairs]
    r = spearman(gs, [p[1] for p in pairs])
    DETAIL.mkdir(parents=True, exist_ok=True)
    json.dump({d: [got.get(d), scores[d]] for d in eval_ids},
              open(DETAIL / f"{task}__{aid}__{channel}.json", "w"))
    return {"R": round(r, 3), "n_eval": len(pairs), "m_hat": rule,
            "glm_sd": round(st.pstdev(gs), 2), "glm_distinct": len(set(gs)),
            "parse_fail": N_EVAL - len(pairs)}


# ------------------------------------------------------------------ main
def main():
    REC.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        for line in open(RESULTS):
            r = json.loads(line)
            if "error" not in r:   # errored channels are RETRIED on resume, not skipped
                done.add((r["task"], r["aspect"], r["channel"]))

    names = {}
    for task in ("press_releases", "math", "patents", "code_review"):
        p = RUNS / task / "aspects.json"
        if p.exists():
            names[task] = {a["aspect_id"]: a for a in json.load(open(p))}

    # ---- channel inventory per task ------------------------------------------------
    jobs = []   # (priority, task, aid, channel, scores, items, rule)

    # press releases (waves v1+v2+v3 share the v1 canonical items)
    pr_items = load_items(OUT / "v1/items_v1.json")
    pr_judge = {}
    for f in ("v1/results_v1.jsonl", "v2/results_v2.jsonl", "v3/results_v3.jsonl"):
        if (OUT / f).exists():
            pr_judge.update(load_judge(OUT / f))
    pr_code_maps = [jload(OUT / f) for f in
                    ("v1/code_scores_v1.json", "v2/code_scores_v2.json",
                     "v3/code_scores_v3.json")]
    hybrids = {}
    for p in list((OUT / "v2").glob("hybrid_scores_*_h0.json")) + \
             list((OUT / "v1").glob("hybrid_scores_*_h0.json")):
        aid = p.stem.replace("hybrid_scores_", "").replace("_h0", "")
        hybrids.setdefault(aid, jload(p))

    def add_task(task, judge, items, code_maps, hyb, prio):
        aids = sorted(judge, key=lambda a: (a not in hyb, a))
        for i, aid in enumerate(aids):
            a = names.get(task, {}).get(aid, {})
            nm, desc = a.get("name", ""), a.get("description", "")
            p = prio - (0.5 if aid in hyb else 0)
            jobs.append((p, task, aid, nm, "judge", judge[aid], items, None))
            if desc:
                jobs.append((p, task, aid, nm, "judge_truedesc", judge[aid], items,
                             f"{nm}: {desc}"))
            fl, cs = best_code(aid, judge[aid], code_maps)
            if cs:
                jobs.append((p, task, aid, nm, f"code_{fl}", cs, items, None))
            if aid in hyb and hyb[aid]:
                jobs.append((p, task, aid, nm, "hybrid", hyb[aid], items, None))
            if i < 5:   # identifiability null control on the judge channel
                perm = list(judge[aid].values())
                random.Random(f"null:{task}:{aid}").shuffle(perm)
                jobs.append((p + 0.25, task, aid, nm, "judge_null",
                             dict(zip(sorted(judge[aid]), perm)), items, None))

    add_task("press_releases", pr_judge, pr_items, pr_code_maps, hybrids, 1)
    for prio, task in ((2, "math"), (3, "patents"), (4, "code_review")):
        base = OUT / "tasks" / task
        if not (base / "results.jsonl").exists():
            continue
        add_task(task, load_judge(base / "results.jsonl"), load_items(base / "items.json"),
                 [jload(base / "code_scores.json")], {}, prio)

    jobs.sort(key=lambda j: j[0])
    print(f"{len(jobs)} channel jobs total, {len(done)} already done", flush=True)

    for prio, task, aid, nm, channel, scores, items, rule in jobs:
        if (task, aid, channel) in done:
            continue
        t0 = time.time()
        try:
            row = run_job(task, aid, nm, channel, scores, items, rule=rule)
        except Exception as e:
            row = {"error": str(e)[:300]}
        row.update({"task": task, "aspect": aid, "name": nm, "channel": channel})
        with open(RESULTS, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{task}/{aid}/{channel}: "
              f"{row.get('R', row.get('skip', row.get('error', '?')))} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if "error" in row and ("quota" in row["error"].lower()
                               or "429" in row["error"]):
            print("quota exhausted — stopping (resume later)", flush=True)
            sys.exit(2)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
