#!/usr/bin/env python3
"""Label the arbiter pair sample with GLM-5.2 (arbiter). Batches of 10, resumable, retry + per-pair
fallback. Writes outputs/analyses/arbiter_labels.jsonl: {pid, label} where 2=same, 1=borderline, 0=different.

CAVEAT: GLM-5.2 is same-family as the GLM-4.7 candidate -> may tilt toward GLM clusters; confirm with Opus.
"""
import json, os, sys, re, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from methods.metric_implementer.experiments.glm_cluster import glm_call

PAIRS = sys.argv[1] if len(sys.argv) > 1 else "outputs/analyses/arbiter_pairs.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/analyses/arbiter_labels.jsonl"
MODEL = os.environ.get("ARBITER_MODEL", "glm-5")   # glm-5 -> glm-5.2 on subscription endpoint
BATCH = 10


def judge_prompt(batch):
    lines = [f"[{p['pid']}] A: \"{p['text_a']}\" | B: \"{p['text_b']}\"" for p in batch]
    return ("You are an expert peer-review rubric judge. For each pair, decide if they express the "
            "SAME evaluation criterion (one is merely a rephrasing of the other — identical judgment in "
            "different words) or DIFFERENT criteria (they evaluate distinct aspects, even if topically related).\n\n"
            "Label: 2 = SAME criterion (rephrased), 1 = related but genuinely different / borderline, "
            "0 = DIFFERENT criteria.\n\n"
            "Judge each pair independently and conservatively (only 2 if truly the same judgment). "
            'Return ONLY a JSON array, one entry per pair in order: [{"pid":0,"label":2},...]\n\nPairs:\n'
            + "\n".join(lines))


def parse_labels(txt, pids):
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for e in arr:
        try:
            out[int(e["pid"])] = int(e["label"])
        except Exception:
            pass
    return out if out and all(p in out for p in pids) else (out or None)


def main():
    pairs = [json.loads(l) for l in open(PAIRS)]
    done = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                o = json.loads(l); done[o["pid"]] = o["label"]
            except Exception:
                pass
    todo = [p for p in pairs if p["pid"] not in done]
    print(f"{len(done)} done, {len(todo)} to label ({len(pairs)} total); model={MODEL}", flush=True)
    fout = open(OUT, "a")
    t0 = time.time()
    for bi in range(0, len(todo), BATCH):
        batch = todo[bi:bi + BATCH]
        pids = [p["pid"] for p in batch]
        labels = None
        for attempt in range(3):
            try:
                txt = glm_call(MODEL, judge_prompt(batch), max_tokens=700, temp=0.0 if attempt == 0 else 0.3)
                labels = parse_labels(txt, pids)
                if labels:
                    break
            except Exception as e:
                print(f"  batch@{bi} attempt{attempt} err: {e}", flush=True)
        if not labels:                                   # per-pair fallback
            labels = {}
            for p in batch:
                try:
                    t = glm_call(MODEL, judge_prompt([p]), max_tokens=60, temp=0.0)
                    mm = re.search(r"\{.*\}", t, re.S)
                    labels[p["pid"]] = (json.loads(mm.group(0))["label"] if mm else 1)
                except Exception:
                    labels[p["pid"]] = 1
        for p in batch:
            fout.write(json.dumps({"pid": p["pid"], "label": labels.get(p["pid"], 1)}) + "\n")
            done[p["pid"]] = labels.get(p["pid"], 1)
        fout.flush()
        if (bi // BATCH + 1) % 5 == 0:
            print(f"  {len(done)}/{len(pairs)} labeled [{time.time()-t0:.0f}s]", flush=True)
    fout.close()
    from collections import Counter
    c = Counter(done.values())
    print(f"DONE {len(done)} labels -> {OUT}  dist: {dict(c)}", flush=True)


if __name__ == "__main__":
    main()
