"""GEPA loop: can Gemma-4 match the Sonnet reference on review-norm extraction?
(user 2026-07-09: kill Sonnet conveyor; GEPA+Gemma is the house extraction stack).

Reference = 16 Sonnet-extracted batches (512 reviews). Train on 4 batches, eval on 3 held-out.
Score per review vs reference: passage-match F1 (normalized token-overlap), polarity accuracy
on matches, verbatim snap-rate as a hard gate. Reflection proposals via a single strong-model
call per round (claude -p), executor = google/gemma-4-31b-it @ temp 0.
"""
import concurrent.futures as cf
import glob
import json
import os
import re
import subprocess
import sys
import urllib.request

D = os.path.dirname(os.path.abspath(__file__))
KEY = open(os.path.expanduser("~/.openrouter-api-key.txt")).read().strip()
MODEL = "google/gemma-4-31b-it"

TRAIN_B = ["batch_002", "batch_003", "batch_005", "batch_008"]
EVAL_B = ["batch_000", "batch_006", "batch_017"]


def load(batches):
    rows = []
    for b in batches:
        inp = {r["review_id"]: r for r in json.load(open(f"{D}/inputs/{b}.json"))}
        for line in open(f"{D}/outputs/{b}.jsonl"):
            rec = json.loads(line)
            if rec["review_id"].startswith("ANCH"):
                continue
            if rec["review_id"] in inp:
                rows.append(dict(review=inp[rec["review_id"]]["review_text"],
                                 review_id=rec["review_id"], ref=rec["passages"]))
    return rows


def call_gemma(prompt, max_tokens=1600):
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "temperature": 0.0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 method="POST", headers={"Authorization": f"Bearer {KEY}",
                                                         "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                obj = json.loads(r.read().decode())
            return obj["choices"][0]["message"]["content"] or ""
        except Exception:
            import time
            time.sleep(3 * (attempt + 1))
    return ""


def parse_passages(txt):
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return None
    try:
        ps = json.loads(m.group(0))
        return [p for p in ps if isinstance(p, dict) and "quote" in p and "polarity" in p]
    except Exception:
        return None


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def score_review(review, ref, hyp):
    """returns (f1, pol_acc, snap, n_ref, n_hyp)"""
    if hyp is None:
        return 0.0, 0.0, 0.0, len(ref), 0
    snap_ok = sum(1 for p in hyp if p["quote"] in review)
    snap = snap_ok / max(1, len(hyp))
    used, tp, pol_ok = set(), 0, 0
    for h in hyp:
        ht = toks(h["quote"])
        best, bj = None, 0.0
        for i, r in enumerate(ref):
            if i in used:
                continue
            rt = toks(r["quote"])
            j = len(ht & rt) / max(1, len(ht | rt))
            if j > bj:
                bj, best = j, i
        if bj >= 0.35:
            used.add(best)
            tp += 1
            if h["polarity"] == ref[best]["polarity"]:
                pol_ok += 1
    prec = tp / max(1, len(hyp))
    rec = tp / max(1, len(ref))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return f1, (pol_ok / tp if tp else 0.0), snap, len(ref), len(hyp)


def evaluate(prompt_tpl, rows, conc=8):
    def one(row):
        out = call_gemma(prompt_tpl.replace("__REVIEW__", row["review"][:8000]))
        hyp = parse_passages(out)
        return row, hyp, score_review(row["review"], row["ref"], hyp)
    with cf.ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(one, rows))
    import statistics as st
    f1s = [s[0] for _, _, s in res]
    pols = [s[1] for _, _, s in res if s[1] > 0]
    snaps = [s[2] for _, _, s in res]
    parse_fail = sum(1 for _, h, _ in res if h is None)
    return dict(f1=st.mean(f1s), pol=st.mean(pols) if pols else 0.0,
                snap=st.mean(snaps), parse_fail=parse_fail, n=len(rows)), res


P0 = """Extract ALL evaluative passages from this academic peer review — every place the reviewer makes a judgment (praise or criticism) grounded in a quality criterion (novelty, clarity, experiments, soundness, reproducibility, significance, related work, presentation, motivation, etc.). Most reviews contain several.

Output ONLY a JSON array. Each element: {"quote": "<VERBATIM contiguous substring of the review, max 300 chars, trimmed to the evaluative core>", "polarity": "pos"|"neg"|"mixed", "aspect": "<2-6 word name of the criterion in the reviewer's own framing>"}

Rules: quotes must be exact substrings (verbatim, no paraphrase, no ellipsis). Output nothing but the JSON array.

REVIEW:
__REVIEW__

JSON:"""

REFLECT = """You are optimizing an extraction prompt for a smaller model (Gemma-4-31B).
Task: extract evaluative passages (verbatim quote + polarity + aspect) from peer reviews, matching a reference standard.

CURRENT PROMPT:
<<<PROMPT
{prompt}
PROMPT>>>

FAILURE EXAMPLES from the current prompt (reference passages it MISSED, spurious ones it ADDED, wrong polarity, non-verbatim quotes, or JSON parse failures):
{failures}

Propose an IMPROVED prompt. Keep the same JSON output contract and the __REVIEW__ placeholder. Focus on fixing the observed failure modes (coverage of all evaluative passages, exact verbatim quoting, polarity calibration, strict JSON). Output ONLY the new prompt text, nothing else."""


def reflect(prompt, res):
    fails = []
    for row, hyp, (f1, pol, snap, nr, nh) in sorted(res, key=lambda x: x[2][0])[:6]:
        missed = [r["quote"][:110] for r in row["ref"]][:4]
        got = [(h.get("quote", "")[:80], h.get("polarity")) for h in (hyp or [])][:4]
        fails.append(f"- review {row['review_id']}: f1={f1:.2f} snap={snap:.2f} "
                     f"parse={'FAIL' if hyp is None else 'ok'}\n  ref: {missed}\n  hyp: {got}")
    msg = REFLECT.format(prompt=prompt, failures="\n".join(fails))
    out = subprocess.run(["claude", "-p", msg, "--model", "sonnet"],
                         capture_output=True, text=True, timeout=300).stdout.strip()
    return out if "__REVIEW__" in out and len(out) > 200 else None


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    train, dev = load(TRAIN_B), load(EVAL_B)
    print(f"train {len(train)} dev {len(dev)} reviews")
    best_p, hist = P0, []
    stats, res = evaluate(P0, train[:40])
    print(f"r0 TRAIN {stats}")
    best_dev, _ = evaluate(P0, dev[:45])
    print(f"r0 DEV   {best_dev}")
    hist.append(("r0", best_dev))
    cur = P0
    for r in range(1, rounds + 1):
        newp = reflect(cur, res)
        if not newp:
            print(f"r{r}: reflection failed, stopping")
            break
        stats, res2 = evaluate(newp, train[:40])
        print(f"r{r} TRAIN {stats}")
        devs, _ = evaluate(newp, dev[:45])
        print(f"r{r} DEV   {devs}")
        hist.append((f"r{r}", devs))
        if devs["f1"] > best_dev["f1"]:
            best_dev, best_p = devs, newp
            cur, res = newp, res2
        json.dump({"best_dev": best_dev, "history": [(k, v) for k, v in hist],
                   "best_prompt": best_p},
                  open(f"{D}/gepa_gemma_state.json", "w"), indent=1)
    print(f"BEST DEV: {best_dev}")
    json.dump({"best_dev": best_dev, "history": [(k, v) for k, v in hist],
               "best_prompt": best_p}, open(f"{D}/gepa_gemma_state.json", "w"), indent=1)
