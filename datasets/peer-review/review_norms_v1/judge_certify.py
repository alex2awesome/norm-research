"""Judge-certification: does a neutral judge (GLM-5.2 and/or Sonnet) certify Gemma-4's
review-norm extractions at parity with Sonnet's? (user 2026-07-09).

Blinded pairwise: for each held-out review, present Gemma-4 extraction and the Sonnet
reference as A/B in randomized (hash-stable) order. Judge rates EACH on coverage / verbatim /
polarity -> certify Y/N, and picks better (A/B/tie). Parity = GLM certifies Gemma ~ Sonnet and
head-to-head ~even.

Stage 1 (local, OpenRouter): generate Gemma-4 extractions on held-out reviews with the
GEPA-best prompt. Stage 2 (--judge): run the judge over the paired file.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

D = os.path.dirname(os.path.abspath(__file__))
KEY = open(os.path.expanduser("~/.openrouter-api-key.txt")).read().strip()
EVAL_B = ["batch_000", "batch_006", "batch_017", "batch_018", "batch_019", "batch_020"]
PAIRS = f"{D}/judge_pairs.jsonl"


def or_call(model, prompt, max_tokens=1600, temp=0.0):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": temp,
                       "reasoning": {"enabled": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 method="POST", headers={"Authorization": f"Bearer {KEY}",
                                                         "Content-Type": "application/json"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=150) as r:
                return json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""
        except Exception:
            import time
            time.sleep(3 * (a + 1))
    return ""


def parse_arr(txt):
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def parse_obj(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def fmt(passages):
    return "\n".join(f"  {i+1}. [{p.get('polarity','?')}] ({p.get('aspect','?')}) "
                     f"\"{p.get('quote','')[:200]}\"" for i, p in enumerate(passages)) or "  (none)"


JUDGE = """You are auditing two systems that extract evaluative passages from an academic peer review. An extraction is a list of {{quote (should be a verbatim substring of the review), polarity, aspect-name}}. A GOOD extraction captures every distinct evaluative judgment (praise or criticism grounded in a criterion) the reviewer made, with exact quotes and correct polarity, without spurious or redundant entries.

REVIEW:
{review}

EXTRACTION A:
{a}

EXTRACTION B:
{b}

For EACH extraction independently, decide if it is CERTIFIABLE (a competent, usable extraction of this review's evaluative content — complete enough, quotes look verbatim, polarities right). Minor boundary/wording differences are fine; judge substance.

Output ONLY this JSON:
{{"A": {{"certify": true/false, "coverage": "high"/"med"/"low", "issues": "<short>"}},
  "B": {{"certify": true/false, "coverage": "high"/"med"/"low", "issues": "<short>"}},
  "better": "A"/"B"/"tie"}}"""


def build_pairs(n_per=12):
    best = json.load(open(f"{D}/gepa_gemma_state.json"))["best_prompt"]
    rows = []
    for b in EVAL_B:
        if not os.path.exists(f"{D}/outputs/{b}.jsonl"):
            continue
        inp = {r["review_id"]: r for r in json.load(open(f"{D}/inputs/{b}.json"))}
        refs = {json.loads(l)["review_id"]: json.loads(l)["passages"]
                for l in open(f"{D}/outputs/{b}.jsonl")}
        cnt = 0
        for rid, ref in refs.items():
            if rid.startswith("ANCH") or rid not in inp or cnt >= n_per:
                continue
            rows.append(dict(review_id=rid, review=inp[rid]["review_text"], sonnet=ref))
            cnt += 1

    def gen(row):
        out = or_call("google/gemma-4-31b-it", best.replace("__REVIEW__", row["review"][:8000]))
        row["gemma"] = parse_arr(out) or []
        return row
    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(gen, rows))
    with open(PAIRS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"built {len(rows)} pairs -> {PAIRS} "
          f"(gemma mean {sum(len(r['gemma']) for r in rows)/len(rows):.1f} vs "
          f"sonnet {sum(len(r['sonnet']) for r in rows)/len(rows):.1f} passages)")


def judge_one(row, judge_fn):
    # blinded, hash-stable A/B assignment
    swap = int(hashlib.md5(row["review_id"].encode()).hexdigest(), 16) % 2
    ga, gb = (row["gemma"], row["sonnet"]) if swap == 0 else (row["sonnet"], row["gemma"])
    who = {"A": "gemma", "B": "sonnet"} if swap == 0 else {"A": "sonnet", "B": "gemma"}
    txt = judge_fn(JUDGE.format(review=row["review"][:8000], a=fmt(ga), b=fmt(gb)))
    v = parse_obj(txt)
    if not v or "A" not in v or "B" not in v:
        return None
    better = v.get("better", "tie")
    return {"gemma_cert": bool(v[[k for k, x in who.items() if x == "gemma"][0]].get("certify")),
            "sonnet_cert": bool(v[[k for k, x in who.items() if x == "sonnet"][0]].get("certify")),
            "better": who.get(better, "tie") if better in ("A", "B") else "tie",
            "gemma_cov": v[[k for k, x in who.items() if x == "gemma"][0]].get("coverage"),
            "review_id": row["review_id"]}


def run_judge(name, judge_fn):
    rows = [json.loads(l) for l in open(PAIRS)]
    with cf.ThreadPoolExecutor(6) as ex:
        res = [r for r in ex.map(lambda x: judge_one(x, judge_fn), rows) if r]
    n = len(res)
    gc = sum(r["gemma_cert"] for r in res) / n
    sc = sum(r["sonnet_cert"] for r in res) / n
    from collections import Counter
    bw = Counter(r["better"] for r in res)
    cov = Counter(r["gemma_cov"] for r in res)
    print(f"\n=== JUDGE: {name}  (n={n}) ===")
    print(f"  Gemma-4 certify rate : {gc:.3f}")
    print(f"  Sonnet  certify rate : {sc:.3f}   (parity gap {gc-sc:+.3f})")
    print(f"  head-to-head better  : gemma {bw['gemma']} / sonnet {bw['sonnet']} / tie {bw['tie']}")
    print(f"  gemma coverage dist  : {dict(cov)}")
    json.dump({"judge": name, "n": n, "gemma_cert": gc, "sonnet_cert": sc,
               "better": dict(bw), "gemma_cov": dict(cov), "rows": res},
              open(f"{D}/judge_cert_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "build":
        build_pairs()
    elif mode == "sonnet":
        run_judge("sonnet", lambda p: subprocess.run(
            ["claude", "-p", p, "--model", "sonnet"], capture_output=True, text=True,
            timeout=200).stdout)
    elif mode == "glm":
        run_judge("glm52", lambda p: or_call("z-ai/glm-5.2", p, max_tokens=700))
