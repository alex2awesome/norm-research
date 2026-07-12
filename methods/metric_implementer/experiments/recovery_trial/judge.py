"""Weak-judge executor for the GEPA trial: Llama-3.3-70B via OpenRouter scores texts on a rubric.
Real API calls, logged raw. Concurrent. Output JSONL: {id, score, reason, raw}."""
import argparse, concurrent.futures as cf, json, os, re, sys, urllib.request

KEY = open(os.path.expanduser("~/.openrouter-api-key.txt")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"


def call(model, prompt, temp):
    body = json.dumps({"model": model, "temperature": temp, "max_tokens": 120,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=90))
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            err = str(e)[:80]
    return f"__ERR__ {err}"


def parse_score(txt):
    if txt is None:
        return None
    m = re.search(r'"score"\s*:\s*([0-9.]+)', txt)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            pass
    m = re.search(r'\b(0?\.\d+|0|1)\b', txt)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric-file", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-chars", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rubric = open(a.rubric_file).read().strip()
    items = [json.loads(l) for l in open(a.texts)]
    # NO .format over the rubric (rubric/LaTeX may contain literal { } -> KeyError). Plain concatenation.
    SUFFIX = ("\n\nScore the following math answer on the rubric above. A high score means it strongly "
              'satisfies the rubric. Output ONLY JSON: {"score": <one of 0.0,0.25,0.5,0.75,1.0>, '
              '"reason": "<=10 words"}.\n\nANSWER:\n')

    def work(it):
        try:
            raw = call(a.model, rubric + SUFFIX + str(it["text"])[:a.max_chars], a.temp)
        except Exception as e:
            raw = f"__ERR_WORK__ {e}"
        if raw is None:
            raw = "__ERR_NONE__"
        return {"id": it["id"], "score": parse_score(raw), "reason": raw[:160], "raw": raw}

    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(work, items))
    res.sort(key=lambda r: r["id"])
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    sc = [r["score"] for r in res if r["score"] is not None]
    import statistics as st
    hist = {v: sum(1 for s in sc if abs(s - v) < 1e-6) for v in (0.0, 0.25, 0.5, 0.75, 1.0)}
    print(f"scored {len(sc)}/{len(res)} | mean={st.mean(sc):.3f} std={st.pstdev(sc):.3f} | hist={hist}")
    print(f"  n_unparsed={sum(1 for r in res if r['score'] is None)} | wrote {a.out}")


if __name__ == "__main__":
    main()
