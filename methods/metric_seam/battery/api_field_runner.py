"""Local API field extractor — OpenRouter (openai format) / z.ai (anthropic format).

Same row/resume semantics as the sk3 scorers: reads {channel, aspect_id, datapoint_id,
prompt} jsonl, appends {.., raw, score} rows, resumes by (channel, aspect_id,
datapoint_id). Deterministic (temperature 0), bounded concurrency, exponential backoff
on 429/5xx/transport errors. Never distorts sampling on retry.

Usage:
  python3 api_field_runner.py --backend zai_anthropic --model glm-4.7 \
      --prompts probe_prompts_pr.jsonl --out probe_results_glm47.jsonl [--limit N]
"""
import argparse, json, os, pathlib, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from urllib import request as ur, error as ue

BACKENDS = {
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions",
                   "keys": ["~/.openrouter-api-key.txt"], "format": "openai"},
    "zai_anthropic": {"url": "https://api.z.ai/api/anthropic/v1/messages",
                      "keys": ["~/.z-ai-api-key.txt",
                               "~/.z-ai-api-key-alexander-spangher.txt",
                               "~/.z-ai-api-key-spangher.txt"], "format": "anthropic"},
}


def read_key(backend, key_file=None):
    cands = [key_file] if key_file else BACKENDS[backend]["keys"]
    for cand in cands:
        p = pathlib.Path(os.path.expanduser(cand))
        if p.exists():
            return p.read_text().strip()
    raise FileNotFoundError(f"no key for {backend} (tried {cands})")


def parse_score(raw):
    m = re.search(r"SCORE:\s*(NA|\d+)", raw, re.IGNORECASE)
    if not m:
        return None
    tok = m.group(1).upper()
    if tok == "NA":
        return "NA"
    v = int(tok)
    return v if 0 <= v <= 10 else None


def key(r):
    return (r.get("channel", ""), r["aspect_id"], r["datapoint_id"])


def call(backend, model, api_key, prompt, max_tokens=60, tries=12):
    cfg = BACKENDS[backend]
    if cfg["format"] == "openai":
        body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        hdrs = {"Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"}
    else:
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}]}
        hdrs = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"}
    data = json.dumps(body).encode()
    delay = 2.0
    for attempt in range(tries):
        try:
            req = ur.Request(cfg["url"], data=data, headers=hdrs)
            with ur.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read())
            if cfg["format"] == "openai":
                if "choices" not in out:
                    raise RuntimeError(f"no choices: {str(out)[:200]}")
                txt = (out["choices"][0]["message"]["content"] or "").strip()
            else:
                parts = out.get("content") or []
                txt = "".join(p.get("text", "") for p in parts
                              if p.get("type") in (None, "text")).strip()
            # GLM's characteristic failure is a 200 with EMPTY content, not an HTTP error.
            # Treat empty as retryable rather than returning silent garbage.
            if not txt:
                raise RuntimeError("empty completion (retryable)")
            return txt
        except ue.HTTPError as e:
            if e.code in (408, 429, 500, 502, 503, 504, 524, 529) and attempt < tries - 1:
                time.sleep(delay + random.uniform(0, delay / 2))  # jitter: desync workers
                delay = min(delay * 2, 180)
                continue
            raise
        except Exception:
            if attempt < tries - 1:
                time.sleep(delay + random.uniform(0, delay / 2))
                delay = min(delay * 2, 180)
                continue
            raise
    raise RuntimeError("unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only first N todo (smoke)")
    ap.add_argument("--key-file", default=None, help="explicit key path (toggle quota)")
    args = ap.parse_args()

    api_key = read_key(args.backend, args.key_file)
    rows = [json.loads(l) for l in open(args.prompts)]
    done = set()
    if os.path.exists(args.out):
        done = {key(json.loads(l)) for l in open(args.out)}
        print(f"resume: {len(done)} already scored", flush=True)
    todo = [r for r in rows if key(r) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} prompts -> {args.model} via {args.backend}", flush=True)

    lock = threading.Lock()
    fout = open(args.out, "a")
    nerr = 0
    ndone = 0

    def work(r):
        nonlocal nerr, ndone
        try:
            raw = call(args.backend, args.model, api_key, r["prompt"])
        except Exception as e:
            with lock:
                nerr += 1
                print(f"ERR {r['aspect_id']}/{r['datapoint_id']}: "
                      f"{str(e)[:120]}", flush=True)
            return
        with lock:
            fout.write(json.dumps({"channel": r.get("channel", ""),
                                   "aspect_id": r["aspect_id"],
                                   "datapoint_id": r["datapoint_id"],
                                   "raw": raw, "score": parse_score(raw)}) + "\n")
            ndone += 1
            if ndone % 100 == 0:
                fout.flush()
                print(f"{ndone}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, todo))
    fout.flush()
    fout.close()
    print(f"DONE ok={ndone} err={nerr}", flush=True)
    sys.exit(1 if nerr > len(todo) * 0.05 else 0)


if __name__ == "__main__":
    main()
