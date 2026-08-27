"""Smoke test v3: aggressive verbatim+norm extraction over 20 RTC sections.

Simulates 4 parallel subagents (4 threads), each handling 5 docs. One extraction
prompt across all of them — we are testing the PROMPT, not exploring variants.
"""
import os, json, sys, time, threading, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

# Allow override; defaults to Sonnet 4.5 for cost.
MODEL = os.environ.get("SMOKE_MODEL", "claude-sonnet-4-5")
INPUT_PATH = "/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/smoke_v3/input_20.jsonl"
OUT_DIR = "/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v2_existing/smoke_v3"
N_AGENTS = 4

EXTRACTION_PROMPT = """You are extracting agency norms from a Response-to-Comments section of a final federal rule.

For EACH (comment, response) pair in the RTC text:

1. comment_verbatim: the exact text describing the commenter's argument. Quote from the source — DO NOT paraphrase.
2. response_verbatim: the exact text of the agency's response. Quote verbatim. Include ALL sentences from the response, even if they seem tangential. The full verbatim response should typically be 100-2000 chars.
3. norms: every distinct normative claim the agency makes in the response.

A NORM is any of:
  - statutory: cites a statutory/regulatory mandate ("Section X requires us to...")
  - principle: agency policy stance ("Our consistent policy is to favor X when Y")
  - balancing: trades off two values ("We weigh X against Y and conclude Z")
  - value: invokes a public/social value ("The public interest favors...", "Worker safety must take priority")
  - philosophy: agency's interpretive stance ("The Agency has historically held that...")
  - cost_benefit: explicit cost-benefit reasoning ("The marginal benefit of X exceeds the cost of Y")
  - procedural: process-of-decisionmaking principle ("Standard practice is to...", "We follow APA notice requirements")

For each norm:
  - norm_verbatim: exact quoted span from the response (10-300 chars). Must appear in response_verbatim.
  - norm_type: one of the categories above
  - norm_summary: one sentence in your own words

Be HIGH RECALL — extract every norm even if multiple norms appear in one sentence. We will filter later. Do NOT merge similar norms; keep them separate.

Output STRICT JSONL: one line per (comment, response) pair, with fields:
{
  "doc_id": "<from input>",
  "agency": "<from input>",
  "pair_idx": <0-indexed within doc>,
  "comment_verbatim": "<quoted commenter argument>",
  "response_verbatim": "<full quoted agency response>",
  "norms": [{"norm_verbatim": "...", "norm_type": "...", "norm_summary": "..."}, ...]
}

Output ONLY the JSONL lines, nothing else — no preamble, no markdown fences, no explanation."""


def load_docs():
    docs = []
    with open(INPUT_PATH) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def chunk(lst, n):
    """Split lst into n roughly-equal chunks."""
    k, m = divmod(len(lst), n)
    out = []
    i = 0
    for j in range(n):
        size = k + (1 if j < m else 0)
        out.append(lst[i:i + size])
        i += size
    return out


def extract_one(client, doc, max_retries=2):
    user_msg = (
        f"doc_id: {doc['doc_id']}\n"
        f"agency: {doc['agency']}\n"
        f"rule_title: {doc['rule_title']}\n\n"
        f"---BEGIN RTC TEXT---\n{doc['rtc_text_full']}\n---END RTC TEXT---"
    )
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            r = client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=EXTRACTION_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            elapsed = time.time() - t0
            text = "".join(b.text for b in r.content if hasattr(b, "text"))
            # Parse JSONL lines
            pairs = []
            parse_errors = 0
            for ln in text.splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("```"):
                    continue
                try:
                    obj = json.loads(ln)
                    pairs.append(obj)
                except json.JSONDecodeError:
                    parse_errors += 1
            return {
                "ok": True,
                "doc_id": doc["doc_id"],
                "agency": doc["agency"],
                "pairs": pairs,
                "parse_errors": parse_errors,
                "raw_chars": len(text),
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "elapsed_sec": round(elapsed, 1),
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return {"ok": False, "doc_id": doc["doc_id"], "error": last_err}


def run_agent(agent_id, doc_chunk):
    """One 'subagent': processes its chunk of docs serially."""
    client = anthropic.Anthropic()
    out_path = os.path.join(OUT_DIR, f"output_part_{agent_id}.jsonl")
    err_path = os.path.join(OUT_DIR, f"errors_part_{agent_id}.txt")
    n_ok = 0
    n_err = 0
    with open(out_path, "w") as fout, open(err_path, "w") as ferr:
        for doc in doc_chunk:
            res = extract_one(client, doc)
            if res["ok"]:
                rec = {
                    "doc_id": res["doc_id"],
                    "agency": res["agency"],
                    "pairs": res["pairs"],
                    "meta": {
                        "parse_errors": res["parse_errors"],
                        "raw_chars": res["raw_chars"],
                        "input_tokens": res["input_tokens"],
                        "output_tokens": res["output_tokens"],
                        "elapsed_sec": res["elapsed_sec"],
                    },
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_ok += 1
                print(f"[agent {agent_id}] OK {res['doc_id']} pairs={len(res['pairs'])} in/out_tok={res['input_tokens']}/{res['output_tokens']} {res['elapsed_sec']}s", flush=True)
            else:
                ferr.write(f"{res['doc_id']}\t{res['error']}\n")
                ferr.flush()
                n_err += 1
                print(f"[agent {agent_id}] ERR {res['doc_id']} {res['error']}", flush=True)
    return {"agent_id": agent_id, "n_ok": n_ok, "n_err": n_err}


def main():
    docs = load_docs()
    print(f"Loaded {len(docs)} docs. Model={MODEL}. Spawning {N_AGENTS} agents.")
    chunks = chunk(docs, N_AGENTS)
    for i, c in enumerate(chunks, 1):
        print(f"  agent {i}: {len(c)} docs: {[d['doc_id'] for d in c]}")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=N_AGENTS) as ex:
        futs = {ex.submit(run_agent, i + 1, chunks[i]): i + 1 for i in range(N_AGENTS)}
        for f in as_completed(futs):
            results.append(f.result())
    print(f"All agents done in {time.time() - t0:.1f}s. Results: {results}")


if __name__ == "__main__":
    main()
