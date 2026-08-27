"""Smoke v1: reason-extraction + canonical-rubric tagging on 20 peer reviews.

Pattern: 4 threads x 5 reviews each. Uses Claude Sonnet 4.5 via Anthropic SDK.

API key: reads from $ANTHROPIC_API_KEY if set, otherwise falls back to the
Claude Code OAuth token in the macOS login keychain ("Claude Code-credentials")
which keeps us on the user's existing subscription.
"""
import os, json, sys, time, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

MODEL = os.environ.get("SMOKE_MODEL", "claude-sonnet-4-5")
SMOKE_DIR = "/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/extracted/smoke_v1"
INPUT_PATH = os.path.join(SMOKE_DIR, "input_20.jsonl")
RUBRIC_PATH = "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_general_r2_expanded.json"
N_AGENTS = int(os.environ.get("SMOKE_N_AGENTS", "4"))
INTER_CALL_SLEEP = float(os.environ.get("SMOKE_INTER_CALL_SLEEP", "0"))


def get_auth():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return {"api_key": key}
    # Fallback: OAuth token from keychain
    try:
        raw = subprocess.check_output(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        d = json.loads(raw)
        return {"auth_token": d["claudeAiOauth"]["accessToken"]}
    except Exception as e:
        raise RuntimeError(f"No ANTHROPIC_API_KEY and keychain lookup failed: {e}")


def load_rubrics():
    with open(RUBRIC_PATH) as f:
        d = json.load(f)
    out = []
    for i, g in enumerate(d["merged_groups"]):
        out.append({
            "id": i,
            "name": g["merged_name"],
            "description": g["merged_description"],
        })
    return out


def load_reviews():
    out = []
    with open(INPUT_PATH) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_rubric_block(rubrics):
    lines = []
    for r in rubrics:
        # Keep description to one compact line
        desc = " ".join(r["description"].split())
        lines.append(f"[{r['id']}] {r['name']}: {desc}")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are extracting reasons from a peer review and tagging them against a canonical rubric taxonomy.

Below is a list of {n_rubrics} canonical peer-review rubrics. Each has an integer ID, a name, and a one-line description.

---BEGIN CANONICAL RUBRICS---
{rubric_block}
---END CANONICAL RUBRICS---

Given a peer review, your task is:

1. Extract EVERY distinct reason the reviewer gives for why the paper is good, bad, or limited. Be HIGH RECALL — extract a separate item for each distinct critique or compliment, even if multiple appear in one paragraph.

2. For each reason, produce:
   - verbatim_span: the EXACT quoted reviewer text (30-500 chars). Must be a substring of the input review. Do not paraphrase, do not summarize. Pick the shortest contiguous span that captures the reason.
   - polarity: "positive" (a strength), "negative" (a weakness/critique), or "mixed" (both).
   - paraphrase: one-sentence summary in your own words.
   - rubric_matches: integer IDs (from the canonical rubric list above) of 1 to 3 rubrics that this reason instantiates. Be STRICT — only tag a rubric if the reason genuinely instantiates the rubric's normative concept. If NONE of the canonical rubrics fit, return an empty list []. An empty list is a useful signal that the taxonomy may be incomplete.

3. Output ONE JSON object (not JSONL — a single object) with this exact shape:

{{
  "review_id": "<from input>",
  "paper_id": "<from input>",
  "venue": "<from input>",
  "decision": "<from input>",
  "review_score": "<from input or null>",
  "reasons": [
    {{
      "verbatim_span": "...",
      "polarity": "positive" | "negative" | "mixed",
      "paraphrase": "...",
      "rubric_matches": [<int>, ...]
    }},
    ...
  ]
}}

Output ONLY the JSON object, no preamble, no markdown fences, no explanation. The JSON must parse with json.loads()."""


def extract_one(client, review, system_prompt, max_retries=6):
    user_msg = (
        f"review_id: {review['review_id']}\n"
        f"paper_id: {review['paper_id']}\n"
        f"venue: {review['venue']}\n"
        f"decision: {review.get('decision')}\n"
        f"review_score: {review.get('review_score')}\n"
        f"is_meta_review: {review.get('is_meta_review')}\n"
        f"title: {review.get('title')}\n\n"
        f"---BEGIN REVIEW TEXT---\n{review['review_text']}\n---END REVIEW TEXT---"
    )
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            r = client.messages.create(
                model=MODEL,
                max_tokens=8000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            elapsed = time.time() - t0
            text = "".join(b.text for b in r.content if hasattr(b, "text")).strip()
            # Strip optional markdown fences
            if text.startswith("```"):
                # Drop first and last fenced lines
                lines = text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines)
            try:
                obj = json.loads(text)
                parse_ok = True
            except json.JSONDecodeError as je:
                obj = {"_parse_error": str(je), "_raw": text[:2000]}
                parse_ok = False
            return {
                "ok": True,
                "parse_ok": parse_ok,
                "review_id": review["review_id"],
                "obj": obj,
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "elapsed_sec": round(elapsed, 1),
            }
        except anthropic.RateLimitError as e:
            last_err = f"RateLimitError: {e}"
            # Exponential-ish backoff: 20, 40, 60, 90, 120, 180
            waits = [20, 40, 60, 90, 120, 180]
            time.sleep(waits[min(attempt, len(waits) - 1)])
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))
    return {"ok": False, "review_id": review["review_id"], "error": last_err}


def chunk(lst, n):
    k, m = divmod(len(lst), n)
    out = []
    i = 0
    for j in range(n):
        size = k + (1 if j < m else 0)
        out.append(lst[i:i + size])
        i += size
    return out


def run_agent(agent_id, reviews, system_prompt, auth):
    client = anthropic.Anthropic(**auth)
    out_path = os.path.join(SMOKE_DIR, f"output_part_{agent_id}.jsonl")
    err_path = os.path.join(SMOKE_DIR, f"errors_part_{agent_id}.txt")
    n_ok = n_err = 0
    with open(out_path, "w") as fout, open(err_path, "w") as ferr:
        for rev in reviews:
            if INTER_CALL_SLEEP > 0:
                time.sleep(INTER_CALL_SLEEP)
            res = extract_one(client, rev, system_prompt)
            if res["ok"]:
                rec = {
                    "review_id": res["review_id"],
                    "parse_ok": res["parse_ok"],
                    "obj": res["obj"],
                    "meta": {
                        "input_tokens": res["input_tokens"],
                        "output_tokens": res["output_tokens"],
                        "elapsed_sec": res["elapsed_sec"],
                    },
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_ok += 1
                n_reasons = len(res["obj"].get("reasons", [])) if isinstance(res["obj"], dict) else "?"
                print(f"[agent {agent_id}] OK {res['review_id']} reasons={n_reasons} in/out={res['input_tokens']}/{res['output_tokens']} {res['elapsed_sec']}s parse_ok={res['parse_ok']}", flush=True)
            else:
                ferr.write(f"{res['review_id']}\t{res['error']}\n")
                ferr.flush()
                n_err += 1
                print(f"[agent {agent_id}] ERR {res['review_id']} {res['error']}", flush=True)
    return {"agent_id": agent_id, "n_ok": n_ok, "n_err": n_err}


def main():
    auth = get_auth()
    auth_kind = "api_key" if "api_key" in auth else "oauth_token"
    rubrics = load_rubrics()
    reviews = load_reviews()
    print(f"Auth: {auth_kind}; Model={MODEL}; rubrics={len(rubrics)}; reviews={len(reviews)}; agents={N_AGENTS}")

    rubric_block = build_rubric_block(rubrics)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(n_rubrics=len(rubrics), rubric_block=rubric_block)
    # Sanity: print prompt size
    print(f"System prompt ~chars={len(system_prompt)}")

    chunks = chunk(reviews, N_AGENTS)
    for i, c in enumerate(chunks, 1):
        print(f"  agent {i}: {len(c)} reviews: {[r['review_id'] for r in c]}")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=N_AGENTS) as ex:
        futs = {ex.submit(run_agent, i + 1, chunks[i], system_prompt, auth): i + 1 for i in range(N_AGENTS)}
        for f in as_completed(futs):
            results.append(f.result())
    print(f"All agents done in {time.time() - t0:.1f}s. Results: {results}")

    # Merge
    merged_path = os.path.join(SMOKE_DIR, "output_all.jsonl")
    with open(merged_path, "w") as fout:
        for i in range(1, N_AGENTS + 1):
            p = os.path.join(SMOKE_DIR, f"output_part_{i}.jsonl")
            if os.path.exists(p):
                with open(p) as fin:
                    for line in fin:
                        fout.write(line)
    print(f"Merged -> {merged_path}")


if __name__ == "__main__":
    main()
