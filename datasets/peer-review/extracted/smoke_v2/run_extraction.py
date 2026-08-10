"""Smoke v2: norm-ADJACENT passage+signal extraction on 20 peer reviews.

Differs from v1 in:
  - Unit is the PASSAGE (paragraph-level critique window), with multiple SIGNALS inside.
  - Signals are evidence FOR norms (complaints/praise/observations/suggestions),
    NOT verbatim norm statements.
  - Multi-label rubric tagging per signal (0-3 rubrics — 0 is a legitimate gap signal).
  - 154 canonical rubrics (built from 184 aspect_clusters via build_rubrics.py).
  - Polarity tracked at both passage and signal level.
"""
import os
import json
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

MODEL = os.environ.get("SMOKE_MODEL", "claude-sonnet-4-5")
SMOKE_DIR = "/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/extracted/smoke_v2"
INPUT_PATH = "/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/extracted/smoke_v1/input_20.jsonl"
RUBRIC_PATH = os.path.join(SMOKE_DIR, "rubrics.jsonl")
N_AGENTS = int(os.environ.get("SMOKE_N_AGENTS", "4"))
INTER_CALL_SLEEP = float(os.environ.get("SMOKE_INTER_CALL_SLEEP", "0"))


def get_auth():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return {"api_key": key}
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
    out = []
    with open(RUBRIC_PATH) as f:
        for line in f:
            out.append(json.loads(line))
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
        desc = " ".join(r["description"].split())
        # Trim long descriptions to keep prompt manageable
        if len(desc) > 250:
            desc = desc[:247] + "..."
        lines.append(f"[{r['rubric_id']}] {r['name']}: {desc}")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are extracting norm-ADJACENT signals from peer reviews and tagging them against a canonical rubric taxonomy.

Below is a list of {n_rubrics} canonical peer-review rubrics. Each has an integer ID, a name, and a one-line description.

---BEGIN CANONICAL RUBRICS---
{rubric_block}
---END CANONICAL RUBRICS---

GOAL
You are NOT extracting verbatim norm statements. Reviewers rarely write canonical norms — they write COMPLAINTS, PRAISE, OBSERVATIONS, and SUGGESTIONS that POINT AT norms.
A signal like "the paper is missing an ablation on n_layers" is evidence FOR the rubric "ablation rigor" without itself being that rubric.

UNITS
- A PASSAGE is a contiguous paragraph or window of sentences (50-1500 chars) where the reviewer is making at least one quality judgment. Every quality-judgment paragraph in the review should become a passage. Skip pure-summary paragraphs.
- A SIGNAL is a specific phrase/claim inside a passage (10-200 chars), and must be an EXACT substring of its passage_text.

TASK
For the input review, output ONE JSON object with this exact shape:

{{
  "review_id": "<from input>",
  "paper_id": "<from input>",
  "venue": "<from input>",
  "decision": "<from input>",
  "review_score": <from input or null>,
  "passages": [
    {{
      "passage_text": "<verbatim contiguous text from the review, 50-1500 chars, MUST be an exact substring of the review>",
      "passage_polarity": "positive" | "negative" | "mixed",
      "signals": [
        {{
          "signal_text": "<exact substring of passage_text, 10-200 chars>",
          "signal_type": "complaint" | "praise" | "observation" | "suggestion",
          "polarity": "positive" | "negative" | "neutral",
          "rubric_matches": [<int rubric_id>, ...]
        }}
      ]
    }}
  ]
}}

GUIDELINES
1. HIGH RECALL on signals. Aim for 3-10 signals per passage when warranted. A single passage often contains multiple distinct complaints/praises — extract them all.
2. passage_text MUST be a verbatim substring of the review text. signal_text MUST be a verbatim substring of its passage_text.
3. polarity definitions:
   - At passage level: "positive" = entirely praise/strengths, "negative" = entirely critique/weaknesses, "mixed" = both
   - At signal level: "positive" = strength/compliment, "negative" = weakness/complaint, "neutral" = observation or suggestion without value judgment
4. signal_type definitions:
   - "complaint" — reviewer flags a problem or weakness
   - "praise"   — reviewer flags a strength
   - "observation" — neutral factual claim about the paper
   - "suggestion" — reviewer recommends a change (regardless of polarity)
5. rubric_matches: 0-3 integer rubric_ids from the canonical list above. Tag a rubric only if the signal genuinely instantiates that rubric's normative concept. An EMPTY list [] is a legitimate and useful signal that the taxonomy may not cover this concept — do not force a match.
6. Skip passages that are pure summary, restatement of the paper, or pure metadata. Only emit passages where the reviewer makes a judgment.

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
                max_tokens=12000,  # Higher than v1: passages+signals are denser
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            elapsed = time.time() - t0
            text = "".join(b.text for b in r.content if hasattr(b, "text")).strip()
            if text.startswith("```"):
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
                if isinstance(res["obj"], dict):
                    n_pass = len(res["obj"].get("passages", []))
                    n_sig = sum(len(p.get("signals", [])) for p in res["obj"].get("passages", []))
                else:
                    n_pass = n_sig = "?"
                print(f"[agent {agent_id}] OK {res['review_id']} passages={n_pass} signals={n_sig} "
                      f"in/out={res['input_tokens']}/{res['output_tokens']} {res['elapsed_sec']}s parse_ok={res['parse_ok']}", flush=True)
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
