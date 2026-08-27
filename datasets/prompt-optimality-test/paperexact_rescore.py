"""Uniform TEST-split rescore of every distinct paperexact candidate (feeds the bound analyses).

Analog of phase4_rescore.py for runs_paperexact/: pool every candidate logged in
runs_paperexact/<bench>/<lm>/<arm>/proposals.jsonl (selection panels differ per arm, so logged
scores are NOT comparable), dedupe by hash, evaluate each ONCE on the full paper test split with
the paper metric, and append to <arm>/rescore.jsonl with per-item scores.

HYGIENE: this is a terminal, post-hoc diagnostic pass — nothing downstream selects on it. The
test split is the right one for bound analysis precisely because no arm's selection ever touched
it (official/inhouse select on train/val; unitrecomb selects on train slices).

  .venv/bin/python paperexact_rescore.py aime --lm-tag Qwen3-8B \
      --task-lm openai/Qwen3-8B --api-base http://127.0.0.1:8077/v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from paperexact_arms import HERE, evaluate_cand, load_bench, cand_hash

import dspy

ARMS = ("official", "inhouse", "unitrecomb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["aime", "hover", "hotpot", "ifbench", "livebench", "pupa"])
    ap.add_argument("--lm-tag", required=True, help="run-dir LM tag, e.g. Qwen3-8B")
    ap.add_argument("--task-lm", required=True)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--n-threads", type=int, default=8)
    a = ap.parse_args()

    key = Path(a.api_key_file).read_text().strip() if a.api_key_file else "EMPTY"
    # deep retries + per-attempt timeout: see paperexact_arms.make_reflection_lm comment
    dspy.configure(lm=dspy.LM(a.task_lm, api_base=a.api_base, api_key=key,
                              temperature=a.temperature, top_p=a.top_p,
                              max_tokens=a.max_tokens, num_retries=10, timeout=300))
    bench, program, metric, _ = load_bench(a.bench)
    test = list(bench.test_set)
    base = HERE / "runs_paperexact" / a.bench / a.lm_tag

    seen: set[str] = set()
    todo: list[tuple[str, str, dict]] = []
    for arm in ARMS:
        p = base / arm / "proposals.jsonl"
        if not p.exists():
            continue
        done_file = base / arm / "rescore.jsonl"
        already = {json.loads(l)["hash"] for l in open(done_file)} if done_file.exists() else set()
        for line in open(p):
            r = json.loads(line)
            h = r.get("hash") or cand_hash(r["candidate"])
            if h in seen or h in already:
                continue
            seen.add(h)
            todo.append((arm, h, r["candidate"]))
    print(f"[{a.bench}|{a.lm_tag}] rescoring {len(todo)} distinct candidates on "
          f"{len(test)} test items", flush=True)

    def endpoint_alive() -> bool:
        # POST-aware for anthropic-style endpoints (z.ai has no GET /models — a GET-only probe
        # would false-abort on candidates that legitimately score 0.0).
        if not a.api_base:
            return True
        import urllib.request
        try:
            if "api.z.ai" in a.api_base:
                req = urllib.request.Request(
                    a.api_base.rstrip("/") + "/v1/messages",
                    data=json.dumps({"model": "glm-4.7", "max_tokens": 8,
                                     "messages": [{"role": "user", "content": "ok"}]}).encode(),
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                             "content-type": "application/json"})
                urllib.request.urlopen(req, timeout=45)
            else:
                urllib.request.urlopen(a.api_base.rstrip("/") + "/models", timeout=10)
            return True
        except Exception:
            return False

    for n, (arm, h, cand) in enumerate(todo):
        t0 = time.time()
        # evaluate_cand logs to a path we choose; give it a throwaway log and write our own row
        # (same schema as runs/ rescore.jsonl: hash + item_scores) so bound tooling can load it.
        tmp_log = base / arm / "rescore_evals.jsonl"
        score = evaluate_cand(program, cand, test, metric, tmp_log, "rescore_test",
                              n_threads=a.n_threads)
        rows = [json.loads(l) for l in open(tmp_log)]
        item_scores = rows[-1]["item_scores"]
        # dspy's max_errors scores dead-endpoint items 0 and reports "success": an all-zero row
        # with a dead endpoint is an OUTAGE artifact, not a measurement — abort, don't record.
        # (This exact failure corrupted 20 rows on 2026-07-20; see rescore.outage-quarantine-*.)
        if sum(item_scores) == 0 and not endpoint_alive():
            raise SystemExit(f"ENDPOINT DOWN while rescoring {arm}/{h} — aborting; rerun to "
                             "resume (already-recorded hashes are skipped)")
        with open(base / arm / "rescore.jsonl", "a") as fh:
            fh.write(json.dumps({"hash": h, "candidate": cand, "mean_score": score,
                                 "item_scores": item_scores, "n_items": len(test),
                                 "split": "paper_test", "ts": time.time()}) + "\n")
        print(f"  [{n + 1}/{len(todo)}] {arm} {h} -> {score:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print("RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
