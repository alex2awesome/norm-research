"""Arm A — OFFICIAL GEPA (pinned, see PIN.txt) on the benchmark JSONLs, with GLM task/reflection
LMs via the z.ai anthropic-compatible subscription endpoint (0-GPU, pure HTTP).

Isolation rule #4: EVERY candidate evaluation is appended to runs/<ds>/official/proposals.jsonl
(timestamp + full candidate text + per-batch scores) — accepted AND rejected candidates, the raw
draw sequence the prompt-optimality estimators need.

  source .venv/bin/activate
  python run_official_gepa.py aime --max-metric-calls 600
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
ZAI_URL = "https://api.z.ai/api/anthropic/v1/messages"
KEY_FILES = [Path.home() / ".z-ai-api-key-alexander-spangher.txt",
             Path.home() / ".z-ai-api-key-spangher.txt",
             Path.home() / ".z-ai-api-key.txt"]


def _key() -> str:
    import os
    env = os.environ.get("ZAI_KEY_FILE")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser().read_text().strip()
    for k in KEY_FILES:
        if k.exists():
            return k.read_text().strip()
    raise RuntimeError("no z.ai key file found")


class GLM:
    """messages-or-prompt -> str against the subscription endpoint, with retry/backoff (z.ai
    intermittently returns overloaded_error 1305; retry clears it)."""

    def __init__(self, model: str, *, max_tokens: int = 2048, temperature: float = 0.6):
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.key = _key()
        self.n_calls = 0

    def __call__(self, prompt_or_messages) -> str:
        if isinstance(prompt_or_messages, str):
            system, msgs = None, [{"role": "user", "content": prompt_or_messages}]
        else:
            system = "\n".join(m["content"] for m in prompt_or_messages if m["role"] == "system") or None
            msgs = [{"role": m["role"], "content": m["content"]}
                    for m in prompt_or_messages if m["role"] != "system"]
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "messages": msgs}
        if system:
            body["system"] = system
        last = ""
        for attempt in range(7):
            try:
                r = requests.post(ZAI_URL, json=body, timeout=240,
                                  headers={"x-api-key": self.key,
                                           "anthropic-version": "2023-06-01"})
                if r.status_code == 200:
                    txt = "".join(b.get("text", "") for b in r.json().get("content", []))
                    if txt.strip():
                        self.n_calls += 1
                        return txt
                last = f"{r.status_code} {r.text[:160]}"
                if r.status_code == 400 and '"1301"' in r.text:
                    # provider content filter rejected the input (e.g. a Wikipedia passage).
                    # Non-retryable and item-specific: return a placeholder so the evaluator
                    # scores it 0 and the RUN survives; never kill 600 calls over one item.
                    self.n_calls += 1
                    return "[provider-content-filter-rejected]"
                if r.status_code == 429:      # request-rate limit: back off much longer
                    time.sleep(min(180, 20 * 2 ** attempt))
                    continue
            except Exception as e:            # noqa: BLE001 — network retry loop
                last = str(e)[:160]
            time.sleep(min(60, 4 * 2 ** attempt))
        raise RuntimeError(f"GLM {self.model} failed after retries: {last}")


# ------------------------------- evaluators (score + textual feedback) -----------------------

def _mk_eval_result(score, feedback):
    from gepa.adapters.default_adapter.default_adapter import EvaluationResult
    return EvaluationResult(score=score, feedback=feedback, objective_scores=None)


def aime_evaluator(data, response):
    tail = response.split("Answer")[-1] if "Answer" in response else response[-200:]
    nums = re.findall(r"-?\d+", tail)
    pred = nums[-1] if nums else None
    gold = str(int(float(data["answer"])))
    if pred == gold:
        return _mk_eval_result(1.0, "Correct final answer.")
    return _mk_eval_result(0.0, f"Incorrect. Parsed '{pred}', correct integer is {gold}. "
                                "End with a line 'Answer: <integer>'.")


def hover_evaluator(data, response):
    up = response.upper()
    pred = ("NOT_SUPPORTED" if "NOT_SUPPORTED" in up or "NOT SUPPORTED" in up
            else "SUPPORTED" if "SUPPORTED" in up else None)
    if pred == data["answer"]:
        return _mk_eval_result(1.0, "Correct verdict.")
    return _mk_eval_result(0.0, f"Incorrect. Model said '{pred}', gold is {data['answer']}. "
                                "Reply with exactly SUPPORTED or NOT_SUPPORTED.")


def hotpot_evaluator(data, response):
    if data["answer"].strip().lower() in response.strip().lower():
        return _mk_eval_result(1.0, "Response contains the correct answer.")
    return _mk_eval_result(0.0, f"Incorrect. The correct answer is '{data['answer']}'. "
                                "Answer concisely with the exact answer phrase.")


# ------------------------------- dataset -> DefaultDataInst ----------------------------------

def _load(ds: str, split: str, cap: int):
    rows = [json.loads(l) for l in open(HERE / "data" / ds / f"{split}.jsonl")][:cap]
    out = []
    for r in rows:
        if ds == "aime2025":
            out.append({"input": r["problem"], "additional_context": {}, "answer": str(r["answer"])})
        elif ds == "hover":
            out.append({"input": r["claim"], "additional_context": {},
                        "answer": "SUPPORTED" if int(r["label"]) == 1 else "NOT_SUPPORTED"})
        elif ds == "hotpotqa":
            ctx = "\n".join(f"[{t}] " + " ".join(s) for t, s in
                            zip(r["context"]["title"], r["context"]["sentences"]))
            out.append({"input": f"Question: {r['question']}\n\nContext:\n{ctx[:8000]}",
                        "additional_context": {}, "answer": r["answer"]})
    return out


SEEDS = {
    "aime2025": ("You are a careful competition mathematician. Solve the problem step by step, "
                 "then finish with a final line formatted exactly as 'Answer: <integer>'."),
    "hover": ("Decide whether the claim is supported by documented facts. Think briefly, then "
              "reply with exactly SUPPORTED or NOT_SUPPORTED."),
    "hotpotqa": ("Answer the question using only the provided context. Reply with just the "
                 "answer phrase, nothing else."),
}
EVALS = {"aime2025": aime_evaluator, "hover": hover_evaluator, "hotpotqa": hotpot_evaluator}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(SEEDS))
    ap.add_argument("--max-metric-calls", type=int, default=600)
    ap.add_argument("--train-n", type=int, default=150)
    ap.add_argument("--val-n", type=int, default=100)
    ap.add_argument("--task-model", default="glm-4.7")
    ap.add_argument("--reflection-model", default="glm-5")     # resolves to glm-5.2 server-side
    a = ap.parse_args()

    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    rundir = HERE / "runs" / a.dataset / "official"
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / "seed.txt").write_text(SEEDS[a.dataset])
    log_path = rundir / "proposals.jsonl"

    class LoggingAdapter(DefaultAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            out = super().evaluate(batch, candidate, capture_traces)
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"ts": time.time(), "candidate": candidate,
                                     "n_batch": len(batch),
                                     "mean_score": (sum(out.scores) / max(len(out.scores), 1)),
                                     "scores": list(out.scores)}) + "\n")
            return out

    task_lm = GLM(a.task_model, max_tokens=2048, temperature=0.2)
    refl_lm = GLM(a.reflection_model, max_tokens=4096, temperature=1.0)
    adapter = LoggingAdapter(model=task_lm, evaluator=EVALS[a.dataset])

    train = _load(a.dataset, "train", a.train_n)
    val = _load(a.dataset, "val", a.val_n)
    print(f"[{a.dataset}] train={len(train)} val={len(val)} budget={a.max_metric_calls} "
          f"task={a.task_model} refl={a.reflection_model}", flush=True)

    res = gepa.optimize(seed_candidate={"system_prompt": SEEDS[a.dataset]},
                        trainset=train, valset=val, adapter=adapter,
                        reflection_lm=refl_lm, max_metric_calls=a.max_metric_calls,
                        run_dir=str(rundir / "gepa_state"), seed=0,
                        display_progress_bar=False, raise_on_exception=False)

    summary = {"dataset": a.dataset, "best_candidate": res.best_candidate,
               "task_lm_calls": task_lm.n_calls, "reflection_lm_calls": refl_lm.n_calls}
    for attr in ("val_aggregate_scores", "best_idx", "total_metric_calls", "num_candidates"):
        try:
            summary[attr] = getattr(res, attr)
        except Exception:
            pass
    (rundir / "result.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[{a.dataset}] DONE best mean-val "
          f"{max(summary.get('val_aggregate_scores') or [-1]):.3f}; wrote {rundir/'result.json'}",
          flush=True)


if __name__ == "__main__":
    main()
