"""Test-set eval for the articulation-STaR loop.

For each LoRA stage (base, iter_00, iter_01, iter_02), do:
  1. Generate ONE rationale per held-out test artifact.
  2. Score with strong judge (rationale-only, logprob mode).
  3. Report acc, average margin, and pos/neg breakdown.

Held-out test = sample of `litbench-to-train.csv.gz` rows that were NOT
included in ANY of iters 0/1/2's training subsamples (random_states
42, 43, 44 each with n=N_TRAIN). Independence is verified before run.

Phases run as separate python subprocess invocations so vLLM tears down
between models. Orchestrated by `scripts/articulation_star/run_test_eval.sh`.

Modes:
  --mode build_split  : write the held-out test jsonl (artifact + y).
  --mode generate     : generate rationales for one stage's LoRA (or base).
  --mode score        : score one stage's rationales with the strong judge.
  --mode summarize    : load all stage results and print the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

from .config import LoopConfig, TASKS
from .prompts import render_gen
from .judge_filter import _score


# ──────────────────────────────────────────────────────────────────
# 1. Build the test split (held out from all training iters)
# ──────────────────────────────────────────────────────────────────

def build_split(cfg: LoopConfig, n_test: int, n_train_iters: int) -> Path:
    """Sample n_test rows from `task.data_path` that were NOT used in
    iters 0..n_train_iters-1 (which used random_states 42, 43, 44, ...)."""
    task = TASKS[cfg.task]
    df = pd.read_csv(task.data_path)[["text", "judgement"]].dropna().reset_index(drop=True)

    used = set()
    for it in range(n_train_iters):
        ids = df.sample(n=cfg.n_train_subsample, random_state=42 + it).index.tolist()
        used.update(ids)

    available = df[~df.index.isin(used)]
    print(f"[build_split] dropna df: {len(df):,} | used in train: {len(used):,} | "
          f"available for test: {len(available):,}")
    test = available.sample(n=n_test, random_state=999).reset_index(drop=False)
    test = test.rename(columns={"index": "global_row_id"})

    out_dir = Path(cfg.output_root) / cfg.task / cfg.run_name / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_artifacts.jsonl"
    with out_path.open("w") as f:
        for i, row in test.iterrows():
            f.write(json.dumps({
                "row_id": i,
                "global_row_id": int(row["global_row_id"]),
                "text": row["text"],
                "y": int(row["judgement"]),
            }) + "\n")
    pos = (test["judgement"] == 1).sum()
    print(f"[build_split] wrote {len(test)} test rows ({pos} pos / {len(test)-pos} neg) "
          f"to {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────
# 2. Generate rationales for one stage
# ──────────────────────────────────────────────────────────────────

def generate(cfg: LoopConfig, stage_name: str, lora_path: str | None) -> Path:
    task = TASKS[cfg.task]
    test_dir = Path(cfg.output_root) / cfg.task / cfg.run_name / "test_eval"
    rows = [json.loads(l) for l in (test_dir / "test_artifacts.jsonl").open()]

    llm_kwargs = dict(
        model=cfg.generator_model,
        gpu_memory_utilization=cfg.vllm_gpu_mem_util,
        max_model_len=cfg.vllm_max_model_len,
        dtype="bfloat16",
        enable_lora=lora_path is not None,
        max_lora_rank=cfg.lora_r if lora_path is not None else 16,
    )
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()

    prompts = []
    for r in rows:
        msgs = render_gen(
            text=r["text"][: cfg.max_text_chars],
            text_type=task.text_type,
            pos=task.positive_label,
            neg=task.negative_label,
        )
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))

    sp = SamplingParams(
        n=1,
        temperature=cfg.gen_temperature,
        top_p=cfg.gen_top_p,
        max_tokens=cfg.gen_max_tokens,
        stop=["\nTherefore", "\nOverall", "\nOn balance"],
    )

    lora_request = None
    if lora_path is not None:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("articulator", 1, lora_path)

    outputs = llm.generate(prompts, sp, lora_request=lora_request)

    out_path = test_dir / f"rationales_{stage_name}.jsonl"
    with out_path.open("w") as f:
        for r, out in zip(rows, outputs):
            f.write(json.dumps({
                "row_id": r["row_id"],
                "y": r["y"],
                "stage": stage_name,
                "completion": out.outputs[0].text.strip(),
            }) + "\n")
    print(f"[generate:{stage_name}] wrote {len(rows)} rationales to {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────
# 3. Score one stage's rationales with the strong judge
# ──────────────────────────────────────────────────────────────────

def score(cfg: LoopConfig, stage_name: str) -> Path:
    task = TASKS[cfg.task]
    test_dir = Path(cfg.output_root) / cfg.task / cfg.run_name / "test_eval"
    rows = [json.loads(l) for l in (test_dir / f"rationales_{stage_name}.jsonl").open()]
    # judge_filter._score expects each row to have a 'completion' field and
    # 'y'. Add a synthetic sample_idx so the shape matches.
    for i, r in enumerate(rows):
        r.setdefault("sample_idx", 0)

    scored = _score(
        rows,
        model=cfg.judge_model,
        text_type=task.text_type,
        pos=task.positive_label.lower(),
        neg=task.negative_label.lower(),
        gpu_mem_util=cfg.vllm_gpu_mem_util,
        max_model_len=cfg.vllm_max_model_len,
    )

    out_path = test_dir / f"scores_{stage_name}.jsonl"
    with out_path.open("w") as f:
        for p in scored:
            f.write(json.dumps(p) + "\n")

    n = len(scored)
    acc = sum(1 for p in scored if p["judge_pred"] == p["y"]) / n
    import statistics
    margins = [p["margin"] for p in scored]
    print(f"[score:{stage_name}] n={n} acc={acc:.1%} "
          f"margin med={statistics.median(margins):.3f} "
          f"mean={statistics.mean(margins):.3f}")
    return out_path


# ──────────────────────────────────────────────────────────────────
# 4. Summarize
# ──────────────────────────────────────────────────────────────────

def summarize(cfg: LoopConfig, stages: list[str]) -> None:
    import statistics
    test_dir = Path(cfg.output_root) / cfg.task / cfg.run_name / "test_eval"
    print("\n" + "=" * 78)
    print(f"TEST-SET RESULTS  ({cfg.task} / {cfg.run_name})")
    print("=" * 78)
    rows = [json.loads(l) for l in (test_dir / "test_artifacts.jsonl").open()]
    n_pos = sum(1 for r in rows if r["y"] == 1)
    print(f"test n={len(rows)} (pos={n_pos}, neg={len(rows)-n_pos}); "
          f"majority baseline = {max(n_pos, len(rows)-n_pos)/len(rows):.1%}")
    print()
    print(f"{'stage':<14} {'acc':>8} {'margin_med':>12} {'margin_mean':>12} "
          f"{'acc_pos':>8} {'acc_neg':>8}")
    print("-" * 78)
    for s in stages:
        f = test_dir / f"scores_{s}.jsonl"
        if not f.exists():
            print(f"{s:<14}  (no scores file)")
            continue
        sc = [json.loads(l) for l in f.open()]
        n = len(sc)
        acc = sum(1 for p in sc if p["judge_pred"] == p["y"]) / n
        med = statistics.median(p["margin"] for p in sc)
        mean = statistics.mean(p["margin"] for p in sc)
        # per-label acc
        pos = [p for p in sc if p["y"] == 1]
        neg = [p for p in sc if p["y"] == 0]
        acc_p = (sum(1 for p in pos if p["judge_pred"] == 1) /
                 len(pos)) if pos else 0.0
        acc_n = (sum(1 for p in neg if p["judge_pred"] == 0) /
                 len(neg)) if neg else 0.0
        print(f"{s:<14} {acc:>8.1%} {med:>12.3f} {mean:>12.3f} "
              f"{acc_p:>8.1%} {acc_n:>8.1%}")
    print("=" * 78)


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative_writing")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--mode", choices=["build_split", "generate", "score", "summarize"],
                    required=True)
    ap.add_argument("--stage", default=None,
                    help="Stage name for generate/score (base, iter00, iter01, iter02).")
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--n_train_iters", type=int, default=3,
                    help="Number of training iters to exclude (random_states 42..42+n-1).")
    ap.add_argument("--n_train_subsample", type=int, default=10000,
                    help="Per-iter training subsample size (must match what was used).")
    ap.add_argument("--stages", default="base,iter00,iter01,iter02",
                    help="Comma-separated list of stages for summarize.")
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    cfg = LoopConfig(task=a.task, run_name=a.run_name)
    cfg.n_train_subsample = a.n_train_subsample

    if a.mode == "build_split":
        build_split(cfg, n_test=a.n_test, n_train_iters=a.n_train_iters)
    elif a.mode == "generate":
        if a.stage is None:
            raise SystemExit("--stage required for generate")
        generate(cfg, a.stage, a.lora_path)
    elif a.mode == "score":
        if a.stage is None:
            raise SystemExit("--stage required for score")
        score(cfg, a.stage)
    elif a.mode == "summarize":
        summarize(cfg, a.stages.split(","))
