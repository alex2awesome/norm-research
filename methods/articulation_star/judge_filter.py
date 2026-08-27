"""Information-bottleneck judge — single and contrastive (weak/strong) modes.

The judge SCORES rationales by extracting the logprob of the two label tokens
under "ANSWER: ", then takes margin = logP(y_true) - logP(y_other). No verdict
text is generated; no parsing is required. This is the literal reading of
"keep rationales that make the answer more likely."

Filter modes:

  * ``single``  : keep iff strong_margin > 0 (judge prefers correct answer).
  * ``contrastive`` : two passes (weak then strong). Keep iff
                  strong_margin > tau_strong AND weak_margin < tau_weak.
                  Drops rationales whose label is decodable from cheap
                  surface cues (weak prefers correct); keeps rationales
                  that require strong-reader inference (strong prefers
                  correct AND weak does not).

To avoid two vLLM engines coexisting on one GPU, the contrastive path runs
in two stages: ``--mode predict --judge weak`` writes weak scores, the
subprocess exits, ``--mode predict --judge strong`` writes strong scores,
and ``--mode combine`` produces the kept jsonl from both score files.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

from vllm import LLM, SamplingParams

from .config import LoopConfig, TASKS
from .prompts import render_judge


# ─────────────────────────────────────────────────────────────────────
# Logprob extraction
# ─────────────────────────────────────────────────────────────────────
#
# For each rationale we ask the judge for ONE token after "ANSWER: " with
# the top-K candidate logprobs. We then find the highest logprob assigned
# to any token that starts the positive label's first word (with or without
# leading space), and similarly for the negative label. Margin = pos - neg
# under y=1, or neg - pos under y=0. Positive margin = judge prefers
# correct answer.

def _label_first_token_ids(tok, word: str) -> set[int]:
    """Token IDs whose decoded form (stripped of leading space) starts with
    ``word``. We probe both leading-space and no-leading-space variants
    because chat templates differ on whether there's a space after the
    assistant header."""
    word = word.lower()
    out: set[int] = set()
    for variant in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        ids = tok.encode(variant, add_special_tokens=False)
        if ids:
            out.add(ids[0])
    return out


def _score(
    rationales: List[dict],
    *,
    model: str,
    text_type: str,
    pos: str,
    neg: str,
    gpu_mem_util: float,
    max_model_len: int,
    top_logprobs: int = 20,
) -> List[dict]:
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    )
    tok = llm.get_tokenizer()

    def _tpl(msgs):
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    # We need the judge to emit a label token at the next position. The judge
    # prompt ends with "ANSWER: ..." instructions; we append "ANSWER: " to
    # the assistant prefix so the next token IS the answer.
    answer_prefix = "ANSWER: "
    prompts = [
        _tpl(render_judge(
            rationale=r["completion"], text_type=text_type, pos=pos, neg=neg,
        )) + answer_prefix
        for r in rationales
    ]

    sp = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=top_logprobs)
    outputs = llm.generate(prompts, sp)

    # discriminating-first-word for each label
    pos_first = next((w for w in pos.split() if w not in neg.split()), pos.split()[0])
    neg_first = next((w for w in neg.split() if w not in pos.split()), neg.split()[0])
    pos_ids = _label_first_token_ids(tok, pos_first)
    neg_ids = _label_first_token_ids(tok, neg_first)
    if not pos_ids or not neg_ids:
        raise RuntimeError(
            f"Could not find label token ids: pos={pos_first!r}->{pos_ids}, "
            f"neg={neg_first!r}->{neg_ids}"
        )

    scored = []
    for r, out in zip(rationales, outputs):
        # vLLM v1 returns logprobs[0] as a dict {token_id: Logprob} for the
        # single emitted token's distribution.
        lp_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
        # Best logprob over any token matching each label's first word.
        # If no candidate appears in top-K, treat as -inf (judge has no
        # mass on this label at all).
        def _best(ids: set[int]) -> float:
            best = float("-inf")
            for tid in ids:
                lp = lp_dict.get(tid)
                if lp is None:
                    continue
                # vLLM Logprob has a .logprob attribute.
                val = lp.logprob if hasattr(lp, "logprob") else float(lp)
                if val > best:
                    best = val
            return best

        lp_pos = _best(pos_ids)
        lp_neg = _best(neg_ids)
        # signed margin: positive if judge prefers correct answer
        if r["y"] == 1:
            margin = lp_pos - lp_neg
        else:
            margin = lp_neg - lp_pos
        scored.append({
            "row_id": r["row_id"],
            "sample_idx": r["sample_idx"],
            "y": r["y"],
            "lp_pos": lp_pos,
            "lp_neg": lp_neg,
            "margin": margin,
            # arg-max prediction for backward-compat / diagnostics
            "judge_pred": 1 if lp_pos > lp_neg else 0,
        })
    return scored


# ─────────────────────────────────────────────────────────────────────
# Single-judge legacy entry point (used by smoke test)
# ─────────────────────────────────────────────────────────────────────

def run(cfg: LoopConfig, iter_idx: int, tau: float = 0.0) -> Path:
    """Single-judge keep iff margin > tau."""
    task = TASKS[cfg.task]
    in_path = cfg.iter_dir(iter_idx) / "rationales.jsonl"
    rows = [json.loads(l) for l in in_path.open()]

    scored = _score(
        rows, model=cfg.judge_model,
        text_type=task.text_type,
        pos=task.positive_label.lower(), neg=task.negative_label.lower(),
        gpu_mem_util=cfg.vllm_gpu_mem_util, max_model_len=cfg.vllm_max_model_len,
    )

    diag_path = cfg.iter_dir(iter_idx) / "judge_diagnostics.jsonl"
    out_path = cfg.iter_dir(iter_idx) / "rationales_kept.jsonl"
    kept = []
    with diag_path.open("w") as df:
        for r, p in zip(rows, scored):
            rec = {**r, **p, "kept": p["margin"] > tau}
            df.write(json.dumps(rec) + "\n")
            if rec["kept"]:
                kept.append(rec)
    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    n_kept = len(kept); pos_n = sum(1 for r in kept if r["y"] == 1)
    print(f"[judge:single] total={len(rows)} kept={n_kept} "
          f"({pos_n} pos / {n_kept - pos_n} neg) at tau={tau}")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# Contrastive entry points (one subprocess per phase)
# ─────────────────────────────────────────────────────────────────────

def run_predict(cfg: LoopConfig, iter_idx: int, which: str) -> Path:
    """Score all rationales under one judge (weak or strong) and write a
    phase-specific jsonl with per-rationale margins."""
    task = TASKS[cfg.task]
    in_path = cfg.iter_dir(iter_idx) / "rationales.jsonl"
    rows = [json.loads(l) for l in in_path.open()]

    if which == "weak":
        model = cfg.weak_judge_model
        suffix = "weak"
    elif which == "strong":
        model = cfg.judge_model
        suffix = "strong"
    else:
        raise ValueError(f"which must be 'weak' or 'strong', got {which!r}")

    scored = _score(
        rows, model=model,
        text_type=task.text_type,
        pos=task.positive_label.lower(), neg=task.negative_label.lower(),
        gpu_mem_util=cfg.vllm_gpu_mem_util, max_model_len=cfg.vllm_max_model_len,
    )

    out_path = cfg.iter_dir(iter_idx) / f"judge_preds_{suffix}.jsonl"
    with out_path.open("w") as f:
        for p in scored:
            f.write(json.dumps(p) + "\n")
    # Sanity: arg-max accuracy AND fraction with positive margin.
    n = len(scored)
    n_acc = sum(1 for p in scored if p["judge_pred"] == p["y"])
    n_pos_margin = sum(1 for p in scored if p["margin"] > 0)
    import statistics
    margins = [p["margin"] for p in scored]
    print(f"[judge:{suffix}] n={n} acc={n_acc}/{n}={n_acc/n*100:.1f}% "
          f"margin>0={n_pos_margin}/{n}={n_pos_margin/n*100:.1f}% "
          f"margin_med={statistics.median(margins):.3f}")
    return out_path


def run_combine(
    cfg: LoopConfig, iter_idx: int,
    tau_strong: float = 0.0, tau_weak: float = 0.0,
    balanced_k_per_label: int | None = None,
) -> Path:
    """Combine weak + strong margins into the contrastive keep set.

    Default rule: keep iff strong_margin > tau_strong AND weak_margin < tau_weak.

    If ``balanced_k_per_label`` is set, instead take the top-K rationales by
    strong_margin from y=1 AND top-K from y=0 (with the weak<tau_weak
    constraint still applied), then balance to equal counts per label. This
    is the STaR-style training-set construction the user asked for: equal
    numbers of examples that help the judge land on positive vs negative.
    """
    iter_dir = cfg.iter_dir(iter_idx)
    rows = [json.loads(l) for l in (iter_dir / "rationales.jsonl").open()]
    weak = {(p["row_id"], p["sample_idx"]): p for p in (
        json.loads(l) for l in (iter_dir / "judge_preds_weak.jsonl").open())}
    strong = {(p["row_id"], p["sample_idx"]): p for p in (
        json.loads(l) for l in (iter_dir / "judge_preds_strong.jsonl").open())}

    diag_path = iter_dir / "judge_diagnostics.jsonl"
    out_path = iter_dir / "rationales_kept.jsonl"
    kept = []
    counts = {"strong_right_weak_wrong": 0, "both_right": 0,
              "only_weak_right": 0, "both_wrong": 0}
    with diag_path.open("w") as df:
        for r in rows:
            k = (r["row_id"], r["sample_idx"])
            wp = weak.get(k, {})
            sp = strong.get(k, {})
            w_margin = wp.get("margin", float("nan"))
            s_margin = sp.get("margin", float("nan"))

            # Categorize by signs first (for diagnostics), then keep by tau.
            if s_margin > 0 and w_margin <= 0:
                category = "strong_right_weak_wrong"
            elif s_margin > 0 and w_margin > 0:
                category = "both_right"
            elif s_margin <= 0 and w_margin > 0:
                category = "only_weak_right"
            else:
                category = "both_wrong"
            counts[category] += 1

            rec = {
                **r,
                "weak_margin": w_margin, "strong_margin": s_margin,
                "weak_pred": wp.get("judge_pred"),
                "strong_pred": sp.get("judge_pred"),
                "category": category,
            }
            df.write(json.dumps(rec) + "\n")
            kept.append(rec)  # will filter below

    n = len(rows)
    # Universe respecting weak constraint and strong > tau.
    eligible = [r for r in kept
                if r["weak_margin"] < tau_weak
                and r["strong_margin"] > tau_strong]

    if balanced_k_per_label is not None:
        pos = sorted([r for r in eligible if r["y"] == 1],
                     key=lambda r: -r["strong_margin"])
        neg = sorted([r for r in eligible if r["y"] == 0],
                     key=lambda r: -r["strong_margin"])
        k = min(len(pos), len(neg), balanced_k_per_label)
        final = pos[:k] + neg[:k]
        print(f"[judge:combine] balanced k_per_label={k} -> {len(final)} total "
              f"(eligible pool: {len(pos)} pos, {len(neg)} neg)")
    else:
        final = eligible
        print(f"[judge:combine] keep all eligible -> {len(final)} "
              f"(strong>{tau_strong}, weak<{tau_weak})")

    for r in final:
        r["kept"] = True
    with out_path.open("w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")

    print(f"[judge:combine] n={n} categories={counts}")
    print(f"[judge:combine] kept = {len(final)} "
          f"({sum(1 for r in final if r['y']==1)} pos / "
          f"{sum(1 for r in final if r['y']==0)} neg)")
    return out_path


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer_review")
    ap.add_argument("--run_name", default="v0")
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--mode", choices=["single", "predict", "combine"],
                    default="single")
    ap.add_argument("--judge", choices=["weak", "strong"], default=None,
                    help="Which judge to run when --mode=predict.")
    ap.add_argument("--tau_strong", type=float, default=0.0)
    ap.add_argument("--tau_weak", type=float, default=0.0)
    ap.add_argument("--balanced_k_per_label", type=int, default=None,
                    help="If set, take top-K per label by strong_margin "
                         "from the eligible pool (STaR-balanced).")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = LoopConfig(task=args.task, run_name=args.run_name)
    if args.mode == "single":
        run(cfg, args.iter, tau=args.tau_strong)
    elif args.mode == "predict":
        if args.judge is None:
            raise SystemExit("--judge weak|strong required when --mode=predict")
        run_predict(cfg, args.iter, args.judge)
    elif args.mode == "combine":
        run_combine(
            cfg, args.iter,
            tau_strong=args.tau_strong, tau_weak=args.tau_weak,
            balanced_k_per_label=args.balanced_k_per_label,
        )
