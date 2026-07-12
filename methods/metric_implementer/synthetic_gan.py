"""GAN-style synthetic data + detectability-GEPA (goal part 3, 2026-06-14). Open models, sk3-only.

A strong open model GENERATES task text that DOES (polarity=+1) or DOES NOT (-1) exhibit a target
metric -> synthetic examples with KNOWN metric labels (for E0-style known-answer validation of a
metric: can a judge+rubric recover the planted label?). A DETECTOR judge predicts AI-vs-human
(P(AI) via vllm_backend.score_binary). We GEPA-iterate the GENERATOR prompt to MINIMIZE
detectability (make synth indistinguishable from real human text) — adversarial, generator vs
detector. Output: realistic labeled synth + the detectability (AUC) trajectory + the best prompt.

ONE resident model serves all three roles (generator / detector / reviser) -> same _engine cache
entry -> ONE GPU ([[feedback_gpu_usage]]). Per-sample prompt variation gives diversity despite
vLLM's fixed-seed decoding.
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional

from .backends import parse_json_obj
from .vllm_backend import make_judge_backend

_GEN_SEED = (
    "Write a single realistic {task_noun} that a real person actually wrote.\n"
    "It must {polarity_clause} this property:\n{metric}\n\n"
    "Write ONLY the {task_noun} itself (no preamble, no quotes), about {nwords} words.\n"
    "Variation #{i}."
)

_DETECT_Q = (
    "Below is a {task_noun}. Was it written by an AI language model rather than by a human?\n\n"
    "{text}\n\nAnswer with one word: YES (AI-written) or NO (human-written)."
)

_REVISE = (
    "You write the INSTRUCTIONS that tell an LLM to generate {task_noun} text. A detector is "
    "catching the generated text as AI-written. Samples it flagged most confidently as AI:\n\n"
    "{flagged}\n\nCurrent instructions:\n\"\"\"\n{gen_prompt}\n\"\"\"\n\n"
    "Rewrite the instructions so the generated {task_noun} is INDISTINGUISHABLE from genuine human "
    "writing (natural voice, concrete specifics, mild imperfection) while STILL {polarity_clause} "
    "this property:\n{metric}\n\nReturn JSON: {{\"instructions\": \"<the rewritten instructions, "
    "keeping the {{task_noun}}/{{polarity_clause}}/{{metric}}/{{nwords}}/{{i}} placeholders>\"}}"
)


def _auc(scores: List[float], labels: List[int]) -> float:
    """Rank-based AUC; label 1 = synth (AI-true), 0 = real (human). 0.5 = indistinguishable."""
    pairs = [(s, y) for s, y in zip(scores, labels) if s == s]   # drop nan
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _polarity_clause(polarity: int) -> str:
    return "clearly exhibit" if polarity > 0 else "clearly NOT exhibit"


def generate_synth(gen, task_noun: str, metric: str, polarity: int, gen_prompt: str,
                   n: int, nwords: int = 200) -> List[str]:
    pc = _polarity_clause(polarity)
    prompts = [gen_prompt.format(task_noun=task_noun, polarity_clause=pc, metric=metric,
                                 nwords=nwords, i=i) for i in range(n)]
    return gen.generate_batch(prompts, max_tokens=int(nwords * 2.2), temperature=0.95)


def detect_pai(det, task_noun: str, texts: List[str]) -> List[float]:
    qs = [_DETECT_Q.format(task_noun=task_noun, text=t[:3000]) for t in texts]
    return det.score_binary(qs, pos="YES", neg="NO")


def gan_loop(task_noun: str, metric: str, real_texts: List[str], *, model: str, cfg,
             rounds: int = 3, n: int = 20, log=print) -> dict:
    """One model, three roles. Returns best (lowest-detectability) gen prompt + labeled synth +
    the AUC trajectory."""
    gen = make_judge_backend(model, cfg, 0.95)
    det = make_judge_backend(model, cfg, 0.0)     # same _engine; logprob detector
    rev = make_judge_backend(model, cfg, 0.7)
    real = real_texts[:n]
    gen_prompt, traj, best = _GEN_SEED, [], None
    for r in range(rounds):
        synth = generate_synth(gen, task_noun, metric, +1, gen_prompt, n)
        synth = [s for s in synth if s and len(s.strip()) > 40] or synth
        texts = synth + real
        labels = [1] * len(synth) + [0] * len(real)
        pai = detect_pai(det, task_noun, texts)
        auc = _auc(pai, labels)
        synth_pai = pai[:len(synth)]
        traj.append({"round": r, "detect_auc": round(auc, 3) if auc == auc else None,
                     "n_synth": len(synth), "mean_pAI_synth": round(
                         sum(x for x in synth_pai if x == x) / max(1, sum(x == x for x in synth_pai)), 3)})
        log(f"[GAN {task_noun}] round {r}: detect_auc={auc:.3f} (0.5=fooled)")
        if best is None or (auc == auc and auc < best["auc"]):
            best = {"auc": auc, "gen_prompt": gen_prompt, "synth": synth}
        # reflect on the most-detected synth and rewrite the generator instructions
        order = sorted(range(len(synth)), key=lambda i: -(synth_pai[i] if synth_pai[i] == synth_pai[i] else 0))
        flagged = "\n---\n".join(synth[i][:500] for i in order[:3])
        out = rev.generate(_REVISE.format(task_noun=task_noun, flagged=flagged,
                                          gen_prompt=gen_prompt, polarity_clause=_polarity_clause(1),
                                          metric=metric), max_tokens=600)
        obj = parse_json_obj(out)
        if obj and isinstance(obj.get("instructions"), str) and "{metric}" in obj["instructions"]:
            gen_prompt = obj["instructions"]
    # build the labeled known-answer set from the best round (both polarities)
    pos = best["synth"]
    neg = generate_synth(gen, task_noun, metric, -1, best["gen_prompt"], n)
    labeled = [{"text": t, "metric_label": 1} for t in pos] + \
              [{"text": t, "metric_label": 0} for t in neg if t and len(t.strip()) > 40]
    return {"task_noun": task_noun, "metric": metric[:200], "best_detect_auc": best["auc"],
            "trajectory": traj, "best_gen_prompt": best["gen_prompt"], "labeled_synth": labeled}


def main(argv=None) -> int:
    from .config import ImplementerConfig, apply_task_preset
    from .manifest import full_manifest, load_corpus, load_metrics
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    ap.add_argument("--task", default="math")            # which dataset entry (by name substring)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="/lfs/skampere3/0/alexspan/norm-research/outputs/"
                    "metric_implementer_scale/gan")
    args = ap.parse_args(argv)
    m = full_manifest(metrics_per_task=3, metric_files_cap=20)
    entry = next((e for e in m.datasets if args.task in e.name or args.task in e.task), m.datasets[0])
    cfg = ImplementerConfig(); apply_task_preset(cfg, entry.task)
    metric = load_metrics(entry)[0]
    real_texts, _ = load_corpus(entry, args.n, seed=0)
    task_noun = getattr(cfg, "item_noun", entry.task.replace("-", " "))
    res = gan_loop(task_noun, metric.body, real_texts, model=args.model, cfg=cfg,
                   rounds=args.rounds, n=args.n)
    import os
    os.makedirs(args.out, exist_ok=True)
    p = f"{args.out}/gan_{entry.name}.json"
    json.dump(res, open(p, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "labeled_synth"}, indent=1))
    print("labeled synth:", len(res["labeled_synth"]), "-> wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
