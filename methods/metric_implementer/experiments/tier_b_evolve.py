"""Tier-B seeded-delta evolution (plan v3/v4; prereg: primary endpoint = DELTA from the
definition seed on human mention labels; Delta_rec > Delta_critic >= 0).

Matched machinery, reward-only manipulation (mirrors metric_seam/roundtrip/run_gepa_program.py,
which uses the OFFICIAL gepa engine per standing directive):
  - candidate = {"rubric": <criterion text>}; seed = the metric's definition (name: description).
  - engine = gepa.optimize, same proposer/reflection LM (Codex gpt-5.6-sol via companion,
    shared by BOTH arms), same budget, same seed policy, same minibatch size.
  - ONLY the reward differs:
      reward=rec    : per-instance rank-agreement of the candidate-rubric scores with M_i
                      (the metric's own definition-verdict by the SAME executor) on probe docs.
      reward=critic : same, target = the frozen qwen-72b critic scores on probe docs.
  - executor = one resident offline vLLM engine per run (rung model passed via --model);
    probes only — mention-y and eval corpora NEVER touched during evolution.
Eval (separate script, after evolution): score evolved rubrics + seed on the corpus with
the same executor; Delta mention-AUC vs seed, paired bootstrap on mean delta + sign test.

Usage: tier_b_evolve.py --task cw --metrics a84,a247 --reward rec --model <dir> \
       --budget 200 [--out-root .../tier_b] [--gpu 0]
Metric list comes from the moderator-flagged + random-control selection (frozen upstream).
"""
import argparse
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

MD = Path(os.environ.get("OCL_DATA", "/lfs/skampere1/0/alexspan/ocl"))
CODEX = Path.home() / ".claude/plugins/cache/openai-codex/codex/1.0.5/scripts/codex-companion.mjs"

TMPL = ("Text:\n{text}\n\nYou are evaluating the above text on ONE specific criterion.\n"
        "Criterion:\n{rubric}\n\nDoes the text satisfy the criterion? "
        "Answer with exactly one word: YES or NO.")


def codex_lm(prompt: str) -> str:
    r = subprocess.run(["node", str(CODEX), "task", prompt, "--fresh"],
                       timeout=900, capture_output=True, text=True)
    return r.stdout or ""


class RubricArmAdapter:
    """gepa GEPAAdapter over rubric text; scores via a resident offline vLLM engine."""

    def __init__(self, llm, tok, probe_texts, target, sampling):
        self.llm, self.tok, self.texts, self.target, self.sampling = \
            llm, tok, probe_texts, target, sampling

    def _score(self, rubric, ids):
        prompts = []
        for d in ids:
            msg = TMPL.format(text=self.texts[d][:6000], rubric=rubric)
            try:
                prompts.append(self.tok.apply_chat_template(
                    [{"role": "user", "content": msg}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                prompts.append(self.tok.apply_chat_template(
                    [{"role": "user", "content": msg}], tokenize=False,
                    add_generation_prompt=True))
        outs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        scores = []
        for o in outs:
            lp = {t.decoded_token.strip().upper(): math.exp(t.logprob)
                  for t in (o.outputs[0].logprobs[0].values() if o.outputs[0].logprobs else [])}
            y, n = lp.get("YES", 0.0), lp.get("NO", 0.0)
            scores.append(y / (y + n) if (y + n) > 0 else np.nan)
        return np.array(scores)

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        got = self._score(candidate["rubric"], batch)
        want = np.array([self.target[d] for d in batch])
        ok = np.isfinite(got) & np.isfinite(want)
        scores = [0.0] * len(batch)
        trajs = [None] * len(batch)
        if ok.sum() >= 2 and got[ok].std() > 0:
            rg = np.argsort(np.argsort(got[ok]))
            rw = np.argsort(np.argsort(want[ok]))
            n = int(ok.sum())
            for j, i in enumerate(np.where(ok)[0]):
                scores[i] = 1.0 - abs(int(rg[j]) - int(rw[j])) / max(1, n - 1)
                if capture_traces:
                    trajs[i] = {"id": batch[i], "got": float(got[i]),
                                "want": float(want[i]),
                                "rank_err": abs(int(rg[j]) - int(rw[j])) / max(1, n - 1)}
        return EvaluationBatch(outputs=list(got), scores=scores,
                               trajectories=trajs if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        rows = []
        tr = [t for t in (eval_batch.trajectories or []) if t]
        tr.sort(key=lambda t: -t["rank_err"])
        for t in tr[:6]:
            rows.append({"Inputs": self.texts[t["id"]][:600],
                         "Generated Outputs": f"score {t['got']:.2f}",
                         "Feedback": f"target gave {t['want']:.2f}; rank disagreement "
                                     f"{t['rank_err']:.2f} — rewrite the criterion so this "
                                     f"document ranks correctly"})
        return {"rubric": rows}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        rows = reflective_dataset["rubric"]
        fb = "\n\n".join(f"--- document (excerpt):\n{r['Inputs']}\nyour criterion scored: "
                         f"{r['Generated Outputs']}\nfeedback: {r['Feedback']}" for r in rows)
        prompt = ("Improve this evaluation criterion so that judging documents with it "
                  "better matches a target ranking. Keep it a single self-contained "
                  "criterion statement (1-4 sentences), concrete and checkable. Do not "
                  "mention the target.\n\nCurrent criterion:\n" + candidate["rubric"] +
                  "\n\nWorst disagreements:\n" + fb +
                  "\n\nReply with ONLY the improved criterion text.")
        txt = codex_lm(prompt).strip()
        txt = re.sub(r"^```.*?\n|```$", "", txt, flags=re.S).strip()
        return {"rubric": txt if 20 < len(txt) < 2000 else candidate["rubric"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--metrics", required=True, help="comma-separated aN ids")
    ap.add_argument("--reward", required=True, choices=("rec", "critic"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--rung", default="llama8b")
    ap.add_argument("--out-root", default=str(MD / "tier_b"))
    args = ap.parse_args()

    import gepa
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    texts = {}
    tf, key = {"peer": ("peer_probe_texts.jsonl", "probe_id"),
               "cw": ("cw_probe_texts.jsonl", "probe_id"),
               "pr": ("pr_probe_texts.jsonl", "probe_id"),
               "humor": ("humor_probe_texts.jsonl", "probe_id")}[args.task]
    for line in open(MD / tf):
        r = json.loads(line)
        texts[r[key]] = r["text"]
    probes = json.load(open(MD / f"ocl_{args.rung}_{args.task}_probes.json"))
    pids = probes["post_ids"]
    crit = defaultdict(dict)
    cf = MD / "critic_all_results.jsonl"
    if cf.exists():
        for line in open(cf):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("score") is not None:
                crit[r["aspect_id"]][r["datapoint_id"]] = float(r["score"])
    man = {e["metric_id"]: e["rubric"] for e in
           json.load(open(MD / f"ocl_{args.task}_manifest.json")) if e["form_idx"] == -1}

    llm = LLM(model=args.model,
              gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.55")),
              max_model_len=8192)
    tok = AutoTokenizer.from_pretrained(args.model)
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=8)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for mid in args.metrics.split(","):
        seed = man.get(mid)
        if not seed:
            print(f"{mid}: no definition rubric, skipped")
            continue
        if args.reward == "rec":
            tgt = {d: float(v) for d, v in
                   zip(pids, probes["scores"].get(f"{mid}__-1", []))}
        else:
            tgt = crit.get(mid, {})
        ids = [d for d in pids if d in texts and np.isfinite(tgt.get(d, np.nan))]
        if len(ids) < 80:
            print(f"{mid}: target coverage too thin ({len(ids)}), skipped")
            continue
        val = ids[::4]
        tr = [d for d in ids if d not in set(val)]
        logp = out_root / f"{args.rung}_{args.task}_{mid}_{args.reward}.json"
        if logp.exists():
            print(f"SKIP {mid} (done)")
            continue
        adapter = RubricArmAdapter(llm, tok, texts, tgt, sampling)
        print(f"=== {args.task}/{mid} reward={args.reward} budget={args.budget} ===",
              flush=True)
        try:
            res = gepa.optimize(seed_candidate={"rubric": seed}, trainset=tr, valset=val,
                                adapter=adapter, reflection_minibatch_size=8,
                                max_metric_calls=args.budget)
            best = res.best_candidate["rubric"]
        except Exception as e:
            json.dump({"error": str(e)[:400]}, open(logp, "w"))
            print(f"  ERROR {str(e)[:120]}")
            continue
        json.dump({"seed": seed, "evolved": best, "reward": args.reward,
                   "rung": args.rung, "changed": best.strip() != seed.strip()},
                  open(logp, "w"), indent=1)
        print(f"  evolved ({'changed' if best.strip() != seed.strip() else 'UNCHANGED'})")


if __name__ == "__main__":
    main()
