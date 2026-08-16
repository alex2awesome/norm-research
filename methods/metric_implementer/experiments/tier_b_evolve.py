"""Tier-B seeded-delta evolution (plan v3/v4; prereg: primary endpoint = DELTA from the
definition seed on human mention labels; Delta_rec > Delta_critic >= 0).

Matched machinery, reward-only manipulation (mirrors metric_seam/roundtrip/run_gepa_program.py,
which uses the OFFICIAL gepa engine per standing directive):
  - candidate = {"rubric": <criterion text>}; seed = the metric's definition (name: description).
  - engine = gepa.optimize, same proposer/reflection LM (Codex gpt-5.6-sol via companion,
    shared by BOTH arms), same budget, same seed policy, same minibatch size.
  - ONLY the reward differs:
      reward=rec    : v2 TRUE THREE-HOP RECOVERY — encode candidate on probes, blind
                      decoder (qwen-72b) reconstructs the criterion from (text, score)
                      pairs, re-execute the reconstruction, reward = rank-agreement of
                      the ROUND-TRIP scores with M_i. Optimum is NOT the seed.
      reward=critic : rank-agreement with the frozen qwen-72b critic scores (one hop).
  - seam-h1 GATE: a candidate only SHIPS if it beats the seed on a held-out probe split
    under its own reward (train/holdout probe split; kills the collapse mode).
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


OPENROUTER_KEY_FILE = Path.home() / ".openrouter-api-key-2.txt"

DECODE_TMPL = """You will see excerpts from {n} documents, each with a numeric score in
[0,1] assigned by an evaluator following ONE consistent hidden criterion. Infer the
criterion.

{pairs}

Reply with ONLY your best reconstruction of the criterion as a single self-contained
instruction (1-3 sentences)."""


def qwen_decode(pairs_txt, n):
    import urllib.request
    key = open(OPENROUTER_KEY_FILE).read().strip()
    body = {"model": "qwen/qwen-2.5-72b-instruct", "max_tokens": 200, "temperature": 0.0,
            "messages": [{"role": "user",
                          "content": DECODE_TMPL.format(n=n, pairs=pairs_txt)}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception:
            import time
            time.sleep(4)
    return None


class RubricArmAdapter:
    """gepa GEPAAdapter over rubric text; scores via a remote vLLM OpenAI server
    (--api-base), so the driver can run where the Codex reflection CLI lives."""

    def __init__(self, api_base, model, probe_texts, target):
        self.api, self.model, self.texts, self.target = api_base, model, probe_texts, target

    def _one(self, msg):
        import urllib.request
        body = {"model": self.model, "max_tokens": 1, "temperature": 0.0,
                "logprobs": True, "top_logprobs": 10,
                "messages": [{"role": "user", "content": msg}]}
        req = urllib.request.Request(f"{self.api}/chat/completions",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer x"})
        for _ in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    obj = json.loads(r.read())
                tls = obj["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                lp = {t["token"].strip().upper(): math.exp(t["logprob"]) for t in tls}
                y, n = lp.get("YES", 0.0), lp.get("NO", 0.0)
                return y / (y + n) if (y + n) > 0 else float("nan")
            except Exception:
                import time
                time.sleep(3)
        return float("nan")

    def _score(self, rubric, ids):
        from concurrent.futures import ThreadPoolExecutor
        msgs = [TMPL.format(text=self.texts[d][:6000], rubric=rubric) for d in ids]
        with ThreadPoolExecutor(8) as ex:
            return np.array(list(ex.map(self._one, msgs)))

    _hop_cache = {}

    def three_hop(self, rubric, ids):
        """candidate -> encode -> blind decode -> re-execute; cached per rubric."""
        key = hash(rubric)
        if key in self._hop_cache:
            hat_scores = self._hop_cache[key]
        else:
            enc = self._score(rubric, ids)
            fin = np.where(np.isfinite(enc))[0]
            if len(fin) < 20:
                self._hop_cache[key] = None
                return None
            order = fin[np.argsort(enc[fin])]
            pick = list(order[:7]) + list(order[-7:])
            pairs = "\n\n".join(
                f"--- doc {j+1} (score {enc[i]:.2f}):\n{self.texts[ids[i]][:450]}"
                for j, i in enumerate(pick))
            hat = qwen_decode(pairs, len(pick))
            if not hat or not (20 < len(hat) < 1500):
                self._hop_cache[key] = None
                return None
            hat_scores = self._score(hat, ids)
            self._hop_cache[key] = hat_scores
        return hat_scores

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        if getattr(self, "reward_mode", "onehop") == "threehop":
            hs = self.three_hop(candidate["rubric"], self.train_ids)
            if hs is None:
                return EvaluationBatch(outputs=[None] * len(batch),
                                       scores=[0.0] * len(batch),
                                       trajectories=None if not capture_traces
                                       else [{"id": b, "got": float("nan"),
                                              "want": float("nan"), "rank_err": 1.0}
                                             for b in batch])
            pos = {d: i for i, d in enumerate(self.train_ids)}
            got = np.array([hs[pos[d]] if d in pos else np.nan for d in batch])
        else:
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
    ap.add_argument("--api-base", required=True, help="vLLM OpenAI base, e.g. http://sk1:8220/v1")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--rung", default="llama8b")
    ap.add_argument("--out-root", default=str(MD / "tier_b"))
    args = ap.parse_args()

    import gepa

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
        tr = [d for d in ids if d not in set(val)][:150]     # cost cap, fixed a priori
        logp = out_root / f"{args.rung}_{args.task}_{mid}_{args.reward}.json"
        if logp.exists():
            print(f"SKIP {mid} (done)")
            continue
        adapter = RubricArmAdapter(args.api_base, args.model, texts, tgt)
        adapter.train_ids = tr
        adapter.val_ids = val
        adapter.reward_mode = "threehop" if args.reward == "rec" else "onehop"
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
        # seam-h1 HOLDOUT GATE: best ships only if it beats the seed on VAL probes
        # under its own reward, computed on ids never used during evolution.
        def val_reward(rubric):
            if args.reward == "rec":
                hs = adapter.three_hop(rubric, val)
                if hs is None:
                    return -2.0
                want = np.array([tgt[d] for d in val])
                okm = np.isfinite(hs) & np.isfinite(want)
                if okm.sum() < 15 or hs[okm].std() == 0:
                    return -2.0
                ra = np.argsort(np.argsort(hs[okm]))
                rw = np.argsort(np.argsort(want[okm]))
                return float(np.corrcoef(ra, rw)[0, 1])
            got = adapter._score(rubric, val)
            want = np.array([tgt[d] for d in val])
            okm = np.isfinite(got) & np.isfinite(want)
            if okm.sum() < 15 or got[okm].std() == 0:
                return -2.0
            ra = np.argsort(np.argsort(got[okm]))
            rw = np.argsort(np.argsort(want[okm]))
            return float(np.corrcoef(ra, rw)[0, 1])

        gated = best.strip() != seed.strip()
        v_seed = v_best = None
        if gated:
            v_seed = val_reward(seed)
            v_best = val_reward(best)
            if v_best <= v_seed:
                gated = False                      # gate closes: ship the seed
        shipped = best if gated else seed
        json.dump({"seed": seed, "evolved": best, "shipped": shipped,
                   "reward": args.reward, "rung": args.rung,
                   "changed": shipped.strip() != seed.strip(),
                   "gate": {"val_seed": v_seed, "val_best": v_best,
                            "passed": bool(gated)}},
                  open(logp, "w"), indent=1)
        print(f"  evolved ({'SHIPPED-change' if gated else 'gated-to-seed'})")


if __name__ == "__main__":
    main()
