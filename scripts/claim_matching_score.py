#!/usr/bin/env python3
"""Score the claim-matching testbed with each bank metric, via one model of the scaling ladder.

For metric M and a testbed pair (claim-element E, reference-span S): a model applies the ARTICULATED
criterion M and rates how strongly S matches/supports/discloses E under that criterion (graded 0-4).
Reconstruction-only: the model NEVER sees the examiner-citation label Y; Y enters only at the
recovery readout. Run once per ladder model (gemma-3-4b/12b/27b, gemma-4-31b, ...); a separate
recovery stage fits the articulability curve across models.

BEST-PRACTICES applied: graded score for AUC resolution; free-form + parse + retry (no structured
decoding); per-metric constancy check (std / value-counts / yes-rate) printed and flagged; text-first
ordering; coverage printed. Offline vLLM batch (thousands of prompts/call).

  CUDA_VISIBLE_DEVICES=N python scripts/claim_matching_score.py --model <path> --tag gemma3_27b
"""
import argparse, json, re, hashlib, os, collections
import numpy as np

BASE = "/lfs/skampere3/0/alexspan/norm-research"
BANK = f"{BASE}/datasets/claim-matching/claim_matching_bank.jsonl"
# v2 = multiple-gold fix (examiner-cited refs never drawn as negatives). Ladder runs before
# 2026-07-10 (gemma3_4b/12b/27b) scored v1; their 59 changed probe negs were rescored separately
# (scores_gemma3_12b_v2negs.jsonl) and overlaid at recovery via --patch.
TESTBED = f"{BASE}/datasets/claim-matching/testbed/pair_testbed_v2.jsonl"
OUTDIR = f"{BASE}/outputs/claim_matching"

SYS = ("You compare a CLAIM to a REFERENCE passage using ONE specific matching CRITERION. "
       "Judge only how well the reference satisfies that criterion for that claim — not overall "
       "quality. Be strict: surface word-overlap is not a match; the reference must actually "
       "satisfy the criterion in substance.")

_SCORE = re.compile(r'"?(?:match|score|strength)"?\s*[:=]\s*([0-4])')
_BARE = re.compile(r"\b([0-4])\b")


def parse_score(raw):
    m = _SCORE.search(raw or "")
    if m:
        return int(m.group(1))
    # fallback: first bare 0-4 in a short answer
    m = _BARE.search((raw or "")[:60])
    return int(m.group(1)) if m else None


def probe_pairs(n_claims):
    """Balanced, stable probe: n_claims claims (each contributes its pos+neg pair), by uid hash."""
    byuid = collections.defaultdict(list)
    for ln in open(TESTBED):
        r = json.loads(ln)
        byuid[r["uid"]].append(r)
    uids = [u for u, v in byuid.items() if len(v) == 2 and {x["y"] for x in v} == {0, 1}]
    uids.sort(key=lambda u: hashlib.md5(f"probe::{u}".encode()).hexdigest())
    uids = uids[:n_claims]
    pairs = []
    for u in uids:
        pairs.extend(byuid[u])
    return pairs


def build_prompt(metric, e, s):
    crit = f"CRITERION: {metric['name']}\n{metric['description']}"
    if metric.get("guidance"):
        crit += f"\nGuidance: {metric['guidance'][:400]}"
    return (f"{crit}\n\nCLAIM (a patent claim element):\n{e[:600]}\n\n"
            f"REFERENCE passage (candidate prior art):\n{s[:1200]}\n\n"
            "Under THIS criterion only, rate how strongly the reference matches/supports/discloses "
            "the claim:\n0 = not at all, 1 = barely, 2 = partial, 3 = substantial, 4 = fully.\n"
            'Reply ONE JSON object: {"match": 0-4, "reason": "<=12 words"}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-claims", type=int, default=1500, dest="n_claims")
    ap.add_argument("--max-metrics", type=int, default=0, dest="max_metrics")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90, dest="gpu_mem_util",
                    help="lower (e.g. 0.5) to co-locate on a GPU that already has a resident job")
    ap.add_argument("--bank-file", default=BANK, dest="bank_file",
                    help="score a different metric set (e.g. discovered candidates) on the same probe")
    ap.add_argument("--pairs-file", default=None, dest="pairs_file",
                    help="score exactly these pair rows (jsonl, testbed schema) instead of the probe")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    bank = [json.loads(l) for l in open(a.bank_file)]
    if a.max_metrics:
        bank = bank[:a.max_metrics]
    if a.pairs_file:
        pairs = [json.loads(l) for l in open(a.pairs_file)]
    else:
        pairs = probe_pairs(a.n_claims)
    print(f"[score:{a.tag}] {len(bank)} metrics x {len(pairs)} pairs = {len(bank)*len(pairs)} judgments",
          flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=4096, enable_prefix_caching=True, trust_remote_code=True, max_num_seqs=512)

    jobs = [(mi, pi) for mi in range(len(bank)) for pi in range(len(pairs))]
    convs = [[{"role": "system", "content": SYS},
              {"role": "user", "content": build_prompt(bank[mi], pairs[pi]["element"], pairs[pi]["span"])}]
             for mi, pi in jobs]
    scores = [None] * len(jobs)
    for seed, temp in ((0, 0.0), (1, 0.5)):
        todo = [k for k in range(len(jobs)) if scores[k] is None]
        if not todo:
            break
        sp = SamplingParams(temperature=temp, max_tokens=40, seed=seed)
        outs = llm.chat([convs[k] for k in todo], sp)
        for k, o in zip(todo, outs):
            scores[k] = parse_score(o.outputs[0].text)
        print(f"[score:{a.tag}] seed {seed}: {sum(s is not None for s in scores)}/{len(jobs)} parsed",
              flush=True)

    # write + per-metric constancy check
    outp = f"{OUTDIR}/scores_{a.tag}.jsonl"
    permetric = collections.defaultdict(list)
    with open(outp, "w") as fh:
        for (mi, pi), sc in zip(jobs, scores):
            m, p = bank[mi], pairs[pi]
            fh.write(json.dumps({"metric_id": m["metric_id"], "domain": m["domain"],
                                 "uid": p["uid"], "y": p["y"], "score": sc}) + "\n")
            permetric[m["metric_id"]].append((sc, p["y"]))
    print(f"[score:{a.tag}] wrote {outp}", flush=True)

    # constancy flags (BEST-PRACTICES: reject std<0.4 or coverage<0.7)
    n_flag = 0
    for mid, vals in permetric.items():
        s = [v for v, _ in vals if v is not None]
        cov = len(s) / len(vals)
        std = float(np.std(s)) if s else 0.0
        yes = float(np.mean([x >= 2 for x in s])) if s else 0.0
        if std < 0.4 or cov < 0.7:
            n_flag += 1
            print(f"  [CONSTANT?] {mid} std={std:.2f} cov={cov:.2f} yesrate={yes:.2f}", flush=True)
    print(f"[score:{a.tag}] {n_flag}/{len(permetric)} metrics flagged constant/low-coverage",
          flush=True)
    print(f"SCORE_{a.tag}_DONE", flush=True)


if __name__ == "__main__":
    main()
