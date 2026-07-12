"""Driver for the granularity search (unit_refine) on real R3 checkpoints — pilot (task #128).

For each metric x rung: pool = cert head_selected ∪ top-K criteria by |corr with M_i| from the aligned
checkpoint (SAME criterion texts across rungs — fixed support); refine() proposes splits (deterministic
clause split) and merges (conjunctive join), scoring new units live with the rung's executor over the
SAME 300 probes as the checkpoint. Outputs per-metric ledger + granularity curve.

Usage (one rung per invocation; chain rungs):
  python -m methods.metric_implementer.experiments.refine_driver \
    --ckpt-dir outputs/r3_cw/aligned_8b_orbit_v2 --cert outputs/r3_cw/_log/cert_8b_v2.json \
    --target-model meta-llama/Llama-3.1-8B-Instruct --task creative-writing \
    --metrics "Protagonist agency" "Core conflict" "Sustained tension" "Prose‑medium" \
              "Dialogue craft" "Diction and figurative" "Scene–summary" "Worldbuilding" \
    --out outputs/unit_refine/cw_8b.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from . import alpha_probe as ap
from . import unit_certificate as uc
from .unit_refine import Unit, refine
from .run_real_test import _load_texts
from .. import config as cfgmod


def _splitter(text: str):
    nodes = [n for n in uc.address_lattice(text, depth=2) if n.level == 2]
    if len(nodes) >= 2:
        return nodes[0].span, nodes[1].span
    words = text.split()
    if len(words) >= 12:                                       # fallback: midpoint split
        h = len(words) // 2
        return " ".join(words[:h]), " ".join(words[h:])
    return None


def _merger(a: str, b: str) -> str:
    return f"BOTH of the following hold together: (i) {a.rstrip('. ')}; and (ii) {b.rstrip('. ')}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--cert", required=True)
    p.add_argument("--target-model", required=True)
    p.add_argument("--task", default="creative-writing")
    p.add_argument("--metrics", nargs="+", required=True, help="name substrings")
    p.add_argument("--pool-size", type=int, default=30)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--eps-accept", type=float, default=0.01)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    cert = {r["name"]: r for r in json.load(open(a.cert))}
    ckpts = {}
    for f in glob.glob(os.path.join(a.ckpt_dir, "*_sigs.npz")):
        z = np.load(f, allow_pickle=True)
        ckpts[str(z["name"])] = f

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    from ..vllm_backend import make_judge_backend
    ex = make_judge_backend(a.target_model, cfgmod.ImplementerConfig(), temperature=None)
    texts, _ = _load_texts(a.task, 60 + a.n_probes, cfg)
    probes = texts[60: 60 + a.n_probes]
    print(f"[refine] probes={len(probes)} executor={a.target_model}", flush=True)

    def score_fn(new_texts):
        return np.vstack([ap.signature(ex, t, probes, cfg.max_text_chars,
                                       template=ap._YESNO_TEXTFIRST) for t in new_texts])

    results = []
    for pat in a.metrics:
        name = next((n for n in ckpts if pat.lower() in n.lower()), None)
        if name is None:
            print(f"[refine] NO MATCH for pattern {pat!r}", flush=True)
            continue
        z = np.load(ckpts[name], allow_pickle=True)
        S = np.asarray(z["sigs"], float)
        prompts = [str(x) for x in z["prompts"]]
        m = np.asarray(z["M_i"], float)
        n = min(len(m), len(probes))
        S, m = S[:, :n], m[:n]
        # pool: cert head ∪ top-|corr| criteria — selected on EVEN probes ONLY (refine's search set),
        # so pool selection cannot leak into the odd-probe report set (winner's-curse discipline).
        even = np.arange(0, n, 2)
        head = list(cert.get(name, {}).get("head_selected") or [])
        sd = S[:, even].std(1)
        cors = np.zeros(len(prompts))
        ok = sd > 1e-9
        cors[ok] = np.abs([np.corrcoef(S[i][even], m[even])[0, 1] for i in np.where(ok)[0]])
        order = list(np.argsort(-cors))
        pool_idx = list(dict.fromkeys([i for i in head if i < len(prompts)] + order))[: a.pool_size]
        units = [Unit(prompts[i], S[i]) for i in pool_idx]
        host_words = sum(u.span_words for u in units)
        res = refine(units, m, score_fn, _splitter, _merger, rounds=a.rounds,
                     eps_accept=a.eps_accept, host_words=host_words)
        out = {"metric": name, "n_pool": len(units),
               "opt_init": res["opt_heldout_init"], "opt_final": res["opt_heldout_final"],
               "n_merges": res["n_merges"], "n_splits": res["n_splits"],
               "ledger": res["ledger"], "opt_curve": res["opt_curve"]}
        results.append(out)
        json.dump(results, open(a.out, "w"), indent=1, default=float)
        print(f"[refine] {name[:48]:48s} init={out['opt_init']:.3f} final={out['opt_final']:.3f} "
              f"merges={out['n_merges']} splits={out['n_splits']}", flush=True)
    print(f"[refine] DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
