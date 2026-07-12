"""Failure-informed GEPA on ONE CW R3 metric, for the prompt-optimality EXPLANATORY figure
(notebooks/2026-07-03__prompt-optimality-gepa-vs-ceiling.ipynb; theory: notes/2026-06-18__
prompt-optimality-theory.md §3 ladder, §6.5 Discovery-to-Selection, §12.6 certificate).

The figure's shared Shannon-bits axis:

  R(p_t) = I(M_i ; binarize(verdict of p_t))   over the SAME 300 probes the certificate used
    - lower bound  g1        = best single Ω criterion (certificate gains[0])
    - upper bound  OPT_Ω + ε = certified checklist ceiling (dotted)
    - info cap     T = H(M)  (verdict deterministic given X ⇒ T is the cap, R is what GEPA moves)

Design (v3 — after the v2 flat run, where seed=NAME scored 0.466b and every structured-checklist
revision scored WORSE, so acceptance kept the seed for 6 rounds):
  - GENERIC SEED: round 0 = "This is a good piece of creative writing." — near-zero R, so the
    figure shows a real CLIMB. The reviser knows the concept NAME (the metric's public identity)
    + failure exemplars; the jump name-knowledge buys is part of the story.
  - ACCEPTANCE: each round the GLM reviser proposes --candidates rubrics; the best-by-R is
    accepted only if it beats the incumbent (real GEPA keeps improving mutations; the earlier
    keep-every-round loop let permissive revisions drag R 0.466→0.03).
  - NO GENERATOR LEAK: the reviser never sees the merged_description (which generated M_i;
    showing it is tautological, R→H(M)). The description IS scored once as a diagnostic
    ("generator_R", the practical single-prompt max), saved to JSON, never fed back.
  - FREE FORM, SHORT: v2 showed the mandated checklist skeleton tanks this executor; v3 asks for
    a concise prose rubric (≤120 words) of short declarative criterion sentences.
  - DISCOVERY-TO-SELECTION: after the loop, ONE GLM call atomizes the accepted prompts into
    independent criteria; each is scored as its own criterion signature; greedy_head runs on the
    GEPA-only pool and on the union with the certificate's freegen pool → "the Ω units broken
    down from the final set of prompts", on the same axis.

Caveat stated in the notebook: acceptance selects on the same 300 probes (≈ rounds×candidates
selections), which can only inflate the trajectory — so "the plateau sits below the certified
ceiling" is conservative.

Run (sk3, 1 GPU; GLM ≈ rounds×candidates calls):
  CUDA_VISIBLE_DEVICES=1 HOME=/lfs/skampere3/0/alexspan VLLM_GPU_MEM_UTIL=0.9 \\
    python -m methods.metric_implementer.experiments.run_gepa_for_plot
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from ..config import ImplementerConfig, apply_task_preset
from ..backends import LLMBackend
from ..vllm_backend import make_judge_backend
from . import alpha_probe as aprobe
from .mine_clusters import r3_groups
from .run_real_test import _load_texts
from .value_census import i_binary
from .value_certificate import greedy_head

_SYS = ("You are an expert creative-writing workshop instructor. You write SHORT evaluation "
        "rubrics: a single prose paragraph that a fresh evaluator reads before answering YES or "
        "NO about a text. The evaluator is literal-minded, so the rubric must state plainly WHAT "
        "quality to look for and WHEN to say NO. Short declarative sentences. No lists, no "
        "headings, no meta-commentary. At most 120 words. Output ONLY the rubric paragraph.")

_USER = """We are refining a YES/NO evaluation rubric for the concept named: {name}

CURRENT RUBRIC:
\"\"\"
{cur}
\"\"\"

Calibration: the concept truly holds for about {target_rate:.0%} of texts, but the current rubric
answers YES on {cur_rate:.0%} — it is too {direction}.

The current rubric scores these excerpts WRONG:

Should be YES (the concept holds) but the rubric answers NO:
{pos}

Should be NO (the concept does not hold) but the rubric answers YES:
{neg}

Revision strategy for this attempt: {hint}

Rewrite the rubric as one prose paragraph, at most 120 words. Keep what works, fix the errors
above. Output ONLY the rubric."""

_HINTS = [
    "state the core quality more precisely, in terms the errors above suggest",
    "add one plain sentence saying when to answer NO, targeted at the false-YES excerpts",
    "make it shorter and more direct — cut anything a literal evaluator could misread",
]

_ATOMIZE = """Below is an evaluation rubric (possibly several versions) for judging creative-writing
excerpts. Break it into its INDEPENDENT atomic criteria: each a single self-contained YES/NO
question about a text, understandable with no other context. One per line, no numbering, no
commentary, 8-30 words each. Do not invent criteria that are not in the rubric.

RUBRIC(S):
{rubrics}"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\n", "", s)
    s = re.sub(r"\n```$", "", s)
    return s.strip()


def _excerpt(i, probes, maxlen=420):
    t = probes[i].replace("\n", " ").strip()
    return (t[:maxlen] + "…") if len(t) > maxlen else t


_GENERIC_SEED = "This is a good piece of creative writing."


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="creative-writing")
    p.add_argument("--bucket", default="general")
    p.add_argument("--gi", type=int, default=29)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--gepa-reserve", type=int, default=60)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--glm-model", default="glm-5")
    p.add_argument("--seed-mode", default="description", choices=["description", "name", "generic"],
                   help="GEPA starting prompt: the metric DESCRIPTION (natural GEPA seed, default), "
                        "the metric NAME, or a totally GENERIC quality sentence.")
    p.add_argument("--out", default="/lfs/skampere3/0/alexspan/tmp_vinfo/gepa_for_plot")
    p.add_argument("--npz", default=None)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = args.npz or (f"/lfs/skampere3/0/alexspan/outputs/r3_cw/aligned_8b_orbit_v2/"
                            f"creative-writing_R3_metric{args.gi}_sigs.npz")

    g = r3_groups(args.task, args.bucket)[args.gi]
    name, desc = g["merged_name"], g["merged_description"]
    metric_id = f"cw_R3_g{args.gi}"
    print(f"[plot-gepa] metric gi={args.gi}: {name}", flush=True)

    cfg = ImplementerConfig()
    apply_task_preset(cfg, args.task)
    cfg.n_oracle_items = 0
    all_texts, _ = _load_texts(args.task, args.gepa_reserve + args.n_probes, cfg)
    probes = all_texts[args.gepa_reserve: args.gepa_reserve + args.n_probes]
    assert len(probes) == args.n_probes, f"only {len(probes)} probes loaded"

    z = np.load(npz_path, allow_pickle=True)
    M = (np.asarray(z["M_i"], float) > 0.5).astype(int)
    assert len(M) == len(probes), f"M_i len {len(M)} != probes {len(probes)}"
    target_rate = float(M.mean())
    print(f"[plot-gepa] probes={len(probes)} M_i mean={target_rate:.3f}", flush=True)

    executor = make_judge_backend(args.model, cfg, 0.0)
    max_chars = getattr(cfg, "max_text_chars", 4000)
    gcfg = ImplementerConfig()
    gcfg.backend = "zai_anthropic"
    gcfg.other_temperature = 0.8
    gcfg.request_timeout_s = 120
    glm = LLMBackend(model=args.glm_model, role="reviser", cfg=gcfg)

    def score_R(prompt_text):
        pyes = aprobe.signature(executor, prompt_text, probes, max_chars, template=None)
        v = (np.nan_to_num(np.asarray(pyes, float), nan=0.5) > 0.5).astype(int)
        return float(i_binary(M, v)), v

    # Reference rungs (diagnostics for the figure): the two human-written single-prompt rungs.
    #   name_R = the metric NAME as prompt;  desc_R = the metric DESCRIPTION as prompt.
    name_R, _ = score_R(name)
    desc_R, _ = score_R(desc)
    print(f"[plot-gepa] rungs: name_R={name_R:.3f}b desc_R={desc_R:.3f}b", flush=True)

    # SEED = the natural GEPA starting point: the metric DESCRIPTION (the human-written definition of
    # the metric), NOT a content-free generic sentence. GEPA then tries to sharpen that definition
    # into a rubric that recovers M_i better. (Earlier 'generic' seed was an artifact: it did not look
    # like a metric, hid where the concept enters, and never even reached the name rung.)
    seed_text = {"description": desc, "name": name, "generic": _GENERIC_SEED}[args.seed_mode]
    traj = []
    R0, v0 = score_R(seed_text)
    traj.append({"round": 0, "operator": "INIT", "prompt": seed_text, "R_bits": R0,
                 "n_yes": int(v0.sum()), "accepted": True, "candidates": [],
                 "seed_mode": args.seed_mode})
    print(f"  r0 INIT({args.seed_mode}) R={R0:.3f}b nyes={int(v0.sum())}/{len(probes)} "
          f"disagree={int((v0 != M).sum())}", flush=True)

    cur, vcur, Rcur = _GENERIC_SEED, v0, R0
    t0 = time.time()
    rng = np.random.default_rng(0)
    for rnd in range(1, args.rounds + 1):
        disagree = np.where(vcur != M)[0]
        pos_all = [int(i) for i in disagree if M[i] == 1]
        neg_all = [int(i) for i in disagree if M[i] == 0]
        pos = list(rng.permutation(pos_all)[:4])
        neg = list(rng.permutation(neg_all)[:4])
        pos_blk = "\n".join(f"- {_excerpt(i, probes)}" for i in pos) or "(none this round)"
        neg_blk = "\n".join(f"- {_excerpt(i, probes)}" for i in neg) or "(none this round)"
        cur_rate = float(vcur.mean())
        direction = "permissive (says YES too often)" if cur_rate > target_rate else \
            "strict (says NO too often)"
        cands = []
        for ci in range(args.candidates):
            user = _USER.format(name=name, cur=cur, target_rate=target_rate, cur_rate=cur_rate,
                                direction=direction, pos=pos_blk, neg=neg_blk,
                                hint=_HINTS[ci % len(_HINTS)])
            new = None
            for attempt in range(2):
                try:
                    new = glm.generate(user, system=_SYS, max_tokens=700, temperature=0.8)
                    break
                except Exception as e:
                    print(f"  r{rnd}c{ci} GLM attempt {attempt} failed: "
                          f"{type(e).__name__}: {str(e)[:120]}", flush=True)
            if not new:
                continue
            new = _strip_fences(new)
            Rn, vn = score_R(new)
            cands.append({"prompt": new, "R_bits": Rn, "n_yes": int(vn.sum()),
                          "hint": _HINTS[ci % len(_HINTS)], "_v": vn})
            print(f"  r{rnd}c{ci} R={Rn:.3f}b nyes={int(vn.sum())}/{len(probes)} "
                  f"words~{len(new.split())}", flush=True)
        best = max(cands, key=lambda c: c["R_bits"]) if cands else None
        accepted = bool(best and best["R_bits"] > Rcur)
        if accepted:
            cur, vcur, Rcur = best["prompt"], best["_v"], best["R_bits"]
        traj.append({"round": rnd, "operator": "REVISE", "prompt": cur, "R_bits": Rcur,
                     "n_yes": int(vcur.sum()), "accepted": accepted,
                     "candidates": [{k: v for k, v in c.items() if k != "_v"} for c in cands]})
        print(f"  r{rnd} {'ACCEPT' if accepted else 'keep-parent'} R={Rcur:.3f}b "
              f"(GLM calls: {glm.stats.n_calls})", flush=True)

    # ---- Discovery-to-Selection (§6.5): Ω units broken out of the accepted prompts ----
    accepted_prompts = [t["prompt"] for t in traj if t["accepted"] and t["round"] > 0]
    accepted_prompts = accepted_prompts or [cur]
    atom_src = "\n\n---\n\n".join(accepted_prompts)
    clause_pool = []
    try:
        atoms = glm.generate(_ATOMIZE.format(rubrics=atom_src), max_tokens=800, temperature=0.3)
        seen = set()
        for ln in _strip_fences(atoms).splitlines():
            c = ln.strip(" -•\t")
            if not (15 < len(c) < 300):
                continue
            k = re.sub(r"[^a-z0-9 ]", "", c.lower())[:80]
            if k not in seen:
                seen.add(k)
                clause_pool.append(c)
    except Exception as e:
        print(f"[plot-gepa] atomize GLM failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
    print(f"[plot-gepa] {len(clause_pool)} atomic criteria from {len(accepted_prompts)} "
          f"accepted prompts", flush=True)
    clause_sigs = np.vstack([
        np.asarray(aprobe.signature(executor, c, probes, max_chars, template=None), float)
        for c in clause_pool]) if clause_pool else np.zeros((0, len(probes)))
    Bg = (np.nan_to_num(clause_sigs, nan=0.5) > 0.5).astype(int)

    gepa_head = greedy_head(Bg, M) if len(Bg) >= 2 else {"gains": [], "selected": [],
                                                         "opt_omega_bits": 0.0}
    print(f"[plot-gepa] GEPA-only Ω: OPT={gepa_head['opt_omega_bits']:.3f}b "
          f"k={len(gepa_head['gains'])}", flush=True)

    # union with the certificate's freegen pool — do GEPA-discovered units enter the head?
    S_pool = np.asarray(z["sigs"], float)
    B_pool = (np.nan_to_num(S_pool, nan=0.5) > 0.5).astype(int)
    pool_prompts = [str(x) for x in z["prompts"]]
    B_union = np.vstack([B_pool, Bg]) if len(Bg) else B_pool
    union_src = ["freegen"] * len(B_pool) + ["gepa"] * len(Bg)
    union_prompts = pool_prompts + clause_pool
    union_head = greedy_head(B_union, M)
    union_sel = [{"idx": int(i), "source": union_src[int(i)],
                  "criterion": union_prompts[int(i)], "gain": float(gv)}
                 for i, gv in zip(union_head["selected"], union_head["gains"])
                 if not isinstance(i, tuple)]
    n_gepa_in_head = sum(1 for s in union_sel if s["source"] == "gepa")
    print(f"[plot-gepa] union Ω head: OPT={union_head['opt_omega_bits']:.3f}b "
          f"k={len(union_head['gains'])} gepa-in-head={n_gepa_in_head}", flush=True)

    np.savez(out / f"gepa_omega_{metric_id}.npz",
             clause_prompts=np.array(clause_pool, dtype=object), clause_sigs=clause_sigs)
    payload = {
        "task": args.task, "bucket": args.bucket, "gi": args.gi, "metric": name,
        "metric_description": desc, "seed_mode": args.seed_mode,
        "model": args.model, "glm_model": args.glm_model,
        "n_probes": len(probes), "M_i_mean": target_rate, "npz": npz_path,
        "rounds": args.rounds, "candidates_per_round": args.candidates,
        "name_R_bits": name_R, "desc_R_bits": desc_R,
        "glm_calls": int(glm.stats.n_calls), "lineage": traj,
        "gepa_omega": {
            "clauses": clause_pool,
            "gains": [float(x) for x in gepa_head["gains"]],
            "selected": [int(i) for i in gepa_head["selected"] if not isinstance(i, tuple)],
            "opt_omega_bits": float(gepa_head["opt_omega_bits"]),
        },
        "union_omega": {
            "opt_omega_bits": float(union_head["opt_omega_bits"]),
            "gains": [float(x) for x in union_head["gains"]],
            "head": union_sel,
            "n_gepa_in_head": n_gepa_in_head,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_json = out / f"gepa_lineage_{metric_id}.json"
    json.dump(payload, open(out_json, "w"), indent=1)
    print(f"\n[plot-gepa] wrote {out_json} ({len(traj)} rounds, {glm.stats.n_calls} GLM calls, "
          f"{payload['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
