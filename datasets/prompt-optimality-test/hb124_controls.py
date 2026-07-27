"""HB124 controls: is 'random units beat GEPA's winner' about THE POOL, or about ANY TEXT?

HB124 found 36/40 random draws from hotpot's frozen pool beat GEPA-official's own winning
prompt (+.038) on a 0.6B executor. Two degenerate explanations must die before that is quotable:

  A. ANY-TEXT.  A 0.6B model may improve whenever more instruction text is appended, regardless
     of content. Control: FOREIGN units drawn from a DIFFERENT benchmark's frozen pool,
     count-matched draw-for-draw against the native draws. If foreign ~= native, HB124 is an
     artifact of text volume and the pool-value claim dies.
  B. SINGLE-PASS INIT.  init was measured once. Everything is referenced to it, so its noise is
     not averaged out anywhere. Control: re-measure init with k passes.

Native draws are re-run here too (same session, same server) so all three arms are strictly
comparable -- the HB121 lesson: never compare across sessions.

Usage:
  python hb124_controls.py hotpot --task-lm openai/Qwen3-0.6B --api-base http://127.0.0.1:PORT/v1 \
      --foreign-bench hover --n-draws 40 --init-passes 5
Writes runs/hb124_controls_<bench>_<lmtag>.json  (resumable: existing draws are kept).
"""
from __future__ import annotations
import argparse, datetime, json, socket
from pathlib import Path
import dspy
import numpy as np
import paperexact_arms as px

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--task-lm", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--lm-tag", default="Qwen3-0.6B")
    ap.add_argument("--pool-tag", default="Qwen3-8B")
    ap.add_argument("--foreign-bench", default="hover")
    ap.add_argument("--n-draws", type=int, default=40)
    ap.add_argument("--include-p", type=float, default=0.5)
    ap.add_argument("--init-passes", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--eval-threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the output file (new session = new file)")
    ap.add_argument("--base", choices=["init", "seed"], default="init",
                    help="what every draw is appended TO. seed = the benchmark seed prompt "
                         "with the searched winner REMOVED (required for a zero-search "
                         "ceiling test; HB124b showed appending to init makes every draw a "
                         "superset of the searched prompt, which licenses nothing).")
    ap.add_argument("--arms", nargs="+",
                    default=["native", "foreign"],
                    choices=["native", "foreign", "shuffled", "seed_units", "foreign_shuffled", "shuffled_entity"],
                    help="shuffled = native clauses with tokens scrambled (kills the "
                         "any-text artifact: same length/format/vocabulary, no semantics). "
                         "seed_units = units appended to the SEED prompt, with GEPA's "
                         "winner REMOVED (the thesis arm: is the pool enough on its own?)")
    a = ap.parse_args()

    lm = dspy.LM(a.task_lm, api_base=a.api_base, api_key="EMPTY", cache=False,
                 temperature=0.6, top_p=0.95, max_tokens=a.max_tokens,
                 num_retries=10, timeout=300, extra_body={"top_k": 20})
    dspy.configure(lm=lm)
    px.EVAL_THREADS = a.eval_threads

    bench, program, metric, _ = px.load_bench(a.bench)
    test = list(bench.test_set)
    seed_cand = px.get_instructions(program)
    bdir = HERE / "runs_paperexact" / a.bench / a.pool_tag
    init_cand = {**seed_cand,
                 **json.loads((bdir / "official" / "result.json").read_text())["best_candidate"]}

    def pool_of(b):
        p = json.loads((HERE / "pools" / f"{b}_{a.pool_tag}_frozen.json").read_text())["units"]
        return [(d["module"], d["unit"]) for d in p]

    native = [u for u in pool_of(a.bench) if u[0] in seed_cand]
    foreign_raw = pool_of(a.foreign_bench)
    # foreign clauses are written for another program's modules; attach them to THIS program's
    # modules round-robin so only the CONTENT differs from the native arm, not the placement.
    mods = [m for m in seed_cand]
    foreign = [(mods[i % len(mods)], c) for i, (_m, c) in enumerate(foreign_raw)]
    # shuffled: destroy semantics, preserve token count, bullet format and vocabulary.
    # This is the control the near-zero corr(#units, score) could NOT provide.
    _rs = np.random.default_rng(a.seed + 991)
    def _shuf(c):
        w = c.split()
        _rs.shuffle(w)
        return " ".join(w)
    shuffled = [(m, _shuf(c)) for m, c in native]
    # entity-preserving scramble: capitalized multi-word spans stay atomic, everything else shuffles.
    # Separates "vocabulary priming" from "entity-name priming" (HB147: word-level scrambling
    # shatters entity names, likely UNDER-estimating the vocabulary share on fact-verification).
    import re as _re
    def _shuf_ent(c):
        toks = _re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)+|\S+", c)
        _rs.shuffle(toks)
        return " ".join(toks)
    shuffled_entity = [(m, _shuf_ent(c)) for m, c in native]
    foreign_shuffled = [(m, _shuf(c)) for m, c in foreign]
    print(f"pools: native={len(native)} foreign({a.foreign_bench})={len(foreign)} "
          f"shuffled={len(shuffled)}", flush=True)

    out = HERE / "runs" / f"hb124_controls_{a.bench}_{a.lm_tag}_{a.base}{a.tag}.json"
    out.parent.mkdir(exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    res = {"bench": a.bench, "lm_tag": a.lm_tag, "foreign_bench": a.foreign_bench,
           "include_p": a.include_p, "seed": a.seed, "n_draws": a.n_draws,
           "base": a.base,
           "fingerprint": {"host": socket.gethostname(), "api_base": a.api_base,
                           "task_lm": a.task_lm, "max_tokens": a.max_tokens, "cache": False,
                           "utc": datetime.datetime.utcnow().isoformat() + "Z"}}
    for k in ("init_k", "native", "foreign"):
        if k in prev:
            res[k] = prev[k]
    log = bdir / "hb124_controls.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    if "init_k" not in res:
        res["init_k"] = px.evaluate_cand(program, init_cand, test, metric, log,
                                         "hb124_init", passes=a.init_passes)
        print(f"init (k={a.init_passes}) = {res['init_k']}", flush=True)
        out.write_text(json.dumps(res, indent=1))

    rng = np.random.default_rng(a.seed)
    # draw the SAME per-draw unit COUNTS for both arms, so text volume is matched exactly
    counts = [int(rng.binomial(len(native), a.include_p)) for _ in range(a.n_draws)]
    BASE = dict(seed_cand) if a.base == "seed" else init_cand
    ARMS = {"native": (native, BASE), "foreign": (foreign, BASE),
            "shuffled": (shuffled, BASE), "shuffled_entity": (shuffled_entity, BASE),
            "foreign_shuffled": (foreign_shuffled, BASE),
            # seed_units always drops the searched winner regardless of --base
            "seed_units": (native, dict(seed_cand))}
    # PAIRED subsets (HB147 fix): one index draw per i, reused by every same-pool arm; foreign
    # arms get their own count-matched draw. Removes pool-composition variance from contrasts.
    _rp = np.random.default_rng(a.seed + 17)
    IDX_NAT = [ _rp.choice(len(native),  size=min(c, len(native)),  replace=False).tolist() for c in counts ]
    IDX_FRN = [ _rp.choice(len(foreign), size=min(c, len(foreign)), replace=False).tolist() for c in counts ]
    PAIRED_IDX = {"native": IDX_NAT, "shuffled": IDX_NAT, "shuffled_entity": IDX_NAT,
                  "seed_units": IDX_NAT, "foreign": IDX_FRN, "foreign_shuffled": IDX_FRN}
    for arm in a.arms:
        pool, base_cand = ARMS[arm]
        rows = res.get(arm, [])
        for i in range(len(rows), a.n_draws):
            idx = PAIRED_IDX[arm][i]
            cand = dict(base_cand)
            for j in idx:
                m, c = pool[j]
                cand[m] = cand[m] + "\n- " + c
            s = px.evaluate_cand(program, cand, test, metric, log, f"hb124_{arm}_{i}")
            rows.append({"i": i, "k_units": k, "score": s})
            res[arm] = rows
            out.write_text(json.dumps(res, indent=1))
            if (i + 1) % 5 == 0:
                print(f"  {arm} {i+1}/{a.n_draws}", flush=True)

    def stat(rows):
        v = [r["score"] for r in rows if r.get("score") is not None]
        return {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
                "min": float(min(v)), "max": float(max(v))} if v else {}
    summ = {"init_k": res["init_k"]}
    got = {}
    for arm in a.arms:
        if arm in res:
            summ[arm] = stat(res[arm])
            v = [r["score"] for r in res[arm] if r.get("score") is not None]
            got[arm] = v
            summ[f"{arm}_above_init"] = sum(1 for x in v if x > res["init_k"])
    # theta: share of the native gain attributable to CONTENT rather than text volume.
    # theta ~ 1 => content carries it; theta ~ 0 => any text would have done.
    nat = got.get("native")
    for ctrl in ("shuffled", "foreign"):
        if nat and got.get(ctrl) and (np.mean(nat) - res["init_k"]) != 0:
            summ[f"theta_vs_{ctrl}"] = float(
                (np.mean(nat) - np.mean(got[ctrl])) / (np.mean(nat) - res["init_k"]))
    res["summary"] = summ
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
