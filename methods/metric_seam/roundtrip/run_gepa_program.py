"""Step 3 — GEPA on the PROGRAM, using the OFFICIAL gepa package (user directive
2026-08-12: base this on the original GEPA, not a re-implementation).

The engine (Pareto-frontier candidate selection over per-instance wins, minibatch
acceptance test, reflective mutation, budget accounting) is gepa.optimize, untouched.
We supply only the two documented integration hooks:
  - CodeArmAdapter: candidate = {"program": <python source>}; evaluate() execs the
    source in a restricted namespace and scores each train instance by within-batch
    rank agreement with the original code channel's score (per-instance score =
    1 - |rank_got - rank_want| / (n-1); set-level spearman is recovered by the mean);
    make_reflective_dataset() shows the worst-disagreeing items with text excerpts.
  - reflection_lm: the Codex companion CLI (fresh thread per call), so all LLM work
    rides the Codex quota, per standing rule.
Data discipline: GEPA sees ONLY train items (channel items minus the frozen 40-item
eval split); its valset is a deterministic train-side split. Eval ids are never loaded
here — final scoring happens once, in eval_roundtrip.py.

Usage: python3 run_gepa_program.py [max_metrics] [max_metric_calls_per_metric]
       defaults: 24 (pilot; worst one-shot train rho first), 150
Output: roundtrip/output_gepa_r1_c0.py (revised programs) + gepa_runs/<job_id>.json logs.
"""
import json
import math
import re
import statistics as st
import subprocess
import sys

from common import (CODEX, WORK, channel_scores_full, load_functions, load_items,
                    ranks, spearman)

RESTRICT = None


def _ns():
    import collections
    import string
    return {"re": re, "math": math, "statistics": st, "string": string,
            "collections": collections, "__builtins__": __builtins__}


def compile_program(src):
    ns = _ns()
    exec(src, ns)
    fns = [v for k, v in ns.items() if k.startswith("score__") or k == "score"]
    if not fns:
        raise ValueError("no score function defined")
    return fns[0]


def codex_lm(prompt: str) -> str:
    r = subprocess.run(["node", CODEX, "task", prompt, "--fresh"],
                       timeout=900, capture_output=True, text=True)
    return r.stdout or ""


def extract_source(reply: str) -> str:
    m = re.findall(r"```(?:python)?\n(.*?)```", reply, re.S)
    return m[-1] if m else reply


def make_adapter(items, train_scores):
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    class CodeArmAdapter(GEPAAdapter):
        def evaluate(self, batch, candidate, capture_traces=False):
            try:
                fn = compile_program(candidate["program"])
            except Exception as e:
                sc = [0.0] * len(batch)
                out = [None] * len(batch)
                traj = [{"error": str(e)[:200]}] * len(batch) if capture_traces else None
                return EvaluationBatch(outputs=out, scores=sc, trajectories=traj)
            got = []
            for d in batch:
                try:
                    v = float(fn(items[d]))
                    got.append(v if v == v else None)
                except Exception:
                    got.append(None)
            want = [train_scores[d] for d in batch]
            ok = [i for i, g in enumerate(got) if g is not None]
            scores = [0.0] * len(batch)
            trajs = [None] * len(batch)
            if len(ok) >= 2 and len({got[i] for i in ok}) > 1:
                rg = ranks([got[i] for i in ok])
                rw = ranks([want[i] for i in ok])
                n = len(ok)
                for j, i in enumerate(ok):
                    scores[i] = 1.0 - abs(rg[j] - rw[j]) / max(1, n - 1)
                    if capture_traces:
                        trajs[i] = {"id": batch[i], "got": got[i], "want": want[i],
                                    "rank_err": abs(rg[j] - rw[j]) / max(1, n - 1)}
            return EvaluationBatch(outputs=got, scores=scores,
                                   trajectories=trajs if capture_traces else None)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            rows = []
            tr = [t for t in (eval_batch.trajectories or []) if t and "id" in t]
            tr.sort(key=lambda t: -t["rank_err"])
            for t in tr[:6]:
                rows.append({"Inputs": items[t["id"]][:700],
                             "Generated Outputs": f"score {t['got']:.2f}",
                             "Feedback": f"target scorer gave {t['want']:.2f}; "
                                         f"rank disagreement {t['rank_err']:.2f} — adjust "
                                         f"the program so this document ranks correctly"})
            return {"program": rows}

        def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            rows = reflective_dataset["program"]
            fb = "\n\n".join(
                f"--- document (excerpt):\n{r['Inputs']}\nyour program: {r['Generated Outputs']}"
                f"\nfeedback: {r['Feedback']}" for r in rows)
            prompt = (
                "Improve this deterministic Python document-scoring program so its RANKING "
                "of documents better matches a target scorer. Only re, math, statistics, "
                "string, collections may be imported; must define exactly one function "
                "named 'score(text)' returning a float 0-10; deterministic; never "
                "constant.\n\nCurrent program:\n```python\n" + candidate["program"] +
                "\n```\n\nWorst disagreements:\n" + fb +
                "\n\nReply with ONLY the improved program in a ```python code block.")
            src = extract_source(codex_lm(prompt))
            src = re.sub(r"def score__[A-Za-z0-9_]+\(", "def score(", src)
            return {"program": src}

    return CodeArmAdapter()


def main(max_metrics=24, budget=150):
    import gepa
    jobs = [j for j in json.load(open(WORK / "jobs_full.json"))
            if j["task"] != "CALIBRATION"]
    base = load_functions("output_rt_c*_codex_t1.py") or load_functions("output_rt_c[0-9].py")
    src_of = {}
    for f in sorted(WORK.glob("output_rt_c*_codex_t1.py")) or sorted(WORK.glob("output_rt_c[0-9].py")):
        txt = open(f).read()
        for m in re.finditer(r"(def score__([A-Za-z0-9_]+)\(.*?)(?=\ndef score__|\nJOB_IDS|\Z)",
                             txt, re.S):
            src_of[m.group(2)] = m.group(1)
    items_cache = {}
    # rank jobs by one-shot train rho (worst first), computed here train-side only
    ranked = []
    for j in jobs:
        jid = j["job_id"]
        if jid not in base or jid not in src_of:
            continue
        t = j["task"]
        items = items_cache.setdefault(t, load_items(t))
        train, _ = channel_scores_full(j)
        ids = [d for d in sorted(train) if d in items]
        try:
            got = [float(base[jid](items[d])) for d in ids]
        except Exception:
            continue
        if len(set(got)) <= 1:
            r0 = -1.0
        else:
            r0 = spearman(got, [train[d] for d in ids])
        ranked.append((r0, j))
    ranked.sort(key=lambda x: x[0])
    (WORK / "gepa_runs").mkdir(exist_ok=True)
    revised = {}
    for r0, j in ranked[:max_metrics]:
        jid = j["job_id"]
        logp = WORK / "gepa_runs" / f"{jid}.json"
        if logp.exists():
            rec = json.load(open(logp))
            if rec.get("best_program"):
                revised[jid] = rec["best_program"]
            print(f"SKIP {jid} (done; train {rec.get('train_r0')}->{rec.get('train_best')})")
            continue
        t = j["task"]
        items = items_cache.setdefault(t, load_items(t))
        train, _ = channel_scores_full(j)
        ids = [d for d in sorted(train) if d in items]
        val = ids[::4]                       # deterministic train-side valset
        tr = [d for d in ids if d not in set(val)]
        src = "def score(text):\n    return _inner(text)\n" \
              if False else re.sub(r"def score__[A-Za-z0-9_]+\(", "def score(", src_of[jid])
        adapter = make_adapter(items, train)
        print(f"=== GEPA {jid} (one-shot train rho {r0:+.3f}; budget {budget}) ===",
              flush=True)
        try:
            res = gepa.optimize(seed_candidate={"program": src}, trainset=tr, valset=val,
                                adapter=adapter, reflection_minibatch_size=8,
                                max_metric_calls=budget)
            best = res.best_candidate["program"]
        except Exception as e:
            json.dump({"error": str(e)[:400], "train_r0": round(r0, 3)}, open(logp, "w"))
            print(f"  ERROR {str(e)[:120]}")
            continue
        # final train-side spearman of the best candidate (keep only if better)
        try:
            fn = compile_program(best)
            got = [float(fn(items[d])) for d in ids]
            r1 = spearman(got, [train[d] for d in ids]) if len(set(got)) > 1 else -1
        except Exception:
            r1 = -1
        keep = r1 > r0
        json.dump({"train_r0": round(r0, 3), "train_best": round(r1, 3), "kept": keep,
                   "best_program": best if keep else None}, open(logp, "w"))
        if keep:
            revised[jid] = best
        print(f"  train {r0:+.3f} -> {r1:+.3f} ({'KEPT' if keep else 'discarded'})",
              flush=True)
    out = WORK / "output_gepa_r1_c0.py"
    with open(out, "w") as f:
        f.write("# AUTO: official-gepa program optimization (train-side only)\n")
        for jid, src in revised.items():
            f.write("\n" + re.sub(r"def score\(", f"def score__{jid}(", src, count=1) + "\n")
        f.write(f"\nJOB_IDS = {sorted(revised)}\n")
    print(f"wrote {out.name}: {len(revised)} revised programs")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24,
         int(sys.argv[2]) if len(sys.argv) > 2 else 150)
