"""y-GEPA seam, stage 3 (ISO push): get an LLM/coded WORKFLOW to isomorphic-or-better
held-out y-AUC vs the holistic GEPA prompt (.759), using the full seam machinery.

Pieces (all selection on TRAIN; test read once at the end by `report`):
  compile   : per-unit fidelity codegen — GLM writes/revises code for EVERY unit against
              its prompt-twin's TRAIN scores (label-free compile, MIGRATE-style), up to
              --attempts rounds with exemplar feedback. Keeps best-fidelity code per unit.
  integrate : GLM writes a doctrine-informed CODE integrator combine(units, text) -> float,
              iterated --rounds times against TRAIN AUC with misranked-case feedback
              (label-aware, like the manual VAT). Also fits sklearn reference combiners.
  report    : evaluates every arm on TEST (single read), headline = TRAIN-argmax arm.

Usage:
  python3 y_gepa_iso.py compile  legal_title_vii --attempts 3
  python3 y_gepa_iso.py integrate legal_title_vii --rounds 4
  python3 y_gepa_iso.py report   legal_title_vii
"""
import argparse, importlib.util, json, pathlib, sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
import battery_common as bc
from certificates import spearman
from y_seam_extend import items_with_y, auc, TASKS as YTASKS
from y_seam_vtier import stratified_auc
from y_gepa import glm_call, OUT

bc.PROGDIR.update({"legal_title_vii": "programs_legal"})


def load_all(task):
    ctx = bc.load_ctx(task)
    iy = items_with_y(task)
    units = json.load(open(OUT / f"{task}_units.json"))
    pcols = {}
    for line in open(OUT / f"{task}_combined_results.jsonl"):
        r = json.loads(line)
        a = r.get("aspect_id", "")
        if a.endswith(".unit") and isinstance(r.get("score"), int):
            pcols.setdefault(a.split(".")[1], {})[r["datapoint_id"]] = r["score"]
    train = [d for d in sorted(ctx["train"]) if iy.get(d, ("", None))[1] in (0, 1)]
    test = [d for d in sorted(ctx["test"]) if iy.get(d, ("", None))[1] in (0, 1)]
    return ctx, iy, units, pcols, train, test


def run_code(src_path, iy, ids):
    spec = importlib.util.spec_from_file_location(src_path.stem, src_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None  # broken/truncated source
    out = {}
    for d in ids:
        try:
            out[d] = float(mod.score(iy[d][0]))
        except Exception:
            out[d] = 0.5
    return out


def cmd_compile(task, attempts):
    ctx, iy, units, pcols, train, test = load_all(task)
    noun, _ = YTASKS[task]
    cdir = OUT / f"{task}_unit_code_v2"; cdir.mkdir(exist_ok=True)
    manifest = {}
    for u in units:
        uid = u["unit_id"]
        target = pcols.get(uid, {})
        tids = [d for d in train if d in target]
        if len(tids) < 40:
            manifest[uid] = {"codable": False, "reason": "no prompt-twin coverage"}; continue
        tv = [target[d] for d in tids]
        best_fid, best_src = -2, None
        prev_src, prev_fid = None, None
        for att in range(attempts):
            fb = ""
            if prev_src is not None:
                # exemplars where the code disagrees most with the prompt twin
                col = manifest.get("_tmpcol", {})
                errs = sorted(tids, key=lambda d: -abs(col.get(d, 0.5) * 10 - target[d]))[:5]
                cases = "\n\n".join(
                    f"[prompt-twin score {target[d]}, your code {col.get(d, 0.5):.2f} "
                    f"(x10 scale mismatch ok — rank matters)]\n{iy[d][0][:500]}" for d in errs)
                fb = (f"\n\nYour previous attempt (train rank-fidelity to the reference "
                      f"scorer: {prev_fid:.3f}):\n```\n{prev_src[:2500]}\n```\n"
                      f"Documents where your code most disagrees with the reference:\n{cases}\n"
                      "Revise to better match the reference scorer's RANKING.")
            system = ("You write a deterministic Python function that mimics a reference "
                      "LLM scorer for ONE factor. Return ONLY Python source (no fences) with "
                      "`def score(text: str) -> float` (any monotone scale), stdlib only "
                      "(re, math, collections), robust (try/except -> 0.5), no catastrophic "
                      "regex. Approximate the factor as well as deterministic text "
                      "processing allows — partial signal is fine.")
            user = (f"Document type: {noun}.\nFactor: {u['name']}\n"
                    f"Question the reference scorer answers 0-10: {u['question']}{fb}\n\n"
                    "Write the code.")
            src = glm_call(system, user, max_tokens=4000).strip()
            if src.startswith("```"):
                parts = src.split("```")
                src = parts[1][6:] if parts[1].startswith("python") else parts[1]
                src = src.strip()
            fp = cdir / f"{uid}_att{att}.py"
            fp.write_text(src)
            col = run_code(fp, iy, tids)
            if col is None:
                print(f"  {uid} att{att}: source broken — skipping", flush=True)
                prev_src, prev_fid = src, -1.0
                continue
            fid = spearman([col[d] for d in tids], tv)
            fid = fid if fid == fid else -1
            print(f"  {uid} att{att}: train fidelity={fid:+.3f}", flush=True)
            manifest["_tmpcol"] = col
            prev_src, prev_fid = src, fid
            if fid > best_fid:
                best_fid, best_src = fid, fp.name
        manifest.pop("_tmpcol", None)
        manifest[uid] = {"codable": best_fid > 0.15, "file": best_src,
                        "train_fidelity": round(best_fid, 4)}
        print(f"{uid} {u['name']}: BEST fidelity={best_fid:+.3f} "
              f"({'ACCEPT' if best_fid > 0.15 else 'reject'})")
    json.dump(manifest, open(OUT / f"{task}_unit_code_v2_manifest.json", "w"), indent=1)


def _feature_frame(task, use_code):
    """unit feature dict per dpid: code column where accepted (use_code), else prompt."""
    ctx, iy, units, pcols, train, test = load_all(task)
    man = json.load(open(OUT / f"{task}_unit_code_v2_manifest.json"))
    cdir = OUT / f"{task}_unit_code_v2"
    cols = {}
    src = {}
    for u in units:
        uid = u["unit_id"]
        if use_code and man.get(uid, {}).get("codable"):
            c = run_code(cdir / man[uid]["file"], iy, train + test)
            if c is None:
                cols[uid] = pcols.get(uid, {}); src[uid] = "P"; continue
            cols[uid] = c
            # code scale ~0-1; prompt scale 0-10 — normalize each column to rank-preserving z later
            src[uid] = "C"
        elif uid in pcols:
            cols[uid] = pcols[uid]
            src[uid] = "P"
    return ctx, iy, units, cols, src, train, test


def cmd_integrate(task, rounds):
    ctx, iy, units, cols, src, train, test = _feature_frame(task, use_code=True)
    noun, q = YTASKS[task]
    uids = [u["unit_id"] for u in units if u["unit_id"] in cols]
    udesc = {u["unit_id"]: u for u in units}
    # normalized train matrix for GLM feedback readability
    def vec(d):
        return {u: round(float(cols[u].get(d, 0)), 3) for u in uids}
    idir = OUT / f"{task}_integrators"; idir.mkdir(exist_ok=True)
    history = []
    best = {"train_auc": -1}
    prompt_ctx = "\n".join(
        f"  {u}: [{src[u]}] {udesc[u]['name']} — {udesc[u]['question'][:140]}" for u in uids)
    prev_src_txt, prev_auc = None, None
    for rd in range(rounds):
        fb = ""
        if prev_src_txt is not None:
            colp = history[-1]["col"]
            scored = sorted(train, key=lambda d: colp[d])
            fpos = [d for d in reversed(scored) if iy[d][1] == 0][:4]
            fneg = [d for d in scored if iy[d][1] == 1][:4]
            cases = "\n".join(
                [f"  predicted-favorable but LOST: units={vec(d)}" for d in fpos] +
                [f"  predicted-unfavorable but WON: units={vec(d)}" for d in fneg])
            fb = (f"\n\nPrevious integrator (TRAIN AUC {prev_auc:.3f}):\n```\n"
                  f"{prev_src_txt[:2600]}\n```\nMisranked TRAIN cases (unit-score vectors):\n"
                  f"{cases}\nRevise. Consider interactions/gating (e.g. one unit changing "
                  "how another matters), thresholds, and robustness — n_train is only 150.")
        system = ("You write a deterministic Python integrator combining unit scores into an "
                  "outcome forecast. Return ONLY Python source (no fences) defining "
                  "`def combine(units: dict, text: str) -> float` where `units` maps unit ids "
                  "to raw scores ([C]=code units, roughly 0..1 scale; [P]=prompt units, 0..10 "
                  "scale — normalize as needed) and `text` is the raw document (use "
                  "sparingly). Higher output = favorable outcome more likely. stdlib only, "
                  "robust (try/except -> 0.5). Domain-informed logic (gating, interactions) "
                  "is encouraged over generic weighted sums.")
        user = (f"Forecast whether {q} for a {noun}.\nUnits available:\n{prompt_ctx}{fb}\n\n"
                "Write the integrator.")
        src_txt = glm_call(system, user, max_tokens=2800).strip()
        if src_txt.startswith("```"):
            parts = src_txt.split("```")
            src_txt = parts[1][6:] if parts[1].startswith("python") else parts[1]
            src_txt = src_txt.strip()
        fp = idir / f"integrator_r{rd}.py"
        fp.write_text(src_txt)
        spec = importlib.util.spec_from_file_location(f"integ_{rd}", fp)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            col = {}
            for d in train:
                try:
                    col[d] = float(mod.combine({u: cols[u].get(d, 0) for u in uids}, iy[d][0]))
                except Exception:
                    col[d] = 0.5
            a, _, _ = auc([col[d] for d in train], [iy[d][1] for d in train])
        except Exception as e:
            print(f"  r{rd}: integrator broken ({type(e).__name__})"); continue
        print(f"  integrator r{rd}: TRAIN AUC={a:.4f}", flush=True)
        history.append({"round": rd, "train_auc": a, "col": col})
        if a > best["train_auc"]:
            best = {"round": rd, "train_auc": a, "file": fp.name}
        prev_src_txt, prev_auc = src_txt, a
    json.dump({"best": best,
               "history": [{k: h[k] for k in ("round", "train_auc")} for h in history],
               "unit_sources": src},
              open(OUT / f"{task}_integrator_state.json", "w"), indent=1)
    print(f"best integrator: r{best.get('round')} train AUC {best['train_auc']:.4f}")


def cmd_report(task):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    ctx, iy, units, cols, src, train, test = _feature_frame(task, use_code=True)
    uids = [u["unit_id"] for u in units if u["unit_id"] in cols]
    ist = json.load(open(OUT / f"{task}_integrator_state.json"))
    holo = json.load(open(OUT / f"{task}_final_eval.json"))

    def matrix(ids):
        X, y, ln = [], [], []
        for d in ids:
            X.append([float(cols[u].get(d, 0)) for u in uids])
            y.append(iy[d][1]); ln.append(len(iy[d][0]))
        return np.array(X, float), np.array(y), ln
    Xtr, ytr, _ = matrix(train)
    Xte, yte, lte = matrix(test)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    arms = {}

    lr = LogisticRegression(max_iter=3000).fit((Xtr - mu) / sd, ytr)
    arms["logistic(mixed units v2)"] = (lr.predict_proba((Xtr - mu) / sd)[:, 1],
                                        lr.predict_proba((Xte - mu) / sd)[:, 1])
    gb = HistGradientBoostingClassifier(max_depth=2, max_iter=60,
                                        learning_rate=0.1).fit(Xtr, ytr)
    arms["gbm(mixed units v2)"] = (gb.predict_proba(Xtr)[:, 1], gb.predict_proba(Xte)[:, 1])
    # GLM-coded integrator (best train round)
    fp = OUT / f"{task}_integrators" / ist["best"]["file"]
    spec = importlib.util.spec_from_file_location("integ_best", fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    def integ(ids):
        out = []
        for d in ids:
            try:
                out.append(float(mod.combine({u: cols[u].get(d, 0) for u in uids}, iy[d][0])))
            except Exception:
                out.append(0.5)
        return np.array(out)
    arms["coded integrator (agentic)"] = (integ(train), integ(test))
    # ensemble: coded integrator + logistic mean-rank
    def rank01(v):
        order = np.argsort(np.argsort(v)); return order / (len(v) - 1)
    arms["ensemble(integrator+logistic)"] = (
        (rank01(arms["coded integrator (agentic)"][0]) + rank01(arms["logistic(mixed units v2)"][0])) / 2,
        (rank01(arms["coded integrator (agentic)"][1]) + rank01(arms["logistic(mixed units v2)"][1])) / 2)

    rows = []
    for name, (ptr, pte) in arms.items():
        atr, _, _ = auc(list(ptr), list(ytr))
        ate, _, _ = auc(list(pte), list(yte))
        sate = stratified_auc(list(pte), list(yte), lte)
        rows.append(dict(arm=name, auc_train=round(atr, 4), auc_test=round(ate, 4),
                         auc_test_strat=round(sate, 4)))
        print(f"  {name:32s} train={atr:.3f} TEST={ate:.3f} strat={sate:.3f}")
    headline = max(rows, key=lambda r: r["auc_train"])
    n_code_units = sum(1 for u in uids if src[u] == "C")
    print(f"\nheadline (TRAIN-argmax): {headline['arm']} -> TEST {headline['auc_test']}")
    print(f"holistic prompt reference: TEST {holo['auc_test']}")
    print(f"unit sources: {src}  (%code units = {n_code_units}/{len(uids)})")
    json.dump(dict(task=task, arms=rows, headline=headline,
                   holistic_ref=holo, unit_sources=src,
                   pct_code_units=round(n_code_units / len(uids), 3)),
              open(OUT / f"{task}_iso_report.json", "w"), indent=1)
    print(f"-> {OUT / f'{task}_iso_report.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compile", "integrate", "report"])
    ap.add_argument("task")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "compile":
        cmd_compile(a.task, a.attempts)
    elif a.cmd == "integrate":
        cmd_integrate(a.task, a.rounds)
    else:
        cmd_report(a.task)
