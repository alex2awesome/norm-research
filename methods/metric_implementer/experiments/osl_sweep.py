"""OSL executor sweep — one executor per invocation (spec: notes/2026-07-07__osl-executor-scaling-spec.md).

Fixed-support recovery across an executor ladder: the frozen humor-40 metrics' criterion TEXTS
(8B-mined silver_r2 checkpoints, sha256 stable-picked to K per metric, frozen inline) are
RE-EXECUTED by the target executor (retarget discipline: texts re-executed, nothing byte-copied);
the Φ-orbit target m̄_ω is re-scored per executor; recovery = greedy-on-EVEN / read-on-ODD
transmission, disattenuated by the form-split reliability of m̄_ω (the X–Y circularity fix). Plus
the frozen anchor battery (declared capability scalar z_E) and the base-measurement vector for the
secondary data-driven latent (PCA/PLS in osl_fit — never the headline axis).

Subcommands:
  --build-freeze   CPU-only: freeze the 40 bank metrics + 5 planted mechanical controls + K=60
                   criteria each (all stable-hash; criteria stored inline -> self-contained artifact)
  (default)        run one executor -> outputs/osl/<short>.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

from . import alpha_probe as ap
from .osl_battery import score_battery, _auc, _wc
from .run_real_test import _load_texts
from .. import config as cfgmod

# short -> (hf id, family, params). Params feed ONLY the within-family monotonicity gate (G2);
# the law's x-axis is the battery scalar. All local vLLM (sk3-only rule; logprob invariance).
EXECUTORS = {
    "llama1b":     ("meta-llama/Llama-3.2-1B-Instruct", "llama", 1.24e9),
    "llama3b":     ("meta-llama/Llama-3.2-3B-Instruct", "llama", 3.21e9),
    "llama8b":     ("meta-llama/Llama-3.1-8B-Instruct", "llama", 8.03e9),
    "llama70b":    ("meta-llama/Llama-3.3-70B-Instruct", "llama", 70.6e9),
    # v3 surgical additions (user 2026-07-08): fill the qwen 14->72 gap, give gemma2/mistral a
    # second rung each (within-family deltas; each new family point shrinks dialect error bars)
    "qwen25-32b":  ("Qwen/Qwen2.5-32B-Instruct", "qwen", 32.8e9),
    "gemma2-9b":   ("google/gemma-2-9b-it", "gemma2", 9.2e9),
    "mistral-24b": ("mistralai/Mistral-Small-24B-Instruct-2501", "mistral", 23.6e9),
    "gemma3-4b":   ("google/gemma-3-4b-it", "gemma3", 4.3e9),
    "gemma3-12b":  ("google/gemma-3-12b-it", "gemma3", 12.2e9),
    "gemma3-27b":  ("google/gemma-3-27b-it", "gemma3", 27.4e9),
    "gemma2-27b":  ("google/gemma-2-27b-it", "gemma2", 27.2e9),
    "qwen35-122b": ("Qwen/Qwen3.5-122B-A10B-FP8", "qwen35", 122e9),
    "qwen25-3b":   ("Qwen/Qwen2.5-3B-Instruct", "qwen25", 3.09e9),
    "qwen25-7b":   ("Qwen/Qwen2.5-7B-Instruct", "qwen25", 7.62e9),
    "qwen25-14b":  ("Qwen/Qwen2.5-14B-Instruct", "qwen25", 14.8e9),
    "qwen25-72b":  ("Qwen/Qwen2.5-72B-Instruct", "qwen25", 72.7e9),
    "mistral7b":   ("mistralai/Mistral-7B-Instruct-v0.3", "mistral", 7.25e9),
    "phi4":        ("microsoft/phi-4", "phi", 14.7e9),
    "gemma4-31b":  ("google/gemma-4-31b-it", "gemma4", 31e9),
    # 5th primary-family rung (curvature probe, user 2026-07-07): FP8, TP=4 via OSL_TP env.
    # Repo resolved at download time (nvidia ModelOpt quant preferred, meta official fallback).
    "llama405b":   (os.environ.get("OSL_405B_REPO", "nvidia/Llama-3.1-405B-Instruct-FP8"),
                    "llama", 405e9),
}


def _stable_pick(strings, k, salt=""):
    """Deterministic subset by sha256 order — immune to list growth (stable-hash-split rule)."""
    ranked = sorted(strings, key=lambda s: hashlib.sha256((salt + s).encode()).hexdigest())
    return ranked[:k]


# -- planted mechanical control metrics (code truth; asymptote must be ~1: gate G4) -----------
def planted_metrics(probe_texts, k_med):
    return [
        {"name": "PLANTED-length-long",
         "rubric": f"This text is longer than {k_med} words.",
         "truth": [(_wc(t) > k_med) for t in probe_texts]},
        {"name": "PLANTED-question",
         "rubric": "This text contains at least one question mark.",
         "truth": [("?" in t) for t in probe_texts]},
        {"name": "PLANTED-digit",
         "rubric": "This text contains at least one digit (0-9).",
         "truth": [bool(re.search(r"\d", t)) for t in probe_texts]},
        {"name": "PLANTED-quote",
         "rubric": "This text contains quoted speech (a quotation mark).",
         "truth": [('"' in t or "“" in t) for t in probe_texts]},
        {"name": "PLANTED-length-short",
         "rubric": f"This text is shorter than {k_med} words.",
         "truth": [(_wc(t) < k_med) for t in probe_texts]},
    ]


def _planted_paraphrases(rubric):
    return [f"Judged purely on this one standard: {rubric}",
            f"The following condition holds for the text above: {rubric}"]


def build_freeze(a):
    """Freeze metrics + criteria (CPU). Criteria inline -> the artifact is self-contained."""
    mg = json.load(open(a.bank_r2))["merged_groups"]
    descs = {g["merged_name"]: (g.get("merged_description") or "").strip()
             for g in mg if g.get("merged_name")}
    ckpts = {}
    for f in sorted(glob.glob(os.path.join(a.src_dir, "*_sigs.npz"))):
        z = np.load(f, allow_pickle=True)
        ckpts[str(z["name"])] = [str(p) for p in z["prompts"]]
    # min-criteria 0 (v2 fleet): mbar-only panels need just name+description, so admit every
    # hierarchy metric with a nonempty description (criteria = whatever sigs exist, maybe none).
    if a.min_criteria == 0:
        usable = [n for n in descs if descs[n]]
    else:
        usable = [n for n in ckpts if descs.get(n) and len(ckpts[n]) >= a.min_criteria]
    names = _stable_pick(usable, a.n_metrics, salt="osl-metric-v1|")
    union = sorted({c for n in names for c in ckpts.get(n, [])})
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    texts, _ = _load_texts(a.task, 60 + a.n_probes, cfg)
    probes = texts[60: 60 + a.n_probes]
    k_med = int(np.median([_wc(t) for t in probes]))
    metrics = [{"name": n, "kind": "bank", "rubric": f"{n}: {descs[n]}",
                "criteria": _stable_pick(ckpts.get(n, []), a.k_criteria, salt="osl-crit-v1|")}
               for n in names]
    for pm in planted_metrics(probes, k_med):
        own = [pm["rubric"]] + _planted_paraphrases(pm["rubric"])
        fill = _stable_pick(union, a.k_criteria - len(own), salt=f"osl-planted|{pm['name']}|")
        metrics.append({"name": pm["name"], "kind": "planted", "rubric": pm["rubric"],
                        "criteria": own + fill})
    blob = {"meta": {"task": a.task, "src_dir": a.src_dir, "n_probes": a.n_probes,
                     "k_criteria": a.k_criteria, "k_med_words": k_med,
                     "probe_window": f"60:{60 + a.n_probes}"},
            "metrics": metrics}
    js = json.dumps(blob, indent=1, sort_keys=True)
    open(a.out, "w").write(js)
    print(f"[freeze] {len(metrics)} metrics ({a.n_metrics} bank + "
          f"{len(metrics) - a.n_metrics} planted) -> {a.out}  "
          f"sha256={hashlib.sha256(js.encode()).hexdigest()}")


# -- recovery: greedy on EVEN probes, transmission read on ODD (winner's-curse discipline) -----
def _ridge_w(X, y, lam=1.0):
    Xc, yc = X - X.mean(0), y - y.mean()
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)


def _cv_corr(X, y, lam=1.0, folds=5, seed=0):
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    preds = np.full(n, np.nan)
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        w = _ridge_w(X[tr], y[tr], lam)
        preds[te] = (X[te] - X[tr].mean(0)) @ w + y[tr].mean()
    if np.std(preds) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(preds, y)[0, 1])


def greedy_recovery(S, y, kmax=8, lam=1.0, min_gain=0.01):
    """S: (n_criteria, n_probes) signatures; y: (n_probes,) target. Returns r_odd, bits, selected."""
    ok = np.isfinite(y)
    S = np.nan_to_num(S[:, ok], nan=0.5)
    y = y[ok]
    n = len(y)
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    live = [i for i in range(S.shape[0]) if S[i, even].std() > 1e-9]
    sel, best = [], -1.0
    for _ in range(min(kmax, len(live))):
        cand = [(_cv_corr(S[sel + [j]][:, even].T, y[even], lam), j)
                for j in live if j not in sel]
        if not cand:
            break
        r, j = max(cand)
        if r <= best + min_gain:
            break
        best = r
        sel.append(j)
    if not sel:
        return {"r_odd": 0.0, "bits": 0.0, "selected": [], "r_even_cv": 0.0}
    w = _ridge_w(S[sel][:, even].T, y[even], lam)
    pred = (S[sel][:, odd].T - S[sel][:, even].T.mean(0)) @ w
    r_odd = 0.0 if pred.std() < 1e-12 else float(np.corrcoef(pred, y[odd])[0, 1])
    r_odd = max(0.0, r_odd)
    bits = float(-0.5 * np.log2(max(1e-9, 1 - min(r_odd, 0.999) ** 2)))
    return {"r_odd": r_odd, "bits": bits, "selected": sel, "r_even_cv": best}


def form_split_reliability(per_form, odd_idx):
    """Reliability of m̄_ω from the Φ-orbit: forms {0,2} vs {1,3} halves on the ODD (readout)
    probes, Spearman-Brown up to the full 4-form average."""
    a = np.nanmean(per_form[0::2], axis=0)[odd_idx]
    b = np.nanmean(per_form[1::2], axis=0)[odd_idx]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or a[m].std() < 1e-9 or b[m].std() < 1e-9:
        return float("nan")
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    return float(np.clip(2 * r / (1 + r) if r > -0.5 else np.nan, -1, 1))


def _setup(a):
    model, family, params = EXECUTORS[a.executor]
    frz = json.load(open(a.freeze))
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), frz["meta"]["task"])
    cfg0 = cfgmod.ImplementerConfig()
    cfg0.vllm_tp_size = int(os.environ.get("OSL_TP", "1"))   # 405B: TP=4 (user-directed probe)
    if a.fake:
        cfg0.vllm_fake = True
    from ..vllm_backend import make_judge_backend
    ex = make_judge_backend(model, cfg0, temperature=None)
    if a.probe_window:
        lo, hi = (int(x) for x in a.probe_window.split(":"))
    else:
        lo, hi = 60, 60 + int(frz["meta"]["n_probes"])
    texts, _ = _load_texts(frz["meta"]["task"], hi, cfg)
    if len(texts) < hi:
        sys.exit(f"corpus too small for probe window {lo}:{hi} ({len(texts)} texts)")
    return model, family, params, frz, cfg, ex, texts[lo:hi]


def run_mbar(a):
    """--mbar-only: score ONLY the orbit target m̄_ω per metric and save the per-probe vectors +
    per-form matrices (the full sweep discards them). Feeds the consensus-agreement y-axis
    (user decision 2026-07-07 after self-recovery was refuted by the planted controls):
    y = agreement of m̄_ω(E) with the leave-one-executor-out consensus. ~6% of full-sweep cost."""
    model, family, params, frz, cfg, ex, probes = _setup(a)
    mx = cfg.max_text_chars
    names, kinds, mbars, pforms = [], [], [], []
    for mi, m in enumerate(frz["metrics"]):
        orb = ap.orbit_metric_verdict(ex, m["rubric"], probes, mx, n_forms=a.n_forms,
                                      template=ap._YESNO_TEXTFIRST)
        pf = np.full((4, len(probes)), np.nan)
        mat = np.asarray(orb["per_form"], float)
        pf[: mat.shape[0]] = mat
        names.append(m["name"])
        kinds.append(m["kind"])
        mbars.append(np.asarray(orb["m_bar"], float))
        pforms.append(pf)
        print(f"[mbar] {mi+1}/{len(frz['metrics'])} {m['name'][:48]}", flush=True)
    np.savez(a.out, executor=a.executor, family=family, params=params,
             names=np.array(names, dtype=object), kinds=np.array(kinds, dtype=object),
             m_bar=np.array(mbars, float), per_form=np.array(pforms, float))
    print(f"[mbar] DONE -> {a.out}", flush=True)


def run_battery_only(a):
    """--battery-only: capability scalar for executors too expensive for a full v1 sweep (405B)."""
    model, family, params = EXECUTORS[a.executor]
    battery = json.load(open(a.battery))
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
    cfg0 = cfgmod.ImplementerConfig()
    cfg0.vllm_tp_size = int(os.environ.get("OSL_TP", "1"))
    if a.fake:
        cfg0.vllm_fake = True
    from ..vllm_backend import make_judge_backend
    ex = make_judge_backend(model, cfg0, temperature=None)
    bat = score_battery(ex, battery["items"], cfg.max_text_chars)
    json.dump({"executor": a.executor, "model": model, "family": family, "params": params,
               "battery": bat, "battery_only": True, "mean_y_dis_bank": None},
              open(a.out, "w"), indent=1, default=float)
    print(f"[battery] {a.executor} auc={bat['auc']:.3f} z={bat['z']:.3f} -> {a.out}", flush=True)


_PAIR_TPL = ("Text A:\n{ta}\n\nText B:\n{tb}\n\n"
             "You are comparing the two texts on ONE specific criterion.\n"
             "Criterion:\n{rubric}\n\n"
             "Which text better satisfies the criterion? Answer with exactly one word: A or B.")


def run_pairwise(a):
    """--pairwise: forced-choice A/B readout over FIXED text pairs — the broad pilot arm
    ALONGSIDE the P(YES) machinery (user 2026-07-07: keep P(YES), pilot pairwise). Rationale:
    absolute P(YES) carries per-executor calibration/acquiescence 'dialect' (dual-polarity
    corr(M+,M-)=+.57; MathlibPR pairwise-A recovered what static-A could not); forced choice
    cancels it by construction. BOTH presentation orders are scored — the order-split is the
    reliability estimate (pairwise analog of the form-split)."""
    model, family, params, frz, cfg, ex, probes = _setup(a)
    mx = max(400, cfg.max_text_chars // 2)          # two texts per prompt: halve each
    rng = np.random.default_rng(0)
    pairs = []
    while len(pairs) < a.n_pairs:                    # deterministic fixed pairs, i != j
        i, j = int(rng.integers(len(probes))), int(rng.integers(len(probes)))
        if i != j:
            pairs.append((i, j))
    done, names, kinds, AB, BA = set(), [], [], [], []
    if os.path.exists(a.out):                        # resumable per metric
        z = np.load(a.out, allow_pickle=True)
        names = [str(x) for x in z["names"]]
        kinds = [str(x) for x in z["kinds"]]
        AB = [r for r in z["pref_ab"]]
        BA = [r for r in z["pref_ba"]]
        done = set(names)
        print(f"[pair] resuming: {len(done)} metrics done", flush=True)

    def _save():
        np.savez(a.out, executor=a.executor, family=family, params=params,
                 names=np.array(names, dtype=object), kinds=np.array(kinds, dtype=object),
                 pref_ab=np.array(AB, float), pref_ba=np.array(BA, float),
                 pair_i=np.array([p[0] for p in pairs]), pair_j=np.array([p[1] for p in pairs]))

    for mi, m in enumerate(frz["metrics"]):
        if m["name"] in done:
            continue
        pr_ab = [_PAIR_TPL.format(ta=probes[i][:mx], tb=probes[j][:mx], rubric=m["rubric"])
                 for i, j in pairs]
        pr_ba = [_PAIR_TPL.format(ta=probes[j][:mx], tb=probes[i][:mx], rubric=m["rubric"])
                 for i, j in pairs]
        v = np.asarray(ex.score_binary(pr_ab + pr_ba, pos="A", neg="B"), float)
        names.append(m["name"])
        kinds.append(m["kind"])
        AB.append(v[: len(pairs)])
        BA.append(v[len(pairs):])
        if (len(names) % 20) == 0 or mi == len(frz["metrics"]) - 1:
            _save()
        print(f"[pair] {mi+1}/{len(frz['metrics'])} {m['name'][:48]} "
              f"nan={float(np.mean(~np.isfinite(v))):.2f}", flush=True)
    _save()
    print(f"[pair] DONE -> {a.out}", flush=True)


def run_executor(a):
    model, family, params = EXECUTORS[a.executor]
    frz = json.load(open(a.freeze))
    battery = json.load(open(a.battery))
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), frz["meta"]["task"])
    cfg0 = cfgmod.ImplementerConfig()
    if a.fake:
        cfg0.vllm_fake = True
    from ..vllm_backend import make_judge_backend
    ex = make_judge_backend(model, cfg0, temperature=None)
    n_probes = int(frz["meta"]["n_probes"])
    texts, _ = _load_texts(frz["meta"]["task"], 60 + n_probes, cfg)
    probes = texts[60: 60 + n_probes]
    mx = cfg.max_text_chars
    print(f"[osl] executor={a.executor} ({model}) probes={len(probes)} "
          f"metrics={len(frz['metrics'])}", flush=True)

    bat = score_battery(ex, battery["items"], mx)
    print(f"[osl] battery auc={bat['auc']:.3f} z={bat['z']:.3f} nan={bat['nan_rate']:.3f}",
          flush=True)

    planted_truth = {m["name"]: m["truth"]
                     for m in planted_metrics(probes, int(frz["meta"]["k_med_words"]))}
    tau0 = ap.noise_floor_tau(ex, frz["metrics"][0]["rubric"], probes[:60], mx,
                              template=ap._YESNO_TEXTFIRST)
    rows, done = [], set()
    if os.path.exists(a.out):                                    # resumable per metric
        prior = json.load(open(a.out))
        rows = prior.get("metrics", [])
        done = {r["name"] for r in rows}
        print(f"[osl] resuming: {len(done)} metrics already done", flush=True)
    for mi, m in enumerate(frz["metrics"]):
        if m["name"] in done:
            continue
        orb = ap.orbit_metric_verdict(ex, m["rubric"], probes, mx, n_forms=4,
                                      template=ap._YESNO_TEXTFIRST)
        m_bar = np.asarray(orb["m_bar"], float)
        nan_rate = float(np.mean(~np.isfinite(m_bar)))
        std = float(np.nanstd(m_bar))
        excluded = bool(std < a.std_floor or nan_rate > 0.10)
        row = {"name": m["name"], "kind": m["kind"], "std": std, "nan_rate": nan_rate,
               "flip_rate": orb["flip_rate"], "excluded": excluded}
        if not excluded:
            S = np.vstack([ap.signature(ex, c, probes, mx, template=ap._YESNO_TEXTFIRST)
                           for c in m["criteria"]])
            rec = greedy_recovery(S, m_bar)
            odd = np.arange(1, len(m_bar), 2)
            rel = form_split_reliability(np.asarray(orb["per_form"], float), odd)
            y_dis = (float(np.clip(rec["r_odd"] / np.sqrt(rel), 0, 1.1))
                     if np.isfinite(rel) and rel > 0.05 else None)
            row.update({"r_odd": rec["r_odd"], "bits": rec["bits"], "rel_form": rel,
                        "y_dis": y_dis, "n_selected": len(rec["selected"]),
                        "sig_nan_rate": float(np.mean(~np.isfinite(S)))})
            if m["kind"] == "planted":
                row["truth_auc"] = _auc(m_bar, np.array(planted_truth[m["name"]], int))
        rows.append(row)
        _dump(a, model, family, params, bat, tau0, rows)
        status = "EXCL" if excluded else "r={:.3f}".format(row.get("r_odd", 0.0))
        print(f"[osl] {mi+1}/{len(frz['metrics'])} {m['name'][:44]:44s} {status}", flush=True)
    _dump(a, model, family, params, bat, tau0, rows, final=True)
    print(f"[osl] DONE -> {a.out}", flush=True)


def _dump(a, model, family, params, bat, tau0, rows, final=False):
    inc = [r for r in rows if not r["excluded"] and r["kind"] == "bank"
           and r.get("y_dis") is not None]
    base = {  # base-measurement vector for the SECONDARY data-driven latent (PCA/PLS in osl_fit)
        "battery_auc": bat["auc"], "battery_nan": bat["nan_rate"], "tau0": float(tau0),
        **{f"battery_{k}": v for k, v in bat["per_family"].items()},
        "mean_std_mbar": float(np.mean([r["std"] for r in rows])) if rows else None,
        "mean_rel_form": (float(np.nanmean([r.get("rel_form", np.nan) for r in rows]))
                          if rows else None),
        "mean_flip_rate": (float(np.nanmean([r.get("flip_rate", np.nan) for r in rows]))
                           if rows else None),
    }
    json.dump({"executor": a.executor, "model": model, "family": family, "params": params,
               "battery": bat, "tau0": float(tau0), "base_measurements": base,
               "n_included_bank": len(inc),
               "mean_y_dis_bank": (float(np.mean([r["y_dis"] for r in inc])) if inc else None),
               "metrics": rows, "final": final},
              open(a.out, "w"), indent=1, default=float)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--build-freeze", action="store_true")
    p.add_argument("--mbar-only", action="store_true",
                   help="score only m_bar per metric, save vectors npz (consensus y-axis input)")
    p.add_argument("--pairwise", action="store_true",
                   help="forced-choice A/B over fixed pairs (pilot arm alongside P(YES))")
    p.add_argument("--battery-only", action="store_true",
                   help="score only the capability battery (expensive executors)")
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--n-forms", type=int, default=4,
                   help="orbit forms for mbar (1 = canonical-only diagnostic)")
    p.add_argument("--probe-window", default=None,
                   help="lo:hi text slice override (e.g. 600:900 for the frontier floor extension)")
    p.add_argument("--src-dir", help="freeze: dir of *_sigs.npz (silver_r2/humor)")
    p.add_argument("--bank-r2", help="freeze: <task>_general_r2_expanded.json")
    p.add_argument("--task", default="humor")
    p.add_argument("--n-metrics", type=int, default=40)
    p.add_argument("--min-criteria", type=int, default=10,
                   help="freeze: min mined criteria per metric; 0 = admit every described "
                        "hierarchy metric (mbar-only panels don't execute criteria)")
    p.add_argument("--k-criteria", type=int, default=60)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--executor", choices=sorted(EXECUTORS))
    p.add_argument("--freeze", help="run: frozen metric file")
    p.add_argument("--battery", help="run: frozen battery file")
    p.add_argument("--std-floor", type=float, default=0.03)
    p.add_argument("--fake", action="store_true", help="FakeVLLM smoke run (CPU)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.build_freeze:
        build_freeze(a)
    elif a.battery_only:
        if not (a.executor and a.battery):
            sys.exit("battery mode needs --executor --battery")
        run_battery_only(a)
    elif a.pairwise:
        if not (a.executor and a.freeze):
            sys.exit("pairwise mode needs --executor --freeze")
        run_pairwise(a)
    elif a.mbar_only:
        if not (a.executor and a.freeze):
            sys.exit("mbar mode needs --executor --freeze")
        run_mbar(a)
    else:
        if not (a.executor and a.freeze and a.battery):
            sys.exit("run mode needs --executor --freeze --battery")
        run_executor(a)


if __name__ == "__main__":
    main()
