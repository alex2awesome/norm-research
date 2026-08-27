#!/usr/bin/env python
"""zxa_fit v1 — first readout of the z×a articulation–capability surface
(spec: notes/2026-07-08__zxa-articulation-capability-exchange-spec.md).

Inputs (sk3): outputs/osl_multi/freeze_zxa_<task>_v1.json,
  mbar_zxa_<task>_<exec>.npz (local, soft P(YES)), mbar_zxaglm_<task>_<short>.npz (GLM, hard),
  battery z from outputs/osl/<exec>.json (fallback osl_multi/<exec>.json).
Alias discipline: skip any npz with alias_skip; executor glm-45air ≡ glm-47 everywhere.

v2 readouts (descriptive; per task). v1 lesson: SAME-ARM consensus measures prompt
DETERMINACY (the mismatched placebo scores as high as the true dossier — everyone following
the same wrong body agrees with each other). Content transmission needs a FIXED REFERENT:
  1. PLANTED GATES  exec × arm truth-acc (external truth) — mismatched must collapse.
  2. FORM-DETERMINACY  frontier inter-agreement per class × arm (kept, correctly named).
  3. y_ref  CONTENT-ANCHORED consensus: reference = strict frontier consensus AT THE DOSSIER
     ARM of the same base metric; y_ref(exec, base, arm) = agreement of the exec's arm row
     with that fixed reference. Mismatched/padded must now underperform if content matters.
  4. EXCHANGE + PLACEBO contrasts on y_ref per class × exec (paired over metrics).
  5. Per-exec diagnostics: yes_rate / nan_rate / degenerate flag (guards gemma2-27b-style
     readout fragility before interpreting its cells).
Frontier = EXPLICIT list (glm-47, glm-52, llama70b, qwen25-72b, hermes405b ∩ available) —
never a z threshold (hard-z and soft-z scales don't mix).
Saves outputs/osl_multi/zxa_fit_v1.json. Binarize local soft at .5; hard rows as-is
(orderings-not-levels across readout types; within-family comparisons only for levels).

v4 adds the FORMAL FORECAST OBJECT (spec section 2026-07-09):
  6. SCALING per family (llama / qwen25 / glm / gemma2, ≥3 non-degenerate rungs):
     isotonic (PAV) y_ref-vs-z per metric × arm; crossing ẑ_c(a) at criterion c=.75 with
     explicit censoring ('<=zmin' / interior / '>zmax'); β̂_j(a) = ẑ_c(name) − ẑ_c(a) when
     both interior, else a lower bound; pooled-β placebo correction
     β_content = β(dossier) − max(0, β(mismatched), β(padded));
     ẑ*_j: observed (name interior; metric-A) or forecast ẑ_c(dossier)+β_pool (name
     right-censored & dossier interior; metric-B), pooling level labeled;
     N* via the SAME family's battery z ~ a+g·log10(params) map (soft-z families only;
     GLM has no honest params map → z* stays on the GLM hard-z rung scale);
     T* = 20·N* printed as Chinchilla ASSUMPTION-ARITHMETIC, never a finding.
     shift-vs-ΔL first pass: top-rung dossier−name gap + dossier plateau flag
     (flat last step while still < c = articulation-resistant asymptote candidate).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

B = "/lfs/skampere3/0/alexspan"
sys.path.insert(0, f"{B}/norm-research")
OM = f"{B}/outputs/osl_multi"
O = f"{B}/outputs/osl"
ARMS = ["name", "definition", "explanation", "dossier", "dossier_mismatched", "definition_padded"]
TASKS = sys.argv[1].split(",") if len(sys.argv) > 1 else \
    ["humor", "creative_writing", "peer_review", "math"]
FRONTIER = ["glm-47", "glm-52", "llama70b", "qwen25-72b", "hermes405b"]


def battery_z(ex):
    for p in (f"{O}/{ex}.json", f"{OM}/{ex}.json"):
        if os.path.exists(p):
            try:
                return float(json.load(open(p))["battery"]["z"])
            except Exception:
                pass
    return float("nan")


def battery_params(ex):
    for p in (f"{O}/{ex}.json", f"{OM}/{ex}.json"):
        if os.path.exists(p):
            try:
                v = float(json.load(open(p)).get("params") or 0)
                return v if v > 0 else float("nan")
            except Exception:
                pass
    return float("nan")


def fam_of(ex):
    for pre, fam in (("llama", "llama"), ("qwen25", "qwen25"), ("glm", "glm"),
                     ("gemma2", "gemma2"), ("mistral", "mistral"), ("hermes", "hermes")):
        if ex.startswith(pre):
            return fam
    return None


# families whose battery z is the soft AUC-logit AND whose param counts are honest —
# only these get a z->params map (GLM z is hard bal-acc logit, params unknown/aliased)
PARAM_MAP_FAMS = ("llama", "qwen25")


def pav(y):
    """Pool-adjacent-violators isotonic regression (non-decreasing), tiny n."""
    out = []
    for i, v in enumerate(y):
        out.append([float(v), 1.0, [i]])
        while len(out) >= 2 and out[-2][0] > out[-1][0]:
            v2, w2, i2 = out.pop()
            v1, w1, i1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, i1 + i2])
    res = [0.0] * len(y)
    for v, _, idx in out:
        for i in idx:
            res[i] = v
    return res


def crossing(zv, yv, c):
    """First c-crossing of the isotonic curve. Returns (kind, z): kind '=' interior
    (linear interp), '<=' left-censored (already >= c at z_min), '>' right-censored."""
    if yv[0] >= c:
        return ("<=", zv[0])
    if yv[-1] < c:
        return (">", zv[-1])
    for i in range(len(zv) - 1):
        if yv[i] < c <= yv[i + 1]:
            if yv[i + 1] == yv[i]:
                return ("=", zv[i + 1])
            t = (c - yv[i]) / (yv[i + 1] - yv[i])
            return ("=", zv[i] + t * (zv[i + 1] - zv[i]))
    return (">", zv[-1])


def fmt_cross(cr):
    return "-" if cr is None else f"{cr[0]}{cr[1]:.2f}"


def load_task(task):
    frz = json.load(open(f"{OM}/freeze_zxa_{task}_v1.json"))
    meta = frz["meta"]
    info = {}
    for m in frz["metrics"]:
        base, arm = m["name"].rsplit("||", 1)
        info[m["name"]] = (base, arm, m["kind"].split("|")[0])
    rows = {}       # exec -> {entry_name: binarized row}
    hard = {}
    for pat, is_hard in ((f"{OM}/mbar_zxa_{task}_*.npz", False),
                         (f"{OM}/mbar_zxaglm_{task}_*.npz", True)):
        for fp in sorted(glob.glob(pat)):
            if "prenanfix" in fp or fp.endswith(".bak.npz"):
                continue          # NaN-repair backups are not executors
            zz = np.load(fp, allow_pickle=True)
            if "alias_skip" in zz.files and int(np.atleast_1d(zz["alias_skip"])[0]):
                continue
            ex = os.path.basename(fp).replace(f"mbar_zxa_{task}_", "").replace(
                f"mbar_zxaglm_{task}_", "").replace(".npz", "")
            ex = {"glm-45air": "glm-47"}.get(ex, ex)
            mb = np.asarray(zz["m_bar"], float)
            names = [str(x) for x in zz["names"]]
            r = {}
            for i, n in enumerate(names):
                if n in info:
                    row = mb[i]
                    r[n] = np.where(np.isfinite(row), (row > 0.5).astype(float), np.nan) \
                        if not is_hard else row
            if r:
                rows[ex] = r
                hard[ex] = is_hard
    return meta, info, rows, hard


def planted_truth(task, meta):
    from methods.metric_implementer.experiments.osl_sweep import planted_metrics
    from methods.metric_implementer.experiments.run_real_test import _load_texts
    from methods.metric_implementer import config as cfgmod
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), meta["task"])
    n = int(meta["n_probes"])
    texts, _ = _load_texts(meta["task"], 60 + n, cfg)
    # truth must be computed on the TRUNCATED text — the view the executor actually saw
    # (long-text tasks: full-text length rules misalign with the shown slice)
    probes = [t[: cfg.max_text_chars] for t in texts[60: 60 + n]]
    return {m["name"]: np.asarray(m["truth"], float)
            for m in planted_metrics(probes, int(meta["k_med_words"]))}


def agree(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(a[m] == b[m])) if m.sum() >= 30 else float("nan")


def kappa(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return float("nan")
    x, y = a[m], b[m]
    po = float(np.mean(x == y))
    pe = float(np.mean(x) * np.mean(y) + (1 - np.mean(x)) * (1 - np.mean(y)))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def balanced(pred, ref, min_per=8):
    """Chance-robust agreement: mean of per-ref-class accuracies. A constant predictor
    scores .5 exactly (kills the base-rate artifact a skewed executor can exploit)."""
    m = np.isfinite(pred) & np.isfinite(ref)
    if m.sum() < 30:
        return float("nan")
    p, r = pred[m], ref[m]
    accs = [float(np.mean(p[r == v] == v)) for v in (0, 1) if (r == v).sum() >= min_per]
    return float(np.mean(accs)) if len(accs) == 2 else float("nan")


def main():
    report = {}
    for task in TASKS:
        try:
            meta, info, rows, hard = load_task(task)
        except FileNotFoundError:
            continue
        if not rows:
            print(f"== {task}: no panels yet =="); continue
        execs = sorted(rows, key=lambda e: (np.nan_to_num(battery_z(e), nan=-9), e))
        zs = {e: battery_z(e) for e in execs}
        truth = planted_truth(task, meta)
        # diagnostics + degeneracy guard (check-judge-distribution discipline)
        diag = {}
        for e in execs:
            allv = np.concatenate([v for v in rows[e].values()])
            yr = float(np.nanmean(allv))
            nr = float(np.mean(~np.isfinite(allv)))
            diag[e] = {"yes_rate": round(yr, 3), "nan_rate": round(nr, 3),
                       "degenerate": bool(abs(yr - .5) > .45 or nr > .10)}
        print(f"\n===== {task}: {len(execs)} executors "
              f"({', '.join(f'{e}[z={zs[e]:.2f}]' for e in execs)}) =====")
        dg = [e for e in execs if diag[e]["degenerate"]]
        print("-- diagnostics: " + "  ".join(
            f"{e}(yes={diag[e]['yes_rate']},nan={diag[e]['nan_rate']})" for e in execs))
        if dg:
            print(f"   DEGENERATE (interpret with caution): {dg}")
        T = {"execs": {e: {"z": zs[e], "hard": hard[e], **diag[e],
                           "n_entries": len(rows[e])} for e in execs}}

        # 1) planted gates
        gate = defaultdict(dict)
        for e in execs:
            for arm in ARMS:
                accs = []
                for n, (base, a, cls) in info.items():
                    if a == arm and cls == "PLANTED" and n in rows[e] and base in truth:
                        b = balanced(rows[e][n], truth[base])
                        if np.isfinite(b):
                            accs.append(b)
                if accs:
                    gate[e][arm] = round(float(np.mean(accs)), 3)
        T["planted_gate"] = dict(gate)
        for e in execs:
            real = [gate[e][a] for a in ("definition", "dossier") if a in gate[e]]
            if real and float(np.mean(real)) < 0.55 and not diag[e]["degenerate"]:
                diag[e]["degenerate"] = True
                diag[e]["degenerate_reason"] = "planted-chance"
        dg2 = [e for e in execs if diag[e]["degenerate"]]
        if dg2:
            print(f"   DEGENERATE after planted check: {dg2}")
        print("-- planted truth-acc (exec x arm; mismatched must collapse) --")
        hdr = "exec".ljust(14) + "".join(a[:12].rjust(13) for a in ARMS)
        print(hdr)
        for e in execs:
            if gate[e]:
                print(e.ljust(14) + "".join(
                    (f"{gate[e].get(a, float('nan')):.3f}" if a in gate[e] else "    -").rjust(13)
                    for a in ARMS))

        # frontier set: explicit, never a z threshold (hard/soft z scales don't mix)
        frontier = [e for e in FRONTIER if e in rows]
        T["frontier"] = frontier

        # 2) frontier inter-agreement per class x arm
        if len(frontier) >= 2:
            fa = defaultdict(list)
            for n, (base, a, cls) in info.items():
                vecs = [rows[f][n] for f in frontier if n in rows[f]]
                if len(vecs) >= 2:
                    ps = [kappa(vecs[i], vecs[j])
                          for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
                    ps = [p for p in ps if np.isfinite(p)]
                    if ps:
                        fa[(cls, a)].append(float(np.mean(ps)))
            T["frontier_agreement"] = {f"{c}|{a}": round(float(np.mean(v)), 3)
                                       for (c, a), v in fa.items() if v}
            print("-- frontier inter-agreement KAPPA (class x arm; low = underdetermined form) --")
            classes = sorted({c for c, _ in fa})
            print("class".ljust(18) + "".join(a[:12].rjust(13) for a in ARMS))
            for c in classes:
                print(c[:17].ljust(18) + "".join(
                    (f"{np.mean(fa[(c, a)]):.3f}" if fa.get((c, a)) else "    -").rjust(13)
                    for a in ARMS))

        # 3) y_ref: content-anchored consensus — reference fixed at the DOSSIER arm
        if len(frontier) >= 2:
            per_base = defaultdict(dict)  # exec -> (base, arm) -> y_ref
            bases = sorted({b for b, _, _ in info.values()})
            for base in bases:
                dn = f"{base}||dossier"
                cls = info[dn][2]
                for e in execs:
                    refs = [f for f in frontier if f != e and dn in rows[f]]
                    if not refs:      # single-ref allowed (marked): frontier's own rows
                        continue      # tonight = pairwise vs the other GLM rung
                    V = np.vstack([rows[f][dn] for f in refs])
                    fin = np.all(np.isfinite(V), axis=0)
                    strict = fin & (np.nanstd(V, axis=0) == 0)  # frontier agrees on dossier
                    if strict.sum() < 30:
                        continue
                    ref_bits = np.where(strict, V[0], np.nan)
                    for a in ARMS:
                        n = f"{base}||{a}"
                        if n not in rows[e]:
                            continue
                        b = balanced(rows[e][n], ref_bits)
                        if np.isfinite(b):
                            per_base[e][(base, a)] = b
            # cell-level constancy (GEPA discrimination-gate practice propagated to the
            # read side, 2026-07-09): a constant row is EVIDENCE-FREE for the cell — it
            # can't fake success (balanced=.5) but it also isn't a measured .5. Flag it
            # so crossings can be sensitivity-checked with collapsed cells masked.
            cell_const = defaultdict(dict)   # exec -> (base, arm) -> bool
            for e in execs:
                for n, vec in rows[e].items():
                    b_, a_, _ = info[n]
                    fin = np.isfinite(vec)
                    cell_const[e][(b_, a_)] = bool(fin.sum() >= 30
                                                   and float(np.nanstd(vec)) == 0.0)
            ymat = defaultdict(dict)
            for e in execs:
                agg = defaultdict(list)
                for (base, a), y in per_base[e].items():
                    agg[(info[f"{base}||{a}"][2], a)].append(y)
                for (cls, a), v in agg.items():
                    ymat[(cls, a)][e] = round(float(np.mean(v)), 3)
            T["y_ref"] = {f"{c}|{a}": v for (c, a), v in ymat.items()}
            print("-- y_ref: agreement with frontier dossier-consensus (class x arm x exec) --")
            classes = sorted({c for c, _ in ymat})
            for cls in classes:
                print(f"  [{cls}]")
                for a in ARMS:
                    v = ymat.get((cls, a), {})
                    cells = "  ".join(f"{e}:{v[e]:.3f}" for e in execs if e in v)
                    if cells:
                        print(f"    {a.ljust(20)} {cells}")
            # 4) paired exchange + placebo contrasts on y_ref
            contr = defaultdict(dict)
            for e in execs:
                by_cls = defaultdict(dict)
                for (base, a), y in per_base[e].items():
                    by_cls[info[f"{base}||{a}"][2]].setdefault(base, {})[a] = y
                for cls, bs in by_cls.items():
                    pairs = [("definition", "name"), ("explanation", "name"),
                             ("dossier", "name"), ("dossier_mismatched", "name"),
                             ("definition_padded", "name"),
                             ("dossier", "dossier_mismatched"),
                             ("dossier", "definition_padded"),
                             ("dossier", "definition")]
                    for hi, lo in pairs:
                        d = [v[hi] - v[lo] for v in bs.values() if hi in v and lo in v]
                        if len(d) >= 3:
                            contr[(cls, f"{hi}-{lo}")][e] = round(float(np.mean(d)), 3)
            T["contrasts_y_ref"] = {f"{c}|{k}": v for (c, k), v in contr.items()}
            print("-- paired contrasts on y_ref (class | contrast : per-exec) --")
            for (cls, k), v in sorted(contr.items()):
                cells = "  ".join(f"{e}:{v[e]:+.3f}" for e in execs if e in v)
                print(f"  {cls[:16].ljust(17)} {k.ljust(26)} {cells}")

            # 5) METRIC-B TABLE: per tacit-candidate, does the PHRASE fail at frontier
            # while the DOSSIER works? (frontier y_ref per arm, averaged over frontier
            # execs with data; best non-degenerate local shown for the transmissibility
            # contrast). Thresholds descriptive only: fail < .60, work >= .70 balanced.
            locals_ok = [e for e in execs if e not in frontier and not diag[e]["degenerate"]]
            best_loc = max(locals_ok, key=lambda e: np.nan_to_num(zs[e], nan=-9)) \
                if locals_ok else None
            mb_rows = []
            for base in bases:
                if info.get(f"{base}||name", (None, None, None))[2] != "TACIT-CANDIDATE":
                    continue
                def fmean(arm):
                    v = [per_base[e][(base, arm)] for e in frontier
                         if (base, arm) in per_base.get(e, {})]
                    return float(np.mean(v)) if v else float("nan")
                nf, df = fmean("name"), fmean("dossier")
                mf = fmean("dossier_mismatched")
                ln = per_base.get(best_loc, {}).get((base, "name"), float("nan"))
                ld = per_base.get(best_loc, {}).get((base, "dossier"), float("nan"))
                verdict = ("PHRASE-FAILS/DOSSIER-WORKS" if nf < .60 and df >= .70 else
                           "PHRASE-SUFFICIENT" if nf >= .70 else
                           "BOTH-FAIL" if np.isfinite(df) and df < .60 else "MID")
                mb_rows.append({"base": base, "frontier_name": round(nf, 3),
                                "frontier_dossier": round(df, 3),
                                "frontier_mismatched": round(mf, 3),
                                "gap": round(df - nf, 3) if np.isfinite(df - nf) else None,
                                "local_name": round(ln, 3) if np.isfinite(ln) else None,
                                "local_dossier": round(ld, 3) if np.isfinite(ld) else None,
                                "verdict": verdict})
            mb_rows.sort(key=lambda r: -(r["gap"] if r["gap"] is not None else -9))
            T["metric_b_table"] = {"best_local": best_loc, "rows": mb_rows}
            if mb_rows:
                print(f"-- METRIC-B table (tacit candidates; frontier y_ref; local={best_loc}) --")
                print("  base".ljust(46) + "f_name f_doss f_mism   gap  l_name l_doss  verdict")
                for r in mb_rows:
                    print(f"  {r['base'][:44].ljust(44)}"
                          f"{r['frontier_name']:7.3f}{r['frontier_dossier']:7.3f}"
                          f"{r['frontier_mismatched']:7.3f}"
                          f"{(r['gap'] if r['gap'] is not None else float('nan')):+7.3f}"
                          f"{(r['local_name'] if r['local_name'] is not None else float('nan')):7.3f}"
                          f"{(r['local_dossier'] if r['local_dossier'] is not None else float('nan')):7.3f}"
                          f"  {r['verdict']}")

            # 6) SCALING: per-family crossings, beta(a), z* forecast, shift-vs-DeltaL.
            # Within ONE family only (same-family-scaling rule); z axes are family-native
            # (llama/qwen soft AUC-logit, glm hard bal-acc logit) and never pooled.
            C_THRESH = 0.75
            # degenerate rungs STAY in the ladder: their balanced y_ref ~= .5 is the honest
            # left edge (the model can't do the task through ANY prompt form); they are
            # never near a crossing, so they anchor rather than bias the isotonic curve.
            fams = defaultdict(list)
            for e in per_base:
                f = fam_of(e)
                if f and np.isfinite(zs[e]):
                    fams[f].append(e)
            scaling = {}
            fc_rows = []           # per metric x family crossing rows
            for fam, members in sorted(fams.items()):
                lad = sorted(members, key=lambda e: zs[e])
                if len(lad) < 3:
                    continue
                zv_all = [zs[e] for e in lad]
                per_metric = {}
                for base in bases:
                    cls = info.get(f"{base}||dossier", (None, None, None))[2]
                    if cls is None:
                        continue
                    arm_cross, arm_top, arm_cross_m = {}, {}, {}
                    for a in ARMS:
                        pts = [(zs[e], per_base[e].get((base, a))) for e in lad
                               if (base, a) in per_base.get(e, {})]
                        pts = [(z, y) for z, y in pts if np.isfinite(y)]
                        if len(pts) < 3:
                            continue
                        zv = [p[0] for p in pts]
                        yv = pav([p[1] for p in pts])
                        arm_cross[a] = crossing(zv, yv, C_THRESH)
                        arm_top[a] = (yv[-1], yv[-2] if len(yv) >= 2 else float("nan"))
                        if a in ("name", "dossier"):
                            # sensitivity refit with collapsed (constant-row) cells masked
                            pl = [(zs[e], per_base[e].get((base, a))) for e in lad
                                  if (base, a) in per_base.get(e, {})
                                  and not cell_const.get(e, {}).get((base, a), False)]
                            pl = [(z, y) for z, y in pl if np.isfinite(y)]
                            if len(pl) >= 3:
                                arm_cross_m[a] = crossing([p[0] for p in pl],
                                                          pav([p[1] for p in pl]),
                                                          C_THRESH)
                    if not arm_cross:
                        continue
                    cn, cd = arm_cross.get("name"), arm_cross.get("dossier")
                    beta = beta_lb = None
                    if cn and cd and cn[0] == "=" and cd[0] == "=":
                        beta = cn[1] - cd[1]
                    elif cn and cd and cn[0] == ">" and cd[0] == "=":
                        beta_lb = cn[1] - cd[1]      # z_max - z_cross(dossier)
                    top_gap = plateau = None
                    if "dossier" in arm_top and "name" in arm_top:
                        top_gap = arm_top["dossier"][0] - arm_top["name"][0]
                        y_t, y_p = arm_top["dossier"]
                        plateau = bool(np.isfinite(y_p) and (y_t - y_p) < 0.02
                                       and y_t < C_THRESH)
                    def _cross_differs(ca, cm):
                        if ca is None or cm is None:
                            return (ca is None) != (cm is None)
                        return ca[0] != cm[0] or abs(ca[1] - cm[1]) > 0.15
                    coll_sens = any(
                        _cross_differs(arm_cross.get(a), arm_cross_m.get(a))
                        for a in ("name", "dossier") if a in arm_cross)
                    n_const_rungs = sum(
                        1 for e in lad
                        if any(cell_const.get(e, {}).get((base, a), False)
                               for a in ("name", "dossier")))
                    per_metric[base] = {"cls": cls, "cross": arm_cross, "beta": beta,
                                        "beta_lb": beta_lb, "top_gap": top_gap,
                                        "plateau": plateau,
                                        "collapse_sensitive": coll_sens,
                                        "n_const_rungs": n_const_rungs,
                                        "cross_masked": arm_cross_m}
                # pooled beta per class (interior-only), and pooled placebo betas
                pool = defaultdict(list)
                for base, pm in per_metric.items():
                    if pm["beta"] is not None:
                        pool[pm["cls"]].append(pm["beta"])
                    for pa in ("dossier_mismatched", "definition_padded"):
                        cp, cn2 = pm["cross"].get(pa), pm["cross"].get("name")
                        if cp and cn2 and cp[0] == "=" and cn2[0] == "=":
                            pool[f"placebo|{pm['cls']}"].append(cn2[1] - cp[1])
                beta_pool = {k: float(np.median(v)) for k, v in pool.items() if v}
                task_pool = [b for k, v in pool.items() if not k.startswith("placebo")
                             for b in v]
                beta_task = float(np.median(task_pool)) if task_pool else None
                scaling[fam] = {
                    "ladder": {e: round(zs[e], 3) for e in lad},
                    "degenerate_in_ladder": [e for e in lad if diag[e]["degenerate"]],
                    "beta_pool_by_class": {k: round(v, 3) for k, v in beta_pool.items()},
                    "beta_task_pooled": round(beta_task, 3) if beta_task is not None else None,
                    "n_interior_beta": len(task_pool)}
                # z->params map (soft-z families with honest params only)
                pmap = None
                if fam in PARAM_MAP_FAMS:
                    zp = [(zs[e], battery_params(e)) for e in lad
                          if np.isfinite(battery_params(e))]
                    if len(zp) >= 3:
                        zz2 = np.array([p[0] for p in zp])
                        ll = np.log10([p[1] for p in zp])
                        g, a0 = np.polyfit(ll, zz2, 1)
                        pmap = (a0, g, float(zz2.max()))
                        scaling[fam]["z_params_map"] = {
                            "a": round(float(a0), 4), "g": round(float(g), 4),
                            "z_max_obs": round(float(zz2.max()), 3),
                            "points": [(e, round(zs[e], 3),
                                        f"{battery_params(e):.2e}") for e in lad]}
                for base, pm in per_metric.items():
                    zstar = zsrc = None
                    z_is_bound = False
                    cn, cd = pm["cross"].get("name"), pm["cross"].get("dossier")
                    if cn and cn[0] == "=":
                        zstar, zsrc = cn[1], "observed"
                    elif cn and cn[0] == ">" and cd and cd[0] == "=":
                        # name right-censored at the top rung: any z* must EXCEED cn[1].
                        # A pooled-beta forecast below that bound is logically impossible —
                        # clamp, and if beta-pool can't push past the bound, report the
                        # bound itself (N* becomes a lower bound, not an estimate).
                        bp = beta_pool.get(pm["cls"], beta_task)
                        plc = max(0.0, beta_pool.get(f"placebo|{pm['cls']}", 0.0))
                        cand = (cd[1] + max(bp - plc, 0.0)) if bp is not None else None
                        if cand is not None and cand > cn[1]:
                            zstar = cand
                            zsrc = ("forecast|beta=" +
                                    ("class" if pm["cls"] in beta_pool else "task") +
                                    ("|placebo-corr" if plc > 0 else ""))
                        else:
                            zstar, zsrc, z_is_bound = cn[1], "bound|name>zmax", True
                    nstar = None
                    if zstar is not None and pmap is not None:
                        a0, g, zmx = pmap
                        nstar = float(10 ** ((zstar - a0) / g))
                    fc_rows.append({"family": fam, "base": base, "cls": pm["cls"],
                                    "z_star_is_lower_bound": z_is_bound,
                                    "cross_name": pm["cross"].get("name"),
                                    "cross_dossier": pm["cross"].get("dossier"),
                                    "cross_mismatched": pm["cross"].get("dossier_mismatched"),
                                    "beta": pm["beta"], "beta_lb": pm["beta_lb"],
                                    "top_gap": pm["top_gap"], "plateau": pm["plateau"],
                                    "z_star": zstar, "z_star_src": zsrc,
                                    "n_star_params": nstar,
                                    "n_star_extrapolated": bool(
                                        zstar is not None and pmap is not None
                                        and zstar > pmap[2]),
                                    "collapse_sensitive": pm["collapse_sensitive"],
                                    "n_const_rungs": pm["n_const_rungs"],
                                    "cross_name_masked": pm["cross_masked"].get("name"),
                                    "cross_dossier_masked": pm["cross_masked"].get("dossier")})
            T["scaling"] = {"c_thresh": C_THRESH, "families": scaling,
                            "per_metric": [
                                {**r,
                                 "cross_name": fmt_cross(r["cross_name"]),
                                 "cross_dossier": fmt_cross(r["cross_dossier"]),
                                 "cross_mismatched": fmt_cross(r["cross_mismatched"]),
                                 "cross_name_masked": fmt_cross(r["cross_name_masked"]),
                                 "cross_dossier_masked": fmt_cross(r["cross_dossier_masked"])}
                                for r in fc_rows]}
            if scaling:
                print(f"-- SCALING (c={C_THRESH} balanced y_ref; isotonic crossings; "
                      "z family-native) --")
                for fam, s in scaling.items():
                    print(f"  [{fam}] ladder " + " ".join(
                        f"{e}:{v}" for e, v in s["ladder"].items())
                        + f"  pooled-beta by class: {s['beta_pool_by_class']}"
                        + f"  task-pooled: {s['beta_task_pooled']}"
                          f" (n={s['n_interior_beta']})")
                shown = [r for r in fc_rows
                         if r["cls"] in ("TACIT-CANDIDATE", "DIALECT-SUSPECT")
                         and (r["cross_dossier"] or r["cross_name"])]
                if shown:
                    print("  metric x family crossings "
                          "(zc_name / zc_doss / zc_mism | beta | z* [src] | N*):")
                    for r in shown:
                        gt = ">" if r.get("z_star_is_lower_bound") else ""
                        ns = (f" N*{gt or '='}{r['n_star_params']:.1e}"
                              + ("(EXTRAP)" if r["n_star_extrapolated"] else "")
                              + f" T*{gt or '~'}{20 * r['n_star_params']:.1e}tok(assump)"
                              if r["n_star_params"] else "")
                        bt = (f"{r['beta']:+.2f}" if r["beta"] is not None else
                              (f">{r['beta_lb']:.2f}" if r["beta_lb"] is not None else "-"))
                        zst = (f"{gt}{r['z_star']:.2f}[{r['z_star_src']}]"
                               if r["z_star"] is not None else "-")
                        cs = (f"  !COLLAPSE-SENS({r['n_const_rungs']}r:"
                              f"nm{fmt_cross(r['cross_name_masked'])}/"
                              f"ds{fmt_cross(r['cross_dossier_masked'])})"
                              if r.get("collapse_sensitive") else "")
                        print(f"   {r['family'][:6].ljust(7)}{r['cls'][:5].ljust(6)}"
                              f"{r['base'][:36].ljust(37)}"
                              f"{fmt_cross(r['cross_name']).rjust(7)}"
                              f"{fmt_cross(r['cross_dossier']).rjust(7)}"
                              f"{fmt_cross(r['cross_mismatched']).rjust(7)}"
                              f"  b={bt.ljust(7)} z*={zst.ljust(22)}{ns}"
                              + ("  PLATEAU" if r["plateau"] else "") + cs)
                    ncs = sum(1 for r in fc_rows if r.get("collapse_sensitive"))
                    print(f"  collapse-sensitivity: {ncs}/{len(fc_rows)} metricxfamily rows "
                          "change crossing status/z by >0.15 when constant-row cells are "
                          "masked (flagged !COLLAPSE-SENS; interpret those crossings as "
                          "articulation-form-limited, not knowledge-limited)")
        report[task] = T
    _p = f"{OM}/zxa_fit_v1.json"
    try:
        _prev = json.load(open(_p))
    except Exception:
        _prev = {}
    _prev.update(report)
    json.dump(_prev, open(_p, "w"), indent=1, default=float)
    print(f"\nsaved {OM}/zxa_fit_v1.json")


if __name__ == "__main__":
    main()
