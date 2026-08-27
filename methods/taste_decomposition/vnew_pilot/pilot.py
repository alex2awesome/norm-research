#!/usr/bin/env python3
"""V_new COMPILATION PILOT (2026-08-11): compile terminal-bank judged criteria into
deterministic code features; certify held-out; refit the frozen Layer-1 stacks.

Cells: jokes_community (priority: read-aloud cadence), press_verdict, nc_responded.
Method: seam-program discipline (label-blind compile from criterion TEXT, held-out
certification vs the judged parent), estimator = the closure campaigns' frozen
closure_core.fit_block (identical md5 across cells; grid {15,31}, lr .06, seeds 0/1/2,
grouped OOF). Compiler fleet: codex `gpt-5.6-luna` (family recorded per artifact).

Subcommands (one cell per process, CPU only):
  bank    -> vnew_out/bank_<cell>.json + judged_<cell>.npz  (criterion inventory)
  prompts -> scratch triage + compile prompt files (label-blind: names/instructions
             + UNLABELED fit-split sample texts only; no y, no judged scores)
  codex   -> run codex exec over pending prompt files
  certify -> exec compiled functions over all rows; certify on MONITOR rows
             (rho vs judged parent, alone-AUC, collapse gate); vnew_out/cert_<cell>.json
  refit   -> fit_block legs V / V_new / V+A / V_new+A; vnew_out/results_<cell>.json
             + ids-carried OOF npz
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent / "closure"
OUT = HERE / "vnew_out"
OUT.mkdir(exist_ok=True)
SCRATCH = Path("/tmp/vnew_pilot_scratch")
SCRATCH.mkdir(exist_ok=True)
MODEL = "gpt-5.6-luna"
RHO_CERT = 0.30          # declared certification floor (held-out Spearman vs judged parent)
MODAL_MAX = 0.98         # enforced collapse gate, mirrored from the campaigns

sys.path.insert(0, str(CLOSURE / "jokes_community"))   # closure_core (md5-identical copies)
import closure_core as CC  # noqa: E402


# ------------------------------------------------------------- cell adapters --
def load_cell(cell):
    if cell in ("jokes_community", "press_verdict"):
        sys.path.insert(0, str(CLOSURE / cell))
        for m in ("cells", "closure_core"):
            if m in sys.modules:
                del sys.modules[m]
        C = importlib.import_module("cells")
        d = C.load(cell)
        ids = [str(s) for s in d["ids"]]
        sp = json.loads((CLOSURE / cell / f"{cell}_splits.json").read_text())
        rows = sp["rows"]
        by_id = {str(r["id"]): r for r in rows}
        split = np.array([by_id[i]["split"] for i in ids], dtype=object)
        out = dict(ids=ids, y=np.asarray(d["y"]).astype(int), groups=np.asarray(d["groups"], dtype=object),
                   texts=[str(t) for t in d["texts"]], V=np.asarray(d["V"], float),
                   v_names=d["v_names"], A=np.asarray(d["A"], float), a_names=d["a_names"],
                   split=split)
        out["mined"] = load_mined_routing(cell, ids)
        return out
    if cell == "nc_responded":
        sys.path.insert(0, str(CLOSURE / cell))
        L = importlib.import_module("nc_closure_lib")
        d = L.load_population()
        ids = [str(s) for s in d["doc_id"]]
        spl = L.load_splits()
        split = np.asarray(spl[1])
        # certification/refit holdout = monitor_full (1,892 rows, docket-disjoint
        # from fit_mine; the 377-row decision monitor is too small for stacks) --
        # DECLARED deviation, both labels kept in the split vector.
        split = np.array([("monitor" if s in ("monitor", "monitor_full") else s)
                          for s in split], dtype=object)
        out = dict(ids=ids, y=np.asarray(d["y"]).astype(int),
                   groups=np.asarray(d["docket"], dtype=object),
                   texts=[str(t) for t in d["texts"]], V=np.asarray(d["V"], float),
                   v_names=[str(s) for s in d["v_names"]],
                   A=np.asarray(d["A"], float),
                   a_names=[str(s) for s in d["a_names"]],
                   split=split)
        out["mined"] = load_mined_nc(ids)
        return out
    raise ValueError(cell)


def load_mined_routing(cell, ids):
    """A-routed, non-collapsed mined criteria across all rounds (incumbent columns)."""
    mined = []
    r = 1
    while (CLOSURE / cell / f"{cell}_r{r}_scores.npz").exists():
        z = np.load(CLOSURE / cell / f"{cell}_r{r}_scores.npz", allow_pickle=True)
        rt = json.loads((CLOSURE / cell / f"{cell}_r{r}_routing_final.json").read_text())
        rep = json.loads((CLOSURE / cell / f"{cell}_r{r}_score_report.json").read_text()) \
            if (CLOSURE / cell / f"{cell}_r{r}_score_report.json").exists() else {"per_criterion": {}}
        cids = [str(s) for s in z["crit_ids"]]
        cnames = [str(s) for s in z["crit_names"]]
        rowids = [str(s) for s in (z["row_id"] if "row_id" in z else z["id"])] if ("row_id" in z or "id" in z) else None
        X = np.asarray(z["X"], float)
        if rowids is not None:
            order = {d_: i for i, d_ in enumerate(rowids)}
            perm = np.array([order[i] for i in ids])
            X = X[perm]
        for x in rt["final"]:
            if x["final_route"] != "A" or x["blind_id"] not in cids:
                continue
            pc = rep["per_criterion"].get(x["blind_id"], {})
            if pc.get("collapsed"):
                continue
            k = cids.index(x["blind_id"])
            mined.append({"uid": f"r{r}:{x['blind_id']}", "name": cnames[k],
                          "instruction": "", "col": X[:, k]})
        r += 1
    return mined


def load_mined_nc(ids):
    mined = []
    for r in range(1, 6):
        f = CLOSURE / "nc_responded" / f"round{r}_scores.npz"
        cf = CLOSURE / "nc_responded" / f"round{r}_criteria_final.json"
        if not (f.exists() and cf.exists()):
            continue
        z = np.load(f, allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        cnames = [str(s) for s in z["crit_names"]]
        rowids = [str(s) for s in z["doc_id"]]
        order = {d_: i for i, d_ in enumerate(rowids)}
        perm = np.array([order[i] for i in ids])
        X = np.asarray(z["X"], float)[perm]
        crit = json.loads(cf.read_text())
        by_name = {c["name"]: c for c in crit["A"]}
        rep_f = CLOSURE / "nc_responded" / f"round{r}_score_report.json"
        rep = json.loads(rep_f.read_text()) if rep_f.exists() else {"per_criterion": {}}
        for k, (cid, nm) in enumerate(zip(cids, cnames)):
            c = by_name.get(nm)
            if c is None:        # B-routed or unmatched -> skip (bank = A side)
                continue
            if rep["per_criterion"].get(cid, {}).get("collapsed"):
                continue
            mined.append({"uid": f"r{r}:{c['id']}", "name": nm,
                          "instruction": c.get("instruction", ""), "col": X[:, k]})
    return mined


def masks(d):
    fitm = d["split"] == "fit_mine"
    monm = d["split"] == "monitor"
    assert fitm.sum() and monm.sum(), f"bad splits: {set(d['split'])}"
    return fitm, monm


# ------------------------------------------------------------------ commands --
def cmd_bank(cell):
    d = load_cell(cell)
    fitm, monm = masks(d)
    y = d["y"]
    from sklearn.metrics import roc_auc_score

    def alone(col, m):
        c = np.asarray(col, float)
        mm = m & np.isfinite(c)
        if mm.sum() < 50 or len(set(y[mm])) < 2 or np.nanstd(c[mm]) == 0:
            return None
        return float(roc_auc_score(y[mm], c[mm]))

    entries, X_j, names = [], [], []
    for j, nm in enumerate(d["a_names"]):
        entries.append({"uid": f"base:{j:02d}", "name": nm, "family_source": "base_bank",
                        "instruction": "", "alone_auc_fitmine": alone(d["A"][:, j], fitm)})
        X_j.append(d["A"][:, j]); names.append(f"base:{j:02d}")
    for m in d["mined"]:
        entries.append({"uid": m["uid"], "name": m["name"], "family_source": "mined",
                        "instruction": m["instruction"][:600],
                        "alone_auc_fitmine": alone(m["col"], fitm)})
        X_j.append(m["col"]); names.append(m["uid"])
    (OUT / f"bank_{cell}.json").write_text(json.dumps(
        {"cell": cell, "n_rows": len(d["ids"]), "n_base": len(d["a_names"]),
         "n_mined_A": len(d["mined"]), "entries": entries}, indent=1))
    np.savez_compressed(OUT / f"judged_{cell}.npz", ids=np.array(d["ids"], dtype=object),
                        X=np.column_stack(X_j) if X_j else np.zeros((len(d["ids"]), 0)),
                        uids=np.array(names, dtype=object),
                        split=d["split"], y=d["y"], groups=d["groups"])
    print(f"[{cell}] bank: {len(entries)} judged criteria "
          f"({len(d['a_names'])} base + {len(d['mined'])} mined-A)")


TRIAGE_RULES = """You are triaging evaluation criteria for CODABILITY: can the criterion be
computed by a DETERMINISTIC Python function of the item text alone (stdlib only, regex,
counting, syllable/stress heuristics, punctuation/structure analysis, word lists)?
NO model calls, NO external data, NO semantic understanding at runtime.
codable=true only if a code approximation could plausibly track the criterion's ranking.
Also assign each criterion ONE family:
  lexical_surface | phonetic_prosodic | structural_format | register_pragmatic |
  semantic_content | relational_context | affective_subjective
Return STRICT JSON: a list of {"uid":..., "codable": true/false, "family":..., "why": "<12 words"}."""


def cmd_prompts(cell, stage):
    bank = json.loads((OUT / f"bank_{cell}.json").read_text())
    d = None
    tag = SCRATCH / cell
    tag.mkdir(exist_ok=True)
    if stage == "triage":
        listing = "\n".join(f'{e["uid"]}\t{e["name"]}' + (f'\t{e["instruction"][:200]}' if e["instruction"] else "")
                            for e in bank["entries"])
        (tag / "prompt_triage.txt").write_text(
            f"{TRIAGE_RULES}\n\nCELL: {cell} (items are "
            f"{'reddit jokes' if 'jokes' in cell else 'press releases' if 'press' in cell else 'regulatory public comments'})"
            f"\n\nCRITERIA (uid<TAB>name<TAB>instruction?):\n{listing}\n\nJSON only.")
        print("wrote", tag / "prompt_triage.txt")
        return
    # compile prompts: batches of 6 codable criteria + 8 unlabeled fit-split sample texts
    tri = json.loads((tag / "out_triage.txt").read_text().strip().strip("`").replace("json\n", "", 1))
    codable = [t["uid"] for t in tri if t.get("codable")]
    d = load_cell(cell)
    fitm, _ = masks(d)
    rng = np.random.default_rng(0)
    samp = rng.choice(np.flatnonzero(fitm), 8, replace=False)
    samples = "\n\n".join(f"--- SAMPLE ITEM {k+1} ---\n{d['texts'][i][:900]}" for k, i in enumerate(samp))
    by_uid = {e["uid"]: e for e in bank["entries"]}
    batches = [codable[i:i + 6] for i in range(0, len(codable), 6)]
    for b, uids in enumerate(batches):
        crit = "\n\n".join(f'UID: {u}\nNAME: {by_uid[u]["name"]}\n'
                           + (f'JUDGE INSTRUCTION: {by_uid[u]["instruction"]}' if by_uid[u]["instruction"] else "")
                           for u in uids)
        fn_list = ", ".join(f'"{u}"' for u in uids)
        (tag / f"prompt_compile_{b:02d}.txt").write_text(f"""Compile each evaluation criterion below into a DETERMINISTIC Python function of the
item text (these items are from the {cell} corpus; samples below). Rules:
- stdlib only (re, math, string, collections). No I/O, no network, no randomness.
- def score(text: str) -> float, higher = MORE of the named property; return 0.0 on
  degenerate input; must never raise.
- Approximate honestly: syllable/stress heuristics, punctuation rhythm, word lists,
  structural counts are all fine. Do NOT return constants; make the function
  discriminate between plausible items.
- Output ONE Python module, nothing else, defining REGISTRY = {{uid: fn}} with exactly
  these uids: [{fn_list}], where each fn is a distinct top-level function
  named score_<sanitized uid>.

CRITERIA:
{crit}

UNLABELED SAMPLE ITEMS (format familiarity only):
{samples}

Python module only, no prose, no markdown fences.""")
    print(f"wrote {len(batches)} compile prompts for {len(codable)} codable criteria")


def cmd_codex(cell, timeout=1500):
    tag = SCRATCH / cell
    for p in sorted(tag.glob("prompt_*.txt")):
        outp = tag / p.name.replace("prompt_", "out_")
        if outp.exists() and len(outp.read_text()) > 200:
            print("skip", outp.name)
            continue
        wd = tag / ("wd_" + p.stem)
        wd.mkdir(exist_ok=True)
        cmd = ["codex", "exec", "--model", MODEL, "-c", "model_reasoning_effort=high",
               "-s", "read-only", "--skip-git-repo-check", "--cd", str(wd), "-"]
        t0 = time.time()
        r = subprocess.run(cmd, input=p.read_text(), capture_output=True, text=True, timeout=timeout)
        raw = r.stdout
        (tag / p.name.replace("prompt_", "raw_")).write_text(raw)
        body = raw
        if "tokens used" in raw:
            body = raw.rsplit("tokens used", 1)[1]
            body = body.split("\n", 2)[-1] if body.strip().split("\n")[0].strip().replace(",", "").isdigit() else body
        outp.write_text(body.strip())
        print(f"[{cell}] {p.name} rc={r.returncode} {time.time()-t0:.0f}s -> {len(body)} chars", flush=True)


def _extract_module(txt):
    m = re.search(r"```(?:python)?\n(.*?)```", txt, re.S)
    return m.group(1) if m else txt


def cmd_certify(cell):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    tag = SCRATCH / cell
    d = load_cell(cell)
    fitm, monm = masks(d)
    y = d["y"]
    jz = np.load(OUT / f"judged_{cell}.npz", allow_pickle=True)
    juids = [str(s) for s in jz["uids"]]
    JX = jz["X"]
    regs = {}
    for f in sorted(tag.glob("out_compile_*.txt")):
        ns = {}
        code = _extract_module(f.read_text())
        try:
            exec(compile(code, str(f), "exec"), ns)
        except Exception as e:
            print(f"MODULE FAIL {f.name}: {e}")
            continue
        regs.update(ns.get("REGISTRY", {}))
    print(f"[{cell}] {len(regs)} compiled functions loaded")
    cert, cols, kept = [], [], []
    texts = d["texts"]
    for uid, fn in regs.items():
        vals = np.full(len(texts), np.nan)
        nfail = 0
        for i, t in enumerate(texts):
            try:
                vals[i] = float(fn(t))
            except Exception:
                nfail += 1
        if not np.isfinite(vals).all():
            vals = np.where(np.isfinite(vals), vals, np.nanmedian(vals))
        modal = np.mean(vals == np.round(np.bincount(np.searchsorted(np.unique(vals), vals)).argmax() * 0 + vals[np.argmax(np.bincount(np.searchsorted(np.unique(vals), vals)))]))
        # simpler modal share:
        _, counts = np.unique(vals, return_counts=True)
        modal = counts.max() / len(vals)
        j = juids.index(uid) if uid in juids else None
        row = {"uid": uid, "n_fail": nfail, "modal_share": float(modal), "compiler": MODEL}
        if j is not None:
            jm = monm & np.isfinite(JX[:, j])
            if jm.sum() >= 50:
                row["rho_monitor"] = float(spearmanr(vals[jm], JX[jm, j]).statistic)
                row["judged_alone_auc_monitor"] = (float(roc_auc_score(y[jm], JX[jm, j]))
                                                  if len(set(y[jm])) == 2 else None)
                row["compiled_alone_auc_monitor"] = (float(roc_auc_score(y[jm], vals[jm]))
                                                     if len(set(y[jm])) == 2 else None)
                jf = fitm & np.isfinite(JX[:, j])
                row["rho_fitmine"] = float(spearmanr(vals[jf], JX[jf, j]).statistic)
        row["certified"] = bool(modal <= MODAL_MAX and abs(row.get("rho_monitor") or 0) >= RHO_CERT)
        cert.append(row)
        if row["certified"]:
            cols.append(vals if (row.get("rho_monitor") or 0) >= 0 else -vals)
            kept.append(uid)
    np.savez_compressed(OUT / f"compiled_{cell}.npz", ids=np.array(d["ids"], dtype=object),
                        X=np.column_stack(cols) if cols else np.zeros((len(texts), 0)),
                        uids=np.array(kept, dtype=object))
    (OUT / f"cert_{cell}.json").write_text(json.dumps(
        {"cell": cell, "compiler": MODEL, "rho_floor": RHO_CERT,
         "n_compiled": len(regs), "n_certified": len(kept), "rows": cert}, indent=1))
    ok = [c for c in cert if c["certified"]]
    print(f"[{cell}] certified {len(ok)}/{len(regs)}; "
          f"median |rho_monitor| = {np.median([abs(c.get('rho_monitor') or 0) for c in cert]):.3f}")


def cmd_refit(cell):
    from sklearn.metrics import roc_auc_score
    d = load_cell(cell)
    fitm, monm = masks(d)
    y, groups = d["y"], d["groups"]
    jz = np.load(OUT / f"judged_{cell}.npz", allow_pickle=True)
    A_term = jz["X"]                       # base + mined-A judged columns
    cz = np.load(OUT / f"compiled_{cell}.npz", allow_pickle=True)
    Cnew = cz["X"]
    V = d["V"]
    legs = {"V": [V], "V_new": [V, Cnew], "VA": [V, A_term], "Vnew_A": [V, Cnew, A_term]}
    res = {"cell": cell, "n_rows": len(d["ids"]), "n_compiled_cols": int(Cnew.shape[1]),
           "estimator": "closure_core.fit_block (frozen campaign estimator, seeds 0/1/2)"}
    oof_store = {}
    for nm, blocks in legs.items():
        r = CC.fit_block(blocks, fitm, monm, y, groups)
        res[nm] = {"n_features": r["n_features"],
                   "lin_MONITOR": float(roc_auc_score(y[monm], r["lin_mon"])),
                   "nl_MONITOR": float(roc_auc_score(y[monm], r["nl_mon"])),
                   "nl_MONITOR_per_seed": [float(roc_auc_score(y[monm], p)) for p in r["nl_mon_seeds"]],
                   "nl_OOF_fitmine": float(roc_auc_score(y[fitm], r["oof_nl_fitmine"]))}
        oof_store[nm + "_oof_nl_fitmine"] = r["oof_nl_fitmine"]
        oof_store[nm + "_nl_mon"] = r["nl_mon"]
        print(f"[{cell}] {nm:7s} nl_MONITOR {res[nm]['nl_MONITOR']:.4f} "
              f"lin {res[nm]['lin_MONITOR']:.4f} oof_fitmine {res[nm]['nl_OOF_fitmine']:.4f}")
    np.savez_compressed(OUT / f"oof_{cell}.npz",
                        ids_fitmine=np.array(np.asarray(d["ids"], dtype=object)[fitm], dtype=object),
                        ids_monitor=np.array(np.asarray(d["ids"], dtype=object)[monm], dtype=object),
                        y=y, split=d["split"], **oof_store)
    (OUT / f"results_{cell}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bank", "prompts", "codex", "certify", "refit"])
    ap.add_argument("cell")
    ap.add_argument("--stage", default="triage")
    a = ap.parse_args()
    {"bank": cmd_bank,
     "prompts": lambda c: cmd_prompts(c, a.stage),
     "codex": cmd_codex,
     "certify": cmd_certify,
     "refit": cmd_refit}[a.cmd](a.cell)
