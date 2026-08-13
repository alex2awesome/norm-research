#!/usr/bin/env python3
"""Patents V_new: compile bank_v1's codable criteria into deterministic code
features (vnew_pilot protocol, 2026-08-13). Fills the ladder's V_new column.

Protocol (mirrors methods/taste_decomposition/vnew_pilot/pilot.py):
  * TRIAGE   codex judges CODABILITY of the 26 non-collapsed criteria.
  * COMPILE  label-blind: criterion name+instruction + 8 UNLABELED train-split
             claims; codex (gpt-5.6, effort=high) writes stdlib-only
             score(text)->float functions. No y, no judged scores in any prompt.
  * CERTIFY  execute over eval+test; certification on the EVAL split (declared
             here, before compile): Spearman rho(compiled, judged parent) >= .30
             AND modal share <= .98.
  * REFIT    V_new = V_content + certified columns; V_new and V_new+A legs via the
             frozen fit_arm frame; deconfounded companions; TEST-split OOF is the
             clean readout (certification touched EVAL only).

CPU + codex CLI. Run: python3 patents_vnew.py > patents_vnew.log 2>&1
"""
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / "v3_claimonly"
SC = HERE.parents[1] / "methods/taste_decomposition/closure/patents_claimonly"
TD = HERE.parents[1] / "methods/taste_decomposition"
SCRATCH = D / "vnew_scratch"
SCRATCH.mkdir(exist_ok=True)
MODEL = "gpt-5.6-luna"
RHO_CERT, MODAL_MAX = 0.30, 0.98

def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m
    spec.loader.exec_module(m); return m

F2 = _mod(TD / "fusion/f2_deconf.py", "f2m_vnew")

bank = json.load(open(D / "bank_v1.json"))["bank"]
rep = json.load(open(SC / "patents_claimonly_r0_score_report.json"))
collapsed = {k for k, v in rep["per_criterion"].items() if v["collapsed"]}
live = [c for c in bank if c["id"] not in collapsed]
print(f"[bank] {len(live)}/{len(bank)} non-collapsed criteria", flush=True)

# ---- codex helpers ------------------------------------------------------------
def codex(prompt, timeout=1500):
    wd = SCRATCH / "wd"; wd.mkdir(exist_ok=True)
    r = subprocess.run(["codex", "exec", "--model", MODEL, "-c",
                        "model_reasoning_effort=high", "-s", "read-only",
                        "--skip-git-repo-check", "--cd", str(wd), "-"],
                       input=prompt, capture_output=True, text=True, timeout=timeout)
    raw = r.stdout
    if "tokens used" in raw:
        body = raw.rsplit("tokens used", 1)[1]
        body = body.split("\n", 2)[-1] if body.strip().split("\n")[0].strip().replace(",", "").isdigit() else body
    else:
        body = raw
    return body.strip()

def extract_json(t):
    s, e = t.find("["), t.rfind("]")
    return json.loads(t[s:e + 1])

def extract_module(t):
    m = re.search(r"```(?:python)?\n(.*?)```", t, re.S)
    return m.group(1) if m else t

# ---- 1. TRIAGE -----------------------------------------------------------------
tri_p = SCRATCH / "out_triage.json"
if not tri_p.exists() or len(tri_p.read_text()) < 50:
    listing = "\n".join(f'{c["id"]}\t{c["name"]}\t{c["instruction"][:200]}' for c in live)
    out = codex(
        "You are triaging evaluation criteria for CODABILITY: can the criterion be "
        "approximated by a deterministic Python function of a PATENT CLAIM ELEMENT's "
        "text alone (regex, word lists, structural counts, punctuation)? Judge each. "
        "Return a JSON list of {\"uid\": ..., \"codable\": true/false, \"why\": \"<10 words\"}.\n\n"
        f"CRITERIA (uid<TAB>name<TAB>instruction):\n{listing}\n\nJSON only.")
    tri_p.write_text(out)
tri = extract_json(tri_p.read_text())
codable = [t["uid"] for t in tri if t.get("codable")]
print(f"[triage] {len(codable)}/{len(live)} codable: {codable}", flush=True)

# ---- 2. COMPILE ----------------------------------------------------------------
train = pd.read_csv(D / "arm_t/split/train.csv")
rng = np.random.default_rng(0)
samp = rng.choice(len(train), 8, replace=False)
samples = "\n\n".join(f"--- SAMPLE ITEM {k+1} ---\n"
                      + train.text.iloc[i].replace("CLAIM ELEMENT:\n", "")[:900]
                      for k, i in enumerate(samp))
by_uid = {c["id"]: c for c in live}
batches = [codable[i:i + 6] for i in range(0, len(codable), 6)]
for b, uids in enumerate(batches):
    outp = SCRATCH / f"out_compile_{b:02d}.py"
    if outp.exists() and len(outp.read_text()) > 200:
        continue
    crit = "\n\n".join(f'UID: {u}\nNAME: {by_uid[u]["name"]}\n'
                       f'JUDGE INSTRUCTION: {by_uid[u]["instruction"]}' for u in uids)
    fn_list = ", ".join(f'"{u}"' for u in uids)
    out = codex(f"""Compile each evaluation criterion below into a DETERMINISTIC Python function of the
item text (items are single PATENT CLAIM ELEMENTS; samples below). Rules:
- stdlib only (re, math, string, collections). No I/O, no network, no randomness.
- def score(text: str) -> float, higher = MORE of the named property; return 0.0 on
  degenerate input; must never raise.
- Approximate honestly: regex for antecedent basis, terms-of-degree word lists,
  punctuation/structure counts are all fine. Do NOT return constants; make the
  function discriminate between plausible claim elements.
- Output ONE Python module, nothing else, defining REGISTRY = {{uid: fn}} with exactly
  these uids: [{fn_list}], where each fn is a distinct top-level function.

CRITERIA:
{crit}

UNLABELED SAMPLE ITEMS (format familiarity only):
{samples}

Python module only, no prose, no markdown fences.""")
    outp.write_text(extract_module(out))
    print(f"[compile] batch {b} -> {len(out)} chars", flush=True)

# ---- 3. CERTIFY ----------------------------------------------------------------
z = np.load(SC / "patents_claimonly_r0_scores.npz", allow_pickle=True)
J, j_ids = z["X"], [str(x) for x in z["row_id"]]
cid_order = [str(c) for c in z["crit_ids"]]
jcol = {c: J[:, k] for k, c in enumerate(cid_order)}
jrow = {rid: i for i, rid in enumerate(j_ids)}

REG = {}
for p in sorted(SCRATCH.glob("out_compile_*.py")):
    ns = {}
    try:
        exec(compile(p.read_text(), str(p), "exec"), ns)  # noqa: S102 — pilot protocol
        REG.update(ns.get("REGISTRY", {}))
    except Exception as e:
        print(f"[certify] {p.name} module error: {e}", flush=True)
print(f"[certify] {len(REG)} compiled functions", flush=True)

rows = []
for sp in ("eval", "test"):
    d = pd.read_csv(D / f"arm_t/split/{sp}.csv")
    pt = pd.read_csv(D / f"arm_t/rm_out_seed42/preds_{sp}.csv")
    d["el"] = d.text.str.replace("CLAIM ELEMENT:\n", "", regex=False)
    d["split"], d["T"] = sp, pt.prob
    rows.append(d[["row_id", "el", "judgement", "group", "split", "T"]])
E = pd.concat(rows, ignore_index=True)

from scipy.stats import spearmanr
cert, cols, names = {}, [], []
ev = (E.split == "eval").values
for uid, fn in REG.items():
    try:
        v = np.array([float(fn(t)) for t in E.el])
    except Exception as e:
        cert[uid] = {"pass": False, "why": f"runtime: {e}"}
        continue
    jidx = np.array([jrow[r] for r in E.row_id])
    parent = jcol[uid][jidx]
    ok = np.isfinite(parent) & ev
    rho = float(spearmanr(v[ok], parent[ok]).statistic) if ok.sum() > 50 else 0.0
    vals, counts = np.unique(v[ev], return_counts=True)
    modal = float(counts.max() / ev.sum())
    ok_cert = bool(abs(rho) >= RHO_CERT and modal <= MODAL_MAX)
    cert[uid] = {"pass": ok_cert, "rho_eval": rho, "modal_eval": modal,
                 "name": by_uid[uid]["name"]}
    if ok_cert:
        cols.append(v if rho >= 0 else -v)
        names.append(uid)
n_pass = sum(1 for c in cert.values() if c.get("pass"))
print(f"[certify] {n_pass}/{len(REG)} certified: {names}", flush=True)
json.dump(cert, open(D / "vnew_cert.json", "w"), indent=1)

# ---- 4. REFIT -------------------------------------------------------------------
DEP = re.compile(r"\bof claim (\d+)\b", re.I)
def content_feats(el):
    words = el.split(); dep = DEP.search(el)
    return [1.0 if dep else 0.0, float(len(el)), float(len(words)),
            float(np.mean([len(w) for w in words])) if words else 0.0,
            float(el.count(",")), float(el.count(";")),
            float(len(re.findall(r"\bwherein\b", el, re.I))),
            float(len(re.findall(r"\d+(?:\.\d+)?", el)))]
y, groups, T = E.judgement.astype(int).values, E.group.astype(str).values, E["T"].values
V = np.array([content_feats(e) for e in E.el])
strata = pd.read_csv(D / "harvest_strata_NEVER_AN_INPUT.csv")
NUIS = pd.concat([strata[strata.split == sp] for sp in ("eval", "test")], ignore_index=True)[
    ["claim_num", "parent_claim_num", "is_dependent", "char_len", "word_len"]].astype(float).values
jidx = np.array([jrow[r] for r in E.row_id])
A = np.column_stack([jcol[c["id"]][jidx] for c in live])

out = {"n_certified": n_pass, "certified_uids": names}
if cols:
    Vn = np.column_stack([V] + cols)
    te = (E.split == "test").values
    from sklearn.metrics import roc_auc_score
    for tag, M in (("V", V), ("V_new", Vn),
                   ("VA", np.column_stack([V, A])), ("Vnew_A", np.column_stack([Vn, A]))):
        r = F2.fit_arm("clean_once", M, T, y, groups)
        out[tag] = {"nl_pooled": r["VA_nl_mean"],
                    "nl_oof_test": float(roc_auc_score(y[te], r["_oof_VA_nl0"][te])),
                    "nl_oof_eval": float(roc_auc_score(y[ev], r["_oof_VA_nl0"][ev]))}
        print(f"[refit] {tag}: pooled {out[tag]['nl_pooled']:.4f} "
              f"test {out[tag]['nl_oof_test']:.4f}", flush=True)
    r_full = F2.fit_arm("clean_once", np.column_stack([Vn, A, NUIS]), T, y, groups)
    out["deconf_Vnew_A"] = {
        "c": r_full["VA_nl_mean"], "d": r_full["VAT_nl_mean"],
        "residual": F2.gboot(y, r_full["_oof_VAT_nl0"], r_full["_oof_VA_nl0"],
                             groups, n_boot=2000)}
json.dump(out, open(D / "vnew_results.json", "w"), indent=1, default=float)
print("PATENTS_VNEW_DONE", flush=True)
