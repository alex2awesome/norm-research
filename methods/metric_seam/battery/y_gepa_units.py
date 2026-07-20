"""y-GEPA seam, stage 2: unit-decompose the y-optimized prompt, compile units to code,
and measure the y-seam the same way the m-seam is measured (user design, 2026-07-19).

  python3 y_gepa_units.py decompose <task>      GLM: frozen prompt -> units JSON
  python3 y_gepa_units.py buildunits <task>     per-unit mini-prompts over TRAIN+TEST
  python3 y_gepa_units.py codegen <task>        GLM: one code implementation per unit
                                                (or NOT_CODABLE with reason)
  python3 y_gepa_units.py eval <task> <unit_results.jsonl>
      ladders (train-fit logistic over unit features, held-out test y-AUC):
        P   : all prompt-units
        C   : all code-units (codable subset)
        MIX : code where codable, prompt-unit otherwise
        + greedy TRAIN-selected swap curve -> %code at iso-AUC
"""
import json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
import battery_common as bc
from y_seam_extend import items_with_y, auc, TASKS as YTASKS
from y_seam_vtier import stratified_auc
from y_gepa import glm_call, MARK, FOOT, MAXCHARS, OUT as GOUT

bc.PROGDIR.update({"legal_title_vii": "programs_legal", "peer_review": "programs_peer",
                   "patents_pa": "programs_pa", "code_review": "programs_code_review"})
OUT = GOUT  # same y_gepa/ dir


def units_path(task):
    return OUT / f"{task}_units.json"


def cmd_decompose(task):
    frozen = json.load(open(OUT / f"{task}_final_frozen.json"))
    noun, q = YTASKS[task]
    system = ("You decompose a forecasting prompt into its distinct evaluation UNITS — the "
              "separable factors/criteria the prompt tells the scorer to weigh. Return STRICT "
              "JSON: a list of objects {\"unit_id\": \"u1\"..., \"name\": short name, "
              "\"question\": a fully self-contained scoring question for JUST this factor "
              "(a scorer who sees only this question and the document must be able to answer "
              "0-10), \"codable_guess\": true/false — could a deterministic Python function "
              "over the raw document text plausibly compute this factor?} "
              "Between 4 and 10 units. No prose outside the JSON.")
    user = (f"The prompt forecasts whether {q} for a {noun}.\n\nPROMPT:\n---\n"
            f"{frozen['prompt']}\n---\n\nDecompose into units. JSON only.")
    raw = glm_call(system, user, max_tokens=3000).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        raw = raw[4:] if raw.startswith("json") else raw
    units = json.loads(raw)
    assert isinstance(units, list) and 4 <= len(units) <= 12
    json.dump(units, open(units_path(task), "w"), indent=1)
    for u in units:
        print(f"  {u['unit_id']}: {u['name']}  (codable_guess={u['codable_guess']})")
    print(f"-> {units_path(task)} ({len(units)} units)")


def cmd_buildunits(task):
    units = json.load(open(units_path(task)))
    ctx = bc.load_ctx(task)
    iy = items_with_y(task)
    noun, _ = YTASKS[task]
    outp = OUT / f"{task}_unit_prompts.jsonl"
    n = 0
    with open(outp, "w") as f:
        for u in units:
            body = (f"You are evaluating ONE factor of a single {noun}.\n\n"
                    f"Factor: {u['name']}\nQuestion: {u['question']}\n\nDocument:\n{MARK}\n\n"
                    "Score this factor only, 0 (strongly unfavorable/absent) to 10 (strongly "
                    "favorable/present). Ignore everything else about the document.")
            for d in sorted(ctx["train"]) + sorted(ctx["test"]):
                if iy.get(d, ("", None))[1] not in (0, 1):
                    continue
                t = iy[d][0][:MAXCHARS]
                f.write(json.dumps({"channel": "field",
                                    "aspect_id": f"{task}.{u['unit_id']}.unit",
                                    "datapoint_id": d,
                                    "prompt": body.replace(MARK, t) + FOOT}) + "\n")
                n += 1
    print(f"wrote {n} rows ({len(units)} units x train+test) -> {outp}")


def cmd_codegen(task):
    units = json.load(open(units_path(task)))
    cdir = OUT / f"{task}_unit_code"; cdir.mkdir(exist_ok=True)
    noun, _ = YTASKS[task]
    manifest = {}
    for u in units:
        system = ("You write a deterministic Python function implementing ONE evaluation factor "
                  "over raw document text. Return ONLY Python source (no fences) defining "
                  "`def score(text: str) -> float` (roughly 0..1, higher = factor more "
                  "favorable/present), stdlib only (re, math, collections), robust "
                  "(try/except -> 0.5), no catastrophic regex. If the factor genuinely CANNOT "
                  "be computed from text by deterministic code, return exactly the single line: "
                  "NOT_CODABLE: <one-line reason>")
        user = (f"Document type: {noun}.\nFactor: {u['name']}\nQuestion: {u['question']}\n\n"
                "Write the code (or NOT_CODABLE line).")
        raw = glm_call(system, user, max_tokens=2500).strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1][6:] if parts[1].startswith("python") else parts[1]
            raw = raw.strip()
        if raw.startswith("NOT_CODABLE"):
            manifest[u["unit_id"]] = {"codable": False, "reason": raw[:200]}
            print(f"  {u['unit_id']} {u['name']}: NOT_CODABLE")
            continue
        fp = cdir / f"{u['unit_id']}.py"
        fp.write_text(raw)
        # smoke: import + run on one doc
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"unit_{u['unit_id']}", fp)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            v = float(mod.score("test document text"))
            manifest[u["unit_id"]] = {"codable": True, "file": fp.name}
            print(f"  {u['unit_id']} {u['name']}: code OK (smoke={v:.2f})")
        except Exception as e:
            manifest[u["unit_id"]] = {"codable": False,
                                      "reason": f"codegen broken: {type(e).__name__}"}
            print(f"  {u['unit_id']} {u['name']}: BROKEN ({type(e).__name__}) -> not codable")
    json.dump(manifest, open(OUT / f"{task}_unit_code_manifest.json", "w"), indent=1)
    nc = sum(1 for m in manifest.values() if m["codable"])
    print(f"-> {nc}/{len(units)} units codable")


def cmd_eval(task, unit_results):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    units = json.load(open(units_path(task)))
    manifest = json.load(open(OUT / f"{task}_unit_code_manifest.json"))
    ctx = bc.load_ctx(task)
    iy = items_with_y(task)
    train, test = sorted(ctx["train"]), sorted(ctx["test"])

    # prompt-unit columns from Gemma results
    pcols = {}
    for line in open(unit_results):
        r = json.loads(line)
        a = r.get("aspect_id", "")
        if a.endswith(".unit") and isinstance(r.get("score"), int):
            pcols.setdefault(a.split(".")[1], {})[r["datapoint_id"]] = r["score"]
    # code-unit columns
    import importlib.util
    ccols = {}
    for uid, m in manifest.items():
        if not m.get("codable"):
            continue
        fp = OUT / f"{task}_unit_code" / m["file"]
        spec = importlib.util.spec_from_file_location(f"unit_{uid}", fp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        col = {}
        for d in train + test:
            txt = iy.get(d, ("", None))[0]
            try:
                col[d] = float(mod.score(txt))
            except Exception:
                col[d] = 0.5
        ccols[uid] = col
    uids = [u["unit_id"] for u in units]
    codable = [u for u in uids if u in ccols]
    print(f"units: {len(uids)} | prompt-scored: {len(pcols)} | codable: {len(codable)} "
          f"({', '.join(codable)})")

    def ladder(feature_cols, label):
        X = {}
        for split, ids in [("tr", train), ("te", test)]:
            rows, ys, ls = [], [], []
            for d in ids:
                lab = iy.get(d, ("", None))[1]
                if lab not in (0, 1):
                    continue
                row = [feature_cols[u].get(d) for u in feature_cols]
                if any(v is None for v in row):
                    continue
                rows.append(row); ys.append(lab); ls.append(len(iy[d][0]))
            X[split] = (np.array(rows, float), np.array(ys), ls)
        (Xtr, ytr, _), (Xte, yte, lte) = X["tr"], X["te"]
        if len(Xtr) < 40 or len(set(ytr)) < 2:
            return None
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=3000).fit((Xtr - mu) / sd, ytr)
        ptr = clf.predict_proba((Xtr - mu) / sd)[:, 1]
        pte = clf.predict_proba((Xte - mu) / sd)[:, 1]
        atr, _, _ = auc(list(ptr), list(ytr))
        ate, _, _ = auc(list(pte), list(yte))
        sate = stratified_auc(list(pte), list(yte), lte)
        print(f"  {label:28s} train={atr:.3f}  TEST={ate:.3f}  strat={sate:.3f} "
              f"(k={Xtr.shape[1]})")
        return dict(label=label, k=int(Xtr.shape[1]), auc_train=round(atr, 4),
                    auc_test=round(ate, 4), auc_test_strat=round(sate, 4))

    results = []
    results.append(ladder({u: pcols[u] for u in uids if u in pcols}, "P: all prompt-units"))
    if codable:
        results.append(ladder({u: ccols[u] for u in codable}, "C: code-units only"))
        mix = {u: (ccols[u] if u in ccols else pcols[u]) for u in uids if u in ccols or u in pcols}
        results.append(ladder(mix, "MIX: code where codable"))
    # greedy TRAIN-selected swap curve
    curve = []
    current = {u: pcols[u] for u in uids if u in pcols}
    swapped = []
    base = ladder(dict(current), "swap0 (all prompt)")
    curve.append(dict(n_code=0, **{k: base[k] for k in ("auc_train", "auc_test")}))
    remaining = [u for u in codable if u in current]
    while remaining:
        best_u, best_a = None, -1
        for u in remaining:
            trial = dict(current); trial[u] = ccols[u]
            r = ladder(trial, f"trial swap {u}")
            if r and r["auc_train"] > best_a:
                best_a, best_u, best_r = r["auc_train"], u, r
        current[best_u] = ccols[best_u]
        swapped.append(best_u); remaining.remove(best_u)
        curve.append(dict(n_code=len(swapped), swapped=list(swapped),
                          auc_train=best_r["auc_train"], auc_test=best_r["auc_test"]))
    out = dict(task=task, n_units=len(uids), n_codable=len(codable),
               pct_codable=round(len(codable) / len(uids), 3),
               ladders=[r for r in results if r], swap_curve=curve)
    json.dump(out, open(OUT / f"{task}_unit_seam.json", "w"), indent=1)
    print(f"-> {OUT / f'{task}_unit_seam.json'}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"decompose": cmd_decompose, "buildunits": cmd_buildunits,
     "codegen": cmd_codegen}.get(cmd, lambda t: cmd_eval(t, sys.argv[3]))(sys.argv[2])
