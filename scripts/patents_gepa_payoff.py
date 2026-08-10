#!/usr/bin/env python3
"""GEPA payoff readout (evaluate-only): does the GEPA-improved disclosure prompt move the metric?

GEPA lifted Gemma-vs-GLM-5.2 disclosure fidelity 0.843 -> 0.929 test_bal on 150 held-out pairs.
This measures whether that atom-level gain transfers (a) to the full localized-pair pool and
(b) to the outcome-level V/A metric (grouped AUC + recovery MI) — Y is used ONLY here, in the
evaluate-only readout; the prompt never saw it (reconstruction-only preserved).

Stages (run ON sk3, HOME=/lfs):
  run     (GPU, one engine session, resumable):
          pairs — re-verify the 12,488 localized (element,span) pos/neg pairs under SEED and
                  GEPA-BEST prompts; pos/neg disclosure rates + balanced separation, split into
                  holdout / dev_test / dev_train (dev_train = GEPA optimization data, quarantined)
          scale — re-verify all option3 refs (59,937 rows x ~8 refs) with the GEPA prompt,
                  checkpointed per chunk -> option3_gepa_verdicts.jsonl
  readout (CPU): recompute disclosure features [n, any, frac, max_overlap] from (a) cached
          original verdicts (sanity: n must equal CSV a_n_disclose) and (b) GEPA verdicts;
          grouped-5fold-by-app AUC (notebook's grouped_auc, verbatim) of V / V+A_orig / V+A_gepa;
          univariate AUC + recovery MI I(disclosure;fell) per arm.
"""
import argparse, json, os, re, sys

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
OUTD = f"{BASE}/outputs/patents_gepa"
LOCALIZED = f"{PROC}/localize_results_scale_gemma.jsonl"
SCALE = f"{PROC}/option3_claims_gemma_scale.jsonl"
VERDICTS_OUT = f"{PROC}/option3_gepa_verdicts.jsonl"
CSV = f"{BASE}/notebooks/data/patents_va_features.csv"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"

_OBJ = re.compile(r"\{[\s\S]*\}")
# field-first parse: survives outputs whose closing brace was truncated by max_tokens
_DISC = re.compile(r'"discloses"\s*:\s*(true|false)', re.I)


def parse_disc(raw):
    m = _DISC.search(raw or "")
    if m:
        return m.group(1).lower() == "true"
    m = _OBJ.search(raw or "")
    if not m:
        return None
    for fix in (lambda s: s, lambda s: re.sub(r",\s*}", "}", s)):
        try:
            o = json.loads(fix(m.group(0)))
            v = o.get("discloses")
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "yes", "1")
            return bool(v) if v is not None else None
        except Exception:
            continue
    return None


def load_prompts(prompts_file="gepa_prompts.json"):
    d = json.load(open(f"{OUTD}/{prompts_file}"))
    return d["seed_prompt"], d["best"]["prompt"]


def join_spans(sp):
    if isinstance(sp, list):
        sp = " ".join(s for s in sp if isinstance(s, str))
    return (sp or "").strip()


def verify_batch(llm, sp_params, prompt, pairs):
    """pairs: [(element, span)] -> [bool|None]"""
    convs = [[{"role": "user", "content": prompt.format(el=e[:600], span=s[:1200])}]
             for e, s in pairs]
    outs = llm.chat(convs, sp_params)
    return [parse_disc(o.outputs[0].text) for o in outs]


# ---------------- stage: pairs ----------------
def run_pairs(llm, sp_params, prompts_file, suffix):
    out_path = f"{OUTD}/payoff_pairs{suffix}.json"
    if os.path.exists(out_path):
        print(f"[pairs] {out_path} exists, skipping", flush=True)
        return
    seed_prompt, best_prompt = load_prompts(prompts_file)
    dev_train = {json.loads(l)["uid"] for l in open(f"{OUTD}/dev_train.jsonl")}
    dev_test = {json.loads(l)["uid"] for l in open(f"{OUTD}/dev_test.jsonl")}
    rows = []
    for ln in open(LOCALIZED):
        r = json.loads(ln)
        el = (r.get("element") or "").strip()
        sp = join_spans(r.get("spans"))
        if len(el) > 15 and sp and el != "(no per-element breakdown)":
            split = ("dev_train" if r["uid"] in dev_train
                     else "dev_test" if r["uid"] in dev_test else "holdout")
            rows.append({"uid": r["uid"], "label": r["label"], "el": el, "sp": sp,
                         "split": split, "orig": str(r.get("discloses")) == "True"})
    print(f"[pairs] {len(rows)} usable pairs "
          f"(holdout {sum(r['split']=='holdout' for r in rows)})", flush=True)
    res = {}
    preds = {"orig": [r["orig"] for r in rows]}
    for arm, prompt in (("seed", seed_prompt), ("gepa", best_prompt)):
        preds[arm] = verify_batch(llm, sp_params, prompt, [(r["el"], r["sp"]) for r in rows])
        cov = sum(p is not None for p in preds[arm]) / len(rows)
        res[f"{arm}/coverage"] = cov
        print(f"[pairs] {arm} parse coverage {cov:.3f}", flush=True)
    for arm in ("orig", "seed", "gepa"):
        for split in ("holdout", "dev_test", "dev_train", "all"):
            sub = [(p, r) for p, r in zip(preds[arm], rows)
                   if p is not None and (split == "all" or r["split"] == split)]
            pos = [p for p, r in sub if r["label"] == "pos"]
            neg = [p for p, r in sub if r["label"] == "neg"]
            if not pos or not neg:
                continue
            pr, nr = sum(pos) / len(pos), sum(neg) / len(neg)
            res[f"{arm}/{split}"] = {"n": len(sub), "pos_rate": pr, "neg_rate": nr,
                                     "bal_sep": (pr + (1 - nr)) / 2}
            print(f"[pairs] {arm:5s} {split:9s} n={len(sub):5d} pos_rate={pr:.3f} "
                  f"neg_rate={nr:.3f} bal_sep={(pr + (1 - nr)) / 2:.3f}", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print("PAIRS_DONE", flush=True)


# ---------------- stage: scale ----------------
def run_scale(llm, sp_params, prompts_file, suffix, chunk_rows=3000):
    verdicts_out = VERDICTS_OUT.replace(".jsonl", f"{suffix}.jsonl")
    _, best_prompt = load_prompts(prompts_file)
    done = set()
    if os.path.exists(verdicts_out):
        for ln in open(verdicts_out):
            try:
                done.add(json.loads(ln)["uid"])
            except Exception:
                pass
    rows = []
    for ln in open(SCALE):
        r = json.loads(ln)
        if r["uid"] not in done:
            rows.append({"uid": r["uid"], "element": r["element"],
                         "spans": [join_spans(ref.get("spans")) for ref in r["refs"]]})
    print(f"[scale] {len(done)} rows done, {len(rows)} to score", flush=True)
    fh = open(verdicts_out, "a", buffering=1)
    for c0 in range(0, len(rows), chunk_rows):
        chunk = rows[c0:c0 + chunk_rows]
        pairs, owner = [], []
        for i, r in enumerate(chunk):
            for j, sp in enumerate(r["spans"]):
                if sp:
                    pairs.append((r["element"], sp)); owner.append((i, j))
        preds = verify_batch(llm, sp_params, best_prompt, pairs)
        verd = [[None] * len(r["spans"]) for r in chunk]
        for (i, j), p in zip(owner, preds):
            verd[i][j] = p
        for r, v in zip(chunk, verd):
            fh.write(json.dumps({"uid": r["uid"], "verdicts": v}) + "\n")
        fh.flush(); os.fsync(fh.fileno())
        print(f"[scale] {min(c0 + chunk_rows, len(rows))}/{len(rows)} rows "
              f"({len(pairs)} pairs this chunk)", flush=True)
    fh.close()
    print("SCALE_DONE", flush=True)


def cmd_run(a):
    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096,
              enable_prefix_caching=True, trust_remote_code=True)
    # 200 tokens: room for a full JSON object even with a wordy reason (60 truncated mid-JSON in v1)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    run_pairs(llm, sp, a.prompts, a.suffix)
    run_scale(llm, sp, a.prompts, a.suffix, chunk_rows=a.chunk_rows)


# ---------------- stage: readout (CPU) ----------------
_WORD = re.compile(r"[a-z]{4,}")


def overlap(element, span):
    e = set(_WORD.findall(element.lower()))
    s = set(_WORD.findall(span.lower()))
    return len(e & s) / max(1, len(e))


# notebook's own auc/grouped_auc, verbatim (see datasets/patents/audit_regroup_va.py)
def auc(y, s):
    import numpy as np
    y = np.asarray(y, float); s = np.asarray(s, float)
    order = np.argsort(s, kind='mergesort'); sr = s[order]
    rk = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    p = (y == 1).sum(); n = (y == 0).sum()
    return (rk[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def grouped_auc(X, y, g):
    import numpy as np
    uniq = np.array(sorted(set(g))); folds = np.array_split(uniq, 5)
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    oof = np.zeros(len(y))
    for k in range(5):
        te = set(folds[k]); m = np.array([x in te for x in g]); tr = ~m
        Xt = X[tr]; mu = Xt.mean(0); sd = Xt.std(0) + 1e-8
        Xb = np.c_[np.ones(len(Xt)), (Xt - mu) / sd]; w = np.zeros(Xb.shape[1])
        for _ in range(2500):
            p = sig(Xb @ w); w -= 0.3 * (Xb.T @ (p - y[tr]) / len(Xt) + 1e-2 * np.r_[0, w[1:]])
        oof[m] = sig(np.c_[np.ones(int(m.sum())), (X[m] - mu) / sd] @ w)
    return auc(y, oof)


def mi_binned(y, m, bins=10):
    import numpy as np

    def entropy(p):
        p = np.asarray(p, float); p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    y = np.asarray(y, int); m = np.asarray(m, float)
    edges = np.unique(np.quantile(m, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(m)  # discrete scores (e.g. n_disclose 0-8): one bin per value
        if len(edges) < 2:
            return 0.0, 0.0
        b = np.searchsorted(edges, m)
    else:
        b = np.clip(np.digitize(m, edges[1:-1]), 0, len(edges) - 2)
    n = len(y)
    Hy = entropy(np.bincount(y) / n)
    Hy_given_m = 0.0
    for bi in np.unique(b):
        mask = b == bi
        Hy_given_m += mask.mean() * entropy(np.bincount(y[mask], minlength=2) / mask.sum())
    return max(0.0, Hy - Hy_given_m), Hy


V_COLS = ['v_max_lexoverlap', 'v_mean_lexoverlap', 'v_count_lexhit', 'v_element_wordlen',
          'v_n_refs', 'v_max_spanlen', 'v_mean_spanlen']


def disclosure_feats(verdicts, spans, element):
    """[n, any, frac, max_overlap-among-disclosed] — same transform for both arms."""
    n_ref = max(1, len(verdicts))
    disc = [i for i, v in enumerate(verdicts) if v is True]
    mo = max((overlap(element, spans[i]) for i in disc), default=0.0)
    return [float(len(disc)), float(bool(disc)), len(disc) / n_ref, mo]


def cmd_readout(a):
    import csv
    import numpy as np
    verdicts_out = VERDICTS_OUT.replace(".jsonl", f"{a.suffix}.jsonl")
    rows = list(csv.DictReader(open(CSV)))
    gep = {}
    for ln in open(verdicts_out):
        r = json.loads(ln)
        gep[r["uid"]] = r["verdicts"]
    orig_f, gepa_f, apps, mism, gold_or, gold_ge, labels = [], [], [], 0, [], [], []
    span_ct = none_ct = 0
    with open(SCALE) as fh:
        for i, ln in enumerate(fh):
            r = json.loads(ln)
            spans = [join_spans(ref.get("spans")) for ref in r["refs"]]
            ov = [str(ref.get("discloses")) == "True" for ref in r["refs"]]
            gv = gep.get(r["uid"]) or [None] * len(r["refs"])
            span_ct += sum(1 for s in spans if s)
            none_ct += sum(1 for k, s in enumerate(spans) if s and gv[k] is None)
            of = disclosure_feats(ov, spans, r["element"])
            gf = disclosure_feats([v is True for v in gv], spans, r["element"])
            mism += int(of[0]) != int(float(rows[i]["a_n_disclose"]))
            orig_f.append(of); gepa_f.append(gf); apps.append(str(r["app_id"]))
            labels.append(1.0 if r["label"] == "pos" else 0.0)
            gi = [k for k, ref in enumerate(r["refs"]) if str(ref.get("is_gold")) == "True"]
            if gi and r["label"] == "pos":
                gold_or.append(any(ov[k] for k in gi))
                gold_ge.append(any(gv[k] is True for k in gi))
    y = np.array([float(c["fell"]) for c in rows])
    assert np.allclose(y, labels), "CSV fell vs jsonl label misaligned — stop"
    print(f"[readout] {len(rows)} rows; n_disclose sanity mismatches vs CSV: {mism}", flush=True)
    cov = 1 - none_ct / max(1, span_ct)
    print(f"[readout] gepa scale parse coverage: {cov:.3f} ({none_ct}/{span_ct} unparsed)", flush=True)
    if cov < 0.98:
        # Codex audit + v1 retraction: `v is True` maps None (parse failure) to "not disclosed",
        # so at low coverage parser failure becomes substantive evidence. Abort instead.
        raise SystemExit(f"ABORT: parse coverage {cov:.3f} < 0.98 — None->False conflation would "
                         "corrupt the payoff (the v1 selection-artifact trap). Fix prompt/parse first.")
    print(f"[readout] gold-ref disclosed | pos: orig {np.mean(gold_or):.3f} "
          f"gepa {np.mean(gold_ge):.3f} (n={len(gold_or)})", flush=True)

    Xv = np.array([[float(c[col]) for col in V_COLS] for c in rows])
    g = np.array(apps)
    Fo, Fg = np.array(orig_f), np.array(gepa_f)
    print(f"\n=== grouped-5fold-by-app AUC (notebook grouped_auc verbatim) ===", flush=True)
    print(f"  V only        : {grouped_auc(Xv, y, g):.4f}", flush=True)
    print(f"  V + A(orig)   : {grouped_auc(np.c_[Xv, Fo], y, g):.4f}", flush=True)
    print(f"  V + A(gepa)   : {grouped_auc(np.c_[Xv, Fg], y, g):.4f}", flush=True)
    print(f"  A(orig) only  : {grouped_auc(Fo, y, g):.4f}", flush=True)
    print(f"  A(gepa) only  : {grouped_auc(Fg, y, g):.4f}", flush=True)
    print(f"\n=== univariate + recovery MI (evaluate-only) ===", flush=True)
    for name, F in (("orig", Fo), ("gepa", Fg)):
        for fi, fn in ((0, "n_disclose"), (2, "frac_disclose")):
            a_ = auc(y, F[:, fi])
            mi, Hy = mi_binned(y, F[:, fi])
            print(f"  {name:4s} {fn:14s} AUC={a_:.4f}  I(M;Y)={mi:.4f} bits "
                  f"({mi / max(Hy, 1e-9):.1%} of H(Y)={Hy:.3f})", flush=True)
    print("PAYOFF_READOUT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--chunk-rows", type=int, default=3000,
                                              dest="chunk_rows")
    r.add_argument("--prompts", default="gepa_prompts.json")
    r.add_argument("--suffix", default="")
    ro = sub.add_parser("readout"); ro.add_argument("--suffix", default="")
    a = ap.parse_args()
    {"run": cmd_run, "readout": cmd_readout}[a.cmd](a)


if __name__ == "__main__":
    main()
