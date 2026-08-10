#!/usr/bin/env python3
"""Deep-metric pilot (notes/2026-07-08__deep-metrics-design.md): GLM-proposed multi-step
programs on one math tag, executed step-synchronously on the resident 70B engine, gated with
the SAME arithmetic as the arm runs (paired-CV gain over the 48-bank, NB-corrected confirm,
Bonferroni over planned programs), plus the depth-premium comparator (program vs its own
flattened one-shot rubric).

  python scripts/tools/deep_metric_pilot.py --tag general-topology --judge-model <path>
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "methods")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.deep_metrics import (
    PROGRAM_PROPOSER_PROMPT, DeepMetricProgram, execute_program, flatten_program)
from metrics_tree_infilling.io_metrics import (
    load_rubric_metrics_from_dir, make_vllm_judge_scorer, three_way_split, _get_offline_engine)


def make_raw_batch(cfg, cache_dir: Path):
    """Batched raw prompt runner on the resident engine, JSONL-cached per (prompt, temp)."""
    cache_path = cache_dir / "deep_raw.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            try:
                rec = json.loads(line)
                cache[rec["key"]] = rec["response"]
            except Exception:
                continue

    def run(prompts, temperature=0.0):
        from vllm import SamplingParams
        keys = [hashlib.sha256(f"{temperature}::{p}".encode()).hexdigest()[:16] for p in prompts]
        miss = [i for i, k in enumerate(keys) if k not in cache]
        if miss:
            llm = _get_offline_engine(cfg)
            sp = SamplingParams(temperature=temperature, max_tokens=400)
            outs = llm.chat([[{"role": "user", "content": prompts[i][:40000]}] for i in miss], sp)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "a") as f:
                for i, o in zip(miss, outs):
                    resp = o.outputs[0].text if o.outputs else ""
                    cache[keys[i]] = resp
                    f.write(json.dumps({"key": keys[i], "response": resp}) + "\n")
        return [cache[k] for k in keys]

    return run


def paired_gain(Xb, xnew, y, seeds):
    """Per-fold paired (AUC, bits) gains of bank+new over bank, across seeded 5-fold CVs."""
    da, db = [], []
    for seed in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xb, y):
            if len(np.unique(y[tr])) < 2:
                continue
            m0 = LogisticRegression(max_iter=2000).fit(Xb[tr], y[tr])
            m1 = LogisticRegression(max_iter=2000).fit(
                np.hstack([Xb[tr], xnew[tr, None]]), y[tr])
            p0 = np.clip(m0.predict_proba(Xb[te])[:, 1], 1e-6, 1 - 1e-6)
            p1 = np.clip(m1.predict_proba(np.hstack([Xb[te], xnew[te, None]]))[:, 1], 1e-6, 1 - 1e-6)
            da.append(roc_auc_score(y[te], p1) - roc_auc_score(y[te], p0))
            db.append((log_loss(y[te], p0, labels=[0, 1]) - log_loss(y[te], p1, labels=[0, 1]))
                      / np.log(2))
    return np.array(da), np.array(db)


def nb_p(diffs, n_train_frac=0.8):
    """Nadeau-Bengio corrected one-sided t-test vs 0."""
    d = np.asarray(diffs)
    k = len(d)
    if k < 2 or d.std(ddof=1) == 0:
        return 1.0
    corr = 1.0 / k + (1 - n_train_frac) / n_train_frac  # NB correction: 1/k + n_test/n_train
    t = d.mean() / (d.std(ddof=1) * np.sqrt(corr))
    from scipy.stats import t as tdist
    return float(1 - tdist.cdf(t, df=k - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k-programs", type=int, default=8)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--proposer-model", default="glm-5.2")
    ap.add_argument("--bank-dir", default="datasets/math/stackexchange/medoid-bank-clean")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"outputs/ctree/deep_pilot/math-{args.tag}")
    out.mkdir(parents=True, exist_ok=True)

    cfg = InfillConfig(materialize_backend="vllm_offline", materialize_model=args.judge_model,
                       max_text_tokens=700, verbose=False,
                       cache_dir="outputs/ctree/B_tree/judge_cache",
                       id_column="question_id", text_column="text", label_column="judgement",
                       group_split_by_id=True, output_dir=str(out))

    df = pd.read_csv(f"datasets/math/stackexchange/by_tag/{args.tag}.csv.gz").dropna(
        subset=["text", "judgement"])
    df["judgement"] = df["judgement"].astype(int)
    df = df.sample(min(args.n + args.n // 2, len(df)), random_state=7).reset_index(drop=True)
    df_d, df_g, df_t = three_way_split(df, cfg)
    dg = pd.concat([df_d, df_g]).reset_index(drop=True)
    y = dg.judgement.to_numpy()
    texts = dg.text.astype(str).tolist()
    print(f"[{args.tag}] discover+guard={len(dg)} base={y.mean():.3f}", flush=True)

    # ---- bank floor (48 clean-bank rubrics, standard judge path; rides shared cache) ----
    bank = load_rubric_metrics_from_dir(args.bank_dir)
    judge = make_vllm_judge_scorer(cfg)
    lv, apl = judge(bank, texts)
    keep = [j for j in range(len(bank)) if apl[:, j].mean() > 0.10 and np.nanstd(lv[:, j]) > 0.05]
    Xb = lv[:, keep]
    mu = np.nanmean(Xb, 0)
    Xb = np.where(np.isnan(Xb), np.where(np.isnan(mu), 0.5, mu), Xb)
    mdl = LogisticRegression(max_iter=2000).fit(Xb, y)
    p = mdl.predict_proba(Xb)[:, 1]
    print(f"bank floor: {len(keep)} viable, in-sample AUC {roc_auc_score(y, p):.3f}", flush=True)

    # ---- residual contrast (per-class quantiles, 6+6, 4000-char clip) ----
    resid = np.abs(y - p)
    def pick(mask):
        r = resid[mask]
        return np.where(mask)[0][r >= np.quantile(r, 0.66)]
    pos_ex = [texts[i][:4000] for i in pick(y == 1)[:6]]
    neg_ex = [texts[i][:4000] for i in pick(y == 0)[:6]]
    known = "\n".join(f"- {bank[j].name}: {bank[j].description[:90]}" for j in keep)

    # ---- propose programs (GLM via z.ai anthropic endpoint) ----
    import os
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                 base_url=os.environ.get("ANTHROPIC_BASE_URL"))
    prompt = PROGRAM_PROPOSER_PROMPT.format(known=known, pos="\n---\n".join(pos_ex),
                                            neg="\n---\n".join(neg_ex), k=args.k_programs)
    r = client.messages.create(model=args.proposer_model, max_tokens=4000, temperature=0.7,
                               messages=[{"role": "user", "content": prompt}])
    rawp = "".join(b.text for b in r.content if hasattr(b, "text"))
    progs = []
    try:
        objs = json.loads(rawp[rawp.find("{"): rawp.rfind("}") + 1])["programs"]
    except Exception:
        objs = []
    for o in objs[: args.k_programs]:
        try:
            progs.append(DeepMetricProgram.from_json(o, raw=json.dumps(o)))
        except Exception as e:
            print(f"  invalid program dropped: {e}", flush=True)
    print(f"programs proposed: {len(progs)}", flush=True)
    json.dump([json.loads(pr.raw) for pr in progs], open(out / "programs.json", "w"), indent=1)

    raw_batch = make_raw_batch(cfg, out)
    m_bonf = args.k_programs * 2  # programs + their flattened comparators are all planned
    ledger = []
    for pr in progs:
        scores = execute_program(pr, texts, lambda ps: raw_batch(ps, 0.0))
        ok = ~np.isnan(scores)
        if ok.mean() < 0.5 or np.nanstd(scores) < 0.05:
            ledger.append(dict(name=pr.name, status="dropped:degenerate",
                               applicability=float(ok.mean())))
            print(f"  [deg ] {pr.name[:50]} appl={ok.mean():.2f}", flush=True)
            continue
        x = np.where(ok, scores, np.nanmean(scores))
        from sklearn.linear_model import LinearRegression
        red = LinearRegression().fit(Xb[ok], scores[ok]).score(Xb[ok], scores[ok])
        da, db = paired_gain(Xb, x, y, seeds=[0])
        cda, cdb = paired_gain(Xb, x, y, seeds=[11, 12, 13, 14, 15])
        p_auc, p_bits = nb_p(cda), nb_p(cdb)
        # reliability: retest judge steps at temp 0.6 with salted prompts on a subsample
        sub = np.random.RandomState(3).choice(len(texts), min(120, len(texts)), replace=False)
        s2 = execute_program(pr, [f"[retest] {texts[i]}" for i in sub],
                             lambda ps: raw_batch(ps, 0.6))
        both = ok[sub] & ~np.isnan(s2)
        retest = float(spearmanr(scores[sub][both], s2[both]).statistic) if both.sum() > 20 else np.nan
        # depth premium: flattened one-shot rubric through the same gain arithmetic
        from metrics_tree_infilling.io_metrics import MetricSpec, _stable_id
        flat_spec = MetricSpec(metric_id=_stable_id("f", pr.name, "flat"), name=f"flat::{pr.name}",
                               description=flatten_program(pr), kind="judge",
                               guidance=flatten_program(pr))
        flv, fapl = judge([flat_spec], texts)
        fx = flv[:, 0]
        fx = np.where(np.isnan(fx), np.nanmean(fx) if not np.isnan(np.nanmean(fx)) else 0.5, fx)
        fda, fdb = paired_gain(Xb, fx, y, seeds=[0])
        rec = dict(name=pr.name, description=pr.description, n_steps=len(pr.steps),
                   n_judge_steps=pr.n_judge_steps, applicability=float(ok.mean()),
                   redundancy_r2=float(red),
                   auc_gain=float(np.mean(da)), bits_gain=float(np.mean(db)),
                   confirm_auc_gain=float(np.mean(cda)), confirm_bits_gain=float(np.mean(cdb)),
                   confirm_p_auc=p_auc, confirm_p_bits=p_bits, retest_spearman=retest,
                   flat_auc_gain=float(np.mean(fda)), flat_bits_gain=float(np.mean(fdb)),
                   depth_premium_bits=float(np.mean(db) - np.mean(fdb)),
                   status="kept" if (np.mean(cdb) >= 0.003 and np.mean(cda) >= 0.005
                                     and p_auc < 0.05 / m_bonf and p_bits < 0.05 / m_bonf)
                          else "dropped:gate")
        ledger.append(rec)
        print(f"  [{rec['status'][:4]:4s}] {pr.name[:44]:44s} bits={rec['bits_gain']:+.4f} "
              f"conf={rec['confirm_bits_gain']:+.4f} p={p_auc:.3g} retest={retest:.2f} "
              f"depth_premium={rec['depth_premium_bits']:+.4f}", flush=True)
    json.dump(ledger, open(out / "ledger.json", "w"), indent=1)
    print(f"DONE -> {out}/ledger.json", flush=True)


if __name__ == "__main__":
    main()
