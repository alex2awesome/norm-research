"""CF editorial-code contamination probe.

Hypothesis: CF editorials are blog posts spanning problems A-E. Our "best code block"
extractor may pick a code block belonging to a SIBLING problem in the same contest,
so the (extracted_code, "editorial") pair labels the wrong problem.

Probe:
  For each confident CF editorial whose contest has >=1 sibling problem with
  candidates AND its own problem has candidates:
    1. Embed the extracted_code with bge-code-v1.
    2. Embed up to N candidate code per own/sibling problem.
    3. own = mean cosine(ed, own_cands); sib_i = mean cosine(ed, sib_i_cands).
       sib_best = max_i sib_i. margin = own - sib_best.
       Contaminated iff margin < 0 (some sibling problem is closer than own).
  Restrict to eval-cell problems too.
"""
import os, sys, time, json, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden, attn_mask):
    left_pad = (attn_mask[:, -1].sum() == attn_mask.shape[0])
    if left_pad:
        return last_hidden[:, -1]
    seq_lens = attn_mask.sum(dim=1) - 1
    bs = last_hidden.size(0)
    return last_hidden[torch.arange(bs, device=last_hidden.device), seq_lens]


def embed_texts(texts, tok, model, batch_size=32, max_len=1024):
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = [t if (isinstance(t, str) and t.strip()) else "EMPTY" for t in texts[i:i+batch_size]]
            enc = tok(batch, max_length=max_len, padding=True, truncation=True,
                      return_tensors="pt", pad_to_multiple_of=8).to("cuda")
            o = model(**enc)
            emb = F.normalize(last_token_pool(o.last_hidden_state, enc["attention_mask"]).float(), p=2, dim=1)
            out[i:i+len(batch)] = emb.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editorials", default="/lfs/skampere3/0/alexspan/norm-research/datasets/competition_unified/editorials_code_extracted.parquet")
    ap.add_argument("--candidates", default="/lfs/skampere3/0/alexspan/norm-research/datasets/competition_unified/candidates.parquet")
    ap.add_argument("--eval-cell", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/comp_fourplatform_cells/cf_pairs.parquet")
    ap.add_argument("--bank-scores", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/comp_fourplatform_cells/cf_bank_scores.parquet")
    ap.add_argument("--labels-dir", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/comp_fourplatform_label_shards/results")
    ap.add_argument("--out-json", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/cf_contamination_probe.json")
    ap.add_argument("--out-md", default="/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/cf_contamination_probe.md")
    ap.add_argument("--cands-per-problem", type=int, default=10)
    ap.add_argument("--max-editorials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260610)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # ---- 1) eligibility ----
    print("loading editorials/candidates ...", flush=True)
    ed = pd.read_parquet(args.editorials)
    ed = ed[(ed.platform=="cf") & (ed.extraction_confidence.isin(["confident","confident_multi"]))].copy()
    ed["contest"] = ed.canonical_pid.str.replace("cf:","",regex=False).str.split("_").str[0]

    cand = pd.read_parquet(args.candidates, columns=["platform","canonical_pid","code","candidate_id"])
    cand = cand[cand.platform=="cf"].copy()
    cand["contest"] = cand.canonical_pid.str.replace("cf:","",regex=False).str.split("_").str[0]

    cand_pids_set = set(cand.canonical_pid.unique())
    cands_by_contest = {c: g for c, g in cand.groupby("contest")}

    # eligible: own problem has candidates and >=1 sibling problem also has candidates
    rows = []
    for _, r in ed.iterrows():
        if r.canonical_pid not in cand_pids_set: continue
        contest_pids = set(cands_by_contest.get(r.contest, pd.DataFrame()).get("canonical_pid", pd.Series(dtype=str)).unique()) if r.contest in cands_by_contest else set()
        sibs = [p for p in contest_pids if p != r.canonical_pid]
        if not sibs: continue
        rows.append({"editorial_id": r.editorial_id, "canonical_pid": r.canonical_pid,
                     "contest": r.contest, "problem_id": r.problem_id,
                     "extracted_code": r.extracted_code, "siblings": sibs})
    print(f"eligible editorials (own in cands + >=1 sibling in cands): {len(rows)}", flush=True)

    if args.max_editorials and len(rows) > args.max_editorials:
        rng.shuffle(rows)
        rows = rows[:args.max_editorials]

    # ---- 2) gather candidate code per problem ----
    needed_pids = set()
    for r in rows:
        needed_pids.add(r["canonical_pid"])
        for s in r["siblings"]: needed_pids.add(s)

    # sample up to args.cands_per_problem per pid
    pid_to_cands = {}
    for pid in needed_pids:
        g = cand[cand.canonical_pid == pid]
        if len(g) > args.cands_per_problem:
            g = g.sample(n=args.cands_per_problem, random_state=args.seed)
        pid_to_cands[pid] = list(zip(g.candidate_id.tolist(), g.code.tolist()))

    # ---- 3) build text list and embed ----
    # ed texts
    ed_texts = [r["extracted_code"] for r in rows]
    ed_keys  = [("ED", r["editorial_id"]) for r in rows]

    # cand texts
    cand_keys = []  # (pid, candidate_id)
    cand_texts = []
    for pid, lst in pid_to_cands.items():
        for cid, code in lst:
            cand_keys.append((pid, cid))
            cand_texts.append(code)

    print(f"to embed: {len(ed_texts)} editorial codes + {len(cand_texts)} candidate codes", flush=True)

    print("loading bge-code-v1 ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("BAAI/bge-code-v1", trust_remote_code=True)
    model = AutoModel.from_pretrained("BAAI/bge-code-v1", trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    print("embedding editorials ...", flush=True)
    t0 = time.time()
    ed_emb = embed_texts(ed_texts, tok, model, batch_size=16, max_len=1024)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print("embedding candidates ...", flush=True)
    t0 = time.time()
    cand_emb = embed_texts(cand_texts, tok, model, batch_size=32, max_len=1024)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # ---- 4) per-editorial scoring ----
    # build pid -> [emb rows]
    pid_to_emb = {}
    for i, (pid, cid) in enumerate(cand_keys):
        pid_to_emb.setdefault(pid, []).append(i)
    pid_to_emb = {p: np.array(idxs, dtype=np.int64) for p, idxs in pid_to_emb.items()}

    results = []
    for i, r in enumerate(rows):
        e = ed_emb[i]
        own = pid_to_emb.get(r["canonical_pid"])
        if own is None or len(own) == 0:
            continue
        own_sims = cand_emb[own] @ e  # already normalized
        own_score = float(own_sims.mean())
        sib_scores = {}
        for s in r["siblings"]:
            idx = pid_to_emb.get(s)
            if idx is None or len(idx) == 0:
                continue
            sib_scores[s] = float((cand_emb[idx] @ e).mean())
        if not sib_scores:
            continue
        best_sib = max(sib_scores, key=sib_scores.get)
        sib_score = sib_scores[best_sib]
        margin = own_score - sib_score
        results.append({
            "editorial_id": r["editorial_id"],
            "problem_id": r["problem_id"],
            "canonical_pid": r["canonical_pid"],
            "contest": r["contest"],
            "own_score": own_score,
            "best_sibling_pid": best_sib,
            "best_sibling_score": sib_score,
            "margin": margin,
            "contaminated": margin < 0,
            "n_siblings_scored": len(sib_scores),
            "extracted_code_snippet": r["extracted_code"][:800],
        })

    res_df = pd.DataFrame(results)
    print("scored editorials:", len(res_df), "contaminated frac:", float(res_df.contaminated.mean()) if len(res_df) else None, flush=True)

    # ---- 5) intersect with eval cell ----
    pairs = pd.read_parquet(args.eval_cell)
    eval_pids = set(pairs.canonical_pid.unique())
    res_df["in_eval_cell"] = res_df.canonical_pid.isin(eval_pids)
    n_cell = res_df.in_eval_cell.sum()
    contam_cell = res_df[res_df.in_eval_cell].contaminated.mean() if n_cell else float("nan")
    print(f"editorials in eval cell: {n_cell}; contaminated within cell: {contam_cell}", flush=True)

    # ---- 6) margin distribution ----
    margins = res_df["margin"].to_numpy()
    margin_stats = {
        "mean": float(margins.mean()),
        "std": float(margins.std()),
        "min": float(margins.min()),
        "p10": float(np.percentile(margins, 10)),
        "p25": float(np.percentile(margins, 25)),
        "p50": float(np.percentile(margins, 50)),
        "p75": float(np.percentile(margins, 75)),
        "p90": float(np.percentile(margins, 90)),
        "max": float(margins.max()),
    } if len(margins) else {}

    # ---- 7) pick 10 flagged examples for eyeballing ----
    flagged = res_df[res_df.contaminated].sort_values("margin").head(10).copy()
    examples = flagged[["editorial_id","problem_id","canonical_pid","contest","own_score","best_sibling_pid","best_sibling_score","margin","extracted_code_snippet"]].to_dict("records")

    # ---- 8) Save raw results ----
    res_df_to_save = res_df.drop(columns=["extracted_code_snippet"])
    res_df_to_save.to_parquet("/lfs/skampere3/0/alexspan/norm-research/outputs/v2_analysis/cf_contamination_probe_per_editorial.parquet")

    overall_contam = float(res_df.contaminated.mean()) if len(res_df) else float("nan")

    payload = {
        "n_editorials_scored": int(len(res_df)),
        "n_editorials_in_eval_cell": int(n_cell),
        "contaminated_frac_overall": overall_contam,
        "contaminated_frac_eval_cell": float(contam_cell) if n_cell else None,
        "margin_distribution": margin_stats,
        "flagged_examples": examples,
        "cands_per_problem": args.cands_per_problem,
        "max_editorials": args.max_editorials,
        "seed": args.seed,
    }

    # ---- 9) If >20%, recompute clean-subset AUC ----
    clean_auc = None
    if overall_contam > 0.20:
        print("contamination >20%, computing clean-subset AUC ...", flush=True)
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import GroupKFold
            from sklearn.metrics import roc_auc_score

            scores = pd.read_parquet(args.bank_scores)
            print("scores cols:", scores.columns.tolist(), "n=", len(scores))

            # gather labels from shards
            label_files = sorted([f for f in os.listdir(args.labels_dir) if f.startswith("shard_cf_") and f.endswith(".jsonl")])
            labels = []
            for fn in label_files:
                with open(os.path.join(args.labels_dir, fn)) as f:
                    for line in f:
                        try:
                            j = json.loads(line)
                            labels.append(j)
                        except Exception:
                            pass
            lab = pd.DataFrame(labels)
            print("label rows:", len(lab), "cols:", lab.columns.tolist()[:10])

            # determine label/pair_id columns
            pair_col = next((c for c in ["pair_id","id"] if c in lab.columns), None)
            label_col = next((c for c in ["claude_label","label","y"] if c in lab.columns), None)
            print("pair_col:", pair_col, "label_col:", label_col)

            # merge scores -> labels -> pairs (for grouping by canonical_pid)
            merged = scores.merge(lab[[pair_col, label_col]], on=pair_col, how="inner")
            merged = merged[merged[label_col].isin([0,1])]
            merged = merged.merge(pairs[["pair_id","canonical_pid"]], on="pair_id", how="left")

            print("merged labelled rows:", len(merged))
            # clean subset: canonical_pid NOT contaminated
            clean_pids = set(res_df[~res_df.contaminated].canonical_pid.unique())
            contam_pids = set(res_df[res_df.contaminated].canonical_pid.unique())
            merged["clean_flag"] = merged.canonical_pid.isin(clean_pids)
            merged["contam_flag"] = merged.canonical_pid.isin(contam_pids)
            n_clean = merged.clean_flag.sum()
            n_contam = merged.contam_flag.sum()
            print("labelled-pair coverage by probe: clean=", int(n_clean), "contam=", int(n_contam))

            feature_cols = [c for c in scores.columns if c not in ("pair_id","canonical_pid","platform")]
            X_all = merged[feature_cols].fillna(0.0).to_numpy()
            y_all = merged[label_col].astype(int).to_numpy()
            groups = merged.canonical_pid.fillna("UNK").to_numpy()

            def grouped_auc(X, y, g):
                if len(np.unique(y)) < 2: return None
                gkf = GroupKFold(n_splits=min(5, len(np.unique(g))))
                lr_aucs, rf_aucs = [], []
                for tr, te in gkf.split(X, y, g):
                    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2: continue
                    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
                    rf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced", n_jobs=4).fit(X[tr], y[tr])
                    lr_aucs.append(roc_auc_score(y[te], lr.predict_proba(X[te])[:,1]))
                    rf_aucs.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:,1]))
                return {"lr_auc_mean": float(np.mean(lr_aucs)) if lr_aucs else None,
                        "rf_auc_mean": float(np.mean(rf_aucs)) if rf_aucs else None,
                        "n_pairs": int(len(X)), "n_groups": int(len(np.unique(g)))}

            # whole sample
            all_auc = grouped_auc(X_all, y_all, groups)
            # clean only
            mask = merged.clean_flag.to_numpy()
            clean_auc_pair = grouped_auc(X_all[mask], y_all[mask], groups[mask]) if mask.sum()>0 else None
            # contam only
            mask = merged.contam_flag.to_numpy()
            contam_auc_pair = grouped_auc(X_all[mask], y_all[mask], groups[mask]) if mask.sum()>0 else None
            clean_auc = {"all_probe_covered": all_auc, "clean_subset": clean_auc_pair, "contam_subset": contam_auc_pair}
        except Exception as e:
            import traceback
            traceback.print_exc()
            clean_auc = {"error": str(e)}

    payload["clean_subset_auc"] = clean_auc

    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("wrote", args.out_json, flush=True)

    # ---- write MD ----
    lines = []
    lines.append("# CF editorial-code contamination probe\n")
    lines.append(f"- editorials scored: **{payload['n_editorials_scored']}**")
    lines.append(f"- editorials in eval cell: **{payload['n_editorials_in_eval_cell']}**")
    lines.append(f"- contaminated overall: **{payload['contaminated_frac_overall']:.1%}**")
    if payload["contaminated_frac_eval_cell"] is not None:
        lines.append(f"- contaminated in eval cell: **{payload['contaminated_frac_eval_cell']:.1%}**")
    lines.append("")
    lines.append("## Margin distribution (own_score - best_sibling_score)\n")
    for k,v in margin_stats.items():
        lines.append(f"  - {k}: {v:.4f}")
    lines.append("")
    lines.append("## 10 worst-flagged examples\n")
    for i, ex in enumerate(examples[:10]):
        lines.append(f"### Flag {i+1}: {ex['canonical_pid']} (problem_id={ex['problem_id']}, contest={ex['contest']})\n")
        lines.append(f"- own_score={ex['own_score']:.4f}; best sibling={ex['best_sibling_pid']} score={ex['best_sibling_score']:.4f}; margin={ex['margin']:.4f}\n")
        snippet_first_lines = "\n".join(ex["extracted_code_snippet"].splitlines()[:15])
        lines.append("```\n" + snippet_first_lines + "\n```\n")
    if clean_auc:
        lines.append("## Clean-subset AUC\n")
        lines.append("```json")
        lines.append(json.dumps(clean_auc, indent=2))
        lines.append("```")
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))
    print("wrote", args.out_md, flush=True)


if __name__ == "__main__":
    main()
