"""E8-ARTIC eval: cross-family articulation convergence vs transport boundness.

Pre-registered (position note SS11, 2026-07-06, BEFORE data): conv(field) = mean
pairwise cos among the 3 family articulations, MINUS the same-task cross-family
different-field background. Criterion conv = mean over its fields. P1:
Spearman(conv, mean transport ratio) < 0. P2: both-swap degraders have lower
median conv. P3 (directional): conv ~ fm. Descriptive only — no gates.

Usage: python3 eval_artic.py
-> outputs/metric_seam_pilot/battery/artic_eval.json
"""
import itertools, json, pathlib, random, statistics as st, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import BASE, ROOT  # noqa: E402
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

FAMS = ("gemma", "llama70", "qwen_toff")
RATIO_FILES = {t: BASE / "tasks" / t / "transport_eval_3fam.json"
               for t in ("creative_writing", "math", "humor", "legal_title_vii")}
RATIO_FILES["press_releases"] = BASE / "v2/transport_eval_3fam.json"


def load_artic(fam):
    out = {}
    p = BASE / f"battery/artic_results_{fam}.jsonl"
    for line in open(p):
        r = json.loads(line)
        raw = (r.get("raw") or "").strip()
        if raw:
            out[r["aspect_id"]] = raw
    return out


def main():
    arts = {f: load_artic(f) for f in FAMS}
    keys = sorted(set.intersection(*(set(a) for a in arts.values())))
    print(f"{len(keys)} fields with all-3-family articulations "
          f"({', '.join(f'{f}:{len(a)}' for f, a in arts.items())})")

    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    texts, idx = [], {}
    for f in FAMS:
        for k in keys:
            idx[(f, k)] = len(texts)
            texts.append(arts[f][k])
    E = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    cos = lambda a, b: float(np.dot(E[idx[a]], E[idx[b]]))

    # background: same-task, cross-family, DIFFERENT-field pairs (sampled)
    rng = random.Random(13)
    by_task = {}
    for k in keys:
        by_task.setdefault(k.split("::")[0], []).append(k)
    bg = {}
    for t, ks in by_task.items():
        vals = []
        for _ in range(min(2000, 4 * len(ks) ** 2)):
            k1, k2 = rng.sample(ks, 2)
            f1, f2 = rng.sample(FAMS, 2)
            vals.append(cos((f1, k1), (f2, k2)))
        bg[t] = st.mean(vals)

    conv_field = {}
    for k in keys:
        t = k.split("::")[0]
        raw = st.mean(cos((f1, k), (f2, k))
                      for f1, f2 in itertools.combinations(FAMS, 2))
        conv_field[k] = {"raw": round(raw, 4), "conv": round(raw - bg[t], 4)}

    # criterion level + transport join
    crit = {}
    for k, v in conv_field.items():
        t, rest = k.split("::")
        aid = rest.split("__", 1)[0]
        crit.setdefault((t, aid), []).append(v["conv"])
    ratios = {}
    for t, p in RATIO_FILES.items():
        if p.exists():
            ratios[t] = json.load(open(p))["aspects"]
    rows = []
    for (t, aid), convs in sorted(crit.items()):
        row = {"task": t, "aid": aid, "n_fields": len(convs),
               "conv": round(st.mean(convs), 4)}
        a = ratios.get(t, {}).get(aid)
        if a:
            rs = [a.get("ratio_llama"), a.get("ratio_qwen")]
            rs = [r for r in rs if r is not None]
            row["mean_ratio"] = round(st.mean(rs), 4) if rs else None
            row["fm"] = a.get("field_marginal")
            row["both_swap_degrader"] = bool(
                (a.get("P_degrade_llama") or 0) >= .95 and
                (a.get("P_degrade_qwen") or 0) >= .95)
        rows.append(row)

    with_r = [r for r in rows if r.get("mean_ratio") is not None]
    p1 = spearman([r["conv"] for r in with_r], [r["mean_ratio"] for r in with_r]) \
        if len(with_r) >= 10 else float("nan")
    deg = [r["conv"] for r in rows if r.get("both_swap_degrader")]
    nond = [r["conv"] for r in rows if r.get("both_swap_degrader") is False]
    with_fm = [r for r in rows if r.get("fm") is not None]
    p3 = spearman([r["conv"] for r in with_fm], [r["fm"] for r in with_fm]) \
        if len(with_fm) >= 10 else float("nan")

    summary = {
        "n_fields": len(keys), "n_criteria": len(rows),
        "background_by_task": {t: round(v, 4) for t, v in bg.items()},
        "P1_spearman_conv_vs_mean_ratio": round(p1, 4) if p1 == p1 else None,
        "P1_n": len(with_r),
        "P2_median_conv_degraders": round(st.median(deg), 4) if deg else None,
        "P2_median_conv_rest": round(st.median(nond), 4) if nond else None,
        "P2_n_degraders": len(deg),
        "P3_spearman_conv_vs_fm": round(p3, 4) if p3 == p3 else None,
        "P3_n": len(with_fm)}
    out = BASE / "battery/artic_eval.json"
    json.dump({"summary": summary, "criteria": rows,
               "fields": conv_field}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
