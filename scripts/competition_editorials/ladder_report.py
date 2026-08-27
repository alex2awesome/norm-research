"""Aggregate Bank LR + TF-IDF + ModernBERT into a single ladder report."""
import json
from pathlib import Path

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
OUT = f"{ROOT}/outputs/v2_analysis"

BANK = f"{OUT}/comp_qwen_phase1_stratified_auc.json"
TFIDF = f"{OUT}/comp_qwen_phase1_tfidf_auc.json"
MB = f"{OUT}/comp_qwen_phase1_modernbert_auc.json"
REPORT = f"{OUT}/comp_qwen_phase1_ladder_report.md"

DECILE_LABELS = ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]


def fmt(x, n=3):
    if x is None or (isinstance(x, float) and (x != x)):
        return "n/a"
    return f"{x:.{n}f}"


def bank_cells():
    """Bank LR reference numbers from the stratified subsample."""
    d = json.load(open(BANK))
    sub = d["subsample"]
    return {
        "pooled": sub.get("combined", {}).get("auc"),
        "lc": sub.get("lc", {}).get("auc"),
        "luogu": sub.get("luogu", {}).get("auc"),
    }


def bank_deciles():
    d = json.load(open(BANK))
    # bank deciles are on the full 46k pool, not the subsample
    return d["deciles"]


def tfidf_cells():
    d = json.load(open(TFIDF))
    return {
        "pooled": d["cells"]["pooled"]["auc"],
        "lc": d["cells"]["lc"]["auc"],
        "luogu": d["cells"]["luogu"]["auc"],
    }


def tfidf_deciles():
    d = json.load(open(TFIDF))
    return d["deciles"]


def mb_cells():
    d = json.load(open(MB))
    out = {}
    for k in ("pooled", "lc", "luogu"):
        cell = d["cells"].get(k, {})
        out[k] = cell.get("auc_mean")
    return out


def mb_cells_std():
    d = json.load(open(MB))
    out = {}
    for k in ("pooled", "lc", "luogu"):
        cell = d["cells"].get(k, {})
        out[k] = cell.get("auc_std")
    return out


def main():
    b = bank_cells()
    t = tfidf_cells()
    try:
        m = mb_cells()
        ms = mb_cells_std()
    except Exception:
        m = {"pooled": None, "lc": None, "luogu": None}
        ms = {"pooled": None, "lc": None, "luogu": None}

    lines = []
    lines.append("# Bank -> TF-IDF -> Dense ladder (Phase 1 stratified subsample)\n")
    lines.append("Subsample: stratified-balanced 500/(platform x decile).")
    lines.append("Bank LR uses 231 a*_score/a*_applied columns (no cosine, no embedding).")
    lines.append("TF-IDF LR uses char_wb 3-5gram on candidate_text alone (no editorial).")
    lines.append("ModernBERT is a cross-encoder predicting label end-to-end from (editorial, candidate).")
    lines.append("All three use StratifiedKFold(5, shuffle=True, seed=42).\n")

    def dlt(name, vals, deltas=None, std=None):
        cols = [name]
        for k in ("pooled", "lc", "luogu"):
            v = vals.get(k)
            s = (std or {}).get(k)
            if s is not None and v is not None:
                cols.append(f"{fmt(v)} +/- {fmt(s)}")
            else:
                cols.append(fmt(v))
        if deltas is None:
            cols.append("0")
        else:
            cols.append(" / ".join(fmt(deltas[k]) for k in ("pooled", "lc", "luogu")))
        return "| " + " | ".join(cols) + " |"

    bank_d = {"pooled": 0.0, "lc": 0.0, "luogu": 0.0}
    tf_d = {k: (t[k] - b[k]) if (t.get(k) is not None and b.get(k) is not None) else None
            for k in ("pooled", "lc", "luogu")}
    mb_d = {k: (m[k] - b[k]) if (m.get(k) is not None and b.get(k) is not None) else None
            for k in ("pooled", "lc", "luogu")}

    lines.append("## Pooled / per-platform AUC\n")
    lines.append("| Method | Pooled AUC | LC AUC | Luogu AUC | Δ vs Bank LR (pooled/lc/luogu) |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(dlt("Bank LR (reference)", b, bank_d))
    lines.append(dlt("TF-IDF char_wb LR", t, tf_d))
    lines.append(dlt("ModernBERT cross-encoder", m, mb_d, std=ms))
    lines.append("")

    # Per-decile
    try:
        bd = bank_deciles()
        td = tfidf_deciles()
    except Exception:
        bd = td = {}
    if bd and td:
        lines.append("## Per-decile AUC (LC)\n")
        lines.append("| Decile | n (TF-IDF) | Bank-LR AUC | TF-IDF AUC |")
        lines.append("|---|---:|---:|---:|")
        for dl in DECILE_LABELS:
            br = bd.get("lc", {}).get(dl, {})
            tr = td.get("lc", {}).get(dl, {})
            lines.append(f"| {dl} | {tr.get('n', 'n/a')} | "
                         f"{fmt(br.get('auc'))} | {fmt(tr.get('auc'))} |")
        lines.append("")
        lines.append("## Per-decile AUC (Luogu)\n")
        lines.append("| Decile | n (TF-IDF) | Bank-LR AUC | TF-IDF AUC |")
        lines.append("|---|---:|---:|---:|")
        for dl in DECILE_LABELS:
            br = bd.get("luogu", {}).get(dl, {})
            tr = td.get("luogu", {}).get(dl, {})
            lines.append(f"| {dl} | {tr.get('n', 'n/a')} | "
                         f"{fmt(br.get('auc'))} | {fmt(tr.get('auc'))} |")
        lines.append("")

    lines.append("## Leakage receipts\n")
    lines.append("- Bank LR: features = 231 a*_score / a*_applied columns; cosine used ONLY for stratification.")
    lines.append("- TF-IDF LR: features = TF-IDF char_wb (3,5) on candidate_text ONLY (editorial_text NOT used).")
    lines.append("- ModernBERT: cross-encoder predicts label end-to-end from (editorial, candidate); no extra pairwise feature.")
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("(filled in by hand after numbers land)")
    Path(REPORT).write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
