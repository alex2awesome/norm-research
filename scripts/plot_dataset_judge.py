"""
Plot dataset/benchmark judge results — venue × year × accept/reject breakdowns.

Outputs (PNG + the aggregate CSVs from aggregate_dataset_judge.py):
  outputs/dataset_judge/plots/<run>__share_by_venue_year.png
  outputs/dataset_judge/plots/<run>__share_by_decision.png
  outputs/dataset_judge/plots/<run>__label_mix_by_venue.png

Run:
  python scripts/plot_dataset_judge.py outputs/dataset_judge/verdicts/full_v1.jsonl
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
PLOTS = ROOT / "outputs/dataset_judge/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

VENUE_ORDER = ["ICLR", "NEURIPS", "ICML", "TMLR", "COLM"]
VENUE_COLOR = {
    "ICLR":    "#2563eb",
    "NEURIPS": "#0f766e",
    "ICML":    "#d97706",
    "TMLR":    "#9333ea",
    "COLM":    "#be185d",
}
LABEL_COLOR = {
    "DATASET":   "#2563eb",
    "BENCHMARK": "#d97706",
    "BOTH":      "#9333ea",
    "NEITHER":   "#cbd5e1",
}


def load(p: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(p)]
    df = pd.DataFrame(rows)
    df = df[df["label"].notna()].copy()
    df["venue_family"] = df["venue"].str.extract(r"^([A-Z]+)", expand=False)
    df["year"] = df["year"].astype("Int64")
    df["is_dsb"] = df["label"].isin(["DATASET","BENCHMARK","BOTH"]).astype(int)
    return df


def plot_share_by_venue_year(df: pd.DataFrame, name: str):
    """% of papers per (venue, year) classified as DATASET/BENCHMARK/BOTH."""
    g = (
        df.groupby(["venue_family", "year"])
          .agg(n=("paper_id", "size"), pct_dsb=("is_dsb", lambda s: 100*s.mean()))
          .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for v in VENUE_ORDER:
        sub = g[g["venue_family"] == v].sort_values("year").dropna(subset=["year"])
        if len(sub) == 0:
            continue
        ax.plot(sub["year"].astype(int), sub["pct_dsb"],
                marker="o", color=VENUE_COLOR[v], label=v, linewidth=2)
        # annotate n
        for _, r in sub.iterrows():
            ax.annotate(f"n={int(r['n'])}", (int(r["year"]), r["pct_dsb"]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7.5, color=VENUE_COLOR[v], alpha=0.7)
    ax.set_xlabel("year")
    ax.set_ylabel("% of papers whose primary contribution is a new dataset/benchmark")
    ax.set_title("Share of dataset/benchmark papers per venue, over time")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = PLOTS / f"{name}__share_by_venue_year.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  wrote {out}")


def plot_share_by_decision(df: pd.DataFrame, name: str):
    """% dataset/benchmark, split by accept vs reject, per venue."""
    g = (
        df[df["decision_unified"].isin(["accept", "reject"])]
        .groupby(["venue_family", "decision_unified"])
        .agg(n=("paper_id","size"), pct_dsb=("is_dsb", lambda s: 100*s.mean()))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    venues = [v for v in VENUE_ORDER if v in g["venue_family"].unique()]
    x = np.arange(len(venues))
    w = 0.36
    for i, dec in enumerate(["accept", "reject"]):
        vals = []
        ns = []
        for v in venues:
            row = g[(g["venue_family"]==v) & (g["decision_unified"]==dec)]
            vals.append(float(row["pct_dsb"].iloc[0]) if len(row) else 0)
            ns.append(int(row["n"].iloc[0]) if len(row) else 0)
        offset = (i - 0.5) * w
        color = "#2563eb" if dec == "accept" else "#dc2626"
        bars = ax.bar(x + offset, vals, w, label=dec, color=color, alpha=0.85)
        for xb, val, n in zip(x + offset, vals, ns):
            ax.text(xb, val + 0.15, f"{val:.1f}%\nn={n:,}", ha="center", va="bottom",
                    fontsize=8, color=color)
    ax.set_xticks(x); ax.set_xticklabels(venues)
    ax.set_ylabel("% of papers whose primary contribution is a new dataset/benchmark")
    ax.set_title("Dataset/benchmark share, accepted vs rejected — by venue")
    ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS / f"{name}__share_by_decision.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  wrote {out}")


def plot_label_mix(df: pd.DataFrame, name: str):
    """Stacked bar: label mix (D/B/Both/Neither) per venue."""
    venues = [v for v in VENUE_ORDER if v in df["venue_family"].unique()]
    mix = (
        df.groupby(["venue_family", "label"]).size().unstack("label", fill_value=0)
          .reindex(venues)
    )
    pct = mix.div(mix.sum(axis=1), axis=0) * 100
    for L in ["DATASET","BENCHMARK","BOTH","NEITHER"]:
        if L not in pct.columns: pct[L] = 0
    pct = pct[["DATASET","BENCHMARK","BOTH","NEITHER"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(pct))
    for L in pct.columns:
        ax.bar(pct.index, pct[L], bottom=bottom, label=L, color=LABEL_COLOR[L], alpha=0.9)
        for i, v in enumerate(pct[L].values):
            if v >= 1.0:
                ax.text(i, bottom[i] + v/2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if L != "NEITHER" else "#475569",
                        fontweight="bold")
        bottom += pct[L].values
    ax.set_ylabel("share of papers (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Label mix per venue — DATASET / BENCHMARK / BOTH / NEITHER")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = PLOTS / f"{name}__label_mix_by_venue.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    p = Path(args.verdicts)
    name = args.name or p.stem
    df = load(p)
    print(f"loaded {len(df):,} parsed verdicts from {p}")
    plot_share_by_venue_year(df, name)
    plot_share_by_decision(df, name)
    plot_label_mix(df, name)


if __name__ == "__main__":
    main()
