# %% [markdown]
# # Explore downloaded guides + extracted rubrics
#
# Loads every GPT-5-mini extraction into two tidy dataframes:
# - `pages_df`  — one row per source page (orientation, subtask, audience, …)
# - `rubrics_df` — one row per extracted rubric (joined back to its source page)
#
# Caches both to parquet on first run so subsequent runs are fast.
# Then walks through coverage stats, sample inspection, subtask breakdowns,
# cross-task duplicate exploration, and rubric content stats.

# %% setup
import json
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
DATASETS = ROOT / "datasets"
CACHE = ROOT / "notebooks" / "_explore_cache"
CACHE.mkdir(parents=True, exist_ok=True)

TASKS = [
    "creative-writing", "peer-review", "math-stackexchange", "news-homepages",
    "press-releases", "code-review", "grant-funding", "humor",
    "legal-outcome-prediction", "notice-and-comment", "patents",
]

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 180)


# %% [markdown]
# ## 1. Load every extraction into tidy frames
#
# Each per-page JSON in `gpt-parsed/gpt-5-mini/` has shape:
# ```
# {path, model, input_tokens, output_tokens, elapsed_s, extracted: {
#   orientation, intended_audience, subtask_short, subtask_description,
#   subtask_keywords, subtask_breadth, error, rubrics_metrics: [{name, description, guidance}, …]
# }}
# ```

# %%
def _collect():
    page_rows = []
    rubric_rows = []
    for task in TASKS:
        d = DATASETS / task / "online-rubrics" / "gpt-parsed" / "gpt-5-mini"
        if not d.is_dir():
            continue
        for fp in d.glob("*.json"):
            if fp.name.startswith("_"):
                continue
            try:
                obj = json.loads(fp.read_text())
            except Exception as e:
                continue
            ex = obj.get("extracted") or {}
            src_name = fp.name[:-5]  # strip ".json"
            # filename is "<source_dir>__<basename>" — pull source dir as first segment
            if "__" in src_name:
                src_dir, src_base = src_name.split("__", 1)
            else:
                src_dir, src_base = "raw", src_name
            # filename prefix hints at which collection wave produced it
            slug = src_base.lower()
            if slug.startswith("rej_"):
                wave_tag = "rej"
            elif slug.startswith("waveh1_") or "waveh1" in slug:
                wave_tag = "h1_wayback"
            elif slug.startswith("waveh2_") or "waveh2" in slug:
                wave_tag = "h2_editions"
            elif slug.startswith("waveh3_") or "waveh3" in slug:
                wave_tag = "h3_archives"
            elif slug.startswith("waveh4_") or "waveh4" in slug:
                wave_tag = "h4_ancient"
            elif slug.startswith("waveh5_") or "waveh5" in slug:
                wave_tag = "h5_20c"
            elif slug.startswith("waveh6_") or "waveh6" in slug:
                wave_tag = "h6_editions"
            elif slug.startswith("wavee_") or "wavee" in slug:
                wave_tag = "wavee_gap"
            elif slug.startswith("wavec_"):
                wave_tag = "wavec_topup"
            elif slug.startswith("waved_"):
                wave_tag = "waved_topup"
            elif slug.startswith("waveb_") or "waveb" in slug:
                wave_tag = "waveb"
            elif slug.startswith("phase"):
                wave_tag = "phase_initial"
            elif slug.startswith("existing_") or src_dir == "claude-parsed":
                wave_tag = "claude_parsed" if src_dir == "claude-parsed" else "existing"
            else:
                wave_tag = "other"

            error = ex.get("error")
            rubrics = ex.get("rubrics_metrics") or []
            page_id = f"{task}::{src_dir}::{src_base}"
            page_rows.append({
                "page_id": page_id,
                "task": task,
                "source_dir": src_dir,   # 'raw' or 'claude-parsed'
                "source_file": src_base,
                "wave_tag": wave_tag,
                "orientation": ex.get("orientation"),
                "intended_audience": ex.get("intended_audience"),
                "subtask_short": ex.get("subtask_short"),
                "subtask_description": ex.get("subtask_description"),
                "subtask_keywords": ex.get("subtask_keywords") or [],
                "subtask_breadth": ex.get("subtask_breadth"),
                "error": error,
                "is_error": bool(error) or ex.get("orientation") == "error",
                "n_rubrics": len(rubrics),
                "input_tokens": obj.get("input_tokens", 0),
                "output_tokens": obj.get("output_tokens", 0),
                "elapsed_s": obj.get("elapsed_s", 0.0),
            })
            for idx, r in enumerate(rubrics):
                rubric_rows.append({
                    "page_id": page_id,
                    "task": task,
                    "source_dir": src_dir,
                    "orientation": ex.get("orientation"),
                    "subtask_short": ex.get("subtask_short"),
                    "subtask_breadth": ex.get("subtask_breadth"),
                    "rubric_idx": idx,
                    "rubric_name": (r.get("name") or "").strip(),
                    "rubric_description": (r.get("description") or "").strip(),
                    "rubric_guidance": (r.get("guidance") or "").strip(),
                    "name_len": len(r.get("name") or ""),
                    "desc_len": len(r.get("description") or ""),
                    "guidance_len": len(r.get("guidance") or ""),
                })
    return pd.DataFrame(page_rows), pd.DataFrame(rubric_rows)


pages_path = CACHE / "pages.parquet"
rubrics_path = CACHE / "rubrics.parquet"

if pages_path.exists() and rubrics_path.exists():
    pages_df = pd.read_parquet(pages_path)
    rubrics_df = pd.read_parquet(rubrics_path)
    print(f"loaded from cache: {len(pages_df):,} pages, {len(rubrics_df):,} rubrics")
else:
    pages_df, rubrics_df = _collect()
    # Persist (subtask_keywords is a list -> store as JSON string for parquet compat)
    pages_df["subtask_keywords"] = pages_df["subtask_keywords"].apply(json.dumps)
    pages_df.to_parquet(pages_path)
    rubrics_df.to_parquet(rubrics_path)
    pages_df["subtask_keywords"] = pages_df["subtask_keywords"].apply(json.loads)
    print(f"built fresh: {len(pages_df):,} pages, {len(rubrics_df):,} rubrics")

# Make sure keyword column is a list, not JSON string (after reload from parquet)
if isinstance(pages_df["subtask_keywords"].iloc[0], str):
    pages_df["subtask_keywords"] = pages_df["subtask_keywords"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )


# %% [markdown]
# ## 2. Coverage & basic counts

# %% pages per task
coverage = (
    pages_df.groupby("task")
    .agg(
        n_pages=("page_id", "size"),
        n_pages_with_rubrics=("n_rubrics", lambda s: (s > 0).sum()),
        n_error_pages=("is_error", "sum"),
        n_rubrics_total=("n_rubrics", "sum"),
        median_rubrics_per_page=("n_rubrics", "median"),
        mean_rubrics_per_page=("n_rubrics", "mean"),
    )
    .sort_values("n_pages", ascending=False)
)
coverage["error_pct"] = (coverage["n_error_pages"] / coverage["n_pages"] * 100).round(1)
coverage["rubric_yield_pct"] = (coverage["n_pages_with_rubrics"] / coverage["n_pages"] * 100).round(1)
print(coverage)

# %% source-dir breakdown (raw vs claude-parsed) per task
src_breakdown = (
    pages_df.groupby(["task", "source_dir"])
    .agg(n_pages=("page_id", "size"), n_rubrics=("n_rubrics", "sum"), error_rate=("is_error", "mean"))
    .round(3)
)
print(src_breakdown)

# %% wave_tag breakdown — which collection wave produced each page
wave_breakdown = (
    pages_df.groupby(["task", "wave_tag"])
    .agg(n_pages=("page_id", "size"), n_rubrics=("n_rubrics", "sum"))
    .reset_index()
    .pivot_table(index="task", columns="wave_tag", values="n_pages", fill_value=0)
)
print("Pages per (task, wave_tag):")
print(wave_breakdown)


# %% [markdown]
# ## 3. Sample inspection
#
# Look at a handful of real outputs per task to spot quality issues.

# %% helper to pretty-print one page + its rubrics
def show_page(page_id, max_rubrics=5, max_desc=200):
    page = pages_df[pages_df.page_id == page_id].iloc[0]
    rubrics = rubrics_df[rubrics_df.page_id == page_id]
    print(f"\n{'='*100}")
    print(f"PAGE: {page['task']}  /  {page['source_dir']}/{page['source_file']}")
    print(f"  orientation       : {page['orientation']}")
    print(f"  intended_audience : {(page['intended_audience'] or '')[:140]}")
    print(f"  subtask_short     : {page['subtask_short']}")
    print(f"  subtask_keywords  : {page['subtask_keywords']}")
    print(f"  subtask_breadth   : {page['subtask_breadth']}")
    if page["error"]:
        print(f"  ERROR             : {page['error']}")
    print(f"  n_rubrics         : {page['n_rubrics']}")
    for _, r in rubrics.head(max_rubrics).iterrows():
        print(f"  [{r['rubric_idx']}] {r['rubric_name'][:90]}")
        print(f"      desc: {r['rubric_description'][:max_desc]}")
        if r["rubric_guidance"]:
            print(f"      guid: {r['rubric_guidance'][:max_desc]}")
    if page["n_rubrics"] > max_rubrics:
        print(f"  ... +{page['n_rubrics']-max_rubrics} more rubrics")


# Sample 2 random pages per task
rng = np.random.default_rng(11)
for task in TASKS:
    sub = pages_df[(pages_df.task == task) & (~pages_df.is_error) & (pages_df.n_rubrics > 0)]
    if len(sub) == 0:
        continue
    for pid in rng.choice(sub.page_id.values, size=min(2, len(sub)), replace=False):
        show_page(pid)


# %% [markdown]
# ## 4. Rubric-count distribution per task

# %% histogram of rubrics-per-page, faceted by task
fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=False)
for ax, task in zip(axes.flat, TASKS):
    s = pages_df.loc[pages_df.task == task, "n_rubrics"]
    ax.hist(s.clip(upper=50), bins=50, color="#4C72B0", alpha=0.85)
    ax.set_title(f"{task}\nμ={s.mean():.1f} med={s.median():.0f} sum={s.sum():,}", fontsize=9)
    ax.set_xlabel("rubrics per page")
    ax.set_yscale("log")
for ax in axes.flat[len(TASKS):]:
    ax.set_visible(False)
fig.suptitle("Rubrics-per-page distribution per task (log y; x-clipped at 50)")
plt.tight_layout()
plt.show()


# %% top 5 highest-rubric pages per task (these are the "anthology" sources)
top5 = (
    pages_df.sort_values(["task", "n_rubrics"], ascending=[True, False])
    .groupby("task")
    .head(5)
    [["task", "source_dir", "source_file", "orientation", "subtask_short", "n_rubrics"]]
)
print(top5.to_string(index=False))


# %% [markdown]
# ## 5. Orientation & breadth breakdown per task

# %%
orient_ct = pd.crosstab(pages_df.task, pages_df.orientation).fillna(0).astype(int)
print("Orientation × task:")
print(orient_ct)

# %%
breadth_ct = pd.crosstab(pages_df.task, pages_df.subtask_breadth).fillna(0).astype(int)
# Order breadth columns sensibly
breadth_order = ["very_narrow", "narrow", "moderate", "broad", "very_broad", ""]
present_cols = [c for c in breadth_order if c in breadth_ct.columns] + [
    c for c in breadth_ct.columns if c not in breadth_order
]
breadth_ct = breadth_ct[present_cols]
print("\nSubtask breadth × task:")
print(breadth_ct)


# %% [markdown]
# ## 6. Subtask vocabulary — what are the page-level subtasks?
#
# `subtask_short` is the per-page subtask label. Top values per task hint at how diverse the subtask space is.

# %%
for task in TASKS:
    sub = pages_df[(pages_df.task == task) & (~pages_df.is_error)]
    counts = sub.subtask_short.fillna("").value_counts().head(15)
    print(f"\n=== {task}  (n={len(sub):,} non-error pages, {sub.subtask_short.nunique():,} distinct subtasks) ===")
    print(counts.to_string())


# %% [markdown]
# ## 7. Top keywords per task
#
# Aggregate `subtask_keywords` over all non-error pages, per task.

# %%
from collections import Counter

for task in TASKS:
    sub = pages_df[(pages_df.task == task) & (~pages_df.is_error)]
    c = Counter()
    for kws in sub.subtask_keywords:
        c.update([k.lower() for k in (kws or [])])
    print(f"\n=== {task}  (top 25 subtask_keywords) ===")
    for k, v in c.most_common(25):
        print(f"  {v:>5d}  {k}")


# %% [markdown]
# ## 8. Cross-task duplicate exploration (cheap first pass)
#
# A first-cut "are these the same rubric?" using **normalized rubric name** as a hash.
# Real linking will need embedding + cross-encoder (see goal 1a). This is just to spot
# obvious overlaps across tasks before we get serious.

# %% normalize rubric names and find ones appearing in multiple tasks
def _norm(name):
    s = (name or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


rubrics_df["rubric_name_norm"] = rubrics_df["rubric_name"].apply(_norm)

# Drop empty names
nm = rubrics_df[rubrics_df.rubric_name_norm.str.len() > 0]

# For each normalized name, count distinct tasks
tasks_per_name = nm.groupby("rubric_name_norm")["task"].nunique().sort_values(ascending=False)
print(f"Distinct normalized rubric names: {len(tasks_per_name):,}")
print(f"Names appearing in >1 task    : {(tasks_per_name > 1).sum():,}")
print(f"Names appearing in >5 tasks   : {(tasks_per_name > 5).sum():,}")

print("\nTop cross-task duplicates (appear in many tasks):")
cross = tasks_per_name[tasks_per_name > 4].head(40)
print(cross.to_string())

# %% show a few examples of cross-task rubric overlaps
print("\nSample of cross-task overlapping rubrics:")
for nm_val in cross.head(8).index:
    matches = rubrics_df[rubrics_df.rubric_name_norm == nm_val][
        ["task", "source_dir", "rubric_name", "rubric_description"]
    ].head(6)
    print(f"\n--- name: '{nm_val}' ---")
    for _, r in matches.iterrows():
        print(f"  [{r['task']:25s}] {r['rubric_name'][:60]}")
        print(f"     desc: {r['rubric_description'][:160]}")


# %% within-task duplicate detection
within_task_dupes = (
    nm.groupby(["task", "rubric_name_norm"]).size().reset_index(name="n_copies")
)
print("\nWithin-task duplicate counts (top 20):")
print(within_task_dupes.sort_values("n_copies", ascending=False).head(20).to_string(index=False))

# Per-task duplicate stats
print("\nWithin-task duplicate summary:")
for task in TASKS:
    s = within_task_dupes[within_task_dupes.task == task]
    if len(s) == 0:
        continue
    n_uniq = len(s)
    n_total = s.n_copies.sum()
    print(f"  {task:30s} unique_names={n_uniq:>6,d} total_occurrences={n_total:>6,d} dup_rate={(n_total-n_uniq)/n_total*100:>5.1f}%")


# %% [markdown]
# ## 9. Rubric content stats
#
# Description / guidance length distributions. The `guidance/description` ratio is one
# tacit-knowledge proxy (more guidance → more contextual fill-in needed).

# %%
rubrics_df["guidance_to_desc"] = rubrics_df["guidance_len"] / rubrics_df["desc_len"].clip(lower=1)

length_stats = (
    rubrics_df.groupby("task")
    .agg(
        n_rubrics=("rubric_name", "size"),
        median_desc_len=("desc_len", "median"),
        mean_desc_len=("desc_len", "mean"),
        median_guidance_len=("guidance_len", "median"),
        mean_guidance_len=("guidance_len", "mean"),
        median_guidance_ratio=("guidance_to_desc", "median"),
        mean_guidance_ratio=("guidance_to_desc", "mean"),
    )
    .round(2)
)
print(length_stats)


# %% plot guidance/description ratio per task
fig, ax = plt.subplots(figsize=(12, 5))
data = [rubrics_df.loc[rubrics_df.task == t, "guidance_to_desc"].clip(upper=4).values for t in TASKS]
ax.boxplot(data, labels=TASKS, showfliers=False)
ax.set_ylabel("guidance_len / description_len (clipped at 4)")
ax.set_title("Tacit-knowledge proxy: per-rubric guidance/description ratio, by task")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 10. Rubric-density by orientation
#
# Are formal_guideline pages more rubric-dense than blog_post pages? (Authority proxy.)

# %%
density_by_orient = (
    pages_df.groupby(["task", "orientation"])
    .agg(n_pages=("page_id", "size"), mean_rubrics=("n_rubrics", "mean"))
    .round(2)
)
print(density_by_orient)


# %% [markdown]
# ## 11. Pages tagged 'error' — quick QA
#
# What's actually in the error pages? (Captcha, 404, empty, paywall…)

# %%
err = pages_df[pages_df.is_error]
print(f"\nError pages by task:")
print(err.task.value_counts())
print(f"\nError-reason samples:")
print(err.error.value_counts().head(25))


# %% [markdown]
# ## 12. Convenience: save a flat CSV for downstream tools

# %%
flat_csv = CACHE / "rubrics_flat.csv"
rubrics_df.drop(columns=["rubric_name_norm"]).to_csv(flat_csv, index=False)
print(f"wrote {flat_csv}  ({flat_csv.stat().st_size/1e6:.1f} MB)")

pages_csv = CACHE / "pages.csv"
pages_export = pages_df.copy()
pages_export["subtask_keywords"] = pages_export["subtask_keywords"].apply(json.dumps)
pages_export.to_csv(pages_csv, index=False)
print(f"wrote {pages_csv}  ({pages_csv.stat().st_size/1e6:.1f} MB)")
