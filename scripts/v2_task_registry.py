"""Per-task config for the v2 validity pipeline.

Each task entry describes:
  - aspects_sources: list of R2-aspects JSON files to concatenate
  - datapoints_csv:  path to the CSV/parquet with labeled examples
  - text_col, label_col: column names within the source
  - label_type: "binary" (0/1) | "continuous" (will be median-split for AUC)
  - n_target_dps: default sample size

The v2 pipeline scripts call `task_config(task_name)` to resolve all paths
under runs/validity_full/v2/<task_name>/...
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V2_ROOT = REPO / "runs" / "validity_full" / "v2"
HIER = REPO / "outputs" / "hierarchy"


def _hier_files(task_slug: str) -> list[Path]:
    """All R2 expanded files for a task across buckets (general/specific/hyper_specific)."""
    out = []
    for bucket in ("general", "specific", "hyper_specific"):
        f = HIER / f"{task_slug}_{bucket}_r2_expanded.json"
        if f.exists():
            out.append(f)
    return out


TASK_REGISTRY: dict[str, dict] = {
    "peer_review": {
        "slug": "peer-review",
        "aspects_sources": _hier_files("peer-review"),
        "datapoints_csv": REPO / "datasets/peer-review/splits/train.csv.gz",
        "text_col": "text",
        "label_col": "judgement",
        "label_type": "binary",
        "n_target_dps": 5000,
        "venue_col": "venue",  # used for stratification
    },
    "creative_writing": {
        "slug": "creative-writing",
        "aspects_sources": _hier_files("creative-writing"),
        # LitBench: each row is one text with binary judgement (upvote-threshold)
        # — the cleanest+most-recent set per user. 87654 rows, 31% positive.
        "datapoints_csv": REPO / "datasets/creative-writing/litbench-to-train.csv.gz",
        "text_col": "text",
        "label_col": "judgement",
        "label_type": "binary",
        "n_target_dps": 5000,
    },
    "press_releases": {
        "slug": "press-releases",
        "aspects_sources": _hier_files("press-releases"),
        # 128131 rows, judgement = was picked up by news
        "datapoints_csv": REPO / "datasets/press-releases/press_release_modeling_dataset.csv.gz",
        "text_col": "text",
        "label_col": "judgement",
        "label_type": "binary",
        "n_target_dps": 5000,
        "stratify_col": "news_article_domain",
    },
    "code_review": {
        "slug": "code-review",
        "aspects_sources": _hier_files("code-review"),
        # Aggregated PR-level dataset: 141K PRs, judgement=pr_merged binary
        # Built from code_review_modeling_dataset.csv.gz (5.3M comment rows)
        # by grouping on (owner, repo, pr_number) and concatenating pr_title
        # + review comment bodies into a single text field.
        "datapoints_csv": REPO / "datasets/code-review/code_review_pr_aggregated.csv.gz",
        "text_col": "text",
        "label_col": "judgement",
        "label_type": "binary",
        "n_target_dps": 5000,
    },
    "math": {
        "slug": "math-stackexchange",
        "aspects_sources": _hier_files("math-stackexchange"),
        # Pulled from sk3 (cleanest-most-recent: text+judgement schema, 150K balanced)
        "datapoints_csv": REPO / "datasets/math-stackexchange/math_se_modeling.csv.gz",
        "text_col": "text",
        "label_col": "judgement",
        "label_type": "binary",
        "n_target_dps": 5000,
    },
    "humor": {
        "slug": "humor",
        "aspects_sources": _hier_files("humor"),
        # 383786 rows balanced (text+judgement), reddit humor dedup. text mean 148 chars (short jokes)
        "datapoints_csv": REPO / "datasets/humor/reddit_humor_modeling_dedup.csv.gz",
        "text_col": "text",
        "label_col": "judgement",
        "label_type": "binary",
        "n_target_dps": 5000,
    },
    "news_homepages": {
        "slug": "news-homepages",
        "aspects_sources": _hier_files("news-homepages"),
        # 183708 rows balanced. text mean 2323. Topic+groupsplit per [[project_homepage_newsworthiness]]
        "datapoints_csv": REPO / "datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz",
        "text_col": "text",
        "label_col": "judgement",
        "label_type": "binary",
        "n_target_dps": 5000,
    },
    "patents": {
        "slug": "patents",
        "aspects_sources": _hier_files("patents"),
        # 500K rows balanced, final outcome. text mean 6786 chars (long patents)
        "datapoints_csv": REPO / "datasets/patents/patents_final_outcome_balanced.csv.gz",
        "text_col": "text",
        "label_col": "judgement",
        "label_type": "binary",
        "n_target_dps": 5000,
    },
    "notice_and_comment": {
        "slug": "notice-and-comment",
        "aspects_sources": _hier_files("notice-and-comment"),
        # 235K rows, judgement binary (substantive vs not). Mean text 117 chars.
        "datapoints_csv": REPO / "datasets/notice-and-comment/notice_and_comment.csv.gz",
        "text_col": "text",
        "label_col": "judgement",
        "label_type": "binary",
        "n_target_dps": 5000,
    },
}


def task_dir(task_name: str) -> Path:
    """Returns runs/validity_full/v2/<task_name>/ (creates if missing)."""
    p = V2_ROOT / task_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def task_config(task_name: str) -> dict:
    """Get task config or raise KeyError with list of known tasks."""
    if task_name not in TASK_REGISTRY:
        raise KeyError(f"unknown task '{task_name}'. Known: "
                       f"{sorted(TASK_REGISTRY)}")
    cfg = dict(TASK_REGISTRY[task_name])
    cfg["task_dir"] = task_dir(task_name)
    return cfg


def shared_dir() -> Path:
    p = V2_ROOT / "_shared"
    p.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cfg = task_config(sys.argv[1])
        print(f"task: {sys.argv[1]}")
        for k, v in cfg.items():
            if isinstance(v, list):
                print(f"  {k}: {len(v)} files")
                for f in v: print(f"    {f}")
            else:
                print(f"  {k}: {v}")
    else:
        print("known tasks:")
        for name in TASK_REGISTRY:
            cfg = task_config(name)
            print(f"  {name:<20} slug={cfg['slug']:<20} "
                  f"aspect_files={len(cfg['aspects_sources'])} "
                  f"data_csv_exists={cfg['datapoints_csv'].exists()}")
