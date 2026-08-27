"""Build shared, label-free item panels for hierarchy prompt/code comparisons.

Both the prompt arm and executable candidate must receive the same ``ctext``.
This builder projects corpus rows to text only, applies the task's declared
prompt-length representation once, exact-deduplicates after that projection,
and freezes disjoint compiler-train and sealed-evaluation files.  Outcome
columns are never emitted or used for sampling.

The code-review corpus registered in ``metric_implementer`` is server-only on
the audited host, so that task uses the existing active metric-seam ``ctext``
items as an explicit fallback.  It is not silently replaced by competitive-code
trial data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from methods.metric_implementer.config import ImplementerConfig, apply_task_preset
from methods.metric_implementer.manifest import DatasetEntry, full_manifest, load_corpus


REPO = Path(__file__).resolve().parents[2]
SCHEMA = "metric-seam.hierarchy-shared-items.v1"
DEFAULT_SALT = "metric-seam-hierarchy-shared-items-v1"
DEFAULT_N = 300

TASKS = (
    "humor",
    "creative-writing",
    "code-review",
    "peer-review",
    "press-releases",
    "news-homepages",
    "grant-funding",
    "legal-outcome-prediction",
    "notice-and-comment",
    "patents",
    "math-stackexchange",
)

CODE_REVIEW_FALLBACK = REPO / "outputs/metric_seam_pilot/tasks/code_review/items.json"
CREATIVE_WRITING_FALLBACK = (
    REPO / "methods/metric_implementer/trial/pool_creative_writing.jsonl.gz"
)


def _rank(salt: str, task: str, text: str) -> str:
    return hashlib.sha256(f"{salt}\x1f{task}\x1f{text}".encode("utf-8")).hexdigest()


def _entry_for_task(task: str) -> DatasetEntry | None:
    return next((entry for entry in full_manifest().datasets if entry.task == task), None)


def _load_code_review_fallback(path: Path = CODE_REVIEW_FALLBACK) -> tuple[list[str], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"code-review fallback is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("code-review fallback must be a JSON list")
    texts = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping) or not isinstance(row.get("ctext"), str):
            raise ValueError(f"code-review fallback row {index} lacks string ctext")
        texts.append(row["ctext"])
    return texts, {
        "kind": "active_metric_seam_ctext_fallback",
        "path": str(path.relative_to(REPO)),
        "text_column": "ctext",
    }


def _load_creative_writing_fallback(
    path: Path = CREATIVE_WRITING_FALLBACK,
) -> tuple[list[str], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"creative-writing fallback is missing: {path}")
    entry = DatasetEntry(
        task="creative-writing",
        name="metric_implementer_trial_creative_writing",
        path=str(path),
        text_column="text",
    )
    texts, _ids = load_corpus(entry, 0, seed=7)
    return texts, {
        "kind": "metric_implementer_trial_fallback",
        "dataset_name": entry.name,
        "path": str(path.relative_to(REPO)),
        "text_column": "text",
        "reason": "canonical creative-writing corpus is not present on this host",
    }


def load_source_texts(
    task: str,
    *,
    requested: int,
    corpus_loader: Callable[[DatasetEntry, int, int], tuple[list[str], list[str]]] = load_corpus,
) -> tuple[list[str], dict]:
    """Load text without returning or consulting any label column."""
    if task not in TASKS:
        raise ValueError(f"unknown hierarchy task: {task}")
    entry = _entry_for_task(task)
    if entry is not None and Path(entry.path).is_file():
        texts, _source_ids = corpus_loader(entry, requested, 7)
        return texts, {
            "kind": "metric_implementer_manifest_corpus",
            "dataset_name": entry.name,
            "path": str(Path(entry.path).resolve().relative_to(REPO)),
            "text_column": entry.text_column,
        }
    if task == "code-review":
        return _load_code_review_fallback()
    if task == "creative-writing":
        return _load_creative_writing_fallback()
    missing = entry.path if entry is not None else "no manifest entry"
    raise FileNotFoundError(f"no local corpus for {task}: {missing}")


def build_task_items(
    task: str,
    *,
    n: int = DEFAULT_N,
    salt: str = DEFAULT_SALT,
    source_texts: Sequence[str] | None = None,
    source_record: Mapping | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    if n < 40:
        raise ValueError("n must be at least 40")
    cfg = apply_task_preset(ImplementerConfig(), task)
    max_chars = int(cfg.max_text_chars)
    if source_texts is None:
        # Oversample before exact-deduplication. Small corpora return their full content.
        source_texts, loaded_record = load_source_texts(task, requested=max(n * 2, n))
        source_record = loaded_record
    if source_record is None:
        source_record = {"kind": "injected_test_source", "path": None, "text_column": "text"}

    projected = []
    for raw in source_texts:
        if not isinstance(raw, str):
            raise ValueError("every source item must be text")
        text = raw[:max_chars]
        if text.strip():
            projected.append(text)
    unique = sorted(set(projected), key=lambda text: (_rank(salt, task, text), text))
    selected = unique[:n]
    if len(selected) < 40:
        raise ValueError(f"{task}: only {len(selected)} unique projected texts")
    train_count = min(150, len(selected) // 2)
    train_texts = selected[:train_count]
    heldout_texts = selected[train_count:]
    train = [
        {"item_key": f"train_{index:04d}", "ctext": text}
        for index, text in enumerate(train_texts, 1)
    ]
    heldout = [
        {"item_key": f"heldout_{index:04d}", "ctext": text}
        for index, text in enumerate(heldout_texts, 1)
    ]
    manifest = {
        "schema": SCHEMA,
        "task": task,
        "representation": {
            "field": "ctext",
            "max_chars": max_chars,
            "projection": "source_text[:max_chars] before exact deduplication",
            "same_bytes_required_for_prompt_and_code": True,
        },
        "selection": {
            "salt": salt,
            "rule": "stable text hash after ctext projection; first n; first half compiler-train",
            "requested_n": n,
            "source_rows_loaded": len(source_texts),
            "nonempty_projected_rows": len(projected),
            "unique_projected_rows": len(unique),
            "selected_n": len(selected),
            "train_n": len(train),
            "heldout_n": len(heldout),
            "outcome_or_reference_values_used": False,
        },
        "source": dict(source_record),
        "policy": {
            "outcome_columns_emitted": False,
            "source_identifiers_emitted": False,
            "compiler_receives_heldout_text": False,
            "external_supervision_used": False,
        },
    }
    return manifest, train, heldout


def validate_task_items(manifest: Mapping, train: Sequence[Mapping],
                        heldout: Sequence[Mapping]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected shared-item schema")
    if manifest.get("policy", {}).get("outcome_columns_emitted") is not False:
        raise ValueError("shared item panel may not emit outcomes")
    expected_train = [f"train_{index:04d}" for index in range(1, len(train) + 1)]
    expected_heldout = [f"heldout_{index:04d}" for index in range(1, len(heldout) + 1)]
    for split, rows, expected in (
        ("train", train, expected_train),
        ("heldout", heldout, expected_heldout),
    ):
        if [row.get("item_key") for row in rows] != expected:
            raise ValueError(f"{split} item keys are not ordered opaque aliases")
        if any(set(row) != {"item_key", "ctext"} for row in rows):
            raise ValueError(f"{split} rows expose fields outside item_key/ctext")
        if any(not isinstance(row["ctext"], str) or not row["ctext"].strip() for row in rows):
            raise ValueError(f"{split} contains invalid ctext")
    train_texts = {row["ctext"] for row in train}
    heldout_texts = {row["ctext"] for row in heldout}
    if train_texts & heldout_texts:
        raise ValueError("compiler-train and heldout ctext overlap")
    selection = manifest.get("selection", {})
    if selection.get("train_n") != len(train) or selection.get("heldout_n") != len(heldout):
        raise ValueError("manifest split counts do not match item files")


def _write(path: Path, payload: Mapping | Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    args = parser.parse_args(argv)
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    summaries = []
    for task in tasks:
        manifest, train, heldout = build_task_items(task, n=args.n, salt=args.salt)
        validate_task_items(manifest, train, heldout)
        task_root = args.out_root / task
        _write(task_root / "manifest.json", manifest)
        _write(task_root / "compiler_train.json", train)
        _write(task_root / "sealed_heldout.json", heldout)
        summaries.append(
            {"task": task, "train_n": len(train), "heldout_n": len(heldout),
             "max_chars": manifest["representation"]["max_chars"]}
        )
    print(json.dumps({"tasks": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
