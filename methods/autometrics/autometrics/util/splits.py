from __future__ import annotations

from pathlib import Path
from typing import Tuple
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FIXED_TRAIN_FRACTION = 0.8
FIXED_EVAL_FRACTION = 0.1
FIXED_TEST_FRACTION = 0.1


def _default_split_dir(data_path: str) -> Path:
    """Choose a deterministic on-disk location for the fixed split."""
    data_file = Path(data_path).resolve()
    stem = data_file.stem
    if stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    return data_file.parent / stem


def _preferred_split_suffix(data_path: str) -> str:
    data_file = Path(data_path).name
    if data_file.endswith(".csv.gz"):
        return ".csv.gz"
    return ".csv"


def _split_paths(split_root: Path, name: str, data_path: str) -> Tuple[Path, Path]:
    """Return (preferred_path, alternate_path) for split files."""
    preferred_suffix = _preferred_split_suffix(data_path)
    preferred = split_root / f"{name}{preferred_suffix}"
    alternate = split_root / f"{name}.csv.gz" if preferred_suffix == ".csv" else split_root / f"{name}.csv"
    return preferred, alternate


def _existing_split_path(split_root: Path, name: str, data_path: str) -> Path | None:
    preferred, alternate = _split_paths(split_root, name, data_path)
    if preferred.exists():
        return preferred
    if alternate.exists():
        return alternate
    return None


def load_fixed_split(
    data_path: str,
    split_dir: str | None = None,
    *,
    create_if_missing: bool = False,
    seed: int = 42,
    label_column: str = "judgement",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load existing on-disk train/eval/test split, or create it if missing.

    Mirrors methods/dense/train_reward_model.py for split semantics.
    """
    if not data_path:
        raise ValueError("data_path is required to locate on-disk train/eval/test splits.")

    split_root = Path(split_dir).resolve() if split_dir else _default_split_dir(data_path)
    train_path = _existing_split_path(split_root, "train", data_path)
    eval_path = _existing_split_path(split_root, "eval", data_path)
    test_path = _existing_split_path(split_root, "test", data_path)

    split_exists = train_path is not None and eval_path is not None and test_path is not None
    if not split_exists:
        existing_count = sum(int(p is not None) for p in (train_path, eval_path, test_path))
        if existing_count > 0:
            raise RuntimeError(
                f"Incomplete split in {split_root}. Expected train/eval/test (csv or csv.gz)."
            )
        if not create_if_missing:
            raise FileNotFoundError(
                f"Could not find on-disk split in {split_root}. "
                "Expected train/eval/test as .csv or .csv.gz."
            )

        # Create split
        split_root.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(data_path)
        stratify_full = df[label_column] if label_column in df and df[label_column].nunique() > 1 else None
        train_eval_df, test_df = train_test_split(
            df,
            test_size=FIXED_TEST_FRACTION,
            random_state=seed,
            stratify=stratify_full,
        )
        eval_fraction_within_remaining = FIXED_EVAL_FRACTION / (FIXED_TRAIN_FRACTION + FIXED_EVAL_FRACTION)
        stratify_train_eval = (
            train_eval_df[label_column] if label_column in train_eval_df and train_eval_df[label_column].nunique() > 1 else None
        )
        train_df, eval_df = train_test_split(
            train_eval_df,
            test_size=eval_fraction_within_remaining,
            random_state=seed,
            stratify=stratify_train_eval,
        )

        preferred_suffix = _preferred_split_suffix(data_path)
        train_path = split_root / f"train{preferred_suffix}"
        eval_path = split_root / f"eval{preferred_suffix}"
        test_path = split_root / f"test{preferred_suffix}"

        train_df.to_csv(train_path, index=False)
        eval_df.to_csv(eval_path, index=False)
        test_df.to_csv(test_path, index=False)
        metadata = {
            "data_path": str(Path(data_path).resolve()),
            "seed": seed,
            "train_fraction": FIXED_TRAIN_FRACTION,
            "eval_fraction": FIXED_EVAL_FRACTION,
            "test_fraction": FIXED_TEST_FRACTION,
            "train_size": int(len(train_df)),
            "eval_size": int(len(eval_df)),
            "test_size": int(len(test_df)),
        }
        with open(split_root / "split_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        return train_df, eval_df, test_df

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    test_df = pd.read_csv(test_path)
    total = max(1, len(train_df) + len(eval_df) + len(test_df))
    observed_train_fraction = len(train_df) / total
    observed_eval_fraction = len(eval_df) / total
    observed_test_fraction = len(test_df) / total
    if (
        not np.isclose(observed_train_fraction, FIXED_TRAIN_FRACTION, atol=2e-2)
        or not np.isclose(observed_eval_fraction, FIXED_EVAL_FRACTION, atol=2e-2)
        or not np.isclose(observed_test_fraction, FIXED_TEST_FRACTION, atol=2e-2)
    ):
        raise RuntimeError(
            "Existing split ratios do not match expected train/eval/test 80/10/10 "
            f"(observed train={observed_train_fraction:.4f}, eval={observed_eval_fraction:.4f}, "
            f"test={observed_test_fraction:.4f}) in {split_root}."
        )
    return train_df, eval_df, test_df
