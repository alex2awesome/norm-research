#!/usr/bin/env python3
"""End-to-end mature V/A bank for the canonical WritingPrompts clean pool.

The judge backend is the authenticated Codex CLI pinned to gpt-5.6-sol.  Each
anonymous story/criterion cell is returned as exactly one member of
{1.0, 0.5, 0.0, NA}.  Labels never enter judge prompts and are not used by the
fidelity pilot or criterion rewrite.

Typical run:
  python score_va.py reconstruct
  python score_va.py pilot --rubrics rubrics_initial.jsonl
  # freeze/refine rubrics.jsonl using only fidelity_pilot.json
  python score_va.py score --rubrics rubrics.jsonl --workers 4
  python score_va.py readout
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v_features import V_NAMES, feature_vector

HERE = Path(__file__).resolve().parent
CW_DIR = HERE.parent
CANONICAL = CW_DIR / "writingprompts_modeling_clean.csv.gz"
PRECURSOR = CW_DIR / "writingprompts_modeling_with_topics.csv.gz"
RECONSTRUCTED = HERE / "writingprompts_modeling_clean_reconstructed.csv.gz"
SAMPLE = HERE / "sample_manifest.csv.gz"
SCORES = HERE / "scores.npz"
BATCH_DIR = HERE / "batches"
SCHEMA = HERE / "scoring_schema.json"
PILOT_OUT = HERE / "fidelity_pilot.json"
RESULTS = HERE / "results.json"
RESULTS_MD = HERE / "RESULTS.md"

MODEL = "gpt-5.6-sol"
SEED = 20260728
CANONICAL_BALANCE_SEED = 42
TARGET_N = 2000
TRUNCATE_SOURCE_CHARS = 6000
TRUNCATE_HEAD_CHARS = 3600
TRUNCATE_TAIL_CHARS = 2400
TRUNCATION_MARKER = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"
BUCKET_EDGES = [0, 500, 1000, 2000, 3000, 5000, 10000, 999999]
MOD_TEMPLATE = re.compile(r"this submission has been removed", re.I)
ALLOWED = {"1.0": 1.0, "0.5": 0.5, "0.0": 0.0, "NA": np.nan}

SYSTEM_TEXT = """You are an expert fiction editor performing a measurement task.
For each anonymous PROMPT+STORY and each criterion, judge only evidence in the supplied
text. Do not predict votes, popularity, labels, authorship, or dataset membership.
Each matrix cell must be EXACTLY ONE token from:
1.0 = clearly satisfies the criterion
0.5 = partly, weakly, inconsistently, or borderline
0.0 = clearly fails or cuts against the criterion
NA = the supplied text gives no evidence bearing on the criterion
Use NA only for genuine inapplicability or unavailable evidence, not uncertainty.
Return JSON matching the supplied schema and nothing else."""


def _sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    ids = [str(r.get("id", "")) for r in rows]
    if not rows or len(ids) != len(set(ids)) or any(not x for x in ids):
        raise ValueError(f"Rubric ids must be present and unique: {path}")
    return rows


def split_text(text: str) -> tuple[str, str]:
    parts = str(text).split("\n\nSTORY: ", 1)
    if len(parts) == 2:
        prompt = parts[0].removeprefix("PROMPT: ").strip()
        return prompt, parts[1]
    return "", str(text)


def prompt_id(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]


def _bucket(length: int) -> int:
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= length < BUCKET_EDGES[i + 1]:
            return i
    return len(BUCKET_EDGES) - 2


def reconstruct_canonical() -> dict[str, Any]:
    """Reproduce rebuild_writingprompts_clean.py exactly from the local precursor."""
    if not PRECURSOR.exists():
        raise FileNotFoundError(PRECURSOR)
    source = pd.read_csv(PRECURSOR, usecols=["text", "judgement"])
    buckets: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    dropped = 0
    for text, judgement in source.itertuples(index=False, name=None):
        prompt, story = split_text(text)
        if MOD_TEMPLATE.search(story):
            dropped += 1
            continue
        row = {
            "text": str(text),
            "judgement": int(judgement),
            "story_len": len(story),
            "prompt_id": prompt_id(prompt),
        }
        buckets[_bucket(len(story))][int(judgement)].append(row)

    rng = random.Random(CANONICAL_BALANCE_SEED)
    balanced: list[dict[str, Any]] = []
    bucket_counts = {}
    for b in sorted(buckets):
        pos = buckets[b][1]
        neg = buckets[b][0]
        bucket_counts[str(b)] = {"neg": len(neg), "pos": len(pos)}
        n = min(len(pos), len(neg))
        rng.shuffle(pos)
        rng.shuffle(neg)
        balanced.extend(pos[:n])
        balanced.extend(neg[:n])
    rng.shuffle(balanced)

    out = pd.DataFrame(balanced)[["text", "judgement", "prompt_id"]]
    out.to_csv(RECONSTRUCTED, index=False, compression="gzip")
    inventory = {
        "precursor_path": str(PRECURSOR.relative_to(CW_DIR.parent.parent)),
        "canonical_path_present": CANONICAL.exists(),
        "reconstructed_path": str(RECONSTRUCTED.relative_to(CW_DIR.parent.parent)),
        "precursor_rows": int(len(source)),
        "moderator_template_rows_removed": int(dropped),
        "balanced_rows": int(len(out)),
        "negative": int((out.judgement == 0).sum()),
        "positive": int((out.judgement == 1).sum()),
        "unique_prompt_ids": int(out.prompt_id.nunique()),
        "canonical_rebuild_seed": CANONICAL_BALANCE_SEED,
        "length_bucket_edges": BUCKET_EDGES,
        "post_filter_bucket_counts": bucket_counts,
    }
    if len(out) != 96080 or inventory["positive"] != 48040:
        raise RuntimeError(f"Canonical reconstruction inventory mismatch: {inventory}")
    (HERE / "reconstruction_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    return inventory


def load_canonical() -> pd.DataFrame:
    path = CANONICAL if CANONICAL.exists() else RECONSTRUCTED
    if not path.exists():
        reconstruct_canonical()
        path = RECONSTRUCTED
    df = pd.read_csv(path)
    required = {"text", "judgement", "prompt_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} lacks {required - set(df.columns)}")
    return df


def make_grouped_sample(target_n: int = TARGET_N) -> pd.DataFrame:
    """Hash-order whole prompt groups; labels play no role in selection."""
    df = load_canonical().copy()
    grouped = {str(pid): frame.index.tolist() for pid, frame in df.groupby("prompt_id")}
    ordered_pids = sorted(grouped, key=lambda pid: _sha(f"cw-va-v2-sample|{pid}", 64))
    chosen: list[int] = []
    chosen_pids = []
    for pid in ordered_pids:
        chosen.extend(grouped[pid])
        chosen_pids.append(pid)
        if len(chosen) >= target_n:
            break
    sample = df.loc[chosen].copy().reset_index(drop=True)
    sample["prompt"] = sample.text.map(lambda x: split_text(x)[0])
    sample["story"] = sample.text.map(lambda x: split_text(x)[1])
    sample["doc_id"] = [
        "cw_" + _sha(f"{pid}|{story}", 20)
        for pid, story in zip(sample.prompt_id, sample.story)
    ]
    sample["sample_order"] = np.arange(len(sample))
    sample.to_csv(SAMPLE, index=False, compression="gzip")
    meta = {
        "target_n": int(target_n),
        "actual_n": int(len(sample)),
        "whole_prompt_groups": True,
        "unique_prompt_ids": int(sample.prompt_id.nunique()),
        "selection": (
            "Sort all complete prompt_id groups by SHA256('cw-va-v2-sample|' + "
            "prompt_id), then take whole groups until row count >= target_n."
        ),
        "selection_uses_labels": False,
        "hash_salt": "cw-va-v2-sample",
    }
    (HERE / "sample_inventory.json").write_text(json.dumps(meta, indent=2) + "\n")
    return sample


def load_sample() -> pd.DataFrame:
    if not SAMPLE.exists():
        return make_grouped_sample()
    return pd.read_csv(SAMPLE)


def truncate_story(story: str) -> str:
    story = str(story)
    if len(story) <= TRUNCATE_SOURCE_CHARS:
        return story
    return story[:TRUNCATE_HEAD_CHARS] + TRUNCATION_MARKER + story[-TRUNCATE_TAIL_CHARS:]


def rubric_block(rubrics: list[dict[str, Any]]) -> str:
    blocks = []
    for i, r in enumerate(rubrics):
        blocks.append(
            f"C{i:02d} [{r['id']}] {r['name']}\n"
            f"DESCRIPTION: {r.get('description', '')}\n"
            f"GUIDANCE: {r.get('guidance', '')}"
        )
    return "\n\n".join(blocks)


def story_block(rows: list[dict[str, str]]) -> str:
    blocks = []
    for i, row in enumerate(rows):
        blocks.append(
            f"S{i:02d}\nPROMPT: {row['prompt']}\n\n"
            f"STORY:\n{truncate_story(row['story'])}"
        )
    return "\n\n==========\n\n".join(blocks)


def make_judge_prompt(
    rows: list[dict[str, str]], rubrics: list[dict[str, Any]]
) -> str:
    return (
        SYSTEM_TEXT
        + "\n\nCRITERIA (columns, preserve this exact order):\n\n"
        + rubric_block(rubrics)
        + "\n\nANONYMOUS TEXTS (rows, preserve this exact order):\n\n"
        + story_block(rows)
        + f"\n\nReturn scores as a {len(rows)}-row by {len(rubrics)}-column matrix. "
        "Every cell must be one allowed token string. Do not omit rows or columns."
    )


def run_codex(prompt: str, timeout: int = 900) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one schema-constrained, read-only gpt-5.6-sol judge request."""
    with tempfile.TemporaryDirectory(prefix="cw_va_judge_") as tmp:
        out = Path(tmp) / "final.json"
        cmd = [
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-rules",
            "--skip-git-repo-check",
            "-C",
            tmp,
            "-s",
            "read-only",
            "-m",
            MODEL,
            "-c",
            "model_reasoning_effort=low",
            "--output-schema",
            str(SCHEMA),
            "-o",
            str(out),
            "-",
        ]
        started = time.time()
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        elapsed = time.time() - started
        meta = {
            "model": MODEL,
            "temperature": 0,
            "elapsed_seconds": round(elapsed, 3),
            "returncode": int(proc.returncode),
            "stderr_tail": proc.stderr[-1000:],
        }
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"codex judge failed rc={proc.returncode}: {proc.stderr[-3000:]}"
            )
        try:
            result = json.loads(out.read_text())
        except Exception as exc:
            raise RuntimeError(f"Invalid judge JSON: {out.read_text()[:2000]}") from exc
        return result, meta


def parse_matrix(
    result: dict[str, Any], n_rows: int, n_cols: int
) -> np.ndarray:
    raw = result.get("scores")
    if not isinstance(raw, list) or len(raw) != n_rows:
        raise ValueError(f"Expected {n_rows} score rows, got {type(raw)} / {len(raw or [])}")
    matrix = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for i, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != n_cols:
            raise ValueError(f"Row {i}: expected {n_cols}, got {len(row or [])}")
        for j, tok in enumerate(row):
            if tok not in ALLOWED:
                raise ValueError(f"Cell {i},{j}: invalid token {tok!r}")
            matrix[i, j] = ALLOWED[tok]
    return matrix


def judge_rows(
    rows: list[dict[str, str]], rubrics: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    result, meta = run_codex(make_judge_prompt(rows, rubrics))
    return parse_matrix(result, len(rows), len(rubrics)), meta


def run_fidelity_pilot(rubric_path: Path, n_stories: int = 16) -> dict[str, Any]:
    """Two order-perturbed passes over anonymous, label-free stories."""
    rubrics = _jsonl(rubric_path)
    df = load_canonical().copy()
    df["prompt"] = df.text.map(lambda x: split_text(x)[0])
    df["story"] = df.text.map(lambda x: split_text(x)[1])
    df["blind_key"] = [
        _sha(f"cw-va-v2-fidelity|{pid}|{story}", 64)
        for pid, story in zip(df.prompt_id, df.story)
    ]
    pilot = df.sort_values("blind_key").head(n_stories)
    rows = [
        {"row_id": f"p{i:03d}", "prompt": r.prompt, "story": r.story}
        for i, r in enumerate(pilot.itertuples())
    ]
    # Pass 1: natural order. Pass 2: deterministic reverse/rotation of both axes.
    m1, meta1 = judge_rows(rows, rubrics)
    row_perm = list(range(n_stories))
    random.Random(SEED + 1).shuffle(row_perm)
    col_perm = list(range(len(rubrics)))
    random.Random(SEED + 2).shuffle(col_perm)
    rows2 = [rows[i] for i in row_perm]
    rubrics2 = [rubrics[j] for j in col_perm]
    m2_perm, meta2 = judge_rows(rows2, rubrics2)
    m2 = np.full_like(m1, np.nan)
    for i2, i1 in enumerate(row_perm):
        for j2, j1 in enumerate(col_perm):
            m2[i1, j1] = m2_perm[i2, j2]

    criterion = []
    for j, rubric in enumerate(rubrics):
        c1, c2 = m1[:, j], m2[:, j]
        same = (np.isnan(c1) & np.isnan(c2)) | (c1 == c2)
        combined = np.concatenate([c1, c2])
        nonna = combined[np.isfinite(combined)]
        values, counts = np.unique(nonna, return_counts=True) if len(nonna) else ([], [])
        offmodal = int(len(nonna) - max(counts)) if len(counts) else 0
        criterion.append(
            {
                "id": rubric["id"],
                "name": rubric["name"],
                "exact_retest": float(np.mean(same)),
                "na_rate": float(np.mean(np.isnan(combined))),
                "off_modal_n": offmodal,
                "mean_non_na": float(np.mean(nonna)) if len(nonna) else None,
                "rewrite_flag": bool(
                    np.mean(same) < 0.80
                    or np.mean(np.isnan(combined)) > 0.75
                    or offmodal < 2
                ),
            }
        )
    out = {
        "label_blind": True,
        "labels_loaded_into_judge_prompt": False,
        "selection_uses_labels": False,
        "n_anonymous_stories": n_stories,
        "n_criteria": len(rubrics),
        "story_selection": "lowest SHA256(cw-va-v2-fidelity|prompt_id|story)",
        "passes": 2,
        "second_pass_row_and_criterion_order_permuted": True,
        "model": MODEL,
        "temperature": 0,
        "truncation": {
            "source_chars": TRUNCATE_SOURCE_CHARS,
            "head_chars": TRUNCATE_HEAD_CHARS,
            "tail_chars": TRUNCATE_TAIL_CHARS,
        },
        "bank_sha256": hashlib.sha256(rubric_path.read_bytes()).hexdigest(),
        "request_meta": [meta1, meta2],
        "criteria": criterion,
        "summary": {
            "mean_exact_retest": float(np.mean([x["exact_retest"] for x in criterion])),
            "mean_na_rate": float(np.mean([x["na_rate"] for x in criterion])),
            "rewrite_flag_ids": [x["id"] for x in criterion if x["rewrite_flag"]],
        },
    }
    PILOT_OUT.write_text(json.dumps(out, indent=2) + "\n")
    return out


def make_anchors(df: pd.DataFrame) -> list[dict[str, str]]:
    """Three fixed blinded anchors. Criteria are already frozen before this runs."""
    positives = df[df.judgement == 1].copy()
    negatives = df[df.judgement == 0].copy()
    positives["akey"] = [
        _sha(f"anchor-high|{pid}|{story}", 64)
        for pid, story in zip(positives.prompt_id, positives.story)
    ]
    # High-score class means source Reddit score >=10; raw scores were not retained.
    high = positives.sort_values("akey").iloc[0]
    rng = np.random.default_rng(SEED)
    low = negatives.iloc[int(rng.integers(0, len(negatives)))]
    tokens = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[.,!?;:]", str(high.story))
    rng_scramble = random.Random(SEED + 99)
    rng_scramble.shuffle(tokens)
    scrambled = " ".join(tokens)
    return [
        {
            "row_id": "__anchor_high",
            "role": "high",
            "prompt": str(high.prompt),
            "story": str(high.story),
        },
        {
            "row_id": "__anchor_random_low",
            "role": "random_low",
            "prompt": str(low.prompt),
            "story": str(low.story),
        },
        {
            "row_id": "__anchor_scrambled",
            "role": "scrambled",
            "prompt": str(high.prompt),
            "story": scrambled,
        },
    ]


def _batch_payload(
    batch_index: int,
    docs: list[dict[str, str]],
    anchors: list[dict[str, str]],
    rubrics: list[dict[str, Any]],
    max_attempts: int,
) -> dict[str, Any]:
    logs = []
    for attempt in range(1, max_attempts + 1):
        combined = [dict(x) for x in docs + anchors]
        random.Random(SEED + 1009 * batch_index + 7919 * attempt).shuffle(combined)
        try:
            matrix, meta = judge_rows(combined, rubrics)
            index = {row["row_id"]: i for i, row in enumerate(combined)}
            anchor_means = {
                a["role"]: float(np.nanmean(matrix[index[a["row_id"]]]))
                for a in anchors
            }
            valid = (
                anchor_means["high"]
                > anchor_means["random_low"]
                > anchor_means["scrambled"]
            )
            logs.append(
                {
                    "attempt": attempt,
                    "parse_valid": True,
                    "anchor_means": anchor_means,
                    "anchor_order_valid": valid,
                    **meta,
                }
            )
            if not valid:
                continue
            doc_scores = {
                row["row_id"]: [
                    None if not np.isfinite(x) else float(x)
                    for x in matrix[index[row["row_id"]]]
                ]
                for row in docs
            }
            return {
                "batch_index": batch_index,
                "n_documents": len(docs),
                "scores": doc_scores,
                "anchor_means": anchor_means,
                "attempts": logs,
                "accepted_attempt": attempt,
            }
        except Exception as exc:
            logs.append(
                {
                    "attempt": attempt,
                    "parse_valid": False,
                    "error": repr(exc),
                }
            )
    raise RuntimeError(
        f"Batch {batch_index} invalid after {max_attempts} attempts: "
        f"{json.dumps(logs, indent=2)[-5000:]}"
    )


def score_full_bank(
    rubric_path: Path,
    batch_size: int = 12,
    workers: int = 4,
    max_attempts: int = 3,
) -> dict[str, Any]:
    rubrics = _jsonl(rubric_path)
    sample = load_sample()
    # The only fields passed into prompts are anonymous id, prompt, and story.
    docs = [
        {
            "row_id": str(r.doc_id),
            "prompt": str(r.prompt),
            "story": str(r.story),
        }
        for r in sample.itertuples()
    ]
    anchors = make_anchors(sample)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]

    pending = []
    for i, batch in enumerate(batches):
        path = BATCH_DIR / f"batch_{i:04d}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if (
                cached.get("n_documents") == len(batch)
                and set(cached.get("scores", {})) == {r["row_id"] for r in batch}
            ):
                continue
        pending.append((i, batch))
    print(
        f"[score] documents={len(docs)} criteria={len(rubrics)} "
        f"batches={len(batches)} pending={len(pending)} workers={workers}",
        flush=True,
    )

    def work(item: tuple[int, list[dict[str, str]]]) -> tuple[int, dict[str, Any]]:
        i, batch = item
        return i, _batch_payload(i, batch, anchors, rubrics, max_attempts)

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, item): item[0] for item in pending}
            for future in as_completed(futures):
                i, payload = future.result()
                path = BATCH_DIR / f"batch_{i:04d}.json"
                path.write_text(json.dumps(payload, indent=2) + "\n")
                print(
                    f"[score] accepted batch {i + 1}/{len(batches)} "
                    f"attempt={payload['accepted_attempt']} anchors={payload['anchor_means']}",
                    flush=True,
                )

    payloads = [
        json.loads((BATCH_DIR / f"batch_{i:04d}.json").read_text())
        for i in range(len(batches))
    ]
    score_by_id = {}
    for payload in payloads:
        score_by_id.update(payload["scores"])
    missing = set(sample.doc_id.astype(str)) - set(score_by_id)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} document scores")

    A = np.array(
        [
            [np.nan if x is None else float(x) for x in score_by_id[str(doc_id)]]
            for doc_id in sample.doc_id
        ],
        dtype=np.float32,
    )
    V = np.array([feature_vector(str(story)) for story in sample.story], dtype=np.float64)
    np.savez_compressed(
        SCORES,
        A=A,
        V=V,
        y=sample.judgement.to_numpy(dtype=np.int8),
        groups=sample.prompt_id.astype(str).to_numpy(),
        doc_id=sample.doc_id.astype(str).to_numpy(),
        a_names=np.array([r["id"] for r in rubrics]),
        a_titles=np.array([r["name"] for r in rubrics]),
        v_names=np.array(V_NAMES),
    )
    summary = {
        "n_documents": int(len(sample)),
        "n_criteria": len(rubrics),
        "n_v_features": len(V_NAMES),
        "n_batches": len(payloads),
        "invalid_attempts_rescored": int(
            sum(p["accepted_attempt"] - 1 for p in payloads)
        ),
        "all_accepted_batches_anchor_valid": all(
            p["anchor_means"]["high"]
            > p["anchor_means"]["random_low"]
            > p["anchor_means"]["scrambled"]
            for p in payloads
        ),
        "anchor_means_across_batches": {
            role: float(np.mean([p["anchor_means"][role] for p in payloads]))
            for role in ("high", "random_low", "scrambled")
        },
        "bank_sha256": hashlib.sha256(rubric_path.read_bytes()).hexdigest(),
        "model": MODEL,
        "temperature": 0,
        "allowed_tokens": ["1.0", "0.5", "0.0", "NA"],
        "batch_size_non_anchor": batch_size,
        "anchors_per_batch": 3,
    }
    (HERE / "scoring_inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _usable_columns(matrix: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], list[int]]:
    keep = []
    for j in range(matrix.shape[1]):
        finite = matrix[:, j][np.isfinite(matrix[:, j])]
        if len(finite) >= 10 and np.unique(finite).size >= 2:
            keep.append(j)
    return matrix[:, keep], [names[j] for j in keep], keep


def _cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, list[float]]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.full(len(y), np.nan, dtype=float)
    fold_aucs = []
    for train, test in splitter.split(X, y, groups):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
        )
        model.fit(X[train], y[train])
        pred[test] = model.predict_proba(X[test])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[test], pred[test])))
    return float(roc_auc_score(y, pred)), fold_aucs


def run_readout() -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    if not SCORES.exists():
        raise FileNotFoundError(SCORES)
    d = np.load(SCORES, allow_pickle=True)
    A = d["A"].astype(float)
    V = d["V"].astype(float)
    y = d["y"].astype(int)
    groups = d["groups"].astype(str)
    a_ids = [str(x) for x in d["a_names"]]
    a_titles = [str(x) for x in d["a_titles"]]
    v_names = [str(x) for x in d["v_names"]]

    Ac, a_used, a_keep = _usable_columns(A, a_ids)
    Vc, v_used, _ = _usable_columns(V, v_names)
    VA = np.column_stack([Vc, Ac])
    readouts = {}
    for name, matrix in (("V", Vc), ("A", Ac), ("V+A", VA)):
        auc, folds = _cv_auc(matrix, y, groups)
        readouts[name] = {
            "auc": auc,
            "fold_aucs": folds,
            "n_features": int(matrix.shape[1]),
        }

    univariate = []
    for j, (aid, title) in enumerate(zip(a_ids, a_titles)):
        col = A[:, j].copy()
        finite = col[np.isfinite(col)]
        if len(finite) == 0 or np.unique(finite).size < 2:
            auc = None
        else:
            col[~np.isfinite(col)] = float(np.median(finite))
            auc = float(roc_auc_score(y, col))
        univariate.append(
            {
                "id": aid,
                "name": title,
                "auc": auc,
                "na_rate": float(np.mean(~np.isfinite(A[:, j]))),
            }
        )
    univariate.sort(key=lambda x: -math.inf if x["auc"] is None else x["auc"], reverse=True)
    for rank, item in enumerate(univariate, 1):
        item["rank"] = rank

    reconstruction = json.loads((HERE / "reconstruction_inventory.json").read_text())
    sampling = json.loads((HERE / "sample_inventory.json").read_text())
    scoring = json.loads((HERE / "scoring_inventory.json").read_text())
    fidelity = json.loads(PILOT_OUT.read_text()) if PILOT_OUT.exists() else None
    result = {
        "pool": {
            "name": "WritingPrompts modeling clean grouped build",
            "source_file": "datasets/creative-writing/writingprompts_modeling_clean.csv.gz",
            "local_materialization": (
                "Exact canonical reconstruction from "
                "writingprompts_modeling_with_topics.csv.gz because the clean gzip was absent locally"
            ),
            "label": "judgement = 1 iff source Reddit story score >= 10",
            "balance": "50/50 after within-story-length-bucket balancing",
            "group": "prompt_id (16-character MD5 of normalized prompt text)",
            "excluded_pool": (
                "litbench-to-train.csv.gz was not used for scoring, sampling, fitting, or AUC"
            ),
        },
        "reconstruction": reconstruction,
        "sample": sampling,
        "inventory": {
            "n": int(len(y)),
            "positive": int(y.sum()),
            "negative": int((y == 0).sum()),
            "unique_prompt_groups": int(len(np.unique(groups))),
        },
        "truncation": {
            "A_judge": (
                "Prompt in full. Story in full when <=6000 characters; otherwise first "
                "3600 + literal omission marker + last 2400 source characters."
            ),
            "source_story_chars_retained": TRUNCATE_SOURCE_CHARS,
            "head_chars": TRUNCATE_HEAD_CHARS,
            "tail_chars": TRUNCATE_TAIL_CHARS,
            "V_features": "full story body, prompt excluded",
        },
        "scoring": scoring,
        "fidelity_optimization": fidelity,
        "readout": {
            "protocol": (
                "5-fold StratifiedGroupKFold(shuffle=True, random_state=20260728), "
                "group=prompt_id; fold-local median imputation, StandardScaler, "
                "LogisticRegression(C=1.0, max_iter=5000); reported AUC is pooled OOF ROC AUC"
            ),
            "V": readouts["V"],
            "A": readouts["A"],
            "V+A": readouts["V+A"],
            "a_features_used": a_used,
            "v_features_used": v_used,
        },
        "per_criterion_univariate_auc": univariate,
        "caveats": [
            (
                "The local canonical clean gzip was absent. The supplied canonical rebuild "
                "script was applied to its local topic-annotated immediate precursor and "
                "reproduced the documented 96,080-row, 48,040/48,040 inventory exactly."
            ),
            (
                "The clean build retains only the binary score>=10 class, not raw Reddit "
                "scores. The mandatory high anchor is therefore a deterministic member of "
                "the high-score class, not the maximum-score story."
            ),
            (
                "A was judged on a deterministic 6,000-source-character head+tail view; "
                "middle-dependent evidence may be unavailable."
            ),
            (
                "The 2,000+ row subsample contains whole prompt groups selected by a "
                "label-blind hash; it is not the full 96,080-row pool."
            ),
            (
                "These numbers belong only to the WritingPrompts clean grouped pool and "
                "must not be merged with the legacy LitBench-derived V/A block."
            ),
        ],
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n")
    write_results_md(result)
    return result


def write_results_md(result: dict[str, Any]) -> None:
    inv = result["inventory"]
    scoring = result["scoring"]
    lines = [
        "# Creative-writing V/A bank v2",
        "",
        "## Pool and inventory",
        "",
        (
            "**Pool for every number in this report:** `writingprompts_modeling_clean` "
            "(direct r/WritingPrompts build; `judgement = score >= 10`; length-bucket "
            "balanced; group = `prompt_id`). The legacy LitBench-derived pool was not used."
        ),
        "",
        (
            f"n = {inv['n']:,}; positive = {inv['positive']:,}; "
            f"negative = {inv['negative']:,}; prompt groups = "
            f"{inv['unique_prompt_groups']:,}."
        ),
        "",
        "## Prompt-grouped CV readout",
        "",
        "| Bank | OOF ROC AUC | Features |",
        "|---|---:|---:|",
    ]
    for key in ("V", "A", "V+A"):
        row = result["readout"][key]
        lines.append(f"| {key} | {row['auc']:.3f} | {row['n_features']} |")
    lines += [
        "",
        (
            "Readout: 5-fold stratified prompt-group CV, fold-local median imputation, "
            "standardization, and logistic regression (C=1)."
        ),
        "",
        "## Anchor check",
        "",
        "| Anchor | Mean A across accepted batches |",
        "|---|---:|",
    ]
    labels = {"high": "High-score-class story", "random_low": "Random low-score story", "scrambled": "Scrambled/degraded text"}
    for role in ("high", "random_low", "scrambled"):
        lines.append(
            f"| {labels[role]} | "
            f"{scoring['anchor_means_across_batches'][role]:.3f} |"
        )
    lines += [
        "",
        (
            f"Ordering high > random-low > scrambled: "
            f"**{'PASS' if scoring['all_accepted_batches_anchor_valid'] else 'FAIL'}** "
            f"for all {scoring['n_batches']} accepted batches. Invalid attempts rescored: "
            f"{scoring['invalid_attempts_rescored']}."
        ),
        "",
        "## Per-criterion univariate AUCs",
        "",
        "| Rank | Criterion | ROC AUC | NA rate |",
        "|---:|---|---:|---:|",
    ]
    for item in result["per_criterion_univariate_auc"]:
        auc = "NA" if item["auc"] is None else f"{item['auc']:.3f}"
        lines.append(
            f"| {item['rank']} | {item['id']}: {item['name']} | "
            f"{auc} | {item['na_rate']:.3f} |"
        )
    lines += [
        "",
        "## Truncation and fidelity",
        "",
        result["truncation"]["A_judge"],
        "",
    ]
    fidelity = result.get("fidelity_optimization")
    if fidelity:
        lines.append(
            f"Label-blind fidelity pilot: {fidelity['n_anonymous_stories']} anonymous "
            f"stories, two order-perturbed passes; mean exact retest = "
            f"{fidelity['summary']['mean_exact_retest']:.3f}, mean NA rate = "
            f"{fidelity['summary']['mean_na_rate']:.3f}."
        )
        lines.append("")
    lines += ["## Caveats", ""]
    for caveat in result["caveats"]:
        lines.append(f"- {caveat}")
        lines.append("")
    RESULTS_MD.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reconstruct")
    p_pilot = sub.add_parser("pilot")
    p_pilot.add_argument("--rubrics", type=Path, default=HERE / "rubrics_initial.jsonl")
    p_pilot.add_argument("--n-stories", type=int, default=16)
    p_score = sub.add_parser("score")
    p_score.add_argument("--rubrics", type=Path, default=HERE / "rubrics.jsonl")
    p_score.add_argument("--batch-size", type=int, default=12)
    p_score.add_argument("--workers", type=int, default=4)
    p_score.add_argument("--max-attempts", type=int, default=3)
    sub.add_parser("readout")
    args = parser.parse_args()

    if args.command == "reconstruct":
        inventory = reconstruct_canonical()
        sample = make_grouped_sample()
        print(json.dumps({**inventory, "sample_rows": len(sample)}, indent=2))
    elif args.command == "pilot":
        print(json.dumps(run_fidelity_pilot(args.rubrics, args.n_stories), indent=2))
    elif args.command == "score":
        print(
            json.dumps(
                score_full_bank(
                    args.rubrics, args.batch_size, args.workers, args.max_attempts
                ),
                indent=2,
            )
        )
    elif args.command == "readout":
        result = run_readout()
        print(json.dumps(result["readout"], indent=2))


if __name__ == "__main__":
    main()
