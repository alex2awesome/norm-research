"""Shared utilities for the part-3 installation-channel experiments.

CPU-only: nothing here imports vLLM/torch. Score-grid conventions mirror the frozen scorer's
npz output (scores matrix rows <-> json meta rows; row vectors over the domain's items).
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np

# Default family layout under notebooks/data/two_faces_20260702 (interim cross-family lanes).
FAMILIES = {
    "llama31": ("family_scores_llama31", "llama31_70b_name_target"),
    "qwen25": ("family_scores_qwen25", "qwen25_72b_name_target"),
    "gemma3": ("family_scores_gemma3", "gemma3_27b_name_target"),
}
DOMAINS = ("notice-and-comment", "humor", "math-stackexchange")

# Gap-conditioned rescue definition (interim tallies; matches the roadmap-note convention).
RHO_FLOOR = 0.70
GAP_FLOOR = 0.10


def parse_bank_cells(bank_path: str) -> dict:
    """Load an arm bank; return {cell_id: cell} with the stringified `arms` field parsed.

    Bank cells were serialized through pandas, so nested fields arrive as python-repr strings
    that may contain `nan` — ast.literal_eval alone rejects them.
    """
    bank = json.loads(open(bank_path).read())
    cells = {}
    for cell in bank["cells"]:
        arms = cell["arms"]
        if isinstance(arms, str):
            arms = eval(  # noqa: S307 - trusted repo artifact; literal_eval rejects nan
                arms, {"__builtins__": {}},
                {"nan": float("nan"), "None": None, "True": True, "False": False})
        parsed = dict(cell)
        parsed["arms"] = arms
        cells[cell["id"]] = parsed
    return cells


def arm_form_prompt(cell: dict, arm_id: str, form_id: str) -> str | None:
    for arm in cell["arms"]:
        if arm["id"] == arm_id:
            for form in arm["forms"]:
                if form["id"] == form_id:
                    return form["prompt"]
    return None


def load_grid(root: str, job: str, domain: str):
    """Load every rep of a job's grid npz for one domain.

    Returns ({(cell_id, arm_id, form): mean-over-reps score vector}, {key: meta-row}).
    np.asarray materializes each compressed array ONCE (NpzFile re-decompresses per indexed
    access otherwise - the OOM landmine).
    """
    vecs, meta = defaultdict(list), {}
    for path in sorted(glob.glob(os.path.join(root, job, f"grid_{domain}_*_rep*.npz"))):
        d = np.load(path, allow_pickle=True)
        scores = np.asarray(d["scores"])
        for i, s in enumerate(d["meta"]):
            m = json.loads(s)
            k = (m["cell_id"], m["arm_id"], m["form"])
            vecs[k].append(scores[i])
            meta[k] = m
    return {k: np.mean(v, axis=0) for k, v in vecs.items()}, meta


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average ties
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) != len(values):
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def spearman(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def cell_stats(tgt: dict, exe: dict, emeta: dict, cell: str) -> dict | None:
    """Adverse-rho stats for one cell: target-name reference, executor name/arms/controls."""
    t_forms = [v for (c, a, f), v in tgt.items() if c == cell and a == "name"]
    if not t_forms:
        return None
    t = np.mean(t_forms, axis=0)

    def adverse(src, arm):
        forms = [v for (c, a, f), v in src.items() if c == cell and a == arm]
        if not forms:
            return None
        return min(spearman(v, t) for v in forms)

    small = adverse(exe, "name")
    big = adverse(tgt, "name")
    substantive = {a for (c, a, f) in exe
                   if c == cell and a != "name"
                   and emeta[(c, a, f)].get("control_for") is None}
    best_arm, best = None, None
    for a in substantive:
        s = adverse(exe, a)
        if s is not None and not math.isnan(s) and (best is None or s > best):
            best_arm, best = a, s
    controls = {a for (c, a, f) in exe
                if c == cell and emeta[(c, a, f)].get("control_for") == best_arm}
    ctrl_scores = [adverse(exe, a) for a in controls]
    beats = all(s is None or math.isnan(s) or s < best for s in ctrl_scores) \
        if best is not None else None
    return {"cell_id": cell, "target_name_rho": big, "exec_name_rho": small,
            "best_arm": best_arm, "best_rho": best,
            "gain": (best - small) if (best is not None and small is not None) else None,
            "gap": (big - small) if (big is not None and small is not None) else None,
            "beats_controls": beats}


def is_conditioned_rescue(row: dict, rho_floor: float = RHO_FLOOR,
                          gap_floor: float = GAP_FLOOR) -> bool:
    """best>=floor AND gain>0 AND beats matched controls AND native gap>gap_floor."""
    return bool(
        row.get("best_rho") is not None and row["best_rho"] >= rho_floor
        and row.get("gain") is not None and row["gain"] > 0
        and row.get("beats_controls") in (True, None)
        and row.get("gap") is not None and row["gap"] > gap_floor
    )


def stable_split(key: str, train_fraction: float = 0.8, salt: str = "tacit_channels_v1") -> str:
    """Deterministic train/eval assignment by content hash — never a seeded shuffle of a
    growing list (feedback_stable_hash_splits)."""
    h = hashlib.sha256(f"{salt}::{key}".encode()).hexdigest()
    return "train" if (int(h[:8], 16) / 0xFFFFFFFF) < train_fraction else "eval"


def write_jsonl(path: str, rows) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]
