"""Artifact context + profile store for the Tacitness Battery.

ArtifactContext exposes lazily-loaded, memoized views over everything that already exists on
disk (target/executor grids, adapter grids, caps, items, embeddings). Probes read ONLY through
the context, so the same expensive loads serve every probe in a run.

Profile store: Hive-partitioned long-format Parquet (cells_v1 pattern,
scripts/build_cells_db.py precedent) with csv.gz fallback; append+dedup, never delete.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from methods.tacit_channels import _apparatus
from methods.tacit_channels.channels.common import (
    FAMILIES, load_grid, spearman, stable_split, _rankdata,
)

PROFILE_ROOT = "outputs/tacit_channels/tacit_profile_v1"
BASE = "notebooks/data/two_faces_20260702"
PACKET_ROOT = f"{BASE}/tacit_breadth_item_partitions_v2"


@dataclass
class ArtifactContext:
    family: str = "qwen25"
    domain: str = "humor"
    exec_jobs: tuple = ()            # empty = all *_executor dirs present
    adapter_root: str = "outputs/tacit_channels/exp_gtk1/scores"  # per-intervention grids
    # test/instrument overrides
    scores_root_override: str | None = None
    target_job_override: str | None = None
    items_override: dict | None = None
    typicality_override: object = None
    _cache: dict = field(default_factory=dict)

    # ---- basic layout -------------------------------------------------------------
    @property
    def scores_root(self) -> str:
        return self.scores_root_override or os.path.join(BASE, FAMILIES[self.family][0])

    @property
    def target_job(self) -> str:
        return self.target_job_override or FAMILIES[self.family][1]

    def executors(self) -> list[str]:
        if self.exec_jobs:
            return list(self.exec_jobs)
        return sorted(j for j in os.listdir(self.scores_root)
                      if j.endswith("_executor")
                      and glob.glob(os.path.join(self.scores_root, j,
                                                 f"grid_{self.domain}_*.npz")))

    # ---- memoized loads -----------------------------------------------------------
    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def grid(self, job: str):
        """(vecs {(cell,arm,form): mean-over-reps vector}, meta) for one job."""
        return self._memo(("grid", job),
                          lambda: load_grid(self.scores_root, job, self.domain))

    def grid_raw_reps(self, job: str):
        """{(cell,arm,form): [vector per rep]} — replicates NOT averaged (ensemble probes)."""
        def load():
            from collections import defaultdict
            vecs = defaultdict(list)
            pattern = os.path.join(self.scores_root, job, f"grid_{self.domain}_*_rep*.npz")
            for path in sorted(glob.glob(pattern)):
                d = np.load(path, allow_pickle=True)
                scores = np.asarray(d["scores"])
                for i, s in enumerate(d["meta"]):
                    m = json.loads(s)
                    vecs[(m["cell_id"], m["arm_id"], m["form"])].append(scores[i])
            return dict(vecs)
        return self._memo(("raw", job), load)

    def adapter_grid(self, tag: str):
        """Grids produced by score_with_adapter under adapter_root/<tag>/."""
        return self._memo(("adapter", tag),
                          lambda: load_grid(self.adapter_root, tag, self.domain))

    def target_name_vector(self, cell: str):
        tgt, _ = self.grid(self.target_job)
        forms = [v for (c, a, f), v in tgt.items() if c == cell and a == "name"]
        return np.mean(forms, axis=0) if forms else None

    def cells(self) -> list[str]:
        tgt, _ = self.grid(self.target_job)
        return sorted({c for (c, a, f) in tgt})

    # ---- items + embeddings -------------------------------------------------------
    def items(self):
        if self.items_override is not None:
            return self.items_override

        def load():
            payload = _apparatus.load_domain_items(
                PACKET_ROOT, self.domain, partitions=["tacit_breadth_search"])
            texts = payload["texts"]
            hashes = payload.get("hashes") or [
                hashlib.sha256(t.encode()).hexdigest() for t in texts]
            return {"texts": list(texts), "hashes": list(hashes)}
        return self._memo(("items",), load)

    def item_half2_mask(self, salt: str = "exp_gtk1") -> np.ndarray:
        hashes = self.items()["hashes"]
        return np.array([stable_split(h, 0.5, salt=salt) != "train" for h in hashes])

    def item_typicality(self) -> np.ndarray:
        """Centroid-cosine typicality per item; embedding if available, lexical fallback."""
        if self.typicality_override is not None:
            return np.asarray(self.typicality_override)

        def compute():
            texts = self.items()["texts"]
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-MiniLM-L6-v2")
                emb = np.asarray(model.encode(texts, batch_size=256,
                                              normalize_embeddings=True))
                centroid = emb.mean(axis=0)
                centroid /= np.linalg.norm(centroid) or 1.0
                return emb @ centroid
            except Exception:
                # lexical fallback: token-Jaccard to pooled vocabulary frequency profile
                import re
                token_sets = [set(re.findall(r"[a-z']+", t.lower())) for t in texts]
                from collections import Counter
                df = Counter(tok for s in token_sets for tok in s)
                common = {t for t, c in df.items() if c >= len(texts) * 0.05}
                return np.array([len(s & common) / (len(s | common) or 1)
                                 for s in token_sets])
        return self._memo(("typicality",), compute)


# ---- shared statistics helpers ----------------------------------------------------
def item_agreement(vec: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-item rank agreement in [0,1]: 1 - |rank residual| / (n-1)."""
    n = len(target)
    r_v, r_t = _rankdata(np.asarray(vec, float)), _rankdata(np.asarray(target, float))
    return 1.0 - np.abs(r_v - r_t) / max(n - 1, 1)


def adverse_rho(vecs: dict, cell: str, arm: str, target: np.ndarray,
                mask: np.ndarray | None = None) -> float | None:
    forms = [v for (c, a, f), v in vecs.items() if c == cell and a == arm]
    if not forms:
        return None
    if mask is not None:
        return min(spearman(v[mask], target[mask]) for v in forms)
    return min(spearman(v, target) for v in forms)


# ---- profile store ------------------------------------------------------------------
def write_profile_rows(rows: list[dict], run_tag: str, domain: str | None = None) -> str:
    """domain joins the filename so per-domain runs never clobber each other.
    NOTE (audit 2026-07-23): each (domain, run_tag) is one immutable file; cross-run dedup is
    NOT implemented — new tags create new files; consumers select by tag."""
    if domain:
        run_tag = f"{domain}_{run_tag}"
    return _write_profile_rows_impl(rows, run_tag)


def _write_profile_rows_impl(rows: list[dict], run_tag: str) -> str:
    """Append long-format stat rows, partitioned by construct-domain; parquet w/ csv.gz
    fallback (cells_v1 pattern). Dedup key = (construct, rung, domain, probe, statistic,
    run_tag): re-runs with the same tag replace, others append."""
    os.makedirs(PROFILE_ROOT, exist_ok=True)
    out_base = os.path.join(PROFILE_ROOT, f"battery_{run_tag}")
    payload = {"schema": "tacit_profile/v1", "run_tag": run_tag, "n_rows": len(rows)}
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out_base + ".parquet", compression="zstd", index=False)
        path = out_base + ".parquet"
    except Exception:
        import csv
        import gzip
        keys = sorted({k for r in rows for k in r})
        with gzip.open(out_base + ".csv.gz", "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        path = out_base + ".csv.gz"
    payload["path"] = path
    payload["sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(out_base + ".manifest.json", "w") as f:
        json.dump(payload, f, indent=2)
    return path
