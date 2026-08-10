"""U_flux from the union ledger — capture-recapture on the VALUE spectrum (MCC §3 Wrap 2,
PO §12.6.3 D1-D3).

Every gated proposal (kept or dropped) is one DRAW from the generator process; the semantic
quotient clusters draws into SPECIES; a species' value is its measured conditional gain
(clipped bits_gain — the supervised v(s|S_g), already in the ledger for every proposal).
The estimators live in metric_implementer.experiments.value_certificate (single source of
truth for D1-D3 + T1 soundness); this module only builds their inputs from run artifacts.

Scope: U_flux is PROCESS-RELATIVE — it bounds what THESE generator arms would articulate by
horizon (1+c)N, not the unconditional articulable ceiling. |G| >= 2 arms is mandatory for a
non-anti-conservative read (MCC §9.1); the merge-precision audit at a second tau and the
singleton fraction f1/N (degeneracy lemma) ship with every read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from metric_implementer.experiments.value_certificate import (
    anytime_delta, flux_certificate, good_toulmin_value, value_spectrum)


def _default_embed(texts: Sequence[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name).encode(list(texts), normalize_embeddings=True)


def species_from_proposals(
    texts: Sequence[str], tau: float = 0.92,
    model_name: str = "all-MiniLM-L6-v2",
    embed_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
) -> np.ndarray:
    """Quotient the draws: average-linkage agglomerative clustering at cosine >= tau (the
    locked rubric-clustering recipe's blend threshold). Returns integer species labels."""
    if len(texts) == 0:
        return np.array([], int)
    if len(texts) == 1:
        return np.array([0])
    emb = embed_fn(texts) if embed_fn else _default_embed(texts, model_name)
    emb = np.asarray(emb, float)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    from sklearn.cluster import AgglomerativeClustering
    cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                 distance_threshold=1.0 - tau)
    return cl.fit_predict(emb)


def _binary_entropy(q: float) -> float:
    q = min(max(float(q), 1e-9), 1 - 1e-9)
    return float(-(q * math.log2(q) + (1 - q) * math.log2(1 - q)))


def flux_from_ledgers(
    ledger_paths: List[str | Path], base_rate: float,
    c: float = 1.0, delta: float = 0.05, tau: float = 0.92, audit_tau: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
    embed_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
) -> dict:
    """Assemble the D1-D3 flux read from union-ledger artifacts.

    Returns a dict with ``flux_tail_bits`` = Ĝ(c) + McDiarmid slack (feeds
    ``certificates.flux_wrap``; the 1/gamma inflation happens there), plus the honesty
    companions: singleton fraction, merge-precision audit, spectrum, B cap provenance.

    - value: v(species) = max over its draws of max(0, bits_gain); nan (viability-dropped
      before measurement) counts as 0 — a draw with no measured gain carries no value.
    - B cap (T1 soundness): max(empirical max, H(Y) - V(S_g)) — an UNSEEN species can carry
      up to the head's residual entropy; the empirical max alone is an invalid McDiarmid
      constant.
    - delta: spent through ``anytime_delta`` (doubling-checkpoint union) so the read stays
      valid if the run is continued and re-issued at a larger N.
    """
    draws, floors = [], []
    for p in ledger_paths:
        with open(p) as f:
            data = json.load(f)
        traj = data.get("guard_bits_trajectory") or [float("nan")]
        floors.append(float(traj[-1]))
        for l in data.get("ledgers", []):
            g = l.get("bits_gain")
            val = float(g) if isinstance(g, (int, float)) and np.isfinite(g) else 0.0
            draws.append({"text": f"{l.get('name', '')}: {l.get('rubric', '')}",
                          "value": max(0.0, val), "generator": l.get("generator", "?")})
    N = len(draws)
    floor_bits = float(np.nanmax(floors)) if floors else 0.0
    if N == 0:
        return {"n_draws": 0, "flux_tail_bits": None,
                "note": "no proposals in ledgers — flux read impossible"}

    texts = [d["text"] for d in draws]
    labels = species_from_proposals(texts, tau, model_name, embed_fn)
    labels_audit = species_from_proposals(texts, audit_tau, model_name, embed_fn)
    n_species = int(labels.max()) + 1
    n_s = np.array([(labels == s).sum() for s in range(n_species)])
    v_s = np.array([max(d["value"] for i, d in enumerate(draws) if labels[i] == s)
                    for s in range(n_species)])

    w = value_spectrum(n_s, v_s)
    B_emp = float(v_s.max()) if len(v_s) else 0.0
    B_cap = max(B_emp, _binary_entropy(base_rate) - max(0.0, floor_bits))
    ad = anytime_delta(N, delta=delta, n_stats=1)
    fc = flux_certificate(w, N, B_cap, delta=ad["delta_eff"])
    g_c = good_toulmin_value(w, c=c, k0=4)
    f1 = int((n_s == 1).sum())
    arms = sorted({d["generator"] for d in draws})

    return {
        "n_draws": N, "n_arms": len(arms), "arms": arms,
        "n_species": n_species, "n_species_audit_tau": int(labels_audit.max()) + 1,
        "tau": tau, "audit_tau": audit_tau,
        "f1": f1, "f1_over_species": f1 / max(1, n_species),
        "singleton_regime": f1 / max(1, n_species) > 0.9,
        "value_spectrum": {str(k): v for k, v in sorted(w.items())},
        "floor_bits": floor_bits, "base_rate": float(base_rate),
        "B_emp": B_emp, "B_cap": B_cap, "delta": delta, "delta_eff": ad["delta_eff"],
        "c": float(c), "good_toulmin_Gc": float(g_c),
        "flux_per_draw": fc["flux"], "mcdiarmid_slack": fc["slack"],
        "flux_tail_bits": float(max(0.0, g_c) + fc["slack"]),
    }
