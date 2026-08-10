"""Residualized contrast — the novelty guard (spec §3).

Inside a gap node, split the node's *discover* items by how well its own metric GLM predicts
them (residual magnitude), NOT by their label:

    WRONG = items the metrics get wrong (|resid| large)
    RIGHT = items the metrics already explain (|resid| small)

We then contrast WRONG vs RIGHT. Because the articulated metrics already account for RIGHT,
the only signal left in WRONG is *orthogonal* to what the metrics capture — this structurally
blocks the LLM from re-deriving an existing or shallower feature. The LLM is fed ``k``
(positive, negative) pairs sampled from within WRONG; pairing is purely a presentation device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .gaps import GapNode
from .mob.glmtree import GapTree


@dataclass
class Contrast:
    node_id: str
    wrong_pos: List[str]            # WRONG items with label 1 (texts)
    wrong_neg: List[str]            # WRONG items with label 0 (texts)
    pairs: List[tuple]             # k sampled (pos_text, neg_text) pairs for the LLM
    wrong_disc_idx: np.ndarray     # discover rows in WRONG (for diagnostics)
    right_disc_idx: np.ndarray     # discover rows in RIGHT
    n_wrong: int = 0


def build_contrast(
    tree: GapTree, gap: GapNode, df_discover, X_discover: np.ndarray, y_discover: np.ndarray,
    cfg, rng: Optional[np.random.Generator] = None,
) -> Optional[Contrast]:
    """Form the WRONG/RIGHT residualized contrast for one gap node."""
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed)
    node = gap.node
    idx = node.indices                      # discover rows at this node
    if len(idx) < 4:
        return None

    p = tree.node_predict_proba(node, X_discover[idx])
    y = y_discover[idx]
    abs_resid = np.abs(y - p)
    pos_mask = y == 1
    neg_mask = y == 0

    # The most-wrong / most-right items WITHIN EACH CLASS. Doing this per class (rather than a
    # single |resid| threshold) is essential: in a pure gap node the model predicts a near-
    # constant base rate, so a single threshold would select almost one class and starve the
    # contrast. Per class, WRONG-pos = positives the metrics score like negatives (and v.v.) —
    # exactly the items whose label the articulated metrics fail to explain.
    def pick(mask, q, top):
        sub = idx[mask]
        if len(sub) == 0:
            return sub
        r = abs_resid[mask]
        thr = np.quantile(r, q)
        keep = (r >= thr) if top else (r <= thr)
        return sub[keep]

    wrong_pos_idx = pick(pos_mask, cfg.wrong_resid_quantile, top=True)
    wrong_neg_idx = pick(neg_mask, cfg.wrong_resid_quantile, top=True)
    right_pos_idx = pick(pos_mask, cfg.right_resid_quantile, top=False)
    right_neg_idx = pick(neg_mask, cfg.right_resid_quantile, top=False)
    if len(wrong_pos_idx) < 1 or len(wrong_neg_idx) < 1:
        return None

    texts = df_discover[cfg.text_column].astype(str)

    def clip(t: str) -> str:
        return t[: cfg.contrast_max_chars]

    wrong_pos = [clip(texts.iloc[i]) for i in wrong_pos_idx]
    wrong_neg = [clip(texts.iloc[i]) for i in wrong_neg_idx]
    wrong_idx = np.concatenate([wrong_pos_idx, wrong_neg_idx])
    right_idx = np.concatenate([right_pos_idx, right_neg_idx])

    pairs = _sample_pairs(wrong_pos, wrong_neg, cfg.contrastive_pairs_k, rng)
    return Contrast(
        node_id=node.node_id, wrong_pos=wrong_pos, wrong_neg=wrong_neg, pairs=pairs,
        wrong_disc_idx=wrong_idx, right_disc_idx=right_idx, n_wrong=len(wrong_idx),
    )


def _sample_pairs(pos: List[str], neg: List[str], k: int, rng: np.random.Generator) -> List[tuple]:
    if not pos or not neg:
        return []
    out = []
    for _ in range(k):
        out.append((pos[rng.integers(len(pos))], neg[rng.integers(len(neg))]))
    return out


def pool_contrasts(contrasts: List[Contrast]) -> Contrast:
    """Union several gap-node contrasts into one population-wide contrast (spec §6).

    De-stratifies the WRONG/RIGHT sets so the LLM sees population-wide variation and returns a
    feature that splits near the root on reinsertion.
    """
    wrong_pos: List[str] = []
    wrong_neg: List[str] = []
    pairs: List[tuple] = []
    wi: List[np.ndarray] = []
    ri: List[np.ndarray] = []
    for c in contrasts:
        wrong_pos += c.wrong_pos
        wrong_neg += c.wrong_neg
        pairs += c.pairs
        wi.append(c.wrong_disc_idx)
        ri.append(c.right_disc_idx)
    return Contrast(
        node_id="+".join(c.node_id for c in contrasts),
        wrong_pos=wrong_pos, wrong_neg=wrong_neg, pairs=pairs,
        wrong_disc_idx=np.concatenate(wi) if wi else np.array([], int),
        right_disc_idx=np.concatenate(ri) if ri else np.array([], int),
        n_wrong=sum(c.n_wrong for c in contrasts),
    )
