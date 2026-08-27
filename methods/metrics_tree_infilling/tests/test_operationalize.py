"""Tests for the GEPA-style operationalization stage (label-free rubric iteration)."""
import numpy as np

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.operationalize import operationalize_rubric


def _mk_judge(noise=0.0, seed=0):
    """Judge whose score = text length signal + optional noise (per-rubric deterministic)."""
    rng = np.random.default_rng(seed)

    def judge(metrics, texts):
        n, m = len(texts), len(metrics)
        lv = np.zeros((n, m))
        ap = np.ones((n, m), dtype=bool)
        for j, met in enumerate(metrics):
            base = np.array([min(1.0, len(t) / 100) for t in texts])
            eps = rng.normal(0, noise, n) if noise else 0
            lv[:, j] = np.clip(base + eps, 0, 1)
        return lv, ap

    return judge


def _proposer_recon(prompt):
    if "reverse-engineering" in prompt:
        return '{"rubric": "Longer, more developed texts score higher."}'
    # rewrite prompt -> returns an improved rubric
    return ("DEFINITION: measures development of the text. GUIDANCE: 1.0 = fully developed; "
            "0.5 = partial; 0.0 = fragmentary. POSITIVE SKETCH: a complete argument. "
            "NEGATIVE SKETCH: a fragment. BOUNDARY NOTE: not mere length.")


def test_reliable_rubric_passes_through_unchanged():
    texts = [("x" * (10 + 7 * i)) for i in range(120)]
    cfg = InfillConfig()
    res = operationalize_rubric("dev", "development", "Score development 0-1.",
                                texts, _mk_judge(noise=0.0), _proposer_recon, cfg)
    assert res.iterations == 0
    assert res.rubric == "Score development 0-1."


def test_unreliable_rubric_gets_rewritten():
    texts = [("x" * (10 + 7 * i)) for i in range(120)]
    cfg = InfillConfig()
    res = operationalize_rubric("dev", "development", "Score vibes 0-1.",
                                texts, _mk_judge(noise=0.45, seed=3), _proposer_recon, cfg,
                                min_retest=0.95)
    assert res.iterations >= 1
    assert len(res.trajectory) == res.iterations + 1


def test_result_fields_populated():
    texts = [("x" * (10 + 7 * i)) for i in range(120)]
    cfg = InfillConfig()
    res = operationalize_rubric("dev", "d", "Score development 0-1.",
                                texts, _mk_judge(), _proposer_recon, cfg)
    assert np.isfinite(res.std)
    assert isinstance(res.trajectory, list) and res.trajectory
