from __future__ import annotations

from methods.codability.lexicon_distill.evaluate_similarity_lora import cohen_kappa, metrics


def _row(truth: int, prediction: int, consistent: bool = True) -> dict:
    probabilities = [0.0, 0.0, 0.0]
    probabilities[prediction] = 1.0
    target = [0.0, 0.0, 0.0]
    target[truth] = 1.0
    return {
        "truth": truth,
        "prediction": prediction,
        "probabilities": probabilities,
        "target_probs": target,
        "order_consistent": consistent,
    }


def test_metrics_perfect_predictions() -> None:
    rows = [_row(0, 0), _row(1, 1), _row(2, 2)]
    result = metrics(rows)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["cohen_kappa"] == 1.0
    assert result["brier"] == 0.0


def test_kappa_penalizes_constant_prediction() -> None:
    assert cohen_kappa([0, 1, 2], [1, 1, 1]) == 0.0
