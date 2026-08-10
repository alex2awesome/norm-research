"""The village's published **Companion Code** — the known/articulated metrics.

Four deterministic code-scorers over the dossier text. These are the criteria the village
wrote down: how big the creature is, what it eats, where it ranges, what its hide is like.
For *canopy* creatures the Code genuinely works — a fitting companion is small, gentle, and
soft — but no single criterion dominates, so it is the *combination* that predicts. For
*cavern* creatures the Code is silent.

Crucially, the Code says **nothing** about the tacit aesthetic norms (glow, song). Recovering
those is the infilling loop's job.
"""

from __future__ import annotations

from typing import List

from methods.metrics_tree_infilling.io_metrics import MetricSpec, _stable_id
from . import world


def _code_metric(attr: str, positive_value: str, name: str, description: str,
                 role: str = "both") -> MetricSpec:
    def fn(text: str):
        v = world.detect(text, attr)
        if v is None:
            return None
        return 1.0 if v == positive_value else 0.0
    return MetricSpec(
        metric_id=_stable_id("code", attr, positive_value),
        name=name, description=description, kind="code", code_fn=fn, role=role,
    )


def companion_code() -> List[MetricSpec]:
    """The known metric set (the published Code).

    Habitat (a 3-way context) is encoded as two binary indicators so the gap-detecting tree
    can isolate each region; size/feeding/pelt are the criteria that govern grove creatures.
    """
    return [
        # habitat = context (a splitting covariate the tree uses to isolate regions)
        _code_metric("habitat", "grove", "Grove-dwelling",
                     "Whether the creature ranges in the groves.", role="context"),
        _code_metric("habitat", "cavern", "Cavern-dwelling",
                     "Whether the creature ranges in the caverns.", role="context"),
        # the criteria that actually predict (the within-node model) = features
        _code_metric("size", "tiny", "Diminutive size",
                     "Whether the creature is small rather than hulking.", role="feature"),
        _code_metric("feeding", "grazer", "Gentle feeder",
                     "Whether the creature grazes/browses rather than hunts live prey.", role="feature"),
        _code_metric("pelt", "furred", "Soft pelt",
                     "Whether the creature is furred rather than scaled.", role="feature"),
    ]
