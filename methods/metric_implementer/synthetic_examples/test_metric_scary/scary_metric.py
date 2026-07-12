"""The `is_scary` metric: a deliberately crude prompt seed plus the canonical description
the oracle scores against. The seed names only the obvious "scary" cue (DREAD), leaving the
other four planted categories as headroom the optimizer can discover and articulate.
"""

from __future__ import annotations

from ...artifact import MetricArtifact

METRIC_ID = "is_scary"

# The canonical construct. The oracle scores items against THIS (not the candidate prompt),
# and it names every planted cue category, so a strong reader recovers the ground-truth
# label. It is also the semantic-match target for reconstruction.
CANONICAL_DESCRIPTION = (
    "The story reads as scary: it evokes fear and dread through concrete cues — ominous "
    "sounds (a creak, footsteps, a scream in the silence), an unseen presence or the sense "
    "of being watched (a figure, a shadow, someone following), bodily fear (a pounding "
    "heart, caught breath, a shiver up the spine), and signs of danger or threat (a knife, "
    "blood, an attack). A non-scary story is calm, warm, and ordinary."
)

# Crude seed: names only the obvious cue. Covers exactly one planted category (DREAD), so it
# detects scary stories only coarsely — the optimizer earns fidelity by articulating the rest.
SEED_PROMPT = (
    "Score whether this story is scary. 1.0 = very scary and frightening; 0.0 = not scary "
    "at all; use intermediate values for in-between cases."
)

# A fully-articulated reference rubric (the planted ceiling) — used for evaluation/contrast,
# never as the optimizer's seed.
REFERENCE_RUBRIC = (
    "Score how scary this story is, in [0,1]. Check for each of these concrete cues and "
    "score higher when more are present:\n"
    "1. SOUND: ominous or sudden sounds — a creak, footsteps, a scream, a scrape, silence.\n"
    "2. PRESENCE: an unseen presence — a figure, a shadow, someone watching or following.\n"
    "3. DREAD: explicit fear, dread, terror, or an ominous feeling of wrongness.\n"
    "4. BODY: bodily fear — a pounding heart, caught breath, a shiver up the spine, trembling.\n"
    "5. DANGER: signs of danger or threat — a knife, blood, an attack, harm.\n"
    "Aggregate: 0.0 if none are present (a calm, warm, ordinary scene); higher as more cue "
    "categories appear."
)

# Invariances the metric SHOULD be robust to (their transforms exist in measures._TRANSFORMS;
# none touches the planted cue markers, so a faithful judge stays stable under them).
INVARIANCES = ["blank_lines", "identifier_rename"]


def seed_artifact() -> "MetricArtifact":
    """The crude prompt seed the optimizer starts from."""
    return MetricArtifact(
        metric_id=METRIC_ID, kind="prompt", body=SEED_PROMPT,
        name="is_scary", description=CANONICAL_DESCRIPTION,
        invariances=list(INVARIANCES),
        meta={"synthetic": True, "planted": True, "ceiling_rubric": "REFERENCE_RUBRIC"})


def reference_artifact() -> "MetricArtifact":
    """The fully-articulated rubric — the planted recovery ceiling, for contrast only."""
    return MetricArtifact(
        metric_id=METRIC_ID, kind="prompt", body=REFERENCE_RUBRIC,
        name="is_scary", description=CANONICAL_DESCRIPTION,
        invariances=list(INVARIANCES), meta={"synthetic": True, "reference": True})
