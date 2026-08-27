"""Future a407 coverage policy for positive witnesses and absence claims.

This module is additive.  It does not change the frozen CodeScope-v3 facts,
candidate, or prepared heldout outputs.  A future v4 instrument can use this
policy to avoid treating "no event detected" as if it were a positive code
certificate.

The policy is deliberately relation-local.  A locally well-formed detection
can witness an exact placeholder surface or a structural collision even when
other files are unavailable.  In contrast, absence requires a complete
observation universe and an input-wide clean parse.  Neither state determines
whether a short name is contextually appropriate or a collision is harmful.
"""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA = "metric-seam.a407-relation-coverage-policy.v4"

EVENT_RELATION_BOUNDARIES = {
    "placeholder_avoidance": {
        "positive_event": "exact configured placeholder surface observed",
        "code_establishes": "surface occurrence at a locally valid identifier site",
        "code_does_not_establish": "whether the short or conventional name is contextually appropriate",
        "quality_interpretation": "prompt_or_hybrid_frontier",
    },
    "collision_and_shadowing": {
        "positive_event": "same-scope declaration collision or visible ancestor shadowing observed",
        "code_establishes": "the structural relation in the visible scope graph",
        "code_does_not_establish": "whether the relation is harmful to readability or behavior",
        "quality_interpretation": "prompt_or_hybrid_frontier",
    },
}

_COMPLETENESS_ZERO_FIELDS = (
    "truncated_input",
    "orphan_fragments",
    "parse_error_files",
    "parse_error_nodes",
    "parse_missing_nodes",
    "unsupported_files_with_added_code",
)


def input_wide_absence_preconditions(
    fact_counts: Mapping[str, Any],
    *,
    relation_observation_universe_complete: bool,
) -> bool:
    """Return whether a negative event claim may cover the visible input.

    ``relation_observation_universe_complete`` is intentionally explicit.  A
    parser can be error-free while a downstream detector still ignores part of
    the construct's universe.  For example, the frozen v3 a407 candidate scores
    declaration names but does not score use-only identifier surfaces, so it
    must not set this flag for criterion-wide placeholder absence.

    Deletion-only supported files do not invalidate an absence claim about
    added identifier evidence.  The required universe and claim scope must be
    declared by the future relation implementation.
    """

    if relation_observation_universe_complete is not True:
        return False
    if int(fact_counts.get("supported_files_analyzed", 0)) <= 0:
        return False
    return all(int(fact_counts.get(field, 0)) == 0 for field in _COMPLETENESS_ZERO_FIELDS)


def classify_event_relation(
    *,
    relation_id: str,
    detected_event_count: int,
    event_local_parse_valid: bool,
    observation_universe_complete: bool,
) -> dict[str, Any]:
    """Classify structural evidence without converting it into a quality score.

    Positive detections need only event-local parser validity and retain a
    relation-local claim scope.  Zero detections require input-wide observation
    completeness.  ``event_local_parse_valid`` is ignored for a zero count;
    callers should derive ``observation_universe_complete`` from the stronger
    input-wide preconditions above.
    """

    if relation_id not in EVENT_RELATION_BOUNDARIES:
        raise ValueError(f"unsupported event relation: {relation_id}")
    if not isinstance(detected_event_count, int) or isinstance(
        detected_event_count, bool
    ) or detected_event_count < 0:
        raise ValueError("detected_event_count must be a nonnegative integer")
    if not isinstance(event_local_parse_valid, bool):
        raise ValueError("event_local_parse_valid must be boolean")
    if not isinstance(observation_universe_complete, bool):
        raise ValueError("observation_universe_complete must be boolean")

    if detected_event_count:
        state = (
            "positive_relation_witness"
            if event_local_parse_valid
            else "uncertified_positive_detection"
        )
    else:
        state = (
            "verified_visible_input_absence"
            if observation_universe_complete
            else "partial_no_event_observed"
        )

    boundary = EVENT_RELATION_BOUNDARIES[relation_id]
    return {
        "schema": SCHEMA,
        "relation_id": relation_id,
        "evidence_state": state,
        "detected_event_count": detected_event_count,
        "positive_relation_witness": state == "positive_relation_witness",
        "verified_visible_input_absence": state
        == "verified_visible_input_absence",
        "claim_scope": (
            "detected_event_only"
            if state == "positive_relation_witness"
            else "visible_input"
            if state == "verified_visible_input_absence"
            else "none"
        ),
        "code_establishes": boundary["code_establishes"],
        "contextual_quality_established": False,
        "quality_interpretation": boundary["quality_interpretation"],
        "scalar_quality_score_emitted": False,
    }
