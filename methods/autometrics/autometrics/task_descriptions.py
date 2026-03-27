"""Registry of task descriptions per dataset.

Each dataset must have an explicit task description that guides the LLM toward
substantive, domain-relevant metrics.  If a dataset is used without a registered
description, an error is raised to prevent accidentally using a generic or
wrong description.

To add a new dataset, add an entry to TASK_DESCRIPTIONS below.
"""

from __future__ import annotations

TASK_DESCRIPTIONS: dict[str, str] = {
    "PressReleaseModeling": (
        "Evaluate press releases for newsworthiness as judged by journalists. "
        "A newsworthy press release has genuine news value: it introduces a "
        "significant development, contains verifiable claims backed by concrete "
        "data or evidence, addresses a timely topic, and is relevant to a broad "
        "audience. Metrics should capture substantive journalistic qualities "
        "(e.g., factual specificity, societal impact scope, source credibility "
        "signals, novelty of the announcement, data/evidence backing claims, "
        "timeliness cues, audience relevance, institutional significance) rather "
        "than surface-level writing features (e.g., clarity, coherence, "
        "readability, word choice, sentence structure, formatting)."
    ),
    "PeerReviewAcceptance": (
        "Predict whether a scientific paper will be accepted or rejected at a "
        "peer-reviewed venue, based on its abstract. An accepted paper typically "
        "presents a clear and significant contribution — a novel method, "
        "theoretical result, or empirical finding — supported by rigorous "
        "methodology and positioned well within the existing literature. "
        "Metrics should capture substantive research quality signals "
        "(e.g., novelty and originality of the contribution, strength of "
        "empirical or theoretical support, clarity of the research question, "
        "significance of the problem addressed, soundness of methodology, "
        "positioning relative to prior work) rather than surface-level "
        "writing features (e.g., grammar, formatting, abstract length)."
    ),
}


def get_task_description(dataset_name: str) -> str:
    """Look up the task description for a dataset.

    Raises ``KeyError`` with an actionable message if the dataset has no
    registered description.
    """
    if dataset_name not in TASK_DESCRIPTIONS:
        raise KeyError(
            f"No task description registered for dataset '{dataset_name}'. "
            f"Add an entry to TASK_DESCRIPTIONS in {__file__} before running. "
            f"Registered datasets: {list(TASK_DESCRIPTIONS.keys())}"
        )
    return TASK_DESCRIPTIONS[dataset_name]
