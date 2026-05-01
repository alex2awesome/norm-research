"""Refactoring-based clustering: replace embed→cluster→label with LLM-based abstraction.

Instead of grouping features by embedding similarity, we iteratively show batches
of raw features to an LLM and ask it to find common abstractions. The abstractions
accumulate into a "principle library" that stabilizes over batches — analogous to
how the verification library builds up a code library from per-example programs.

Same interface as build_canonical_vocabulary(): takes weighted features, returns
CanonicalFeatures with rubrics.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .rubric_generation import CanonicalFeature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 1: Deduplicate & group observations that say the same thing
# ---------------------------------------------------------------------------

DEDUP_PROMPT = """\
You are analyzing evaluation criteria for a {text_type}.

Here are specific observations extracted from individual examples. Many say \
the same thing in different words.

{observations}

Group observations that express the same underlying point. For each group:
- "canonical": a clear, concise canonical statement of the point
- "members": the original observations that express this point
- "polarity": "positive" (argues for {positive_label}), "negative" (argues for \
{negative_label}), or "mixed"

Return a JSON list. Every observation should appear in exactly one group. \
Observations that are unique get their own singleton group.

```json
[
  {{
    "canonical": "The statistical analysis is underpowered",
    "members": ["sample size too small", "underpowered study", "insufficient N for claimed effect"],
    "polarity": "negative"
  }}
]
```

Return ONLY the JSON list wrapped in ```json ... ``` markers."""


# ---------------------------------------------------------------------------
# Step 2: Abstract — find common structure across canonical groups
# ---------------------------------------------------------------------------

ABSTRACTION_PROMPT = """\
You are analyzing evaluation criteria for a {text_type}.

{existing_library_section}\
Here are {n_new} canonical evaluation criteria (deduplicated from raw observations):

{observations}

Find **common abstractions** across these criteria. An abstraction is a reusable \
conceptual building block that captures structure shared across multiple criteria.

Abstractions might take many forms, including but not limited to:
- **Generalizations**: a broader concept that subsumes several criteria
- **Compositions**: multiple criteria that combine into one higher-level judgment
- **Conditions / exceptions**: "X applies, unless Y" or "X only when Z"
- **Dependencies**: "X only matters if Y is already satisfied"
- **Scope constraints**: "applies to empirical work but not surveys"
- **Polarity patterns**: the same property argues for/against depending on degree
- **Severity levels**: some criteria are dealbreakers, others are minor
- **Default expectations**: assumed baseline unless stated otherwise

For each abstraction, output:
- "name": short descriptive name
- "description": 2-3 sentences explaining what this captures and how to evaluate it
- "type": the kind of abstraction (e.g. "generalization", "composition", \
"condition", "dependency", "scope", "severity", or describe your own)
- "member_criteria": list of canonical criteria this abstraction covers
- "polarity": "positive", "negative", or "mixed"

Return {target_k} to {target_k_max} abstractions as a JSON list. \
Cover all criteria if possible.

```json
[
  {{
    "name": "Statistical methodology concerns",
    "description": "The study's statistical approach has issues that undermine confidence. Includes power, test selection, and multiple comparisons.",
    "type": "generalization",
    "member_criteria": ["underpowered analysis", "wrong statistical test", "no multiple comparison correction"],
    "polarity": "negative"
  }}
]
```

Return ONLY the JSON list wrapped in ```json ... ``` markers."""


# ---------------------------------------------------------------------------
# Step 3: Merge abstractions across batches
# ---------------------------------------------------------------------------

MERGE_PROMPT = """\
You are consolidating evaluation criteria for a {text_type}.

Here is the current set of abstractions:

{existing_abstractions}

Here are new abstractions from the latest batch:

{new_abstractions}

Merge into a single consolidated set:
- **Equivalent** abstractions: merge them (combine descriptions and members)
- **Specializations**: keep the general one, enrich its description with the specific case
- **Novel** abstractions: add them
- **Overlapping** abstractions: merge if the overlap is substantial
- **Too vague**: remove abstractions that are generic enough to apply to anything

Preserve the "type" field. If merging changes the type (e.g., a generalization \
absorbs a condition), update accordingly.

Target: {target_k} to {target_k_max} total abstractions.

Return a JSON list with fields: name, description, type, member_criteria, polarity. \
Return ONLY the JSON list wrapped in ```json ... ``` markers."""


# ---------------------------------------------------------------------------
# Step 4: Convert abstractions to scoring rubrics
# ---------------------------------------------------------------------------

RUBRIC_FROM_ABSTRACTION_PROMPT = """\
You are creating an evaluation rubric for the following criterion used to judge \
a {text_type}:

Criterion: {name}
Type: {abstraction_type}
Description: {description}

Example observations that fall under this criterion:
{members}

{structural_note}

Write a binary rubric that an evaluator can use to score any {text_type}.

Output as JSON:
```json
{{
  "yes": "What constitutes YES for this criterion (2-3 sentences). Be specific.",
  "no": "What constitutes NO for this criterion (2-3 sentences). Be specific."
}}
```

Return ONLY the JSON wrapped in ```json ... ``` markers."""


RUBRIC_FROM_ABSTRACTION_PROMPT = """\
You are creating a binary evaluation rubric for the following criterion \
used to evaluate a {text_type}:

Criterion: {name}
Description: {description}
Example observations that fall under this criterion:
{members}

Write a binary rubric that an evaluator can use to score any {text_type} on this criterion.

Output as JSON:
```json
{{
  "yes": "Description of what constitutes YES for this criterion (2-3 sentences)",
  "no": "Description of what constitutes NO for this criterion (2-3 sentences)"
}}
```"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(response: str) -> Optional[list]:
    """Extract JSON list from ```json ... ``` blocks or raw text."""
    pattern = r"```json\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    text = match.group(1).strip() if match else response.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return None
    except json.JSONDecodeError:
        return None


def _extract_json_dict(response: str) -> Optional[dict]:
    """Extract JSON dict from response."""
    pattern = r"```json\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    text = match.group(1).strip() if match else response.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        return None


def _make_feature_id(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:16]


def _format_observations(
    features: List[str],
    weights: List[float],
    polarities: Optional[Dict[str, str]] = None,
) -> str:
    """Format feature observations for the prompt."""
    lines = []
    for feat, w in zip(features, weights):
        pol = ""
        if polarities and feat in polarities:
            pol = f" [{polarities[feat]}]"
        sign = "+" if w >= 0 else ""
        lines.append(f"- {feat} (weight: {sign}{w:.2f}){pol}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_canonical_vocabulary_via_refactoring(
    weighted_features: List[Tuple[str, float]],
    text_type: str,
    positive_label: str,
    negative_label: str,
    generate_fn: Callable[[str], str],
    output_dir: str,
    feature_polarity: Optional[Dict[str, str]] = None,
    target_k: int = 30,
    batch_size: int = 50,
) -> List[CanonicalFeature]:
    """Replace embed-cluster-label with LLM-based refactoring.

    Same output type as build_canonical_vocabulary: List[CanonicalFeature].

    Args:
        weighted_features: flat list of (feature_text, weight).
        text_type: e.g. "scientific paper abstract".
        positive_label: e.g. "accepted".
        negative_label: e.g. "rejected".
        generate_fn: LLM text generation callable (prompt -> response).
        output_dir: directory for artifacts.
        feature_polarity: optional dict mapping feature text to "for"/"against".
        target_k: target number of canonical features.
        batch_size: features per refactoring batch.

    Returns: list of CanonicalFeature objects ready for binary scoring.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Check cache
    canon_path = out_path / "canonical_features_refactored.json"
    if canon_path.exists():
        logger.info("Loading cached refactored canonical features from %s", canon_path)
        with open(canon_path) as f:
            cached = json.load(f)
        return [
            CanonicalFeature(
                feature_id=item.get("feature_id") or _make_feature_id(item["name"]),
                name=item["name"],
                description=item["description"],
                rubric=item["rubric"],
                aggregate_weight=item.get("aggregate_weight", 0.0),
                cluster_size=item.get("cluster_size", 0),
                member_features=item.get("member_features", []),
            )
            for item in cached
        ]

    # Deduplicate features
    feature_weights: Dict[str, float] = {}
    for feat, weight in weighted_features:
        feat = feat.strip()
        if feat:
            feature_weights[feat] = feature_weights.get(feat, 0.0) + weight

    features = list(feature_weights.keys())
    weights = [feature_weights[f] for f in features]
    logger.info("Refactoring pipeline: %d unique features", len(features))

    # Sort by |weight| descending — most discriminative first
    sorted_pairs = sorted(zip(features, weights), key=lambda x: abs(x[1]), reverse=True)
    features = [f for f, _ in sorted_pairs]
    weights = [w for _, w in sorted_pairs]

    target_k_max = int(target_k * 1.5)
    n_batches = (len(features) + batch_size - 1) // batch_size

    # Save snapshots for convergence analysis
    snapshots_dir = out_path / "refactoring_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 1: Deduplicate — group observations that say the same thing
    # -----------------------------------------------------------------------
    logger.info("Step 1: Deduplicating %d features across %d batches", len(features), n_batches)
    all_canonical_groups: List[dict] = []

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(features))
        batch_features = features[start:end]
        batch_weights = weights[start:end]

        observations_text = _format_observations(
            batch_features, batch_weights, feature_polarity,
        )

        dedup_prompt = DEDUP_PROMPT.format(
            text_type=text_type,
            positive_label=positive_label,
            negative_label=negative_label,
            observations=observations_text,
        )
        response = generate_fn(dedup_prompt)
        groups = _extract_json(response)
        if groups:
            all_canonical_groups.extend(groups)
            logger.info("Batch %d: %d groups from %d observations", batch_idx, len(groups), len(batch_features))
        else:
            logger.warning("Dedup failed for batch %d, treating each as singleton", batch_idx)
            for f, w in zip(batch_features, batch_weights):
                all_canonical_groups.append({
                    "canonical": f,
                    "members": [f],
                    "polarity": feature_polarity.get(f, "mixed") if feature_polarity else "mixed",
                })

    # Save dedup results
    dedup_path = snapshots_dir / "step1_deduplicated.json"
    with open(dedup_path, "w") as f:
        json.dump(all_canonical_groups, f, indent=2)
    logger.info("Step 1 complete: %d canonical groups", len(all_canonical_groups))

    # -----------------------------------------------------------------------
    # STEP 2: Abstract — find common abstractions across canonical groups
    # -----------------------------------------------------------------------
    logger.info("Step 2: Finding abstractions across %d canonical groups", len(all_canonical_groups))

    # Format canonical groups as observations for abstraction
    canonical_texts = [g["canonical"] for g in all_canonical_groups]
    # Compute weight per canonical group by summing member weights
    canonical_weights = []
    for g in all_canonical_groups:
        w = sum(feature_weights.get(m, 0.0) for m in g.get("members", [g["canonical"]]))
        canonical_weights.append(w)

    # Process in batches, merging incrementally
    current_abstractions: Optional[list] = None
    canon_batch_size = min(batch_size, len(canonical_texts))
    n_canon_batches = (len(canonical_texts) + canon_batch_size - 1) // canon_batch_size

    for batch_idx in range(n_canon_batches):
        start = batch_idx * canon_batch_size
        end = min(start + canon_batch_size, len(canonical_texts))
        batch_texts = canonical_texts[start:end]
        batch_ws = canonical_weights[start:end]

        observations_text = _format_observations(batch_texts, batch_ws, feature_polarity)

        existing_section = ""
        if current_abstractions is not None:
            existing_section = (
                "Here are abstractions discovered in previous batches:\n\n"
                f"```json\n{json.dumps(current_abstractions, indent=2)}\n```\n\n"
                "Consider these when identifying abstractions. "
                "New criteria may fit existing abstractions or reveal new ones.\n\n"
            )

        prompt = ABSTRACTION_PROMPT.format(
            text_type=text_type,
            positive_label=positive_label,
            negative_label=negative_label,
            existing_library_section=existing_section,
            n_new=len(batch_texts),
            observations=observations_text,
            target_k=target_k,
            target_k_max=target_k_max,
        )
        response = generate_fn(prompt)
        new_abstractions = _extract_json(response)

        if new_abstractions is None:
            logger.error("Failed to parse abstractions from batch %d", batch_idx)
            continue

        if current_abstractions is None:
            current_abstractions = new_abstractions
        else:
            # Merge
            merge_prompt = MERGE_PROMPT.format(
                text_type=text_type,
                existing_abstractions=json.dumps(current_abstractions, indent=2),
                new_abstractions=json.dumps(new_abstractions, indent=2),
                target_k=target_k,
                target_k_max=target_k_max,
            )
            merge_response = generate_fn(merge_prompt)
            merged = _extract_json(merge_response)
            if merged is not None:
                current_abstractions = merged
            else:
                logger.warning("Merge failed, appending new abstractions")
                current_abstractions = current_abstractions + new_abstractions

        # Save snapshot
        snapshot_path = snapshots_dir / f"step2_abstractions_batch_{batch_idx:03d}.json"
        with open(snapshot_path, "w") as f:
            json.dump(current_abstractions, f, indent=2)

        logger.info(
            "After abstraction batch %d: %d abstractions",
            batch_idx, len(current_abstractions),
        )

    if current_abstractions is None:
        logger.error("No abstractions produced")
        return []

    # -----------------------------------------------------------------------
    # STEP 3: Generate rubrics — convert abstractions to scoring questions
    # -----------------------------------------------------------------------
    logger.info("Step 3: Generating rubrics for %d abstractions", len(current_abstractions))
    canonical_features = []

    for abstraction in current_abstractions:
        name = abstraction.get("name", "unnamed")
        description = abstraction.get("description", "")
        members = abstraction.get("member_criteria", abstraction.get("member_observations", []))
        polarity = abstraction.get("polarity", "mixed")
        abs_type = abstraction.get("type", "generalization")

        members_text = "\n".join(f"- {m}" for m in members[:10])

        # Type-aware structural notes for the rubric prompt
        structural_notes = {
            "condition": "This criterion has conditions/exceptions. The rubric should specify when it applies and when exceptions override it.",
            "dependency": "This criterion depends on other criteria being satisfied. The rubric should note what must be true for this criterion to be relevant.",
            "scope": "This criterion only applies to certain types of inputs. The rubric should specify when to apply it and when to skip it.",
            "composition": "This criterion combines multiple sub-criteria. The rubric should address all components.",
            "severity": "This criterion has severity implications. The rubric should indicate whether this is a dealbreaker or a minor concern.",
        }
        structural_note = structural_notes.get(abs_type, "")
        if structural_note:
            structural_note = f"Note: {structural_note}"

        rubric_prompt = RUBRIC_FROM_ABSTRACTION_PROMPT.format(
            text_type=text_type,
            name=name,
            abstraction_type=abs_type,
            description=description,
            members=members_text,
            structural_note=structural_note,
        )
        rubric_response = generate_fn(rubric_prompt)
        rubric = _extract_json_dict(rubric_response)
        if rubric is None:
            rubric = {"yes": description, "no": f"Does not exhibit: {description}"}

        # Aggregate weight from member features
        agg_weight = sum(
            feature_weights.get(m, 0.0) for m in members
        )

        cf = CanonicalFeature(
            feature_id=_make_feature_id(name),
            name=name,
            description=description,
            rubric=rubric,
            aggregate_weight=agg_weight,
            cluster_size=len(members),
            member_features=members,
        )
        canonical_features.append(cf)

    # Sort by |aggregate_weight| descending
    canonical_features.sort(key=lambda f: abs(f.aggregate_weight), reverse=True)

    # Trim to target_k
    if len(canonical_features) > target_k:
        canonical_features = canonical_features[:target_k]

    # Cache
    cache_data = [
        {
            "feature_id": f.feature_id,
            "name": f.name,
            "description": f.description,
            "rubric": f.rubric,
            "aggregate_weight": f.aggregate_weight,
            "cluster_size": f.cluster_size,
            "member_features": f.member_features,
        }
        for f in canonical_features
    ]
    with open(canon_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    logger.info("Saved %d canonical features to %s", len(canonical_features), canon_path)

    return canonical_features
