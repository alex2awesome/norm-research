"""Robust parsers for semi-structured LLM outputs."""

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StarLocalResult:
    """Parsed output from Approach B's blind extraction + deliberation."""
    features_for: List[str]
    features_against: List[str]
    prediction: str              # normalized: "positive" or "negative"
    confidence: str              # "HIGH", "MEDIUM", or "LOW"
    raw_deliberation: str = ""   # free-form thinking before the structured output


def _extract_json_from_text(text: str) -> Optional[str]:
    """Find a JSON blob in text, handling markdown fences."""
    # Strip markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Try to find a JSON array
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        return match.group(0)

    # Try to find a JSON object
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return None


def parse_feature_list_json(raw_text: str) -> List[str]:
    """Parse a JSON list of feature strings from LLM output.

    Used for: Approach A prior estimation & rationalization.

    Tries multiple strategies:
    1. Direct JSON array parse
    2. Extract from {"features": [...]} wrapper
    3. Regex for quoted strings in array pattern
    4. Fallback: split by newlines, strip bullets
    """
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.strip()

    # Strategy 1: Try direct JSON parse
    for attempt_text in [text, _extract_json_from_text(text) or ""]:
        if not attempt_text:
            continue
        try:
            parsed = json.loads(attempt_text)
            if isinstance(parsed, list):
                return [str(f).strip() for f in parsed if f]
            if isinstance(parsed, dict) and "features" in parsed:
                return [str(f).strip() for f in parsed["features"] if f]
        except (json.JSONDecodeError, TypeError):
            pass

        # Strategy 2: ast.literal_eval fallback
        try:
            parsed = ast.literal_eval(attempt_text)
            if isinstance(parsed, list):
                return [str(f).strip() for f in parsed if f]
        except (ValueError, SyntaxError):
            pass

    # Strategy 3: Regex for quoted strings in array
    matches = re.findall(r'"([^"]{5,})"', text)
    if len(matches) >= 2:
        return [m.strip() for m in matches]

    # Strategy 4: Bullet-point / numbered list fallback
    lines = text.split("\n")
    features = []
    for line in lines:
        line = line.strip()
        # Strip common prefixes: "- ", "1. ", "* ", "• "
        line = re.sub(r"^[-*\u2022]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        line = line.strip().strip('"').strip("'")
        if len(line) > 10 and not line.startswith("{") and not line.startswith("["):
            features.append(line)
    return features


def parse_json_dict(raw_text: str) -> Dict:
    """Parse a JSON dict from LLM output. Used for cluster naming, rubrics, etc."""
    if not raw_text:
        return {}

    text = raw_text.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Try full text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try nested braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _extract_bulleted_section(text: str, header_pattern: str) -> List[str]:
    """Extract bulleted items following a header line."""
    # Find the header
    match = re.search(header_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return []

    # Get text after header until next section header or end
    start = match.end()
    # Look for next section: FEATURES_AGAINST, PREDICTION, or end
    next_section = re.search(
        r"^(FEATURES_(?:FOR|AGAINST)|PREDICTION|CONFIDENCE)\s*[:(]",
        text[start:],
        re.IGNORECASE | re.MULTILINE,
    )
    end = start + next_section.start() if next_section else len(text)
    section_text = text[start:end]

    # Extract bulleted items
    features = []
    # Section header keywords that should never appear as feature content (parser
    # otherwise picks them up as a feature when the model emits the next section's
    # header on its own line).
    SECTION_TOKENS = {
        "features_for", "features_against", "prediction", "confidence",
        "features for", "features against",
    }
    for line in section_text.split("\n"):
        line = line.strip()
        line = re.sub(r"^[-*\u2022]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        line = line.strip()
        # Skip empty, very short, or bracket-only lines
        if len(line) <= 5 or line.startswith("[") or line == "...":
            continue
        # Remove leading "feature N:" pattern
        line = re.sub(r"^feature\s*\d+\s*:\s*", "", line, flags=re.IGNORECASE)
        # Drop lines that are themselves section markers (with or without trailing : or ()
        cmp = line.lower().rstrip(":(").strip()
        cmp = re.sub(r"\s*\([^)]*\)\s*$", "", cmp).strip()
        if cmp in SECTION_TOKENS:
            continue
        # Drop empty-padding negations the model emits when it has nothing critical
        # to say (e.g. "There are no obvious flaws in the abstract that suggest rejection")
        # — these are noise, not real features. Match common phrasings.
        if re.match(
            r"^there (?:is|are)\s+no\b.*(?:flaw|weakness|feature|issue|concern|red flag|drawback|problem|negative)",
            line, re.IGNORECASE,
        ):
            continue
        if re.match(
            r"^no obvious\s+(?:flaw|weakness|feature|issue|concern|red flag|drawback|problem)",
            line, re.IGNORECASE,
        ):
            continue
        features.append(line)
    return features


def parse_star_local_output(
    raw_text: str,
    positive_label: str,
    negative_label: str,
) -> StarLocalResult:
    """Parse Approach B's structured output.

    Extracts FEATURES_FOR, FEATURES_AGAINST, PREDICTION, CONFIDENCE
    from the model's deliberation + structured summary.
    """
    if not raw_text:
        return StarLocalResult(
            features_for=[], features_against=[],
            prediction="", confidence="LOW",
        )

    text = raw_text.strip()

    # Extract features_for
    features_for = _extract_bulleted_section(
        text, r"FEATURES_FOR\s*(?:\([^)]*\))?\s*:?"
    )

    # Extract features_against
    features_against = _extract_bulleted_section(
        text, r"FEATURES_AGAINST\s*(?:\([^)]*\))?\s*:?"
    )

    # Extract prediction
    prediction = ""
    pred_match = re.search(
        r"PREDICTION\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE
    )
    if pred_match:
        pred_raw = pred_match.group(1).strip().lower()
        if positive_label.lower() in pred_raw:
            prediction = "positive"
        elif negative_label.lower() in pred_raw:
            prediction = "negative"
        else:
            prediction = pred_raw

    # Extract confidence
    confidence = "MEDIUM"
    conf_match = re.search(
        r"CONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE
    )
    if conf_match:
        confidence = conf_match.group(1).upper()

    # Everything before FEATURES_FOR is deliberation
    deliberation = ""
    for_match = re.search(r"FEATURES_FOR", text, re.IGNORECASE)
    if for_match:
        deliberation = text[: for_match.start()].strip()

    return StarLocalResult(
        features_for=features_for,
        features_against=features_against,
        prediction=prediction,
        confidence=confidence,
        raw_deliberation=deliberation,
    )


def parse_similarity_labels(raw_text: str, n_pairs: int) -> Dict[int, bool]:
    """Parse Yes/No labels for similarity pairs.

    Returns: dict mapping pair index (0-based) -> True (similar) / False (different).
    """
    result = {}

    # Try JSON parse first
    parsed = parse_json_dict(raw_text)
    if parsed:
        for key, value in parsed.items():
            # Extract pair number from key like "pair_1", "1", etc.
            num_match = re.search(r"(\d+)", str(key))
            if num_match:
                idx = int(num_match.group(1)) - 1  # 0-based
                result[idx] = str(value).strip().lower().startswith("yes")
        if result:
            return result

    # Fallback: line-by-line search
    for line in raw_text.split("\n"):
        num_match = re.search(r"(?:pair\s*)?(\d+)\s*:\s*(yes|no)", line, re.IGNORECASE)
        if num_match:
            idx = int(num_match.group(1)) - 1
            result[idx] = num_match.group(2).lower() == "yes"

    return result
