import re
from typing import Any, Dict

from dspy.adapters import JSONAdapter
from dspy.adapters.utils import parse_value
from dspy.utils.exceptions import AdapterParseError

__all__ = ["LenientJSONAdapter"]


class LenientJSONAdapter(JSONAdapter):
    """JSONAdapter with a fallback parser for plain-text Reasoning/Score outputs."""

    def parse(self, signature, completion: str) -> Dict[str, Any]:
        try:
            return super().parse(signature, completion)
        except AdapterParseError:
            text = completion or ""
            fields: Dict[str, Any] = {}

            # Reasoning (optional)
            if "reasoning" in signature.output_fields:
                reasoning = ""
                if "Reasoning:" in text:
                    reasoning = text.split("Reasoning:", 1)[1].split("Score:", 1)[0].strip()
                fields["reasoning"] = reasoning

            # Score (required for our judge signatures)
            if "score" in signature.output_fields:
                score_match = re.search(r"Score\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+)", text, re.IGNORECASE)
                if not score_match:
                    score_match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", text)
                if score_match:
                    raw_score = score_match.group(1)
                    try:
                        fields["score"] = parse_value(raw_score, signature.output_fields["score"].annotation)
                    except Exception:
                        fields["score"] = raw_score

            # Fill missing fields with None to satisfy adapter contract
            for field in signature.output_fields.keys():
                if field not in fields:
                    fields[field] = None

            if any(v is not None for v in fields.values()):
                return fields

            # No usable fallback parse; re-raise original error
            raise
