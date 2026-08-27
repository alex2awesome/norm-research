"""u29 Physical Prop Spectacle: code detects physical-action/object verb density as a structural corroboration; LLM fields carry the named prop/stunt extraction and the safety-tone judgment code cannot reach from text alone."""

import re

LLM_FIELDS = {
    "prop_or_stunt": (
        "In <=8 words, name the physical prop, object, or physical "
        "stunt/action central to this joke's comedic effect, or say 'none'."
    ),
    "safety_tone": (
        "Is the described physical action played as safe slapstick fun, or "
        "as genuinely risky/harmful? Answer: safe, risky, or none."
    ),
}

_PHYSICAL_ACTION_RE = re.compile(
    r"\b(threw|throws|throwing|dropped|drops|dropping|fell|falls|falling|"
    r"slipped|slips|tripped|trips|crashed|crashes|smash\w*|juggl\w*|"
    r"balanc\w*|swung|swings|kick\w*|punch\w*|explod\w*|collaps\w*|"
    r"grab\w*|hurl\w*|flip\w*|spin\w*|spun|shatter\w*)\b",
    re.IGNORECASE,
)


def _classify_safety(raw):
    if not raw:
        return None
    s = raw.lower()
    if "safe" in s or "slapstick" in s or "fun" in s:
        return "safe"
    if "risky" in s or "harm" in s or "danger" in s or "reckless" in s:
        return "risky"
    if "none" in s or "n/a" in s:
        return "none"
    return None


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.5
        extracted = extracted if isinstance(extracted, dict) else {}

        try:
            norm = ops.normalize(text)
        except Exception:
            norm = text

        # --- primary construct: comedic effect must rely on a physical
        # prop/object or physical stunt, not verbal-only humor. Field names
        # it; code checks presence and corroborates with an action-verb
        # density proxy, since most of this corpus is verbal-only jokes
        # with no prop mechanism at all. ---
        prop = str(extracted.get("prop_or_stunt", "") or "").strip().lower()
        has_prop = bool(prop) and prop not in ("none", "n/a")
        base = 0.75 if has_prop else 0.25

        n_action = len(_PHYSICAL_ACTION_RE.findall(norm))
        struct = min(0.1, 0.04 * n_action)
        if not has_prop:
            struct *= 0.3  # weak on its own; topic/action words are a known weak proxy

        # --- secondary nuance: "while balancing safety" -- reward
        # slapstick-safe handling, penalize gratuitously risky/harmful
        # physical content. ---
        safety = _classify_safety(str(extracted.get("safety_tone", "") or ""))
        if safety == "safe":
            safety_adj = 0.08
        elif safety == "risky":
            safety_adj = -0.12
        else:
            safety_adj = 0.0

        s = base + struct + safety_adj
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.5
