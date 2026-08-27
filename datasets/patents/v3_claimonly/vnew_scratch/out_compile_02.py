import re

_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")
_MEANS_RE = re.compile(r"\bmeans\s+for\b", re.I)
_FUNCTIONAL_RE = re.compile(
    r"\b(?:configured|adapted|operative| operable|capable)\s+to\b|"
    r"\bcapable\s+of\b|\bfor\s+(?:receiving|processing|controlling|"
    r"detecting|generating|transmitting|treating|improving|reducing|"
    r"increasing|supporting|providing)\b|\bso\s+as\s+to\b",
    re.I,
)
_TRANSITION_RE = re.compile(
    r"\b(?:comprising|comprises|comprise|including|includes|include|"
    r"consisting\s+(?:essentially\s+)?of|consists\s+(?:essentially\s+)?of|"
    r"having|characterized\s+by)\b",
    re.I,
)
_CLAIM_REF_RE = re.compile(r"\bclaims?\s+\d+(?:\s*(?:to|-)\s*\d+)?\b", re.I)

_STRUCTURAL_TERMS = {
    "apparatus", "assembly", "body", "bracket", " chamber", "channel",
    "circuit", "circuitry", "component", "composition", "container",
    "controller", "coupler", "cover", "device", "display", "element",
    "filter", "housing", "layer", "matrix", "member", "module", "opening",
    "panel", "processor", "polypeptide", "portion", "receiver", "reed",
    "reservoir", "sensor", "server", "shaft", "signal", "substrate",
    "support", "system", "tab", "terminal", "valve", "wall", "wire",
}
_ACTION_TERMS = {
    "adjust", "adjusting", "apply", "applying", "calculate", "calculating",
    "compare", "comparing", "control", "controlling", "couple", "coupling",
    "define", "defining", "detect", "detecting", "determine", "determining",
    "dispos", "disposing", "encrypt", "encrypting", "generate", "generating",
    "measure", "measuring", "mount", "mounting", "operate", "operating",
    "position", "positioning", "receive", "receiving", "select", "selecting",
    "sense", "sensing", "set", "setting", "store", "storing", "transmit",
    "transmitting", "treat", "treating", "using", "providing", "form",
    "forming", "extend", "extending", "connect", "connecting",
}
_PURPOSE_TERMS = {
    "configured", "adapted", "purpose", "use", "using", "treating",
    "improving", "reducing", "increasing", "maximize", "maximizing",
    "minimize", "minimizing", "desired", "result", "outcome", "solution",
    "field", "application",
}
_RESULT_TERMS = {
    "improve", "improving", "enhance", "enhancing", "optimize", "optimizing",
    "achieve", "achieving", "provide", "providing", "obtain", "obtaining",
    "desired", "result", "solution", "performance", "efficiency", "quality",
}
_PURPOSE_RE = re.compile(
    r"\b(?:for\s+use\s+in|adapted\s+for|configured\s+for|used\s+for|"
    r"operative\s+for|to\s+(?:improve|reduce|increase|treat|provide|"
    r"achieve|maximize|minimize|optimize))\b",
    re.I,
)
_CUMULATIVE_REF_RE = re.compile(
    r"\b(?:claims?\s+\d+\s+and\s+(?:claims?\s+)?\d+|"
    r"claims?\s+\d+(?:\s*,\s*\d+)+\s+and\s+\d+)\b",
    re.I,
)
_ALTERNATIVE_REF_RE = re.compile(
    r"\b(?:any\s+one\s+of|one\s+of|either\s+of|any\s+of)\s+claims?\b|"
    r"\bclaim\s+\d+\s+or\s+(?:claim\s+)?\d+\b",
    re.I,
)


def _tokens(text):
    return [x.lower() for x in _WORD_RE.findall(text)]


def _clamp(value):
    return float(max(0.0, min(10.0, value)))


def _valid_text(text):
    return isinstance(text, str) and bool(re.search(r"[A-Za-z0-9]", text))


def score_pb19(text: str) -> float:
    try:
        if not _valid_text(text):
            return 0.0
        lower = text.lower()
        if _MEANS_RE.search(text):
            return 10.0

        tokens = _tokens(text)
        structural = sum(token in _STRUCTURAL_TERMS for token in tokens)
        structural += len(re.findall(
            r"\b(?:configured|adapted|coupled|disposed|mounted|positioned|"
            r"connected|formed|defined)\b", lower
        ))
        functional = len(_FUNCTIONAL_RE.findall(text))
        vague = len(re.findall(
            r"\b(?:means|module|unit|mechanism|element|device)\s+to\b", lower
        ))

        if functional == 0:
            return _clamp(7.0 + min(3.0, structural * 0.35))
        return _clamp(
            6.0 + min(2.5, structural * 0.35)
            - min(5.5, functional * 1.35)
            - min(2.0, vague * 0.8)
        )
    except Exception:
        return 0.0


def score_pb20(text: str) -> float:
    try:
        if not _valid_text(text):
            return 0.0
        tokens = _tokens(text)
        actions = sum(
            any(token.startswith(prefix) for prefix in _ACTION_TERMS)
            for token in tokens
        )
        mechanical = len(re.findall(
            r"\b(?:coupled|connected|disposed|mounted|positioned|"
            r"extending|defining|opening|closing|equal|within|between)\b",
            text,
            re.I,
        ))
        sequence = len(re.findall(
            r"\b(?:then|subsequently|thereafter|in response to|based on|"
            r"when|while|until|if|responsive to|comprises|including)\b",
            text,
            re.I,
        ))
        outcome = len(_RESULT_TERMS.intersection(tokens))
        outcome += len(re.findall(
            r"\b(?:such that|to thereby|thereby|in order to|so as to)\b",
            text,
            re.I,
        ))

        return _clamp(
            1.0 + actions * 1.25 + mechanical * 0.3 + sequence * 0.65
            - outcome * 1.35
        )
    except Exception:
        return 0.0


def score_pb22(text: str) -> float:
    try:
        if not _valid_text(text):
            return 0.0
        lower = text.lower()
        tokens = _tokens(text)

        structure = sum(token in _STRUCTURAL_TERMS for token in tokens)
        structure += len(re.findall(
            r"\b(?:comprising|comprises|consisting|includes|having|"
            r"coupled|connected|disposed|mounted|defines|extends|"
            r"contains|formed|positioned)\b",
            lower,
        ))
        structure += len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:mm|nm|cm|%|°c)?\b", lower))
        active = len(re.findall(
            r"\b(?:receiving|determining|adjusting|generating|transmitting|"
            r"controlling|measuring|selecting|operating|applying|encrypting)\b",
            lower,
        ))
        purpose = len(_PURPOSE_RE.findall(text))
        purpose += sum(token in _PURPOSE_TERMS for token in tokens)
        result_only = len(re.findall(
            r"\b(?:newly\s+discovered|desired\s+result|field\s+of\s+use|"
            r"improve|improving|treat|treating)\b",
            lower,
        ))

        return _clamp(
            1.0 + structure * 0.9 + active * 0.75
            - purpose * 0.85 - result_only * 0.7
        )
    except Exception:
        return 0.0


def score_pb23(text: str) -> float:
    try:
        if not _valid_text(text):
            return 0.0
        lower = text.lower()
        transitions = _TRANSITION_RE.findall(text)
        if not transitions:
            return _clamp(
                0.5 + min(2.0, len(re.findall(r"\bwherein\b|\bfurther\b", lower)) * 0.4)
            )

        malformed = len(re.findall(
            r"\b(?:comprising|comprises|including|includes|consisting)\s+of\b",
            lower,
        ))
        recognized = len(transitions)
        boundary = len(re.findall(r"[,;:]|\bwherein\b", lower))
        return _clamp(
            7.5 + min(1.5, recognized * 0.7)
            + min(1.0, boundary * 0.15)
            - malformed * 3.0
        )
    except Exception:
        return 0.0


def score_pb24(text: str) -> float:
    try:
        if not _valid_text(text):
            return 0.0
        lower = text.lower()
        strong = bool(re.search(
            r"\bselected\s+from\s+the\s+group\s+consisting\s+of\b", lower
        ))
        weak = bool(re.search(
            r"\b(?:selected\s+from|chosen\s+from)\b", lower
        ))
        alternatives = (
            lower.count(",")
            + len(re.findall(r"\bor\b|\band\b", lower))
        )
        tail = lower.split("consisting of", 1)[-1] if "consisting of" in lower else ""
        tail_items = len(_tokens(tail))

        if strong and tail_items >= 2 and alternatives >= 1:
            return _clamp(8.5 + min(1.5, alternatives * 0.35))
        if strong:
            return 7.0
        if weak and alternatives >= 1:
            return _clamp(3.5 + min(2.5, alternatives * 0.5))
        if alternatives:
            return _clamp(1.0 + min(2.0, alternatives * 0.35))
        return 0.0
    except Exception:
        return 0.0


def score_pb26(text: str) -> float:
    try:
        if not _valid_text(text):
            return 0.0
        refs = _CLAIM_REF_RE.findall(text)
        if not refs:
            return 0.0

        lower = text.lower()
        if _CUMULATIVE_REF_RE.search(text):
            return 2.0

        alternative = bool(_ALTERNATIVE_REF_RE.search(text))
        dependent_cue = len(re.findall(
            r"\b(?:wherein|further\s+comprising|further\s+including|"
            r"depending\s+from|according\s+to)\b",
            lower,
        ))
        range_ref = bool(re.search(r"\bclaims?\s+\d+\s*(?:to|-)\s*\d+\b", lower))

        if len(refs) == 1 and not range_ref:
            return _clamp(8.5 + min(1.5, dependent_cue * 0.6))
        if alternative:
            return _clamp(8.0 + min(1.5, dependent_cue * 0.4))
        if range_ref:
            return _clamp(4.0 + min(2.0, dependent_cue * 0.5))
        return _clamp(4.0 + min(2.0, dependent_cue * 0.5))
    except Exception:
        return 0.0


REGISTRY = {
    "pb19": score_pb19,
    "pb20": score_pb20,
    "pb22": score_pb22,
    "pb23": score_pb23,
    "pb24": score_pb24,
    "pb26": score_pb26,
}