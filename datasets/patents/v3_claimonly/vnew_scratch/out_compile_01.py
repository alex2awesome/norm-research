import re
import math
import string
from collections import Counter

_PARAM_RE = re.compile(
    r"\b(?:amount|quantity|temperature|pressure|radius|diameter|length|width|height|"
    r"thickness|density|concentration|voltage|current|power|frequency|wavelength|"
    r"flow rate|rate|speed|capacity|energy|force|mass|volume|distance|position|"
    r"percentage|ratio|proportion|range|set point|setpoint|q\s*cop(?:max)?)\b",
    re.I,
)

_UNIT_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:mm|cm|m|µm|um|nm|kg|g|mg|ml|l|v|a|w|hz|khz|"
    r"mhz|pa|kpa|mpa|°c|celsius|kelvin|%|percent)\b|"
    r"\b(?:millimeter|millimeters|meter|meters|kelvin|celsius|percent)\b)",
    re.I,
)

_MEASURE_RE = re.compile(
    r"\b(?:measure|measured|measuring|measurement|determine|determining|"
    r"calculate|calculated|calculating|define|defined|derive|derived|"
    r"using|based on|according to|in terms of|ratio|relative to)\b",
    re.I,
)

_VAGUE_PATTERNS = (
    (r"\bhigh[- ]quality\b", 1.4),
    (r"\b(?:optimized|optimised)\b", 1.1),
    (r"\b(?:suitable|appropriate|desired|preferable)\b", 0.8),
    (r"\bsubstantially\b", 0.8),
    (r"\bapproximately|approximate|about\b", 0.9),
    (r"\bat or near\b", 1.0),
    (r"\bnear(?:ly)?\b", 0.7),
    (r"\b(?:rapidly|quickly|efficiently|effectively)\b", 0.7),
    (r"\bas needed\b", 1.0),
    (r"\bto a desired degree\b", 1.1),
    (r"\b(?:and the like|or the like)\b", 0.9),
    (r"\b(?:relatively|generally|typically|normally)\b", 0.5),
    (r"\b(?:any suitable|任意)\b", 1.0),
)

_HARDWARE_RE = re.compile(
    r"\b(?:apparatus|device|system|processor|controller|computer|server|sensor|"
    r"detector|valve|reed valve|tab|chamber|micro[- ]?lens|matrix|display|"
    r"surface|pump|housing|substrate|layer|electrode|circuit|conduit|channel|"
    r"opening|shaft|spring|nozzle|injector|reservoir|membrane|module|unit|"
    r"assembly|plate|element|polypeptide|composition|fuel injector)\b",
    re.I,
)

_MATERIAL_RE = re.compile(
    r"\b(?:steel|stainless steel|aluminum|aluminium|silicon|polymer|plastic|"
    r"ceramic|glass|rubber|carbon|alloy|copper|gold|silver|metal|oxide|"
    r"semiconductor|resin|fiber|fibre|fabric)\b",
    re.I,
)

_PHYSICAL_ACTION_RE = re.compile(
    r"\b(?:inject|injected|injecting|mount|mounted|mounting|couple|coupled|"
    r"coupling|connect|connected|connecting|transmit|transmitted|transmitting|"
    r"receive|received|receiving|operate|operating|control|controlling|"
    r"generate|generating|store|storing|encode|encoding|switch|switching|"
    r"apply|applied|applying|measure|measuring|detect|detecting|open|opening|"
    r"close|closed|closing|rotate|rotating|heat|heating|cool|cooling|"
    r"compress|compressing|flow|flowing|position|positioning|form|forming)\b",
    re.I,
)

_MENTAL_RE = re.compile(
    r"\b(?:mentally|mental|think|thinking|reason|reasoning|imagine|imagining|"
    r"remember|remembering|believe|believing|opine|opining|judge|judging|"
    r"decide|deciding|consider|considering|interpret|interpreting|"
    r"understand|understanding|cognit|contemplate|contemplating)\b",
    re.I,
)

_ABSTRACT_ACTION_RE = re.compile(
    r"\b(?:select|selecting|identify|identifying|determine|determining|"
    r"analyze|analyzing|classify|classifying|evaluate|evaluating|"
    r"compare|comparing|calculate|calculating|infer|inferring|"
    r"categorize|categorizing)\b",
    re.I,
)

_MATH_RE = re.compile(
    r"(?:[=<>≤≥+\-*/^]|"
    r"\b(?:ratio|proportion|percentage|sum|product|minimum|maximum|min|max|"
    r"average|mean|equal|equals|less than|greater than|at least|at most|"
    r"range|calculated|formula|equation|algorithm|mathematical|"
    r"q\s*cop(?:max)?)\b)",
    re.I,
)

_APPLICATION_RE = re.compile(
    r"\b(?:fuel|injection|valve|sensor|processor|server|payment|card|"
    r"chamber|temperature|display|optical|imaging|composition|polypeptide|"
    r"device|apparatus|system|substrate|layer|fluid|material|"
    r"transmit|inject|control|measure|detect|heat|cool|compress|"
    r"manufacture|operate)\b",
    re.I,
)

_STRUCTURAL_REL_RE = re.compile(
    r"\b(?:defines?|extends?|through|within|between|adjacent|contiguous|"
    r"coupled|mounted|disposed|positioned|surrounding|embedded|formed|"
    r"attached|connected|faces?|toward|away from|overlapping|underlying)\b",
    re.I,
)

_CONCEPTUAL_RE = re.compile(
    r"\b(?:concept|principle|property|characteristic|quality|function|"
    r"configured to|adapted to|capable of|means for|optimized|desired|"
    r"abstract|general|any suitable)\b",
    re.I,
)

_FUNCTION_RE = re.compile(
    r"\b(?:configured to|adapted to|operable to|capable of|for\s+\w+ing|"
    r"perform|provide|enable|allow|control|process|transmit|receive|"
    r"generate|store|measure|detect|inject|adjust|select|determine|"
    r"calculate|switch|open|close|support|reduce|increase|maintain)\b",
    re.I,
)

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _clean(text):
    try:
        return text if isinstance(text, str) else ""
    except Exception:
        return ""


def _tokens(text):
    return re.findall(r"[A-Za-z0-9µ°]+", text.lower())


def _count(pattern, text):
    try:
        return len(pattern.findall(text))
    except Exception:
        return 0


def _clip(value, low=0.0, high=10.0):
    try:
        return float(max(low, min(high, value)))
    except Exception:
        return 0.0


def pb09(text: str) -> float:
    try:
        text = _clean(text).strip()
        if not text:
            return 0.0

        parameters = list(_PARAM_RE.finditer(text))
        math_terms = _count(_MATH_RE, text)
        if not parameters and not math_terms:
            tokens = len(_tokens(text))
            return _clip(2.0 + min(3.0, tokens / 18.0))

        supported = 0.0
        for match in parameters:
            start = max(0, match.start() - 75)
            end = min(len(text), match.end() + 75)
            window = text[start:end]
            evidence = 0.0
            if _UNIT_RE.search(window):
                evidence += 1.0
            if _MEASURE_RE.search(window):
                evidence += 0.8
            if re.search(r"\b(?:by|via|using|with)\s+(?:a|an|the)?\s*\w+", window, re.I):
                evidence += 0.5
            supported += min(1.0, evidence)

        denominator = len(parameters) + (1.0 if math_terms else 0.0)
        completeness = supported / denominator if denominator else 0.0
        if math_terms:
            relation_support = min(
                1.0,
                (_count(_UNIT_RE, text) + _count(_MEASURE_RE, text)) / math_terms,
            )
            completeness = 0.7 * completeness + 0.3 * relation_support

        return _clip(1.0 + 9.0 * completeness)
    except Exception:
        return 0.0


def pb10(text: str) -> float:
    try:
        text = _clean(text).strip()
        if not text:
            return 0.0

        tokens = _tokens(text)
        if not tokens:
            return 0.0

        vague = 0.0
        for pattern, weight in _VAGUE_PATTERNS:
            vague += weight * len(re.findall(pattern, text, re.I))

        concrete = (
            1.4 * _count(_NUMBER_RE, text)
            + 1.2 * _count(_UNIT_RE, text)
            + 0.9 * _count(_HARDWARE_RE, text)
            + 0.7 * _count(_MATERIAL_RE, text)
            + 0.5 * _count(_STRUCTURAL_REL_RE, text)
        )
        specificity = min(1.0, concrete / (1.0 + len(tokens) / 10.0))
        vague_density = vague / (1.0 + len(tokens) / 14.0)
        score = 10.0 * (0.42 + 0.58 * specificity) * math.exp(-0.22 * vague_density)
        return _clip(score)
    except Exception:
        return 0.0


def pb12(text: str) -> float:
    try:
        text = _clean(text).strip()
        if not text:
            return 0.0

        hardware = _count(_HARDWARE_RE, text)
        physical = _count(_PHYSICAL_ACTION_RE, text)
        mental = _count(_MENTAL_RE, text)
        abstract = _count(_ABSTRACT_ACTION_RE, text)
        numeric = _count(_NUMBER_RE, text) + _count(_UNIT_RE, text)

        physical_evidence = 1.8 * hardware + 1.1 * physical + 0.35 * numeric
        mental_evidence = 2.5 * mental + 0.55 * abstract
        if physical_evidence == 0 and mental_evidence == 0:
            return _clip(min(4.0, len(_tokens(text)) / 12.0))

        score = 10.0 * physical_evidence / (
            physical_evidence + mental_evidence + 1.5
        )
        if hardware and physical:
            score += 0.8
        return _clip(score)
    except Exception:
        return 0.0


def pb15(text: str) -> float:
    try:
        text = _clean(text).strip()
        if not text:
            return 0.0

        math_count = _count(_MATH_RE, text)
        application = _count(_APPLICATION_RE, text)
        hardware = _count(_HARDWARE_RE, text)
        physical = _count(_PHYSICAL_ACTION_RE, text)
        abstract = _count(_CONCEPTUAL_RE, text) + _count(_ABSTRACT_ACTION_RE, text)

        if math_count:
            practical = 1.2 * application + 1.0 * hardware + 0.8 * physical
            bare = 1.4 * abstract + 0.45 * math_count
            score = 10.0 * practical / (practical + bare + 1.0)
            if practical >= 2.0:
                score += 1.0
            return _clip(score)

        practical = application + hardware + physical
        if practical == 0:
            return _clip(5.0 - min(3.5, 0.6 * abstract))
        return _clip(6.0 + 4.0 * practical / (practical + abstract + 4.0))
    except Exception:
        return 0.0


def pb17(text: str) -> float:
    try:
        text = _clean(text).strip()
        if not text:
            return 0.0

        hardware = _count(_HARDWARE_RE, text)
        materials = _count(_MATERIAL_RE, text)
        relations = _count(_STRUCTURAL_REL_RE, text)
        numbers = _count(_NUMBER_RE, text)
        units = _count(_UNIT_RE, text)
        conceptual = _count(_CONCEPTUAL_RE, text)

        structural = (
            1.4 * hardware
            + 1.6 * materials
            + 0.75 * relations
            + 0.6 * numbers
            + 0.7 * units
        )
        score = 10.0 * structural / (structural + 1.3 * conceptual + 2.0)
        return _clip(score)
    except Exception:
        return 0.0


def pb18(text: str) -> float:
    try:
        text = _clean(text).strip()
        if not text:
            return 0.0

        hardware_matches = list(_HARDWARE_RE.finditer(text))
        hardware = len(hardware_matches)
        boundaries = _count(_STRUCTURAL_REL_RE, text)
        functions = _count(_FUNCTION_RE, text)

        paired = 0
        for match in hardware_matches:
            start = max(0, match.start() - 90)
            end = min(len(text), match.end() + 130)
            if _FUNCTION_RE.search(text[start:end]):
                paired += 1

        if hardware == 0:
            return _clip(1.2 + 1.1 * min(functions, 4))

        structure_strength = 1.0 - math.exp(
            -0.34 * (hardware + 0.8 * boundaries)
        )
        pairing_strength = 1.0 - math.exp(
            -0.45 * (paired + 0.35 * boundaries)
        )
        unsupported_function = max(0.0, functions - paired - boundaries / 2.0)
        score = 10.0 * (
            0.34 * structure_strength + 0.66 * pairing_strength
        ) * math.exp(-0.12 * unsupported_function)
        return _clip(score)
    except Exception:
        return 0.0


REGISTRY = {
    "pb09": pb09,
    "pb10": pb10,
    "pb12": pb12,
    "pb15": pb15,
    "pb17": pb17,
    "pb18": pb18,
}