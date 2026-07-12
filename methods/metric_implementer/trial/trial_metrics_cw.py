"""Creative-writing trial metric: deliberately THICK, with a deliberately crude seed.

``distinctive_voice`` is chosen as a canonically tacit criterion (the rater-cognition
literature's standing example: raters recognize voice instantly and articulate it
post-hoc). The prompt seed is crude on purpose — the optimizer must earn any fidelity —
and the code seed is the classic crude proxy stack (sentence-length variance +
type-token ratio + punctuation diversity), expected to diverge from the judge: that
divergence IS the thickness evidence.
"""

from __future__ import annotations

from ..artifact import MetricArtifact

_VOICE_DESC = (
    "The story is told in a distinctive narrative voice: a recognizable personality "
    "lives in the prose itself — diction, rhythm, attitude, perspective — rather than "
    "generic, interchangeable storytelling. A reader shown a second story by the same "
    "narrator would recognize it."
)

_VOICE_PROMPT_SEED = """Score whether this story has a distinctive narrative voice. \
1.0 = strongly distinctive, instantly recognizable voice; 0.0 = generic, flat, \
interchangeable prose; intermediate values for partial cases."""

_VOICE_CODE_SEED = '''
import re
import statistics


def score(text: str) -> float:
    """Crude voice proxy: sentence-length variance + type-token ratio + punctuation
    diversity, each squashed to [0,1] and averaged."""
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sents) < 3:
        return None
    lens = [len(s.split()) for s in sents]
    var = statistics.pstdev(lens) / (statistics.mean(lens) + 1e-9)      # burstiness
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 50:
        return None
    ttr = len(set(words)) / len(words) ** 0.72                          # length-corrected
    punct = len(set(re.findall(r"[;:\\-—…\\"\']", text))) / 7.0
    return max(0.0, min(1.0, 0.4 * min(var, 1.5) / 1.5 + 0.4 * min(ttr, 1.2) / 1.2
                        + 0.2 * min(punct, 1.0)))
'''


# ---- thin + mid rungs (added 2026-06-12 for the E7 pilot ladder) -------------------------

_ADVERB_DESC = ("The prose exercises adverb restraint: actions and dialogue carry their "
                "own weight instead of leaning on -ly adverbs ('she said angrily'), a "
                "standing craft norm (King, Leonard).")

_ADVERB_PROMPT_SEED = """Score this story's adverb restraint. 1.0 = verbs and dialogue \
carry tone with almost no -ly adverb propping; 0.0 = -ly adverbs constantly do the work \
('walked slowly', 'said angrily')."""

_ADVERB_CODE_SEED = '''import re

def score(text):
    words = re.findall(r"[a-zA-Z']+", text)
    if len(words) < 80:
        return None
    ly = [w for w in words if w.lower().endswith("ly") and len(w) > 4]
    rate = len(ly) / len(words)
    return max(0.0, min(1.0, 1.0 - rate / 0.04))
'''

_SHOW_DESC = ("The story shows emotion and stakes through concrete sensory detail and "
              "action rather than telling the reader abstract emotional states.")

_SHOW_PROMPT_SEED = """Score how much this story SHOWS rather than TELLS. 1.0 = emotion \
and stakes conveyed through concrete sensory detail, action, and dialogue; 0.0 = the \
narrator names abstract emotional states ('she was sad', 'it was terrifying') instead."""

_SHOW_CODE_SEED = '''import re

TELL = [r"\\bwas\\s+(sad|happy|angry|afraid|scared|terrified|nervous|excited|upset)",
        r"\\bfelt\\s+\\w+", r"\\bvery\\s+\\w+"]
SENSE = [r"smell|scent|reek", r"cold|warm|hot|chill", r"whisper|creak|hum|roar",
         r"taste|bitter|sweet", r"flicker|glint|shadow|glow"]

def score(text):
    words = len(text.split())
    if words < 80:
        return None
    tell = sum(len(re.findall(p, text, re.I)) for p in TELL)
    sense = sum(len(re.findall(p, text, re.I)) for p in SENSE)
    raw = (sense - 1.5 * tell) / (words / 200.0)
    return max(0.0, min(1.0, 0.5 + raw / 10.0))
'''


def trial_metrics_cw():
    """thin (adverb_restraint) -> mid (show_dont_tell) -> thick (distinctive_voice)."""
    spec = [
        ("adverb_restraint", "Adverb restraint", _ADVERB_DESC,
         _ADVERB_PROMPT_SEED, _ADVERB_CODE_SEED, ["blank_lines"]),
        ("show_dont_tell", "Shows rather than tells", _SHOW_DESC,
         _SHOW_PROMPT_SEED, _SHOW_CODE_SEED, ["blank_lines"]),
        ("distinctive_voice", "Distinctive narrative voice", _VOICE_DESC,
         _VOICE_PROMPT_SEED, _VOICE_CODE_SEED, ["blank_lines"]),
    ]
    out = []
    for mid, name, desc, prompt, code, inv in spec:
        out.append((
            MetricArtifact(metric_id=mid, kind="prompt", body=prompt.strip(),
                           name=name, description=desc, invariances=inv),
            MetricArtifact(metric_id=mid, kind="code", body=code,
                           name=name, description=desc, invariances=inv),
        ))
    return out
