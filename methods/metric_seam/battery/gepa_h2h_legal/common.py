"""Shared constants + helpers for the GEPA-H2H harness (seam note Sec 9).

Arm G = a SINGLE Gemma-31B scoring prompt (one call/doc, 0-10 int score), GEPA-optimized
against TRAIN judge verdicts only (own-verdict reconstruction, never external labels).
Arm H = the certified hybrid program (code + <=2 Gemma field extractions), already frozen
in outputs/metric_seam_pilot/battery/agentic_cert.json.

This module centralizes the constants every other script needs so they can't drift out of
sync with each other (prompt-format footer, document marker token, dev-set seed/size, the
12-criterion list, state.json path). Only stdlib + battery_common are imported here.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
BASE = ROOT / "outputs/metric_seam_pilot"

sys.path.insert(0, str(ROOT / "methods/metric_seam/battery"))
from battery_common import load_ctx  # noqa: E402,F401  (re-exported for the other scripts)

# ---- the 12 criteria (battery/agentic_cert.json CANDS minus math.a132) --------------------
# LEGAL variant of the GEPA-H2H harness (arm G only, for the S4 waterfall F extension).
# 9 legal_title_vii criteria spanning a spread of code-baseline r_base (.0 -> .71) so the
# code<->prompt seam is visible across the tacit range. base + ceiling come from the legal
# hybrid_gate_report.json; this harness produces the missing arm-G held-out rho.
CRITERIA = [
    ("legal_title_vii", "a8"), ("legal_title_vii", "a3"), ("legal_title_vii", "a26"),
    ("legal_title_vii", "a10"), ("legal_title_vii", "a39"), ("legal_title_vii", "a15"),
    ("legal_title_vii", "a0"), ("legal_title_vii", "a13"), ("legal_title_vii", "a28"),
]

TASK_DOMAIN = {
    "legal_title_vii": "U.S. Title VII employment-discrimination court opinion",
}

# arm H candidate program filenames (mirrors cert_agentic.py CANDS, same 12 rows)
AGENTIC_FNAME = {
    ("press_releases", "a119"): "a119_agentic.py",
    ("press_releases", "a115"): "a115_agentic.py",
    ("press_releases", "a87"): "a87_agentic.py",
    ("creative_writing", "a90"): "a90cw_agentic.py",
    ("creative_writing", "a72"): "a72cw_agentic.py",
    ("creative_writing", "a99"): "a99cw_agentic.py",
    ("creative_writing", "a342"): "a342cw_agentic.py",
    ("math", "a198"): "a198math_agentic.py",
    ("math", "a42"): "a42math_agentic.py",
    ("humor", "a351"): "a351humor_agentic.py",
    ("humor", "a135"): "a135humor_agentic.py",
    ("humor", "a153"): "a153humor_agentic.py",
}

PROGDIR = {"press_releases": "programs_v2", "creative_writing": "programs_cw",
           "math": "programs_math", "humor": "programs_humor",
           "legal_title_vii": "programs_legal"}
AGENTIC_DIR = ROOT / "methods/metric_seam/hybrids/programs_agentic"

# ---- dev-set sampling (own-verdict reconstruction; TRAIN only, never TEST) -----------------
SEED = 13
N_DEV = 40

# ---- prompt-template contract -------------------------------------------------------------
# The proposer (GLM) may freely rewrite the BODY text stored in state.json, but MUST preserve
# this literal marker exactly once (build_round.py substitutes the document text at that spot).
# Plain token (no braces) so no str.format()/f-string escaping headaches anywhere downstream.
DOC_MARKER = "<<<DOCUMENT>>>"
MAXCHARS = 6000  # matches reference_gepa_pr.py's MAXCHARS convention

# The reply-format instruction is NEVER handed to the proposer and NEVER stored in state.json's
# body text -- it is appended by build_round.py at prompt-assembly time, every round, so a GEPA
# rewrite can never break the scorer's parse regex (gemma_score_v1.py: `SCORE:\s*(NA|\d+)`).
# NOTE: this deviates from the seam-note literal wording ("reply with ONLY the integer") --
# see NOTES.md / README.md "deviations" for why (must match the actual offline scorer in repo).
FOOTER = "\n\nReply with exactly one line: SCORE: <integer 0-10>"

SEED_TEMPLATE = """You are scoring a single {domain} on one specific quality criterion.

Criterion: {criterion_name} -- {criterion_description}

Document:
{marker}

Score how well this document satisfies the criterion above, from 0 (fails completely) to \
10 (fully satisfies it). Judge the criterion only; ignore unrelated aspects of quality."""


def crit_key(task, aid):
    return f"{task}.{aid}"


def improver_pack_path(task, aid):
    d = "v2" if task == "press_releases" else f"tasks/{task}"
    return BASE / d / "improver_packs" / f"{aid}.json"


def load_improver_pack(task, aid):
    return json.load(open(improver_pack_path(task, aid)))


STATE_PATH = HERE / "state.json"


def load_state():
    return json.load(open(STATE_PATH))


def save_state(state):
    json.dump(state, open(STATE_PATH, "w"), indent=1)


def build_doc_prompt(body_template, text):
    """Assemble the final per-document prompt: GLM-revisable BODY (with DOC_MARKER
    substituted) + the fixed, never-revised reply-format FOOTER."""
    t = text[:MAXCHARS] + ("\n...[truncated]" if len(text) > MAXCHARS else "")
    if DOC_MARKER not in body_template:
        # defensive: should never happen (propose.py validates this before saving state.json)
        body_template = body_template.rstrip() + "\n\nDocument:\n" + DOC_MARKER
    return body_template.replace(DOC_MARKER, t) + FOOTER


# ---- local GLM key (propose.py only) -------------------------------------------------------
# Mirrors the z.ai key-search convention in methods/metric_implementer/backends.py._read_key,
# but this harness talks to the API directly (dependency-light: stdlib only) rather than
# importing that module. Fill in / reorder if your local key file differs.
KEY_PATHS = [
    "~/.z-ai-api-key-alexander-spangher.txt",
    "~/.z-ai-api-key-spangher.txt",
    "~/.z-ai-api-key.txt",
]
ZAI_URL = "https://api.z.ai/api/anthropic/v1/messages"
GLM_MODEL = "glm-5.2"
