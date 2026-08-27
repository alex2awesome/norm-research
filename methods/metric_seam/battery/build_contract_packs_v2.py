"""WS1a: build contract-authoring packs (v2) for the E2 PANEL EXTENSION (16 new criteria:
peer_review a100/a114/a29/a172/a14/a128/a163/a214 + math a234/a180/a24/a174/a228/a72/a0/a84).

Same output shape as build_contract_packs.py (task, aspect_id, panel_role, r_hyb_frozen,
r_base_frozen, author_instructions, criterion_name, criterion_description, contract,
judge_reliability, attenuation_ceiling) but:
  - reads outputs/metric_seam_pilot/battery/effort_ladder/panel_extension_v2.json instead of
    panel_freeze.json (this is the panel EXTENSION, not the original frozen panel)
  - AUTHOR_INSTRUCTIONS upgraded to v2, folding in the E2-sweep instrument lessons (frozen
    2026-07-11, notes/2026-07-10__seam-agentic-program-runbook.md): probe-flaw / corpus-absent
    axes (a204/a222), mention-only gameability (PR a41/humor a342), authoring artifacts
    (a216), contract-blindness / L-channel (a216, a60, a90, a351, a198)
  - math: sourced directly from tasks/math/improver_packs/<aid>.json (already present, 8/8)
  - peer_review: tasks/peer_review/improver_packs/ did not exist before this run. Rebuilt via
    the task-generic recipe (methods/metric_seam/hybrids/build_packs_task.py peer_review),
    which produced 6/8 target aspects (a100, a29, a172, a14, a128, a163). The remaining two
    (a114 n=87, a214 n=42) failed that script's n>=100 quality gate for PACK-BUILDING
    (reliability too noisy to trust for an autonomous improver fleet) but are reconstructed
    here directly from the same primary source (results.jsonl pass1/pass2, same spearman +
    attenuation_ceiling formula) with the sub-100 n flagged honestly in `source_note` — NOT
    fabricated, just lower-confidence.

Usage: python3 build_contract_packs_v2.py
-> outputs/metric_seam_pilot/battery/effort_ladder/contract_packs_v2/<task>__<aid>.json
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
from certificates import spearman, attenuation_ceiling  # noqa: E402
from gen_improver_pack import CONTRACT  # noqa: E402

BASE = ROOT / "outputs/metric_seam_pilot"
EL = BASE / "battery/effort_ladder"

PACK_DIR = {
    "math": BASE / "tasks/math/improver_packs",
    "peer_review": BASE / "tasks/peer_review/improver_packs",
}

# task label used in panel_extension_v2.json -> actual task id (tasks/<task>/ dir)
GROUP_TO_TASK = {"peer_review": "peer_review", "math_expansion": "math"}

# Reconstructed directly from tasks/peer_review/results.jsonl pass1/pass2 (same recipe as
# build_packs_task.py: spearman(pass1,pass2) over datapoints with int scores in both passes,
# attenuation_ceiling(rel1, k=2)), for the 2 aspects that fell under build_packs_task.py's
# n>=100 pack-building gate. criterion_name/description from runs/validity_full/v2/peer_review
# /aspects.json (same source build_packs_task.py reads).
PEER_REVIEW_LOWN_OVERRIDE = {
    "a114": {
        "criterion_name": "Sample, Population and Inclusion Criteria",
        "criterion_description": (
            "Participant inclusion/exclusion, sampling versus population coverage, sample "
            "size, and study-design inclusion choices."),
        "judge_reliability": 0.792,
        "attenuation_ceiling": 0.94,
        "source_note": "reliability/ceiling computed on n=87 pass1/pass2 pairs (<100 quality"
                        " gate used by build_packs_task.py) -- lower confidence, not fabricated.",
    },
    "a214": {
        "criterion_name": "Quotation Usage",
        "criterion_description": (
            "Correct use of quotations and quotation marks for definitions and discussed "
            "terms."),
        "judge_reliability": 0.554,
        "attenuation_ceiling": 0.844,
        "source_note": "reliability/ceiling computed on n=42 pass1/pass2 pairs (<100 quality"
                        " gate used by build_packs_task.py) -- lower confidence, not fabricated.",
    },
}


def recompute_lown_override():
    """Sanity-check PEER_REVIEW_LOWN_OVERRIDE against results.jsonl directly (no caching)."""
    p1, p2 = {}, {}
    for line in open(BASE / "tasks/peer_review/results.jsonl"):
        r = json.loads(line)
        if not isinstance(r["score"], int) or r["channel"] not in ("pass1", "pass2"):
            continue
        (p1 if r["channel"] == "pass1" else p2).setdefault(
            r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    for aid, ov in PEER_REVIEW_LOWN_OVERRIDE.items():
        both = [d for d in p1.get(aid, {}) if d in p2.get(aid, {})]
        rel1 = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        ceil = attenuation_ceiling(max(0.0, min(1.0, rel1)), 2)
        assert abs(rel1 - ov["judge_reliability"]) < 0.005, (aid, rel1, ov)
        assert abs(ceil - ov["attenuation_ceiling"]) < 0.005, (aid, ceil, ov)
        assert len(both) < 100, (aid, len(both))


AUTHOR_INSTRUCTIONS_V2 = """\
You are authoring a FROZEN CONSTRUCT CONTRACT for one evaluative criterion. This contract
guards a multi-round compiler crew against construct drift: every candidate implementation
must keep passing it, at every effort rung. Author from the criterion definition ONLY —
never from item labels or score patterns.

v2 additions (frozen 2026-07-11 from the E2 instrument-defect sweep — read carefully, these
are not stylistic suggestions, they close specific holes that produced unusable contracts):

(a) CORPUS-PRESENT axes only. Every probe must test an axis that actually occurs in this
    task's corpus and that the JUDGE is known to discriminate on — not a plausible-sounding
    axis you invented from the definition alone. (Lesson: math a204's probes differed only by
    LaTeX newline placement; a broadened corpus scan found ZERO genuine wraps in 150 docs —
    the axis was corpus-absent. math a222's axis was corpus-present but UNREPRESENTABLE: text
    extraction destroys layout before the axis survives to string form. Both wasted a full
    compiler cell on an unwinnable contract.) For EACH probe, state which real corpus
    phenomenon it represents in a new `corpus_phenomenon` field — a one-line pointer concrete
    enough that someone could go find 2-3 matching documents, not a restatement of the why.

(b) MENTION-ONLY near-misses required. At least 1-2 of your 4-6 probes must pair a genuine
    positive against a text_neg that MENTIONS the same trigger vocabulary/keywords as the
    positive but does not actually perform/satisfy the construct — not just an unrelated or
    opposite-topic negative. (Lesson: PR a41 and humor a342 were both killed because their
    contract let regexes fire full-strength on mention-only prose — identical score to
    genuine occurrences — a gap only a mention-vs-use probe would have caught.) Tag such
    probes with `"probe_type": "mention_only"` in the probe object; tag ordinary contrastive
    probes `"probe_type": "genuine_contrast"`.

(c) NO AUTHORING ARTIFACTS. Do not let your text_pos examples share an incidental surface
    marker (a name-drop, a fixed phrase, a boilerplate opener) that isn't actually part of
    the construct — a candidate could pattern-match the artifact and pass every probe while
    scoring the wrong thing (the a216 lesson: a spurious shared marker across positives is
    as dangerous as a missing negative). Vary surface form across your positives; if two
    probes need a shared setup for realism, vary the specific wording/entities each time.

(d) CODE-VISIBLE vs L-CHANNEL. Prefer probes that exercise something a deterministic code
    channel could plausibly detect (structure, extractable fields, countable patterns) over
    probes that only a free-text LLM-extracted field can resolve. Tag each probe's primary
    channel with `"channel": "CODE"` or `"channel": "L"`. If the construct is INHERENTLY
    L-channel — i.e. you cannot write even one CODE-primary probe for it without trivializing
    the construct — say so EXPLICITLY in boundary_notes (e.g. "this construct is judged from
    free-text semantic content the code channel cannot see; expect the code path to be near-
    zero and rely on the LLM_FIELDS extractor"). This is not a failure to avoid — 5 of the
    first 20 E2 cells were exactly this (contract-blind at-ceiling controls, e.g. PR a2,
    humor a351/a90/a216, math a198) and mislabeling them as tacitness failures wasted the
    verifier pass. Naming it up front lets the compiler crew route straight to the L-channel
    instead of re-discovering blindness empirically.

Return JSON: {"construct_definition": <verbatim criterion description>,
 "cf_probes": [{"text_pos": ..., "text_neg": ..., "why": ...,
   "corpus_phenomenon": <one-line pointer to the real corpus phenomenon this axis represents>,
   "probe_type": "mention_only" | "genuine_contrast",
   "channel": "CODE" | "L"} x4-6]  (minimal pairs: the two texts differ ONLY in the construct;
   short, realistic for this task's genre; at least 1-2 probes must be probe_type
   "mention_only" per (b) above),
 "discrimination_checks": {"min_std": 0.05, "max_frac_at_mode": 0.85},
 "boundary_notes": <2-3 sentences: what neighboring constructs this one is NOT; if the
   construct is inherently L-channel, say so explicitly here per (d) above>}"""


def build_pack(task, group, role, r):
    aid = r["aspect"]
    src = PACK_DIR[task] / f"{aid}.json"
    pack = {"task": task, "aspect_id": aid, "panel_role": role,
            "r_hyb_frozen": r["r_hyb"], "r_base_frozen": r["r_base"],
            "author_instructions": AUTHOR_INSTRUCTIONS_V2}
    if src.exists():
        ip = json.load(open(src))
        pack.update({k: ip.get(k) for k in
                     ("criterion_name", "criterion_description", "contract",
                      "judge_reliability", "attenuation_ceiling")})
        return pack, "sourced"
    if task == "peer_review" and aid in PEER_REVIEW_LOWN_OVERRIDE:
        ov = PEER_REVIEW_LOWN_OVERRIDE[aid]
        pack.update({
            "criterion_name": ov["criterion_name"],
            "criterion_description": ov["criterion_description"],
            "contract": CONTRACT,
            "judge_reliability": ov["judge_reliability"],
            "attenuation_ceiling": ov["attenuation_ceiling"],
            "source_note": ov["source_note"],
        })
        return pack, "reconstructed-lown"
    pack["MISSING_SOURCE"] = str(src)
    return pack, "missing"


def main():
    recompute_lown_override()
    panel = json.load(open(EL / "panel_extension_v2.json"))
    out_dir = EL / "contract_packs_v2"
    out_dir.mkdir(exist_ok=True)
    report = {"sourced": [], "reconstructed-lown": [], "missing": []}
    for group, roles in panel["panel_extension"].items():
        task = GROUP_TO_TASK[group]
        for role, rows in roles.items():
            for r in rows:
                pack, status = build_pack(task, group, role, r)
                aid = r["aspect"]
                json.dump(pack, open(out_dir / f"{task}__{aid}.json", "w"), indent=1)
                report[status].append(f"{task}__{aid}")
    n = sum(len(v) for v in report.values())
    print(f"packs: {n} total -> {out_dir}")
    for status, items in report.items():
        print(f"  {status}: {len(items)} {items}")
    json.dump(report, open(out_dir / "_build_report.json", "w"), indent=1)
    print(f"build report -> {out_dir / '_build_report.json'}")


if __name__ == "__main__":
    main()
