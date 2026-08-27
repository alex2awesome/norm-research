"""WS1a WAVE 1: build contract-authoring PACKS (v3) for the panel-v3 census extension
(115 new criteria across 7 tasks: press_releases, creative_writing, math, humor,
legal_title_vii, peer_review, legal_ss_disability).

Same output shape as build_contract_packs_v2.py (task, aspect_id, panel_role, r_hyb_frozen,
r_base_frozen, author_instructions, criterion_name, criterion_description, contract,
judge_reliability, attenuation_ceiling) PLUS two v3 additions:

  - `genre_note`: one-line description of what this task's items actually ARE and what the
    dataset `judgement` label means (mandatory per the peer-review genre bug: contract
    authors previously mis-modeled peer_review items as full reviews when they are the
    reviewed papers' ABSTRACTS -- this field prevents that class of error recurring for
    every task, not just peer_review). Verified 2026-07-12 by reading 2-3 real items.json
    rows per task + the task's datasets/<name>/README.md Task section; see _build_report.json
    "genre_note_verification" for the datapoint_ids inspected.

  - `r_base_frozen`: panel_v3_census.json (unlike panel_freeze.json / panel_extension_v2.json)
    carries only {aspect, r_hyb, band} -- no r_base. This is not an oversight: the census
    header's own `coding_task` field ("joins on judge-pass completion (sk3)") says the code
    channel for these 115 criteria has not been built yet, so no hybrid r_base exists to
    report. Set to null with an explicit r_base_note rather than fabricating or silently
    dropping the field.

Sourcing (in priority order, per criterion):
  1. improver_packs/<aid>.json where they exist (task-specific dir, see PACK_DIR below) --
     copies criterion_name, criterion_description, contract, judge_reliability,
     attenuation_ceiling verbatim. These already satisfy build_packs_task.py's n>=100 pairs /
     rel1>0.05 quality gate.
  2. Else reconstruct directly from primary sources (aspects registry
     runs/validity_full/v2/<task>/aspects.json for name/description; results.jsonl pass1/pass2
     + certificates.spearman/attenuation_ceiling for judge_reliability/attenuation_ceiling,
     same recipe as build_packs_task.py) with any n<100 flagged in `source_note` -- NOT
     fabricated, just lower-confidence. This path is used for peer_review a186 (n=99, one
     pair short of the n>=100 gate used when tasks/peer_review/improver_packs/ was first
     built) and, if applicable, legal_ss_disability aids not covered by a fresh
     build_packs_task.py run (see below).
  3. Else MISSING_SOURCE, flagged not fabricated.

Task-specific sourcing notes:
  - press_releases has no tasks/press_releases/ dir (PR uses the older V1/V2 harness layout);
    its improver packs live flat at outputs/metric_seam_pilot/v2/improver_packs/<aid>.json and
    already cover all 14 new PR criteria 14/14.
  - creative_writing, math, humor, legal_title_vii: tasks/<task>/improver_packs/ already covers
    all new criteria (26/26, 17/17, 23/23, 14/14 respectively) -- no reconstruction needed.
  - peer_review: tasks/peer_review/improver_packs/ covers 7/8; a186 reconstructed (n=99).
  - legal_ss_disability: tasks/legal_ss_disability/improver_packs/ did NOT exist before this
    run (fleet plumbing -- items.json, results.jsonl, code_scores.json, aspects_used.json,
    field_results.jsonl, hybrid_gate_report.json, methods/metric_seam/hybrids/programs_ssdis/,
    runs/validity_full/v2/legal_ss_disability/{aspects.json,codegen_claude/} -- all verified
    PRESENT, no plumbing gap). Ran build_packs_task.py legal_ss_disability once (upstream,
    unmodified) which produced 16/20 aspect packs (a4/a5/a6/a12 failed its own gates: no
    runnable code flavor / n<100 / rel1 undefined); all 13 census-new aids landed inside the
    successful 16, so 13/13 sourced with no reconstruction needed.

Usage: python3 build_contract_packs_v3.py
-> outputs/metric_seam_pilot/battery/effort_ladder/contract_packs_v3/<task>__<aid>.json
"""
import glob
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
from certificates import spearman, attenuation_ceiling  # noqa: E402
from gen_improver_pack import CONTRACT  # noqa: E402

BASE = ROOT / "outputs/metric_seam_pilot"
EL = BASE / "battery/effort_ladder"
CENSUS_PATH = EL / "panel_v3_census.json"
OUT_DIR = EL / "contract_packs_v3"

PACK_DIR = {
    "press_releases": BASE / "v2/improver_packs",
    "creative_writing": BASE / "tasks/creative_writing/improver_packs",
    "math": BASE / "tasks/math/improver_packs",
    "humor": BASE / "tasks/humor/improver_packs",
    "legal_title_vii": BASE / "tasks/legal_title_vii/improver_packs",
    "peer_review": BASE / "tasks/peer_review/improver_packs",
    "legal_ss_disability": BASE / "tasks/legal_ss_disability/improver_packs",
}

# results.jsonl location for the direct-reconstruction fallback (task-generic layout; PR is
# never expected to hit this path since v2/improver_packs already covers it 14/14).
RESULTS_PATH = {t: BASE / "tasks" / t / "results.jsonl" for t in PACK_DIR if t != "press_releases"}
ASPECTS_PATH = {t: ROOT / "runs/validity_full/v2" / t / "aspects.json" for t in PACK_DIR}

# ---------------------------------------------------------------------------------------
# GENRE NOTES (mandatory, from the peer-review genre bug). One line per task, written after
# reading 2-3 real items.json rows (see _build_report.json genre_note_verification for the
# exact datapoint_ids inspected) + each task's datasets/<name>/README.md Task section.
GENRE_NOTES = {
    "press_releases": (
        "items are scraped press-release-body web pages (nav/footer chrome common, a minority "
        "are non-PR pages e.g. product/blog pages); the dataset `judgement` label is whether "
        "the release was picked up by a top-domain news outlet (newsworthiness) -- unrelated "
        "to the per-criterion rubric judge scores used for r_hyb/judge_reliability here."),
    "creative_writing": (
        "items are r/WritingPrompts (prompt, story) pairs -- ctext is the story text only, "
        "often carrying prompt echoes/author notes/markdown artifacts; `judgement` is a "
        "crowd-vote label (non-trivial reader upvotes vs not), not a curatorial/editorial "
        "quality label."),
    "math": (
        "items are Math Stack Exchange question+answer threads, ctext literally formatted "
        "'Question: ... Answer: ...' with LaTeX-ish markup; `judgement` is community answer "
        "quality (accepted-answer AND score>=3 vs score<=0), a crowd-revealed label, not the "
        "per-criterion judge scores here."),
    "humor": (
        "items are short (median ~500-char) reddit-sourced jokes/humorous texts, often with "
        "title echoes and edit notes; `judgement` is whether the piece was rewarded (upvoted) "
        "by its native taste community, source-specific reward signal, not the rubric judge."),
    "legal_title_vii": (
        "items are de-leaked, EX-ANTE Title VII employment-discrimination case-facts "
        "narratives (plaintiff/defendant facts as pleaded, BEFORE any ruling; no disposition "
        "text present); `judgement` is the eventual outcome (plaintiff win vs defendant win), "
        "held out of the facts text itself."),
    "peer_review": (
        "items are the reviewed papers' ABSTRACTS (not the reviews, not the full paper PDF "
        "text), typically 1-2K chars of abstract prose; `judgement` is accept/reject."),
    "legal_ss_disability": (
        "items are de-leaked, EX-ANTE Social Security disability-appeal case-facts narratives "
        "(claimant history/impairments/procedural posture as found, BEFORE any ruling; no "
        "disposition text present); `judgement` is the eventual outcome (remand-reverse vs "
        "affirm), held out of the facts text itself."),
}

GENRE_NOTE_VERIFICATION = {
    "press_releases": {"datapoint_ids_read": ["d03229", "d03532", "d00337"],
                        "source": "outputs/metric_seam_pilot/v1/items_v1.json + "
                                  "datasets/press-releases/README.md Task section"},
    "creative_writing": {"datapoint_ids_read": ["d03302", "d03603"],
                          "source": "outputs/metric_seam_pilot/tasks/creative_writing/items.json"
                                    " + datasets/creative-writing/README.md Task section"},
    "math": {"datapoint_ids_read": ["d03509", "d03828"],
             "source": "outputs/metric_seam_pilot/tasks/math/items.json + "
                       "datasets/math/README.md (stackexchange row)"},
    "humor": {"datapoint_ids_read": ["d03155", "d03445"],
              "source": "outputs/metric_seam_pilot/tasks/humor/items.json + "
                        "datasets/humor/README.md Task section"},
    "legal_title_vii": {"datapoint_ids_read": ["d00788", "d00861"],
                         "source": "outputs/metric_seam_pilot/tasks/legal_title_vii/items.json"
                                   " + datasets/legal-outcome-prediction/README.md (Title VII "
                                   "row)"},
    "peer_review": {"datapoint_ids_read": ["d03173", "d03465", "d00332"],
                     "source": "outputs/metric_seam_pilot/tasks/peer_review/items.json "
                               "(length ~1-2K chars, abstract-shaped prose; domain/venue "
                               "fields confirm paper-level not review-level) + "
                               "datasets/peer-review/README.md Task section"},
    "legal_ss_disability": {"datapoint_ids_read": ["d01730", "d00788"],
                             "source": "outputs/metric_seam_pilot/tasks/legal_ss_disability/"
                                       "items.json + datasets/legal-outcome-prediction/"
                                       "README.md (SS disability row)"},
}

# ---------------------------------------------------------------------------------------
AUTHOR_INSTRUCTIONS_V2_1 = """\
You are authoring a FROZEN CONSTRUCT CONTRACT for one evaluative criterion. This contract
guards a multi-round compiler crew against construct drift: every candidate implementation
must keep passing it, at every effort rung. Author from the criterion definition ONLY —
never from item labels or score patterns.

Before anything else, read this pack's `genre_note` field. It states what this task's items
actually ARE and what the dataset's `judgement` label means — get this wrong (e.g. treating
peer_review items as full reviews instead of the reviewed papers' abstracts) and every probe
you write will be genre-confused no matter how careful the construct logic is.

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

v2.1 addition (frozen 2026-07-12 from the WAVE 1 pack-build):

(e) GREP-VERIFY corpus_phenomenon, with a hit count. A `corpus_phenomenon` line is not
    credited unless it reports an ACTUAL grep/keyword-scan hit count against this task's
    ctext corpus — e.g. "grep-verified: 'closed-form' occurs in 3/250 items (d04991, d03446,
    d04628)" — not a plausible-sounding but unchecked pointer. An unverified pointer is
    exactly the corpus-absent failure mode (a) exists to catch, just relocated into the
    phenomenon field instead of the probe axis. Run the scan yourself before writing the
    field; report the count AND 1-2 real datapoint_ids, for both the genuine-positive
    phenomenon and (for mention_only probes) the near-miss phenomenon. This applies on top
    of, not instead of, (a)-(d) above: mention-only probes are still required at 1-2 of 4-6,
    channel tags are still required on every probe, positives still must not share an
    incidental surface marker, and an inherently-L construct must still say so explicitly in
    boundary_notes.

Return JSON: {"construct_definition": <verbatim criterion description>,
 "cf_probes": [{"text_pos": ..., "text_neg": ..., "why": ...,
   "corpus_phenomenon": <one-line pointer to the real corpus phenomenon this axis represents,
     GREP-VERIFIED with an explicit hit count per (e) above>,
   "probe_type": "mention_only" | "genuine_contrast",
   "channel": "CODE" | "L"} x4-6]  (minimal pairs: the two texts differ ONLY in the construct;
   short, realistic for this task's genre per genre_note above; at least 1-2 probes must be
   probe_type "mention_only" per (b) above),
 "discrimination_checks": {"min_std": 0.05, "max_frac_at_mode": 0.85},
 "boundary_notes": <2-3 sentences: what neighboring constructs this one is NOT; if the
   construct is inherently L-channel, say so explicitly here per (d) above>}"""


def existing_aids(dirpath):
    out = {}
    for f in glob.glob(os.path.join(dirpath, "*.json")):
        base = os.path.basename(f)[:-5]
        if "__" not in base:
            continue
        task, aid = base.split("__", 1)
        out.setdefault(task, set()).add(aid)
    return out


def compute_new_list(census):
    v1 = existing_aids(str(EL / "contracts"))
    v2 = existing_aids(str(EL / "contracts_v2"))
    new = {}
    reconciliation = {"n_total": 0, "n_already": 0, "n_new": 0}
    for task, items in census.items():
        have = v1.get(task, set()) | v2.get(task, set())
        rows = [it for it in items if it["aspect"] not in have]
        new[task] = rows
        reconciliation["n_total"] += len(items)
        reconciliation["n_already"] += len(items) - len(rows)
        reconciliation["n_new"] += len(rows)
    return new, reconciliation


_ASPECTS_CACHE = {}
_RESULTS_CACHE = {}


def load_aspects(task):
    if task not in _ASPECTS_CACHE:
        p = ASPECTS_PATH[task]
        _ASPECTS_CACHE[task] = ({x["aspect_id"]: x for x in json.load(open(p))}
                                 if p.exists() else {})
    return _ASPECTS_CACHE[task]


def load_pass_scores(task):
    if task not in _RESULTS_CACHE:
        p1, p2 = {}, {}
        rp = RESULTS_PATH.get(task)
        if rp and rp.exists():
            for line in open(rp):
                r = json.loads(line)
                if not isinstance(r["score"], int) or r["channel"] not in ("pass1", "pass2"):
                    continue
                (p1 if r["channel"] == "pass1" else p2).setdefault(
                    r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
        _RESULTS_CACHE[task] = (p1, p2)
    return _RESULTS_CACHE[task]


def reconstruct_from_primary(task, aid):
    """Direct reconstruction fallback when no improver_pack exists: aspects registry for
    name/description, results.jsonl pass1/pass2 for judge_reliability/attenuation_ceiling,
    generic CONTRACT template. Flags n<100 in source_note; does not fabricate."""
    aspects = load_aspects(task)
    a = aspects.get(aid)
    if a is None:
        return None, None, "no aspects.json entry"
    p1, p2 = load_pass_scores(task)
    both = [d for d in p1.get(aid, {}) if d in p2.get(aid, {})]
    if len(both) < 5:
        return None, None, f"no usable pass1/pass2 pairs (n={len(both)})"
    rel1 = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
    if rel1 != rel1:
        return None, None, "rel1 undefined (nan)"
    rel1c = max(0.0, min(1.0, rel1))
    ceil = attenuation_ceiling(rel1c, 2)
    fields = {
        "criterion_name": a["name"],
        "criterion_description": a["description"],
        "contract": CONTRACT,
        "judge_reliability": round(rel1, 3),
        "attenuation_ceiling": round(ceil, 3),
    }
    note = None
    if len(both) < 100:
        note = (f"reliability/ceiling computed on n={len(both)} pass1/pass2 pairs (<100 "
                f"quality gate used by build_packs_task.py) -- lower confidence, not "
                f"fabricated.")
    return fields, len(both), note


def build_pack(task, row):
    aid = row["aspect"]
    pack = {
        "task": task, "aspect_id": aid, "panel_role": row["band"],
        "r_hyb_frozen": row["r_hyb"], "r_base_frozen": None,
        "r_base_note": ("not available in panel_v3_census.json (census carries only "
                        "{aspect, r_hyb, band}); the census header's own coding_task note "
                        "('joins on judge-pass completion (sk3)') says the code channel for "
                        "these criteria has not been built yet, so no hybrid r_base exists "
                        "yet to report -- not fabricated."),
        "author_instructions": AUTHOR_INSTRUCTIONS_V2_1,
        "genre_note": GENRE_NOTES[task],
    }
    src = PACK_DIR[task] / f"{aid}.json"
    if src.exists():
        ip = json.load(open(src))
        pack.update({k: ip.get(k) for k in
                     ("criterion_name", "criterion_description", "contract",
                      "judge_reliability", "attenuation_ceiling")})
        return pack, "sourced", None
    fields, n, note = reconstruct_from_primary(task, aid)
    if fields is not None:
        pack.update(fields)
        if note:
            pack["source_note"] = note
            return pack, "reconstructed-lown", n
        return pack, "reconstructed", n
    pack["MISSING_SOURCE"] = f"{src} (also unreconstructable: {note})"
    return pack, "missing", None


def main():
    census_doc = json.load(open(CENSUS_PATH))
    census = census_doc["census"]
    new_lists, reconciliation = compute_new_list(census)

    print("=== count reconciliation vs census ===")
    print(f"census n_total={census_doc['n_total']} n_already_contracted="
          f"{census_doc['n_already_contracted']} n_new_contracts_needed="
          f"{census_doc['n_new_contracts_needed']}")
    print(f"computed  n_total={reconciliation['n_total']} n_already="
          f"{reconciliation['n_already']} n_new={reconciliation['n_new']}")
    assert reconciliation["n_new"] == census_doc["n_new_contracts_needed"], "RECONCILIATION MISMATCH"
    print("RECONCILED OK")
    print()

    OUT_DIR.mkdir(exist_ok=True)
    report = {
        "date": "2026-07-12", "source_census": str(CENSUS_PATH),
        "reconciliation": reconciliation,
        "genre_note_verification": GENRE_NOTE_VERIFICATION,
        "per_task": {},
        "plumbing_check": {
            "legal_ss_disability": (
                "VERIFIED PRESENT (no gap): outputs/metric_seam_pilot/tasks/"
                "legal_ss_disability/{items.json,results.jsonl,code_scores.json,"
                "aspects_used.json,field_results.jsonl,hybrid_gate_report.json} all exist; "
                "methods/metric_seam/hybrids/programs_ssdis/ (20 a*_h0.py fleet programs) "
                "exists; runs/validity_full/v2/legal_ss_disability/{aspects.json,"
                "codegen_claude/} exists. tasks/legal_ss_disability/improver_packs/ did NOT "
                "exist before this run -- built via unmodified "
                "methods/metric_seam/hybrids/build_packs_task.py legal_ss_disability, which "
                "produced 16/20 aspect packs (a4: no runnable code flavor; a5: n=21 pairs "
                "<100; a6: n=46 pairs <100; a12: rel1=nan). All 13 census-new aids for this "
                "task (a0,a1,a2,a3,a7,a8,a9,a10,a11,a13,a15,a16,a19) fall inside the "
                "successful 16, so 13/13 sourced with the standard n>=100 gate already "
                "satisfied -- no reconstruction / no flagging needed."),
        },
        "missing": [], "reconstructed_lown": [], "sourced": [],
    }

    total_built = 0
    for task, rows in new_lists.items():
        counts = {"n_new": len(rows), "sourced": 0, "reconstructed": 0,
                  "reconstructed_lown": 0, "missing": 0}
        for row in rows:
            pack, status, n = build_pack(task, row)
            aid = row["aspect"]
            out_path = OUT_DIR / f"{task}__{aid}.json"
            json.dump(pack, open(out_path, "w"), indent=1)
            total_built += 1
            if status == "sourced":
                counts["sourced"] += 1
                report["sourced"].append(f"{task}__{aid}")
            elif status == "reconstructed":
                counts["reconstructed"] += 1
            elif status == "reconstructed-lown":
                counts["reconstructed_lown"] += 1
                report["reconstructed_lown"].append(
                    {"id": f"{task}__{aid}", "n_pairs": n, "note": pack["source_note"]})
            else:
                counts["missing"] += 1
                report["missing"].append({"id": f"{task}__{aid}", "reason": pack["MISSING_SOURCE"]})
        report["per_task"][task] = counts
        print(f"{task}: {counts}")

    report["n_packs_built"] = total_built
    json.dump(report, open(OUT_DIR / "_build_report.json", "w"), indent=1)
    print()
    print(f"{total_built} packs -> {OUT_DIR}")
    print(f"build report -> {OUT_DIR / '_build_report.json'}")


if __name__ == "__main__":
    main()
