"""Generate per-aspect materials for the reflective hybrid-improver agents (round 0).

Each pack: criterion, contract + ops docs, stratified TRAIN examples (judge score + best
baseline code score + canonical-text excerpt), baseline program source, known failure pattern.
Improvers see TRAIN only; the gate runs on held-out TEST.
"""
import json, pathlib, statistics as st
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from harness import load_judge, split_ids, OUT, ROOT

V2 = ROOT / "runs/validity_full/v2/press_releases"
PACKS = OUT / "improver_packs"
PACKS.mkdir(exist_ok=True)

TARGETS = {
    "a80": {"baseline": "v1_structure",
            "failure": ("Adjudicated v0: regex credits page-chrome contact strings (nav 'Contact "
                        "Us', subscription emails) on NON-releases; misses real media-contact "
                        "blocks with OCR-dropped '@', spaced phone digits ('202 219 7499'), or "
                        "email-free named blocks. Contact blocks usually sit at the END.")},
    "a86": {"baseline": "v0_keyword",
            "failure": ("Adjudicated v0: presence-proxy. Fires on scare quotes, terms-of-use "
                        "strings, JSON/HTML escape artifacts; misses real executive/analyst "
                        "quotes behind mojibake curly quotes or 'said'-verb-free attribution. "
                        "Judge scores QUALITY (insightful, excerptable, role-appropriate), "
                        "not presence. A CF gate will inject a generic excited-CEO quote — "
                        "your score must NOT rise on it.")},
    "a110": {"baseline": "v1_structure",
             "failure": ("Boilerplate ('About X' + footer metadata) repeats nearly verbatim "
                         "across releases from the same issuer — the retrieval op can detect "
                         "shared-footer mass directly.")},
    "a105": {"baseline": "v1_structure",
             "failure": ("Expected A-layer: readability formulas track sentence mechanics, but "
                         "the judge scores audience-appropriate clarity/jargon. Try your best "
                         "shot anyway — an honest failure is a finding.")},
}

CONTRACT = """You must write a single Python module implementing a HYBRID metric channel:

  LLM_FIELDS = {"field_name": "one-line instruction for an LLM extractor", ...}  # OPTIONAL, max 2
  def score(text: str, extracted: dict, ops) -> float   # return in [0.0, 1.0]

- `text` is the canonical document text (may contain scraped chrome/mojibake).
- `extracted[field_name]` is the LLM extractor's SHORT answer (<=20 words) for this document,
  or "" if it answered NONE. Only fields you declare in LLM_FIELDS exist. Use them for
  THICK-INPUT grounding (things regex can't see), keep the PREDICATE in code.
- `ops` provides:
    ops.normalize(text) -> str        # fixes mojibake/smart quotes/nbsp  [computation op]
    ops.extract_dates(text) -> list   # surface date strings              [computation op]
    ops.sent_stats(text) -> (n_sent, mean_words_per_sent, frac_long_words) [computation op]
    ops.retrieve_similar(text, k=5, exclude_id=None) -> [(similarity, datapoint_id)]
                                      # TF-IDF over the 5k-corpus         [EVIDENCE op]
- Deterministic, stdlib + provided ops only (re, math, statistics ok; no network, no files).
- Target: reproduce the JUDGE's ranking of documents on this criterion (Spearman on held-out).
- Robustness notes: scraped corpus contains non-releases (blogs, news articles, nav chrome);
  mojibake is common; contact/boilerplate blocks sit near the END of the text."""


def main():
    judge, _, _ = load_judge()
    train, _ = split_ids()
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(OUT / "items_v1.json"))}
    aspects = {x["aspect_id"]: x for x in json.load(open(V2 / "aspects.json"))}
    code = json.load(open(OUT / "code_scores_v1.json"))

    for aid, cfg in TARGETS.items():
        a = aspects[aid]
        base_col = code.get(f"{aid}_{cfg['baseline']}") or {}
        rows = [(d, judge[aid][d]) for d in train if d in judge.get(aid, {})]
        rows.sort(key=lambda t: t[1])
        n = len(rows)
        strata = rows[:10] + rows[n // 2 - 5: n // 2 + 5] + rows[-10:]
        examples = [{
            "datapoint_id": d, "judge_score_0_1": round(s, 2),
            "baseline_code_score": (round(base_col[d], 2)
                                    if base_col.get(d) is not None else None),
            "text_excerpt_head": items[d][:2200],
            "text_excerpt_tail": items[d][-1200:],
        } for d, s in strata]
        base_src = ""
        p = V2 / "codegen_claude" / f"{aid}_{cfg['baseline']}.py"
        if p.exists():
            base_src = p.read_text()[:4000]
        pack = {"aspect_id": aid, "criterion_name": a["name"],
                "criterion_description": a["description"],
                "contract": CONTRACT, "known_failure_pattern": cfg["failure"],
                "baseline_flavor": cfg["baseline"], "baseline_source": base_src,
                "train_examples": examples}
        json.dump(pack, open(PACKS / f"{aid}.json", "w"), indent=1)
        print(f"{aid}: pack with {len(examples)} train examples "
              f"(judge n_train={n})")


if __name__ == "__main__":
    main()
