"""Wave-2 improver packs: 20 aspects, stratified TRAIN examples + best-flavor baseline."""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(ROOT / "methods/metric_seam/pilot"))
from harness import split_ids
from analyze_v2 import load_judge_v2, OUT, V1
from gen_improver_pack import CONTRACT

V2DIR = ROOT / "runs/validity_full/v2/press_releases"
PACKS = OUT / "improver_packs"
PACKS.mkdir(exist_ok=True)

GENERIC_FAILURE = (
    "Known corpus hazards (from wave-1 adjudication): scraped pages include NON-releases "
    "(news articles, blogs, nav chrome) — the judge scores those ~0 on most criteria; mojibake "
    "curly quotes and OCR-dropped '@' are common (ops.normalize first); contact/boilerplate "
    "blocks sit near the END; keyword presence is often a PROXY for the judged quality — "
    "decouple presence from quality in your predicate.")


def main():
    judge, _, _ = load_judge_v2()
    train, _ = split_ids()
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(V1 / "items_v1.json"))}
    aspects = {x["aspect_id"]: x for x in json.load(open(V2DIR / "aspects.json"))}
    seam = {r["aspect"]: r for r in json.load(open(OUT / "seam_table_v2.json"))}
    code = json.load(open(OUT / "code_scores_v2.json"))

    for aid in json.load(open(OUT / "wave2_aspects.json")):
        a, s = aspects[aid], seam.get(aid, {})
        fl = s.get("best_flavor") or "v0_keyword"
        base_col = code.get(f"{aid}_{fl}") or {}
        rows = sorted(((d, judge[aid][d]) for d in train if d in judge.get(aid, {})),
                      key=lambda t: t[1])
        n = len(rows)
        if n < 40:
            print(f"{aid}: SKIP (n_train={n})")
            continue
        strata = rows[:10] + rows[n // 2 - 5: n // 2 + 5] + rows[-10:]
        pack = {"aspect_id": aid, "criterion_name": a["name"],
                "criterion_description": a["description"], "contract": CONTRACT,
                "known_failure_pattern": GENERIC_FAILURE,
                "judge_reliability": s.get("rel1"),
                "attenuation_ceiling": s.get("ceiling"),
                "baseline_flavor": fl,
                "baseline_source": (V2DIR / "codegen_claude" / f"{aid}_{fl}.py"
                                    ).read_text()[:3500],
                "train_examples": [{
                    "datapoint_id": d, "judge_score_0_1": round(sc, 2),
                    "baseline_code_score": (round(base_col[d], 2)
                                            if base_col.get(d) is not None else None),
                    "text_excerpt_head": items[d][:2200],
                    "text_excerpt_tail": items[d][-1200:]} for d, sc in strata]}
        json.dump(pack, open(PACKS / f"{aid}.json", "w"), indent=1)
        print(f"{aid}: pack ok (rel1={s.get('rel1')}, ceiling={s.get('ceiling')})")


if __name__ == "__main__":
    main()
