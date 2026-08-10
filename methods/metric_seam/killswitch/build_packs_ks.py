"""Improver packs for the kill-switch hybrid round (Arm S synthetic channels).

Mirrors gen_packs_v2.py exactly: stratified 30 train examples (bottom/mid/top 10 by channel
score), best description-compiled flavor as baseline, same CONTRACT, same corpus-hazards
note. Channel = mean(pass1,pass2)/10 from channels_synth.jsonl. Split = harness seed-7
150 train / 100 test over the same v1 250 ids. Truth is never included (blinding).
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from harness import split_ids, spearman   # noqa: E402
from gen_improver_pack import CONTRACT    # noqa: E402
from certificates import attenuation_ceiling  # noqa: E402

OUT = ROOT / "outputs/metric_seam_pilot/killswitch"
PACKS = OUT / "improver_packs"
PACKS.mkdir(exist_ok=True)

GENERIC_FAILURE = (
    "Known corpus hazards (from wave-1 adjudication): scraped pages include NON-releases "
    "(news articles, blogs, nav chrome) — the judge scores those ~0 on most criteria; mojibake "
    "curly quotes and OCR-dropped '@' are common (ops.normalize first); contact/boilerplate "
    "blocks sit near the END; keyword presence is often a PROXY for the judged quality — "
    "decouple presence from quality in your predicate.")


def load_channel():
    p1, p2 = {}, {}
    for line in open(OUT / "channels_synth.jsonl"):
        r = json.loads(line)
        d = p1 if r["channel"] == "pass1" else p2
        d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]
    combined, rel = {}, {}
    for aid in p1:
        both = [d for d in p1[aid] if d in p2.get(aid, {})]
        rel[aid] = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        combined[aid] = {d: (p1[aid][d] + p2[aid].get(d, p1[aid][d])) / 2 / 10.0
                         for d in p1[aid]}
    return combined, rel


def main():
    judge, rel = load_channel()
    train, _ = split_ids()
    items = {x["datapoint_id"]: x["ctext"]
             for x in json.load(open(ROOT / "outputs/metric_seam_pilot/v1/items_v1.json"))}
    plants = {p["aspect_id"]: p for p in json.load(open(OUT / "plants.json"))}
    cs = json.load(open(OUT / "code_scores_ks.json"))["scores"]

    for aid in sorted(plants):
        ch = judge.get(aid)
        if not ch:
            print(f"{aid}: no channel yet, skip")
            continue
        best_fl, best_rho = None, -2
        for fl in ["v0_keyword", "v1_structure", "v2_holistic"]:
            col = cs.get(f"{aid}_{fl}") or {}
            sel = [d for d in train if d in ch and col.get(d) is not None]
            r = spearman([col[d] for d in sel], [ch[d] for d in sel])
            if r == r and r > best_rho:
                best_fl, best_rho = fl, r
        base_col = cs.get(f"{aid}_{best_fl}") or {}
        rows = sorted(((d, ch[d]) for d in train if d in ch), key=lambda t: t[1])
        n = len(rows)
        strata = rows[:10] + rows[n // 2 - 5: n // 2 + 5] + rows[-10:]
        r1 = max(0.0, min(1.0, rel[aid]))
        pack = {"aspect_id": aid, "criterion_name": plants[aid]["name"],
                "criterion_description": plants[aid]["description"],
                "contract": CONTRACT, "known_failure_pattern": GENERIC_FAILURE,
                "judge_reliability": round(rel[aid], 3),
                "attenuation_ceiling": round(attenuation_ceiling(r1, 2), 3),
                "baseline_flavor": best_fl,
                "baseline_train_rho": round(best_rho, 3),
                "baseline_source": (ROOT / "methods/metric_seam/killswitch/codegen" /
                                    f"{aid}_{best_fl}.py").read_text()[:3500],
                "train_examples": [{
                    "datapoint_id": d, "judge_score_0_1": round(sc, 2),
                    "baseline_code_score": (round(base_col[d], 2)
                                            if base_col.get(d) is not None else None),
                    "text_excerpt_head": items[d][:2200],
                    "text_excerpt_tail": items[d][-1200:]} for d, sc in strata]}
        json.dump(pack, open(PACKS / f"{aid}.json", "w"), indent=1)
        print(f"{aid}: pack ok (rel1={rel[aid]:.3f}, base {best_fl} train rho={best_rho:.3f})")


if __name__ == "__main__":
    main()
