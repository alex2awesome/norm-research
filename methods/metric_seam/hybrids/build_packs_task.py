"""Task-generic improver packs (wave-2 protocol) for any surveyed task.

Usage: python3 build_packs_task.py <task>
Split: 150 train / 100 test by sorted datapoint_id, seed 7 (same recipe as the PR harness,
computed over THIS task's items). Packs -> outputs/metric_seam_pilot/tasks/<task>/improver_packs/.
Skips degenerate/unreliable aspects (needs rel1 > 0.05 and >=40 train cells).
"""
import json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman, attenuation_ceiling  # noqa: E402
from gen_improver_pack import CONTRACT                  # noqa: E402

HAZARDS = {
    "creative_writing": (
        "Corpus notes: WritingPrompts-style short fiction scraped from the web; texts may "
        "contain prompt echoes, author notes ('[WP]', edit notes, thank-yous), markdown "
        "artifacts (*, **, &gt;), and hard line-wrapping; '[...]' marks an elided middle. "
        "Keyword presence is usually a WEAK proxy for judged literary quality — decouple "
        "surface markers from quality in your predicate; the judge scores craft, not topic."),
    "math": (
        "Corpus notes: Math StackExchange answers with LaTeX-ish markup ($...$, $$...$$, "
        "\\begin{align}, broken delimiters, unicode math); code blocks and quoted question "
        "text appear; '[...]' marks an elided middle. Keyword presence is often a PROXY for "
        "the judged quality — decouple presence from quality."),
    "humor": (
        "Corpus notes: reddit-style jokes and short humorous pieces; texts are SHORT (median "
        "~500 chars — a complete joke, not a truncated document), may carry title echoes, "
        "quotation marks, edit notes, and occasional offensive content (the criterion, not "
        "your taste, decides how that scores). Beware: topic words (blonde, bar, lawyer) and "
        "profanity are WEAK proxies for judged craft; structure (setup/punchline placement, "
        "economy, escalation) is usually the real signal. Many criteria are about stage "
        "performance — for text, score the textual analog the criterion admits, and rely on "
        "your LLM fields for constructs code cannot reach."),
    "legal_title_vii": (
        "Corpus notes: ex-ante Title VII case-facts narratives (plaintiff/defendant facts as "
        "pleaded, BEFORE any outcome; no ruling text present). Statutory quantities (dates, "
        "day-counts, employee counts, dollar amounts) are extractable and genuinely "
        "load-bearing for the mechanical criteria — real date arithmetic beats keyword "
        "counting there. For doctrinal-construct criteria (pretext, inference of "
        "discrimination, hostile environment) beware: legal TERM presence is a weak proxy "
        "for the construct being FACTUALLY PRESENT; use LLM fields for the construct and "
        "code for the quantities and temporal structure."),
}


def split_task_ids(items):
    ids = sorted(x["datapoint_id"] for x in items)
    random.Random(7).shuffle(ids)
    return set(ids[:150]), set(ids[150:])


def main():
    task = sys.argv[1]
    OUT = ROOT / "outputs/metric_seam_pilot/tasks" / task
    V2 = ROOT / "runs/validity_full/v2" / task
    PACKS = OUT / "improver_packs"
    PACKS.mkdir(exist_ok=True)

    items_l = json.load(open(OUT / "items.json"))
    items = {x["datapoint_id"]: x["ctext"] for x in items_l}
    train, _ = split_task_ids(items_l)
    aspects = {x["aspect_id"]: x for x in json.load(open(V2 / "aspects.json"))}
    code = json.load(open(OUT / "code_scores.json"))

    p1, p2 = {}, {}
    for line in open(OUT / "results.jsonl"):
        r = json.loads(line)
        if not isinstance(r["score"], int) or r["channel"] not in ("pass1", "pass2"):
            continue
        (p1 if r["channel"] == "pass1" else p2).setdefault(
            r["aspect_id"], {})[r["datapoint_id"]] = r["score"]

    built = 0
    for aid in json.load(open(OUT / "aspects_used.json")):
        both = [d for d in p1.get(aid, {}) if d in p2.get(aid, {})]
        if len(both) < 100:
            print(f"{aid}: skip (n={len(both)})")
            continue
        rel1 = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        if rel1 != rel1 or rel1 <= 0.05:
            print(f"{aid}: skip (rel1={rel1})")
            continue
        judge = {d: (p1[aid][d] + p2[aid][d]) / 2 / 10.0 for d in both}

        best_fl, best_rho = None, -2
        for fl in ["v0_keyword", "v1_structure", "v2_holistic"]:
            col = code.get(f"{aid}_{fl}") or {}
            sel = [d for d in train if d in judge and col.get(d) is not None]
            if len(sel) < 30:
                continue
            r = spearman([col[d] for d in sel], [judge[d] for d in sel])
            if r == r and r > best_rho:
                best_fl, best_rho = fl, r
        if best_fl is None:
            print(f"{aid}: skip (no runnable flavor)")
            continue
        base_col = code.get(f"{aid}_{best_fl}") or {}
        rows = sorted(((d, judge[d]) for d in train if d in judge), key=lambda t: t[1])
        n = len(rows)
        if n < 40:
            print(f"{aid}: skip (n_train={n})")
            continue
        strata = rows[:10] + rows[n // 2 - 5: n // 2 + 5] + rows[-10:]
        a = aspects[aid]
        pack = {"aspect_id": aid, "criterion_name": a["name"],
                "criterion_description": a["description"], "contract": CONTRACT,
                "known_failure_pattern": HAZARDS.get(task, ""),
                "judge_reliability": round(rel1, 3),
                "attenuation_ceiling": round(
                    attenuation_ceiling(max(0.0, min(1.0, rel1)), 2), 3),
                "baseline_flavor": best_fl, "baseline_train_rho": round(best_rho, 3),
                "baseline_source": (V2 / "codegen_claude" / f"{aid}_{best_fl}.py"
                                    ).read_text()[:3500],
                "train_examples": [{
                    "datapoint_id": d, "judge_score_0_1": round(sc, 2),
                    "baseline_code_score": (round(base_col[d], 2)
                                            if base_col.get(d) is not None else None),
                    "text_excerpt_head": items[d][:2200],
                    "text_excerpt_tail": items[d][-1200:]} for d, sc in strata]}
        json.dump(pack, open(PACKS / f"{aid}.json", "w"), indent=1)
        built += 1
        print(f"{aid}: pack ok (rel1={rel1:.2f}, base {best_fl} train rho={best_rho:+.2f}) "
              f"{a['name'][:45]}")
    print(f"{built} packs -> {PACKS}")


if __name__ == "__main__":
    main()
