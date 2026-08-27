"""Round-1 feedback: top TRAIN disagreements between the h0 hybrid and the judge.
(TEST is never shown to improvers.)"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from harness import OUT, load_judge, load_scope, split_ids

ASPECTS = sys.argv[1:] or ["a80", "a110", "a105"]

def main():
    judge, _, _ = load_judge()
    train, _ = split_ids()
    in_scope, scope_scores = load_scope()
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(OUT / "items_v1.json"))}
    for aid in ASPECTS:
        hyb = json.load(open(OUT / f"hybrid_scores_{aid}_h0.json"))
        rows = [(d, hyb[d], judge[aid][d]) for d in train
                if d in judge.get(aid, {}) and hyb.get(d) is not None]
        rows.sort(key=lambda t: -abs(t[1] - t[2]))
        fb = [{"datapoint_id": d, "hybrid_score": round(h, 2),
               "judge_score": round(j, 2), "scope_score": scope_scores.get(d),
               "in_scope": d in in_scope,
               "text_head": items[d][:2500], "text_tail": items[d][-1500:]}
              for d, h, j in rows[:12]]
        json.dump(fb, open(OUT / "improver_packs" / f"{aid}_round1_feedback.json", "w"),
                  indent=1)
        print(f"{aid}: {len(fb)} disagreement cells "
              f"(max gap {abs(rows[0][1]-rows[0][2]):.2f})")

if __name__ == "__main__":
    main()
