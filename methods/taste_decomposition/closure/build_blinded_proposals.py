#!/usr/bin/env python3
"""Pool the round-1 Track-A and Track-B proposals, strip provenance, and emit the
blinded file that the independent routing auditor sees (prereg step 4).

Order is a stable sha256 sort over a fixed salt + criterion name -- no seeded
shuffle.  The blinded file carries ONLY a neutral id, the criterion name and its
scoring instruction: the authoring rationales are withheld because they name the
track's mindset outright ('shortcut channel', 'spurious').
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SALT = "layer3-closure-round1-blind"


def main():
    pool = []
    for track in ("a", "b"):
        d = json.loads((HERE / f"round1_track_{track}.json").read_text())
        for c in d["criteria"]:
            pool.append({"src_id": c["id"], "track": track.upper(), **c})

    pool.sort(key=lambda c: hashlib.sha256(f"{SALT}|{c['name']}".encode()).hexdigest())
    for k, c in enumerate(pool):
        c["blind_id"] = f"P{k + 1:02d}"

    (HERE / "round1_proposals_blinded.json").write_text(
        json.dumps(
            {
                "n": len(pool),
                "scale": "0-10 integer, NA if the text gives no evidence bearing on the criterion",
                "task": (
                    "Each entry is a candidate criterion for judging machine-learning "
                    "conference paper abstracts. Classify each as quality-relevant or "
                    "incidental."
                ),
                "criteria": [
                    {"id": c["blind_id"], "name": c["name"], "instruction": c["instruction"]}
                    for c in pool
                ],
            },
            indent=1,
        )
    )
    (HERE / "round1_proposals_provenance.json").write_text(
        json.dumps(
            [{"blind_id": c["blind_id"], "src_id": c["src_id"], "track": c["track"],
              "name": c["name"]} for c in pool],
            indent=1,
        )
    )
    print(f"wrote {len(pool)} blinded proposals")


if __name__ == "__main__":
    main()
