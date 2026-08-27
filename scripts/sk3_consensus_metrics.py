"""Cross-source consensus metrics on the locked tau-0.825 clustering.

A cluster of 5 forms from ONE expert guide is intra-document repetition; 5
forms from 5 different guides is a genuine cross-source norm. Cluster size
alone conflates the two. Each form's provenance key is
`task::source_dir::source_file::idx`, so the source file separates them.

Per cluster: distinct source files among its members. Reports, per task, how
many multi-member clusters are single-source vs multi-source, and the concepts
with the widest source consensus (the strongest shared norms).
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"

import json
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
FORMS = WORK / "canon_all_real_forms.jsonl"
MATCH_OUT = WORK / "match_out"
OUT = MATCH_OUT / "metrics"

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]


def main():
    forms_by_task = defaultdict(list)
    for line in FORMS.open():
        r = json.loads(line)
        forms_by_task[r["task"]].append(r)

    summary = {}
    print(f"{'=' * 80}")
    print("CROSS-SOURCE CONSENSUS  (multi-member clusters only)")
    print(f"{'=' * 80}")
    print(f"{'task':<26}{'multi-cl':>9}{'1-src':>8}{'>=2src':>8}"
          f"{'>=3src':>8}{'>=5src':>8}{'maxsrc':>8}")
    all_concepts = []
    for task in TASKS:
        rows = forms_by_task[task]
        cl = json.loads((MATCH_OUT / f"clusters_{task}.json").read_text())
        members = defaultdict(list)
        for r in rows:
            members[cl[r["key"]]].append(r)
        rec = {"1src": 0, "ge2": 0, "ge3": 0, "ge5": 0, "multi": 0, "maxsrc": 0}
        for cid, ms in members.items():
            if len(ms) < 2:
                continue
            rec["multi"] += 1
            srcs = {m["key"].split("::")[2] for m in ms}
            ns = len(srcs)
            rec["maxsrc"] = max(rec["maxsrc"], ns)
            rec["1src"] += ns == 1
            rec["ge2"] += ns >= 2
            rec["ge3"] += ns >= 3
            rec["ge5"] += ns >= 5
            rep = Counter(m["canonical"] or "" for m in ms).most_common(1)[0][0]
            all_concepts.append((ns, len(ms), task, rep))
        summary[task] = rec
        print(f"{task:<26}{rec['multi']:>9}{rec['1src']:>8}{rec['ge2']:>8}"
              f"{rec['ge3']:>8}{rec['ge5']:>8}{rec['maxsrc']:>8}")

    tot_multi = sum(r["multi"] for r in summary.values())
    tot_1src = sum(r["1src"] for r in summary.values())
    print(f"\n{tot_1src}/{tot_multi} multi-member clusters "
          f"({tot_1src / tot_multi * 100:.0f}%) are single-source "
          f"(intra-document repetition, not cross-source consensus)")

    print(f"\n{'=' * 80}")
    print("WIDEST-CONSENSUS CONCEPTS  (most distinct expert sources agreeing)")
    print(f"{'=' * 80}")
    for ns, sz, task, rep in sorted(all_concepts, reverse=True)[:35]:
        print(f"  {ns:>3} sources ({sz:>3} forms) [{task[:14]:<14}] {rep[:74]}")

    (OUT / "consensus.json").write_text(json.dumps(summary, indent=1))
    top = [{"n_sources": ns, "n_forms": sz, "task": t, "rep": rep}
           for ns, sz, t, rep in sorted(all_concepts, reverse=True)[:200]]
    (OUT / "consensus_top_concepts.json").write_text(json.dumps(top, indent=1))
    print(f"\nwrote consensus.json + consensus_top_concepts.json -> {OUT}")


if __name__ == "__main__":
    main()
