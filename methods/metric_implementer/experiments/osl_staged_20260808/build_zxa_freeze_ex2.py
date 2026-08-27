"""1c-v2 replication arms (ICL reconciliation, user directive 2026-08-06):
  exemplars_fmt  — SAME 8 items as the `exemplars` arm but in canonical interleaved demo
                   format ("Text: ... -> Yes/No"), testing whether the weak corpus-exemplar
                   channel was a formatting artifact.
  exemplars_shuf — SAME 8 items, labels deterministically shuffled (~half wrong): the exact
                   Min-et-al.-2022 control. If y(shuf) ~ y(exemplars), the exemplar lift is
                   format/anchoring, not label content.
Only the 14 crowd-decisive bases (those with corpus exemplars). Reads the selected item
indices straight from freeze_zxa_ex_humor_v1.json so items are IDENTICAL across arms.
"""
import hashlib
import json
import sys

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer import config as cfgmod

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"

v1 = json.load(open(f"{OM}/freeze_zxa_ex_humor_v1.json"))
cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
texts, _ = _load_texts("humor", 360, cfg)
probes = texts[60:360]

entries = []
for e in v1["metrics"]:
    if e["zxa"]["arm"] != "exemplars":
        continue
    base, cls = e["zxa"]["base"], e["zxa"]["class"]
    idx = e["zxa"]["exemplar_idx"]
    # recover pos/neg split from the rubric ordering: first 4 listed = satisfy block.
    # safer: reparse from the rubric text itself
    rub = e["rubric"]
    sat_block = rub.split("Examples that satisfy this criterion:")[1].split(
        "Examples that do NOT satisfy it:")[0]
    not_block = rub.split("Examples that do NOT satisfy it:")[1]
    pos_texts = [l[2:].strip() for l in sat_block.strip().split("\n") if l.startswith("- ")]
    neg_texts = [l[2:].strip() for l in not_block.strip().split("\n") if l.startswith("- ")]
    items = [(t, "Yes") for t in pos_texts] + [(t, "No") for t in neg_texts]
    # deterministic interleave for fmt arm
    order = sorted(range(len(items)),
                   key=lambda i: hashlib.md5((base + str(i)).encode()).hexdigest())
    fmt = "\n\n".join("Text: %s\nDoes it satisfy the criterion? %s" % (items[i][0], items[i][1])
                      for i in order)
    rub_fmt = ("%s\nDecide whether a text satisfies this criterion. Worked examples:\n\n%s"
               % (base, fmt))
    # shuffled labels: same items, permute labels by rotating them one position in `order`
    labs = [items[i][1] for i in order]
    labs_shuf = labs[1:] + labs[:1]
    shuf = "\n\n".join("Text: %s\nDoes it satisfy the criterion? %s" % (items[i][0], l)
                       for i, l in zip(order, labs_shuf))
    rub_shuf = ("%s\nDecide whether a text satisfies this criterion. Worked examples:\n\n%s"
                % (base, shuf))
    n_wrong = sum(1 for a, b_ in zip(labs, labs_shuf) if a != b_)
    for arm, rub_ in (("exemplars_fmt", rub_fmt), ("exemplars_shuf", rub_shuf)):
        entries.append({"name": f"{base}||{arm}", "kind": f"{cls}|{arm}", "rubric": rub_,
                        "criteria": [],
                        "zxa": {"base": base, "arm": arm, "class": cls,
                                "exemplar_idx": idx, "mismatch_src": None,
                                "n_wrong_labels": n_wrong if arm == "exemplars_shuf" else 0}})

out = {"meta": dict(v1["meta"], replication_freeze="ex2",
                    note="fmt = canonical interleaved demos; shuf = Min-et-al shuffled-label "
                         "control, same items; mask exemplar_idx at fit time"),
       "metrics": entries}
json.dump(out, open(f"{OM}/freeze_zxa_ex2_humor_v1.json", "w"), indent=1)
wrongs = [e["zxa"]["n_wrong_labels"] for e in entries if e["zxa"]["arm"] == "exemplars_shuf"]
print(f"entries: {len(entries)} ({len(entries)//2} bases); shuf wrong-label counts: {wrongs}")
