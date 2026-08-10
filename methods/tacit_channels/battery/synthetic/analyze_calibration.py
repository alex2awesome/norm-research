"""Score the GLM calibration cycle against the mechanical oracles.

Per (construct, variant), with truth = oracle (E/G tiers):
  tf           accuracy vs oracle                    (instrument sanity; E-tier bar .90)
  negated      accuracy vs NOT oracle                (concept-level NOT compliance)
  exclusion_v1 accuracy vs NOT oracle                (deployed wording — defect cost)
  exclusion_fx accuracy vs NOT oracle                (W1c candidate wording)
  composed     accuracy vs AND(oracles)              (composition compliance)
  confidence   mean/std + scale-use gate; on E-tier the stated answer is the oracle's,
               so confidence SHOULD be high and non-degenerate

Also emits the instrument-level verdicts: the exclusion defect cost = acc(fx) − acc(v1)
on E-tier, and the leak analog (agreement of exclusion answers WITH tf answers = failed
inversion) for continuity with the W1a statistic.

  python -m methods.tacit_channels.battery.synthetic.analyze_calibration \
      --results outputs/tacit_channels/battery_calibration/results_cycle1.jsonl \
      --out outputs/tacit_channels/battery_calibration/report_cycle1.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from methods.tacit_channels.battery.stats import confidence_scale_valid
from methods.tacit_channels.battery.synthetic.constructs import (
    COMPOSED_PAIRS, CONSTRUCTS, ITEMS, oracle_vector,
)

YESNO = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)
INT = re.compile(r"\b(\d{1,3})\b")


def parse_yesno(raw: str):
    m = YESNO.search(raw or "")
    return None if not m else m.group(1).upper() == "YES"


def parse_conf(raw: str):
    for m in INT.finditer(raw or ""):
        v = int(m.group(1))
        if 0 <= v <= 100:
            return float(v)
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by = defaultdict(dict)   # (cid, variant) -> {item_idx: raw}
    for line in open(args.results):
        r = json.loads(line)
        cid, variant = r["aspect_id"].rsplit("::", 1)
        by[(cid, variant)][int(r["datapoint_id"])] = r["raw"]

    def acc(cid, variant, truth):
        raws = by.get((cid, variant), {})
        pairs = [(parse_yesno(raws[j]), truth[j]) for j in raws if j < len(truth)]
        pairs = [(p, t) for p, t in pairs if p is not None]
        if len(pairs) < 20:
            return None, len(pairs)
        return float(np.mean([p == t for p, t in pairs])), len(pairs)

    report = {"constructs": {}, "instrument_verdicts": {}}
    for cid, c in CONSTRUCTS.items():
        vec = oracle_vector(cid)
        row = {"tier": c["tier"]}
        if vec is not None:
            inv = [not x for x in vec]
            row["tf_acc"], row["n"] = acc(cid, "tf", vec)
            row["negated_acc"], _ = acc(cid, "negated", inv)
            row["exclusion_v1_acc"], _ = acc(cid, "exclusion_v1", inv)
            row["exclusion_fx_acc"], _ = acc(cid, "exclusion_fx", inv)
        # leak analog: exclusion answers agreeing with TF ANSWERS (not oracle)
        tf_raws = by.get((cid, "tf"), {})
        tf_ans = {j: parse_yesno(v) for j, v in tf_raws.items()}
        for ev in ("exclusion_v1", "exclusion_fx"):
            raws = by.get((cid, ev), {})
            same = [parse_yesno(raws[j]) == tf_ans.get(j) for j in raws
                    if parse_yesno(raws[j]) is not None and tf_ans.get(j) is not None]
            row[f"{ev}_leak_rate"] = float(np.mean(same)) if len(same) >= 20 else None
        conf_raws = by.get((cid, "confidence"), {})
        if conf_raws:
            cv = np.array([parse_conf(conf_raws[j]) for j in sorted(conf_raws)])
            row["conf_mean"] = float(np.nanmean(cv))
            row["conf_n_unique"] = int(len(np.unique(cv[np.isfinite(cv)])))
        report["constructs"][cid] = row

    for a, b in COMPOSED_PAIRS:
        va, vb = oracle_vector(a), oracle_vector(b)
        truth = [x and y for x, y in zip(va, vb)]
        cacc, n = acc(f"{a}&&{b}", "composed", truth)
        report["constructs"][f"{a}&&{b}"] = {"tier": "E-composed",
                                             "composed_acc": cacc, "n": n}

    e = [r for r in report["constructs"].values() if r["tier"] == "E"]
    mean_of = lambda k: (float(np.mean([r[k] for r in e if r.get(k) is not None]))
                         if any(r.get(k) is not None for r in e) else None)
    verdicts = {
        "E_tf_acc_mean": mean_of("tf_acc"),
        "E_negated_acc_mean": mean_of("negated_acc"),
        "E_exclusion_v1_acc_mean": mean_of("exclusion_v1_acc"),
        "E_exclusion_fx_acc_mean": mean_of("exclusion_fx_acc"),
        "exclusion_defect_cost": (
            mean_of("exclusion_fx_acc") - mean_of("exclusion_v1_acc")
            if mean_of("exclusion_fx_acc") is not None
            and mean_of("exclusion_v1_acc") is not None else None),
        "tf_bar_pass": (mean_of("tf_acc") or 0) >= 0.90,
    }
    # confidence scale-use across all constructs (one matrix)
    conf_rows = []
    for cid in CONSTRUCTS:
        raws = by.get((cid, "confidence"), {})
        if raws:
            conf_rows.append([parse_conf(raws.get(j, "")) for j in range(len(ITEMS))])
    if conf_rows:
        verdicts["confidence_scale_gate"] = confidence_scale_valid(
            np.array(conf_rows), min_unique=8, min_cell_std=2.0)
    report["instrument_verdicts"] = verdicts

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(verdicts, indent=2))
    for cid, r in report["constructs"].items():
        print(cid, json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in r.items() if k != "tier"}))


if __name__ == "__main__":
    main()
