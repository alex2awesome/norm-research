"""Build the E-S1 kill-switch: synthetic planted channels + Arm-J judge prompts + label chunks.

Phase 1 (no args): local truths (p901/902/903/906/907), calibrated 2-pass synthetic channels,
  plants.json, prompts_judge.jsonl (7 plants x 250 items x 2 passes, same T1/T2 as real
  surveys), and 10 label chunks for the Claude-labeled truths (p904 quoted speakers,
  p905 authenticity).
Phase 2 (--phase2): reads claude_labels.json {dpid: {"n_quoted_speakers": int,
  "authenticity_0_10": int}} -> appends p904/p905 truths + synthetic channels.

Everything agent-visible mirrors the real-survey formats exactly (blinding).
"""
import argparse, json, pathlib, random, statistics as st, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
sys.path.insert(0, str(ROOT / "methods/metric_seam/pilot"))
sys.path.insert(0, str(ROOT / "methods/metric_seam/hybrids"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from certificates import spearman                       # noqa: E402
from build_task import T1, T2, ROLE                     # noqa: E402
from plants import (PLANTS, REL_TARGET, TRUTH_TYPE, map_p904,   # noqa: E402
                    truth_p901_raw, truth_p902, truth_p907_raw)

V1 = ROOT / "outputs/metric_seam_pilot/v1"
OUT = ROOT / "outputs/metric_seam_pilot/killswitch"
OUT.mkdir(parents=True, exist_ok=True)


def decile_rank(vals):
    """value list -> 0-10 ints by rank (uniform spread, ties share)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0] * len(vals)
    for r, i in enumerate(order):
        out[i] = round(10 * r / max(1, len(vals) - 1))
    return out


def calibrated_passes(truth, target_rel, seed):
    """Two independent discretized noisy passes hitting spearman(p1,p2) ~= target_rel.
    Fixed base noise vectors; binary search on sigma (monotone in rel1)."""
    rng = random.Random(seed)
    n = len(truth)
    e1 = [rng.gauss(0, 1) for _ in range(n)]
    e2 = [rng.gauss(0, 1) for _ in range(n)]

    def passes(sig):
        p1 = [max(0, min(10, round(t + sig * e))) for t, e in zip(truth, e1)]
        p2 = [max(0, min(10, round(t + sig * e))) for t, e in zip(truth, e2)]
        return p1, p2

    lo, hi = 0.05, 12.0
    for _ in range(40):
        mid = (lo + hi) / 2
        p1, p2 = passes(mid)
        r = spearman(p1, p2)
        if r > target_rel:
            lo = mid
        else:
            hi = mid
    sig = (lo + hi) / 2
    p1, p2 = passes(sig)
    return p1, p2, sig, spearman(p1, p2)


def emit_channels(f, aid, ids, p1, p2):
    for dpid, a, b in zip(ids, p1, p2):
        f.write(json.dumps({"channel": "pass1", "aspect_id": aid,
                            "datapoint_id": dpid, "raw": "", "score": a}) + "\n")
        f.write(json.dumps({"channel": "pass2", "aspect_id": aid,
                            "datapoint_id": dpid, "raw": "", "score": b}) + "\n")


def build_channel(aid, truth_by_id, ids, fout, meta):
    truth = [truth_by_id[d] for d in ids]
    if len(set(truth)) < 3:
        print(f"  !! {aid}: DEGENERATE truth ({len(set(truth))} distinct) — inspect")
    tgt = REL_TARGET[aid]
    if tgt is None:                                     # p906 null: independent noise
        r1, r2 = random.Random(f"{aid}-1"), random.Random(f"{aid}-2")
        p1 = [r1.randint(0, 10) for _ in ids]
        p2 = [r2.randint(0, 10) for _ in ids]
        sig, rel = float("nan"), spearman(p1, p2)
    else:
        p1, p2, sig, rel = calibrated_passes(truth, tgt, seed=f"ks-{aid}")
    emit_channels(fout, aid, ids, p1, p2)
    meta[aid] = {"truth_type": TRUTH_TYPE[aid], "rel_target": tgt,
                 "sigma": None if sig != sig else round(sig, 3),
                 "rel1_achieved": round(rel, 3),
                 "truth_distinct": len(set(truth)),
                 "truth_mean": round(st.mean(truth), 2)}
    print(f"  {aid} [{TRUTH_TYPE[aid]}]: rel1={rel:.3f} (target {tgt}), "
          f"sigma={sig if sig == sig else 'NA'}, distinct={len(set(truth))}, "
          f"truth mean={st.mean(truth):.2f}")


def phase1():
    items = json.load(open(V1 / "items_v1.json"))
    ids = [x["datapoint_id"] for x in items]
    ctext = {x["datapoint_id"]: x["ctext"] for x in items}

    print("computing local truths ...")
    raw_dens = [truth_p901_raw(ctext[d]) for d in ids]
    truths = {"p901": dict(zip(ids, decile_rank(raw_dens))),
              "p902": {d: truth_p902(ctext[d]) for d in ids}}
    raw_len = [truth_p907_raw(ctext[d]) for d in ids]
    truths["p907"] = dict(zip(ids, decile_rank(raw_len)))

    print("computing p903 truth (evidence op: retrieval over 5k corpus) ...")
    from ops import Ops
    ops = Ops(corpus_path=str(ROOT / "runs/validity_full/v2/press_releases/datapoints.json"))
    maxsim = []
    for d in ids:
        top = ops.retrieve_similar(ctext[d], k=1, exclude_id=d)
        maxsim.append(top[0][0] if top else 0.0)
    truths["p903"] = dict(zip(ids, [10 - v for v in decile_rank(maxsim)]))
    truths["p906"] = {d: None for d in ids}             # null: no truth by design

    meta = {}
    with open(OUT / "channels_synth.jsonl", "w") as f:
        for aid in ["p901", "p902", "p903", "p906", "p907"]:
            tb = truths[aid] if aid != "p906" else {d: 5 for d in ids}
            build_channel(aid, tb, ids, f, meta)

    json.dump({"truths": truths, "meta": meta, "items": ids},
              open(OUT / "truth.json", "w"), indent=1)
    json.dump(PLANTS, open(OUT / "plants.json", "w"), indent=1)

    role, doctype = ROLE["press_releases"]
    n = 0
    with open(OUT / "prompts_judge.jsonl", "w") as f:
        for p in PLANTS:
            for d in ids:
                for ch, T in [("pass1", T1), ("pass2", T2)]:
                    f.write(json.dumps({
                        "channel": ch, "aspect_id": p["aspect_id"], "datapoint_id": d,
                        "prompt": T.format(role=role, doctype=doctype, name=p["name"],
                                           description=p["description"],
                                           text=ctext[d])}) + "\n")
                    n += 1
    print(f"prompts_judge.jsonl: {n} prompts (scope reused from v1)")

    chunks = OUT / "label_chunks"
    chunks.mkdir(exist_ok=True)
    for i in range(0, len(items), 25):
        json.dump([{"datapoint_id": x["datapoint_id"], "ctext": x["ctext"]}
                   for x in items[i:i + 25]],
                  open(chunks / f"chunk_{i // 25:02d}.json", "w"))
    print(f"label chunks: {len(range(0, len(items), 25))} x 25 docs")


def phase2():
    labels = json.load(open(OUT / "claude_labels.json"))
    tr = json.load(open(OUT / "truth.json"))
    ids = tr["items"]
    missing = [d for d in ids if d not in labels]
    if missing:
        print(f"WARNING: {len(missing)} items missing labels; first: {missing[:3]}")
    ids = [d for d in ids if d in labels]
    truths = {"p904": {d: map_p904(int(labels[d]["n_quoted_speakers"])) for d in ids},
              "p905": {d: max(0, min(10, int(labels[d]["authenticity_0_10"])))
                       for d in ids}}
    meta = tr["meta"]
    with open(OUT / "channels_synth.jsonl", "a") as f:
        for aid in ["p904", "p905"]:
            build_channel(aid, truths[aid], ids, f, meta)
    tr["truths"].update(truths)
    tr["meta"] = meta
    json.dump(tr, open(OUT / "truth.json", "w"), indent=1)
    print("phase 2 complete: p904/p905 channels appended")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2", action="store_true")
    a = ap.parse_args()
    phase2() if a.phase2 else phase1()
