#!/usr/bin/env python3
"""(V+A)_new criteria mining for the unified-X cells — sealed dual-track fleet
harness, the BBC/peer machinery generalized to three cells:
  mathse_bounty, so_bounty (CURATED: manual bounty award)
  so_accepted (VERDICT: asker accepted)

slice  (sk3): top-N |dense pct − articulated pct| disagreement rows, EVAL-split
        rows only (test split reserved for the gain readout — the FITMINE/MONITOR
        discipline adapted to the grouped-OOF frame).  Cards carry the answer text
        (truncated), both percentiles, NO labels; blindness regex enforced.
build  (laptop): one sealed prompt per (track, proposer) in scratch, per-proposer
        card-order salt.  Fleet = codex gpt-5.6-luna x4 + GLM-5.2 x4 (2-family
        floor, recorded; claude legs added as sealed CLI sessions when cap allows).
collect(laptop): the BBC tolerant parser verbatim (all four observed formats).

  python3 harness_unified.py slice   --cell mathse_bounty --round 1     (sk3)
  python3 harness_unified.py build   --cell mathse_bounty --round 1     (laptop)
  python3 harness_unified.py collect --cell mathse_bounty --round 1     (laptop)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TD = HERE.parents[1]
NR = TD.parents[1]
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/unified_closure")
K_A, K_B = 15, 10
N_SLICE = 60
CARD_CHARS = 700
FLEET = [("codex_luna", "a"), ("codex_luna", "b"), ("codex_luna", "c"),
         ("codex_luna", "d"), ("glm", "a"), ("glm", "b"), ("glm", "c"), ("glm", "d")]

CELLS = {
    "mathse_bounty": dict(
        oof="results/mathse_bounty_va_oof.npz",
        dense="datasets/math-se/mathse_bounty/dense_standard_mathse_bounty",
        pop="datasets/math-se/mathse_bounty/population.csv.gz",
        item="mathematics answer",
        corpus="answers posted to questions on a mathematics question-and-answer site, "
               "on questions where the asker attached a reputation bounty",
        construct="whether the bounty-setter chose to AWARD the bounty to this answer "
                  "by a deliberate manual decision (system auto-awards are excluded) — "
                  "a considered individual judgment of which answer most deserved the "
                  "reward, as opposed to the crowd's vote totals",
        bank_desc="a bank of 31 explicitly-worded quality criteria (rigor, clarity, "
                  "completeness, pedagogy families) plus surface features"),
    "so_bounty": dict(
        oof="results/so_bounty_va_oof.npz",
        dense="datasets/stackoverflow-votes/so_bounty/dense_standard_so_bounty",
        pop="datasets/stackoverflow-votes/so_bounty/population.csv.gz",
        item="programming answer",
        corpus="answers posted to python questions on a programming question-and-answer "
               "site, on questions where the asker attached a reputation bounty",
        construct="whether the bounty-setter chose to AWARD the bounty to this answer "
                  "by a deliberate manual decision (system auto-awards are excluded)",
        bank_desc="a bank of 39 explicitly-worded quality criteria (correctness, "
                  "engagement with the asker's code, robustness, explanation families) "
                  "plus surface features"),
    "so_accepted": dict(
        oof="results/so_accepted_va_oof.npz",
        dense="datasets/stackoverflow-votes/so_accepted/dense_standard_so_accepted_qtrunc",
        pop="datasets/stackoverflow-votes/va/population.csv.gz",
        item="programming answer",
        corpus="answers posted to python questions on a programming question-and-answer site",
        construct="whether the person who ASKED the question marked this answer as "
                  "accepted — the asker's own verdict that it solved their problem, as "
                  "opposed to the crowd's votes",
        bank_desc="a bank of 39 explicitly-worded quality criteria plus surface features"),
}

COMMON = """You are helping to audit a measurement instrument. You will be shown {n} items:
{item_plural}. Each carries two scores on a 0-1 PERCENTILE scale, computed by two
different systems that DISAGREE about it:

  * `dense percentile` -- a neural model that reads the raw text and was trained on
    this outcome. It is accurate overall but its reasoning is opaque.
  * `articulated percentile` -- a transparent instrument built from {bank_desc}.

The items below are the ones where the two systems disagree MOST. The construct both are
trying to measure is: {construct}.

The corpus is {corpus}.

IMPORTANT CONSTRAINTS:
  * You are NOT told the outcome for any item, and you must not guess it item-by-item.
    Your job is to name GENERAL, TESTABLE properties, not to label these specific rows.
  * Do not propose anything that requires information outside the item text itself.
  * Each property must be judgeable by a careful reader from the item text alone, and
    must be phrased so two independent judges would score it the same way.

THE ITEMS:

{cards}
"""

TRACK_A = """
YOUR TASK (Track A). Propose exactly {k} candidate QUALITY-RELEVANT criteria: properties
that plausibly bear on the construct itself and that the criteria families listed above
do not already capture. Composite / interaction criteria are allowed and welcome --
"X together with Y", "X in the absence of Y".

Return EXACTLY {k} items, one per line, in this format and nothing else:

NAME: <=8 words | DESCRIPTION: one or two sentences a judge could apply, stating what
scores high, what scores low, and when it does not apply | RATIONALE: <=20 words on why
this bears on the construct
"""

TRACK_B = """
YOUR TASK (Track B). Propose exactly {k} candidate SPURIOUS channels: textual properties
that would PREDICT the outcome without being part of the construct's merit -- length or
format proxies, boilerplate, house-style markers, topic markers, temporal tells,
formatting habits, markup density.

Work in UPSTREAM-REASONING mode:
  1. enumerate factors BEYOND the text that could causally affect the outcome (who
     answers this kind of question, answer order and timing, the asker's own vote
     behavior, cross-posting, reputation dynamics, the bounty amount and timing);
  2. for each, ask what textual FINGERPRINT it would leave in the item;
  3. propose those fingerprints as channels.

TAG each channel with its conjectured upstream parent, or "surface-only" if none. If a
channel's parent plausibly causes GENUINE merit as well, tag it MIXED -- do not force it
to one side.

Return EXACTLY {k} items, one per line, in this format and nothing else:

NAME: <=8 words | DESCRIPTION: one or two sentences a judge could apply | PARENT:
<conjectured upstream factor, or surface-only> | MIXED: yes|no
"""


def slice_cmd(cell, round_no):
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    cfg = CELLS[cell]
    z = np.load(TD / cfg["oof"], allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    pos = {r: i for i, r in enumerate(ids)}
    va = z["VA_nl"].astype(float)
    sp = pd.read_csv(NR / cfg["dense"] / "split" / "eval.csv")
    per_seed = []
    for s in (42, 1, 2):
        p = pd.read_csv(NR / cfg["dense"] / f"rm_out_seed{s}" / "preds_eval.csv")
        assert (p["judgement"].values == sp["judgement"].values).all()
        per_seed.append(p["prob"].values.astype(float))
    dense = np.mean(per_seed, axis=0)
    rids = sp["row_id"].astype(str).tolist()
    keep = [i for i, r in enumerate(rids) if r in pos]
    rids = [rids[i] for i in keep]
    dense = dense[keep]
    va_e = np.array([va[pos[r]] for r in rids])
    dp = rankdata(dense) / len(dense)
    ap_ = rankdata(va_e) / len(va_e)
    gap = np.abs(dp - ap_)
    banned = set()
    for rr in range(1, round_no):
        pj = HERE / f"{cell}_r{rr}_slice.json"
        if pj.exists():
            banned |= set(json.loads(pj.read_text())["row_ids"])
    order = np.argsort(-gap)
    chosen = [i for i in order if rids[i] not in banned][:N_SLICE]

    popdf = pd.read_csv(NR / cfg["pop"]).set_index("row_id")
    popdf.index = popdf.index.astype(str)
    text_col = "text" if "text" in popdf.columns else "body"
    lines = []
    for k, i in enumerate(chosen):
        t = str(popdf.loc[rids[i], text_col]).replace("\n", " ")[:CARD_CHARS]
        lines.append(f"[{k+1:02d}] dense={dp[i]:.2f} articulated={ap_[i]:.2f}")
        lines.append(f"     {t}")
    (HERE / f"{cell}_r{round_no}_cards.txt").write_text("\n".join(lines) + "\n")
    n_d_hi = int(sum(1 for i in chosen if dp[i] > ap_[i]))
    (HERE / f"{cell}_r{round_no}_slice.json").write_text(json.dumps({
        "cell": cell, "round": round_no, "n_slice": len(chosen),
        "row_ids": [rids[i] for i in chosen],
        "median_abs_gap": float(np.median(gap[chosen])),
        "n_dense_higher": n_d_hi, "n_artic_higher": len(chosen) - n_d_hi,
        "slice_source": "EVAL split only (test reserved for the gain readout)",
        "banned_prior_rounds": len(banned)}, indent=1))
    print(f"[{cell} r{round_no}] slice {len(chosen)} cards, median|gap| "
          f"{float(np.median(gap[chosen])):.3f}, dense-higher {n_d_hi}")


def build(cell, round_no):
    cfg = CELLS[cell]
    cards_path = HERE / f"{cell}_r{round_no}_cards.txt"
    sj = json.loads((HERE / f"{cell}_r{round_no}_slice.json").read_text())
    raw = cards_path.read_text().rstrip("\n").split("\n")
    cards = ["\n".join(raw[i:i + 2]) for i in range(0, len(raw), 2)]
    assert len(cards) == sj["n_slice"]
    # Blindness patterns, corpus-adapted: the BBC regex banned bare "y=" and
    # "accepted", but in math/code corpora those are benign algebra and prose
    # ("x+y=z", "the widely accepted way"). Ban only LABEL-SHAPED strings: the
    # judgement column name, y=0/1 literals, explicit award/acceptance markers.
    # In math/code corpora "y = 1" is algebra and "label" is a plot axis — the
    # BBC-style content patterns false-positive constantly here. The seal's real
    # guarantee is STRUCTURAL (cards are built from the text column and the two
    # percentiles, nothing else — see slice_cmd); the assert checks only for the
    # label column name and explicit award/acceptance phrases.
    bad = re.compile(r"judgement|bounty\s+award|marked\s+as\s+accepted"
                     r"|answer\s+was\s+accepted", re.I)
    for c in cards:
        assert not bad.search(c), f"label leak at seal: {c[:60]}"
    d = SCRATCH / f"{cell}_r{round_no}"
    d.mkdir(parents=True, exist_ok=True)
    note = (f"DIRECTION NOTE: on {sj['n_artic_higher']} of {sj['n_slice']} items the "
            f"ARTICULATED instrument scores higher; on {sj['n_dense_higher']} the dense "
            "model scores higher. Consider both directions.")
    manifest = {"cell": cell, "round": round_no, "P": len(FLEET),
                "families": sorted({f for f, _ in FLEET}),
                "n_families": len({f for f, _ in FLEET}), "target_families": 3,
                "degraded": True,
                "degradation_reason": "2-family floor at launch; claude legs appended "
                                      "as sealed CLI sessions if cap allows",
                "k_A": K_A, "k_B": K_B,
                "slice_sha1": hashlib.sha1(cards_path.read_bytes()).hexdigest(),
                "proposers": []}
    for fam, pid in FLEET:
        name = f"{fam}_{pid}"
        salt = f"{cell}-r{round_no}-{name}"
        rng = random.Random(int(hashlib.sha256(salt.encode()).hexdigest()[:12], 16))
        order = cards[:]
        rng.shuffle(order)
        body = COMMON.format(n=len(order), item_plural=cfg["item"] + "s",
                             bank_desc=cfg["bank_desc"], construct=cfg["construct"],
                             corpus=cfg["corpus"], cards="\n".join(order))
        body += "\n" + note + "\n"
        for track, tmpl, k in (("A", TRACK_A, K_A), ("B", TRACK_B, K_B)):
            (d / f"prompt_{track}_{name}.txt").write_text(body + tmpl.format(k=k))
        manifest["proposers"].append({"name": name, "family": fam, "salt": salt})
    (d / "fleet_manifest.json").write_text(json.dumps(manifest, indent=1))
    (HERE / f"{cell}_fleet_manifest_r{round_no}.json").write_text(json.dumps(manifest, indent=1))
    print(f"[{cell} r{round_no}] sealed {2*len(FLEET)} prompts under {d}")


FIELD = r"(?:RATIONALE|PARENT|MIXED|DESCRIPTION)"
LINE_STRICT = re.compile(r"^NAME:\s*(?P<name>[^|]+?)\s*\|\s*DESCRIPTION:\s*(?P<desc>.+)",
                         re.I | re.S)
LINE_LOOSE = re.compile(r"^(?P<name>[^:|]{3,90}?):\s*(?P<desc>.+)", re.S)
LINE_PIPE = re.compile(r"^(?P<name>[^|:]{3,90}?)\s*\|\s*(?P<desc>.+)", re.S)
HAS_FIELD = re.compile(r"\|\s*" + FIELD + r"\s*:", re.I)


def collect(cell, round_no):
    d = SCRATCH / f"{cell}_r{round_no}"
    man = json.loads((d / "fleet_manifest.json").read_text())
    out, missing = [], []
    for p in man["proposers"]:
        for track in ("A", "B"):
            f = d / f"out_{track}_{p['name']}.txt"
            if not f.exists() or len(f.read_text().strip()) < 100:
                missing.append(f"{track}/{p['name']}")
                continue
            for ln in f.read_text().splitlines():
                ln = ln.strip().lstrip("-*0123456789. ")
                bare = None
                if not HAS_FIELD.search(ln):
                    parts = [q.strip() for q in ln.split("|")]
                    if len(parts) >= 3 and parts[-1].lower() in ("yes", "no"):
                        bare = parts
                    else:
                        continue
                m = (LINE_STRICT.match(ln) or LINE_LOOSE.match(ln) or LINE_PIPE.match(ln))
                if not m and not bare:
                    continue
                if bare:
                    mb = LINE_STRICT.match(bare[0]) or LINE_LOOSE.match(bare[0])
                    if not mb:
                        continue
                    out.append({"track": track, "proposer": p["name"],
                                "family": p["family"],
                                "name": mb.group("name").strip().strip("*")[:120],
                                "description": mb.group("desc").strip()[:600],
                                "parent": bare[1][:200] if len(bare) > 2 else "",
                                "mixed": bare[-1].strip().lower()})
                    continue
                rest = m.group("desc")
                item = {"track": track, "proposer": p["name"], "family": p["family"],
                        "name": m.group("name").strip().strip("*")[:120]}
                for fld in ("RATIONALE", "PARENT", "MIXED"):
                    mm = re.search(fld + r":\s*([^|]+)", rest, re.I)
                    if mm:
                        item[fld.lower()] = mm.group(1).strip()[:200]
                item["description"] = re.split(r"\|\s*(RATIONALE|PARENT|MIXED):",
                                               rest, flags=re.I)[0].strip()[:600]
                out.append(item)
    res = {"cell": cell, "round": round_no, "n_proposals": len(out),
           "missing_slots": missing,
           "by_track": {t: sum(1 for o in out if o["track"] == t) for t in "AB"},
           "by_proposer": {p["name"]: sum(1 for o in out if o["proposer"] == p["name"])
                           for p in man["proposers"]},
           "proposals": out}
    (HERE / f"{cell}_r{round_no}_proposals.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "proposals"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["slice", "build", "collect"])
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    {"slice": slice_cmd, "build": build, "collect": collect}[a.cmd](a.cell, a.round)
