#!/usr/bin/env python3
"""Score the REBUILT homepage-curation A bank (rubrics_v2.jsonl, 29 criteria) with the
local Gemma-4-31B judge, offline-batch vLLM, one token per (item, criterion).

WHY THIS FILE EXISTS
--------------------
The census bank scored in `outputs/va_gemma_banks_scaleupC/homepage_curation_*` FAILED
its coherent-vs-scrambled gate at .387 (below chance: word salad scored HIGHER than real
headlines). Registry diagnosis: "census bank FAILS coherence (scrambled .387: entity
detectors, not reading instruments)". This file scores the repaired bank on the SAME
frozen population. Nothing in the scaleupC output directory is touched or overwritten;
this run writes to `outputs/va_gemma_banks_homepage_v2/`.

WHAT CHANGED, AND WHY
---------------------
1. NA IS NO LONGER A TOPIC BIT. The old system prompt defined
   `NA = the headline gives no evidence bearing on this criterion`, so an NA meant
   "wrong section" -- 28% of cells, and 62-72% on the crisis / economic / legal
   criteria. The press-verdict closure showed exactly where that leads: on that cell
   the applicability MASK alone reached .7322 while the judged levels were worth .0014
   over it (notes/2026-08-10__closure_press.md 2.2). Here NA is reserved for empty
   input only; "the dimension is absent" is now a real 0.0.
2. THAT ALSO FIXES THE COHERENCE FAILURE. Under the old rule a scrambled headline drew
   NA on the criteria it could not satisfy, those cells were excluded from the row mean,
   and the surviving cells were the entity detectors that fire on token presence -- so
   scrambling two headlines together RAISED the row mean. The v2 prompt says in terms:
   unintelligible text scores 0.0 on every criterion, never NA.
3. FIVE CRITERIA (b01-b05) ARE AN EXPLICIT COHERENCE BACKBONE requiring an actor-action
   RELATION, which scrambling destroys while preserving tokens.
4. THREE CRITERIA (b26-b28) ARE PAGE-RELATIVE: they rank the focal headline against the
   other headlines on the same capture. That is the charge's "rank quality WITHIN
   story-type" -- the comparison set is the same day's story mix, so a score cannot be
   earned by belonging to a favoured genre.

VALIDITY MACHINERY
------------------
* Scoring loop / shard checkpointing / per-shard 3-row blinded anchors with re-draw /
  NA parsing: imported VERBATIM from `score_va_gemma_banks.py` (never re-typed).
* K>=50 extended battery: imported VERBATIM from `score_scaleupC_banks.py`, and then
  EXTENDED here with (a) a PER-CRITERION coherent-vs-scrambled AUC, which the old
  battery never computed and without which "which criteria are entity detectors" was
  unanswerable, and (b) an explicit count of all-NA anchor rows BY TAG -- the old
  battery dropped 16 all-NA rows silently, and 16 of the 50 dropped rows were scrambled.
* Judge score-distribution collapse check (per-criterion mean / NA rate / modal share /
  distinct values), the guided-JSON all-min guard.
* --triage mode reruns the LEGACY 14 census criteria through the SAME per-criterion
  battery so the salvage decision is measured, not asserted.

GPU: one GPU only (CUDA_VISIBLE_DEVICES set by the caller). --auto-util sizes
gpu_memory_utilization from free memory AT ENGINE-INIT TIME (the CW-expert landmine:
free memory at claim time and at init time are different numbers).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import multiprocessing as _mp  # noqa: E402

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_va_gemma_banks as S  # noqa: E402
import score_scaleupC_banks as C  # noqa: E402

REPO = S.REPO
OUT = Path(os.environ.get("VA_OUT_HP2", str(REPO / "outputs/va_gemma_banks_homepage_v2")))
SEED = 20260809
HOMEPAGE_DIR = REPO / "datasets/news-homepages"


# ============================== system prompt ================================
# Differences from SYS_HOMEPAGE (scaleupC) are marked in the module docstring.
SYS_HOMEPAGE_V2 = (
    "You are a senior news editor performing a measurement task. You are given ONE "
    "headline from a news organisation's home page, the other headlines that appeared "
    "on the same capture as context, and ONE criterion. Decide how strongly the FOCAL "
    "headline satisfies that criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, or borderline\n"
    "  0.0 = fails the criterion, INCLUDING the case where the headline contains "
    "nothing the criterion could attach to\n"
    "  NA = the input is empty\n"
    "\n"
    "RULES, all of which matter:\n"
    "1. NA IS FOR EMPTY INPUT ONLY. If the criterion's subject matter is simply absent "
    "from the headline, that is a FAILURE of the criterion and scores 0.0. Do not use "
    "NA to mean 'this criterion is about a different kind of story'.\n"
    "2. IF THE TEXT IS NOT A WELL-FORMED HEADLINE -- an unordered pile of words, a "
    "fragment, or a string whose words do not compose into a statement -- score 0.0 on "
    "EVERY criterion. A text that asserts nothing satisfies nothing. Do not award "
    "credit merely because a recognisable name, place, number, or topic word appears in "
    "it: a name with no coherent claim attached to it is worth 0.0.\n"
    "3. JUDGE THE HEADLINE, NOT ITS SUBJECT. Do not reward or penalise a headline for "
    "the section it would be filed under. Two headlines about the same event must be "
    "separable by these criteria, and a sports or entertainment headline must be able to "
    "score as highly as a politics one when it satisfies the criterion as well.\n"
    "4. Judge ONLY the focal headline. The context list is the rest of that page. Never "
    "score a context headline in the focal headline's place. Three criteria explicitly "
    "ask you to COMPARE the focal headline with the context list; every other criterion "
    "ignores the context.\n"
    "5. Do not infer or predict where the story was placed on the page, how high or low "
    "it ran, which outlet published it, how popular it was, or whether it belongs to any "
    "dataset.\n"
    "\n"
    "Output only the token."
)


# ============================== bank builder =================================
def _build(rubrics_path: Path, name: str):
    import pandas as pd
    vf = S.load_module(HOMEPAGE_DIR / "va/v_features.py", "vf_homepage_v2")
    df = pd.read_csv(HOMEPAGE_DIR / "va/population.csv.gz")
    items = []
    for r in df.itertuples():
        t = str(r.text)
        head = vf.headline_of(t)
        ctxs = t.split("\n\nCONTEXT: ", 1)[1] if "\n\nCONTEXT: " in t else ""
        items.append({"id": r.row_id, "group": str(r.snapshot_id), "outlet": r.outlet,
                      "snapshot_id": str(r.snapshot_id), "headline": head,
                      "context": ctxs, "judgement": int(r.judgement)})

    rubrics = [json.loads(l) for l in open(rubrics_path) if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        return (f"FOCAL HEADLINE: {r['headline']}\n\n"
                f"OTHER HEADLINES ON THE SAME PAGE (context only): {r['context'][:1800]}")

    def vvec(r):
        return vf.vector(r["headline"])

    def anchors(shard):
        # Frozen anchor construction, identical in form to the scaleupC builder so the
        # v2 battery number is comparable with the .387 it replaces.
        rng = random.Random(SEED + 701 * shard)
        pos = dict(rng.choice([r for r in items if r["judgement"] == 1]))
        neg = dict(rng.choice([r for r in items if r["judgement"] == 0]))
        scr = dict(neg)
        scr["headline"] = S.scramble([pos["headline"], neg["headline"]], rng)
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"top_half_placement": np.array([r["judgement"] for r in items])}
    return dict(
        name=name, items=items, rubrics=rubrics, blocks=blocks,
        sys=SYS_HOMEPAGE_V2, ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES),
        anchors=anchors, ys=ys, n_shards=6,
        extra_cols={"outlet": np.array([r["outlet"] for r in items], dtype=object),
                    "snapshot_id": np.array([r["snapshot_id"] for r in items],
                                            dtype=object)},
        meta={"population": "datasets/news-homepages/va/population.csv.gz",
              "bank": str(rubrics_path.relative_to(REPO)),
              "n_criteria": len(rubrics),
              "system_prompt_version": "v2 (NA=empty-input-only; word-salad=0.0; "
                                       "judge-the-headline-not-the-section)",
              "group_column": "snapshot_id (STORY-GROUPED, matching the corrected dense "
                              "design; the OUTLET column is carried as a secondary)",
              "secondary_group_column": "outlet",
              "n_groups": int(df["snapshot_id"].nunique()),
              "outlets": sorted(df["outlet"].unique().tolist()),
              "snapshot_ids": df["snapshot_id"].astype(str).tolist(),
              "outlet_of_item": df["outlet"].tolist(),
              "supersedes": "outputs/va_gemma_banks_scaleupC/homepage_curation_* "
                            "(census bank, coherent-vs-scrambled .387 = FAILED)",
              "weak_instrument_flag":
                  "y is homepage spatial placement, jointly determined with layout/ad/"
                  "image constraints; outlet identity alone predicts it strongly. Every "
                  "number from this cell carries this flag."})


def build_homepage_v2():
    return _build(HOMEPAGE_DIR / "va/rubrics_v2.jsonl", "homepage_curation_v2")


def build_homepage_legacy_triage():
    """The 14 census criteria, scored through the V2 SYSTEM PROMPT is NOT what we want --
    the triage must reproduce the legacy instrument. This builder therefore restores the
    ORIGINAL scaleupC system prompt so the per-criterion coherence numbers describe the
    bank that actually failed."""
    b = _build(HOMEPAGE_DIR / "va/rubrics.jsonl", "homepage_curation_legacy")
    b["sys"] = C.SYS_HOMEPAGE
    b["meta"]["system_prompt_version"] = "legacy scaleupC SYS_HOMEPAGE (verbatim)"
    b["meta"]["purpose"] = ("TRIAGE ONLY -- per-criterion coherent-vs-scrambled AUC for "
                            "the 14 census criteria, to make the salvage decision "
                            "measured rather than asserted. Not used as an A matrix.")
    return b


BUILDERS = {"homepage_v2": build_homepage_v2,
            "homepage_legacy_triage": build_homepage_legacy_triage}


# ==================== per-criterion battery (NEW) ============================
def run_battery_percriterion(llm, sp, bank, k, outdir, tag_suffix=""):
    """K>=50 per class, but reported PER CRITERION as well as per row.

    The row-mean statistic (imported verbatim as C.run_battery) is what the campaign
    quotes. This function adds the decomposition: for each criterion, the AUC separating
    coherent anchors (pos+neg) from scrambled ones. A criterion below .5 here is an
    entity detector -- it scores word salad above real headlines -- and that is the
    measurement the census bank never had.
    """
    from sklearn.metrics import roc_auc_score
    rows, tags = [], []
    for j in range(k):
        for r in bank["anchors"](900_000 + j):
            rows.append(r)
            tags.append(r["anchor_tag"])
    convs = []
    for r in rows:
        c = bank["ctx"](r)
        for blk in bank["blocks"]:
            convs.append([{"role": "user", "content": f"{bank['sys']}\n\n{c}\n\n{blk}"}])
    print(f"[battery-pc:{bank['name']}] {len(rows)} anchors x {len(bank['blocks'])} "
          f"= {len(convs)} prompts", flush=True)
    outs = llm.chat(convs, sp)
    X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                 dtype=float).reshape(len(rows), len(bank["blocks"]))
    tags = np.array(tags)
    names = [m["name"] for m in bank["rubrics"]]
    ids = [m.get("rubric_id", f"c{i:02d}") for i, m in enumerate(bank["rubrics"])]

    coh = (tags != "anchor_scram")
    per = []
    for ci, nm in enumerate(names):
        col = X[:, ci]
        ok = np.isfinite(col)
        y = coh[ok].astype(int)
        v = col[ok]
        auc = (float(roc_auc_score(y, v)) if len(set(y.tolist())) == 2 else float("nan"))
        pv = col[(tags == "anchor_pos") & ok]
        nv = col[(tags == "anchor_neg") & ok]
        sv = col[(tags == "anchor_scram") & ok]
        pn = (float(roc_auc_score([1] * len(pv) + [0] * len(nv),
                                  np.concatenate([pv, nv])))
              if len(pv) and len(nv) and len(set(np.concatenate([pv, nv]).tolist())) > 1
              else float("nan"))
        per.append({
            "rubric_id": ids[ci], "criterion": nm,
            "coherent_vs_scrambled_auc": auc,
            "pos_vs_neg_auc": pn,
            "mean_pos": float(np.mean(pv)) if len(pv) else float("nan"),
            "mean_neg": float(np.mean(nv)) if len(nv) else float("nan"),
            "mean_scram": float(np.mean(sv)) if len(sv) else float("nan"),
            "na_rate_on_anchors": float(np.isnan(col).mean()),
            "na_rate_on_scrambled": float(np.isnan(col[tags == "anchor_scram"]).mean()),
            "is_entity_detector": bool(np.isfinite(auc) and auc < 0.5),
        })
    per.sort(key=lambda d: (-(d["coherent_vs_scrambled_auc"]
                             if np.isfinite(d["coherent_vs_scrambled_auc"]) else -9)))

    # all-NA rows BY TAG -- the statistic the old battery hid
    with np.errstate(invalid="ignore"):
        item_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=1)
    all_na = ~np.isfinite(item_mean)
    by_tag = {t: {"n": int((tags == t).sum()),
                  "n_all_NA_dropped": int((all_na & (tags == t)).sum())}
              for t in ("anchor_pos", "anchor_neg", "anchor_scram")}

    res = {"k_per_class": k, "n_criteria": len(names),
           "all_NA_rows_by_tag": by_tag,
           "n_criteria_below_chance": int(sum(d["is_entity_detector"] for d in per)),
           "per_criterion": per}
    p = outdir / f"anchor_battery_percriterion{tag_suffix}.json"
    payload = json.loads(p.read_text()) if p.exists() else {}
    payload[bank["name"]] = res
    p.write_text(json.dumps(payload, indent=1))
    bad = [d["rubric_id"] for d in per if d["is_entity_detector"]]
    print(f"[battery-pc:{bank['name']}] below-chance criteria "
          f"{len(bad)}/{len(names)}: {bad}", flush=True)
    return res


def distribution_check(bank, outdir):
    """Judge score-distribution collapse guard, run on the assembled shards."""
    Xs, idss = [], []
    for si in range(bank["n_shards"]):
        p = outdir / f"{bank['name']}_shard{si}.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=True)
        Xs.append(z["X"])
        idss.append(z["ids"])
    X = np.vstack(Xs)
    names = [m["name"] for m in bank["rubrics"]]
    cols, flags = [], []
    for ci, nm in enumerate(names):
        col = X[:, ci]
        fin = col[np.isfinite(col)]
        vals, cnts = np.unique(fin, return_counts=True)
        modal = float(cnts.max() / max(len(col), 1)) if len(cnts) else 1.0
        d = {"criterion": nm, "mean": float(np.nanmean(col)) if len(fin) else float("nan"),
             "na_rate": float(np.isnan(col).mean()), "modal_share": modal,
             "n_distinct": int(len(vals)),
             "std": float(np.nanstd(col)) if len(fin) else 0.0}
        d["near_constant"] = bool(d["std"] < 1e-9 or modal > 0.98)
        cols.append(d)
        if d["near_constant"]:
            flags.append(nm)
    res = {"n_items": int(X.shape[0]), "n_criteria": len(names),
           "na_rate_overall": float(np.isnan(X).mean()),
           "n_near_constant": len(flags), "near_constant_criteria": flags,
           "all_min_collapse": bool(np.nanmean(X) < 1e-6),
           "half_pinned_to_one_value": bool(
               sum(c["modal_share"] > 0.9 for c in cols) >= len(names) / 2),
           "per_criterion": cols}
    p = outdir / "distribution_check.json"
    payload = json.loads(p.read_text()) if p.exists() else {}
    payload[bank["name"]] = res
    p.write_text(json.dumps(payload, indent=1))
    print(f"[dist:{bank['name']}] NA {res['na_rate_overall']:.4f} | near-constant "
          f"{res['n_near_constant']}/{len(names)} | all-min {res['all_min_collapse']} | "
          f"half-pinned {res['half_pinned_to_one_value']}", flush=True)
    if res["all_min_collapse"] or res["half_pinned_to_one_value"]:
        print(f"[dist:{bank['name']}] *** DISTRIBUTION CHECK FAILED ***", flush=True)
    return res


# ================================== main =====================================
def _auto_util(cap, min_gib, headroom_gib):
    import torch
    free, total = torch.cuda.mem_get_info()
    free_g, total_g = free / 2**30, total / 2**30
    usable = max(free_g - headroom_gib, 0.0)
    if usable < min_gib:
        print(f"[auto-util] only {usable:.1f} GiB usable (< min {min_gib}); ABORT",
              flush=True)
        sys.exit(4)
    util = min(cap, usable / total_g)
    print(f"[auto-util] free {free_g:.1f} / {total_g:.1f} GiB -> util {util:.3f} "
          f"({util * total_g:.1f} GiB)", flush=True)
    return util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="homepage_v2")
    ap.add_argument("--util", type=float, default=0.93)
    ap.add_argument("--auto-util", action="store_true")
    ap.add_argument("--min-gib", type=float, default=80.0)
    ap.add_argument("--headroom-gib", type=float, default=6.0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--battery", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--battery-only", action="store_true",
                    help="run only the anchor batteries (triage mode)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    banks = []
    for t in [x for x in a.tasks.split(",") if x]:
        b = BUILDERS[t]()
        print(f"[build] {t}: {len(b['items'])} items, {len(b['blocks'])} criteria, "
              f"{len(set(str(r['group']) for r in b['items']))} groups", flush=True)
        banks.append(b)

    from vllm import LLM, SamplingParams
    util = _auto_util(a.util, a.min_gib, a.headroom_gib) if a.auto_util else a.util
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    for b in banks:
        if a.smoke:
            rows = b["items"][:a.smoke]
            convs = []
            for r in rows:
                c = b["ctx"](r)
                for blk in b["blocks"]:
                    convs.append([{"role": "user",
                                   "content": f"{b['sys']}\n\n{c}\n\n{blk}"}])
            outs = llm.chat(convs, sp)
            X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                         dtype=float).reshape(len(rows), len(b["blocks"]))
            print(f"[smoke:{b['name']}] n={len(rows)} NA={np.isnan(X).mean():.4f} "
                  f"mean={np.nanmean(X):.3f}", flush=True)
            for ci, m in enumerate(b["rubrics"]):
                col = X[:, ci]
                vals, cnts = np.unique(col[np.isfinite(col)], return_counts=True)
                modal = float(cnts.max() / max(len(col), 1)) if len(cnts) else 1.0
                print(f"  {m.get('rubric_id', ci)} {m['name'][:52]:54s} "
                      f"mean={np.nanmean(col):.3f} na={np.isnan(col).mean():.3f} "
                      f"modal={modal:.2f} distinct={len(vals)}", flush=True)
            run_battery_percriterion(llm, sp, b, max(a.battery, 12), OUT,
                                     tag_suffix="_smoke")
            C.run_battery(llm, sp, b, max(a.battery, 12), OUT)
            print("SMOKE_DONE", flush=True)
            continue
        if not a.battery_only:
            S.score_bank(llm, sp, b, OUT)
            distribution_check(b, OUT)
        if a.battery:
            C.run_battery(llm, sp, b, a.battery, OUT)          # verbatim, comparable
            run_battery_percriterion(llm, sp, b, a.battery, OUT)  # new decomposition
    print("HOMEPAGE_V2_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
