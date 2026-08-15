"""Objective-comparison experiment driver (plan: notes/2026-08-15__objective-comparison-
experiment-plan.md, v2; prereg frozen in the plan BEFORE this ran).

Tier A machinery. Subcommands (run on sk3):
  s1        (CPU)  freeze pools + hash doc-split + form-sensitivity from EXISTING per-form
                   score matrices; write objective_comparison_v1/{pools,split,sensitivity}.json
  s2_prompts (CPU) build definition-prompt (m_desc / M_i) scoring shards for the free GPU
  s3_critic  (CPU) build critic prompts (qwen-2.5-72b via OpenRouter — GLM window is
                   contested by the pupa parity arm; family independence preserved: critic
                   qwen / arbiter gpt / executor llama+gemma) for objective-half docs
  select     (CPU) compute the three arms' selections per prereg
Tasks in Tier A: peer (8B instrument), cw/pr/humor (Gemma-4 instrument). crx EXCLUDED from
Tier A: no stored multi-form pool (single-form scores only) — disclosed in the plan.
Doc-split: md5("ocsplit:<task>:<doc>") even/odd -> objective-half / eval-half. Objectives
NEVER read eval-half docs or any y.
"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
OUT = Path("/lfs/skampere3/0/alexspan/outputs/objective_comparison_v1")
HIER = Path("/lfs/skampere3/0/alexspan/norm-research/outputs/hierarchy")
BANKDIR = Path("/lfs/skampere3/0/alexspan/data/silver_match_v3_20260712_faithful/banks")
HIERFILE = {"cw": "creative-writing", "pr": "press-releases", "humor": "humor"}
TEXTS = {"peer": ("peer_paper_texts.jsonl", "paper_id"),
         "cw": ("cw_story_texts.jsonl", "post_id"),
         "pr": ("pressrel_score_texts.jsonl", "source_id"),
         "humor": ("humor_score_texts.jsonl", "source_id")}

TEMPLATE = """Text:
{text}

You are evaluating the above text on ONE specific criterion.
Criterion:
{rubric}

How well does the text satisfy the criterion? Reply with exactly "SCORE: N" where N is an integer 0-10 (0 = not at all, 10 = fully)."""


def obj_half(task, doc):
    return int(hashlib.md5(f"ocsplit:{task}:{doc}".encode()).hexdigest(), 16) % 2 == 0


def construct_of(task):
    if task in HIERFILE:
        g = json.load(open(HIER / f"{HIERFILE[task]}_general_r2_expanded.json"))["merged_groups"]
        return {f"a{i}": (x.get("merged_name", f"a{i}"), x.get("merged_description", "")[:600])
                for i, x in enumerate(g)}
    bank = json.load(open(BANKDIR / "peer-review.json"))["metrics"]
    return {f"a{i}": (m["name"], m.get("description", "")[:600]) for i, m in enumerate(bank)}


def form_scores(task):
    """Objective substrate = the 300-probe panels ({task}_probes_g4.json): every form of
    every metric jointly finite on all 300 probes; probes are disjoint from eval corpora
    by construction (separate probe_texts files)."""
    d = json.load(open(MD / f"{task}_probes_g4.json"))
    ids = d["post_ids"]; S = d["scores"]
    per = defaultdict(dict)
    for k, v in S.items():
        m = re.match(r"(a\d+)__(\-?\d+)$", k)
        if m:
            per[m.group(1)][k] = np.asarray(v, float)
    return ids, per


def s1():
    OUT.mkdir(parents=True, exist_ok=True)
    pools, sens, split = {}, {}, {}
    for task in ("peer", "cw", "pr", "humor"):
        try:
            ids, per = form_scores(task)
        except FileNotFoundError as e:
            print(f"{task}: form scores missing ({e}) — skipped"); continue
        oh = list(range(len(ids)))          # probes are all objective-side
        split[task] = {"n_docs": len(ids), "n_obj_half": len(oh)}
        kept = {}
        for mid, forms in per.items():
            if len(forms) < 8:
                continue
            mat = np.vstack([v[oh] for v in forms.values()])
            ok = np.isfinite(mat).all(axis=0)
            if ok.sum() < 40:
                continue
            m2 = mat[:, ok]
            cors = []
            for i in range(len(m2)):
                for j in range(i + 1, len(m2)):
                    if m2[i].std() > 0 and m2[j].std() > 0:
                        cors.append(np.corrcoef(m2[i], m2[j])[0, 1])
            if len(cors) < 10:
                continue
            kept[mid] = sorted(forms)
            sens.setdefault(task, {})[mid] = round(float(np.percentile(cors, 75)
                                                         - np.percentile(cors, 25)), 4)
        pools[task] = kept
        print(f"{task}: {len(kept)} metrics with >=8-form pools; obj-half {len(oh)}/{len(ids)} docs")
    json.dump(pools, open(OUT / "pools.json", "w"), indent=1)
    json.dump(sens, open(OUT / "form_sensitivity.json", "w"), indent=1)
    json.dump(split, open(OUT / "split.json", "w"), indent=1)


def s2_prompts():
    """Definition-prompt (m_desc; doubles as M_i) shards: every pooled metric x ALL docs."""
    pools = json.load(open(OUT / "pools.json"))
    for task, kept in pools.items():
        con = construct_of(task)
        tf, key = TEXTS[task]
        texts = {}
        for line in open(MD / tf):
            r = json.loads(line)
            texts[r[key]] = r["text"]
        n = 0
        with open(OUT / f"defpass_{task}_prompts.jsonl", "w") as f:
            for mid in kept:
                nm, desc = con.get(mid, (mid, ""))
                rubric = f"{nm}: {desc}" if desc else nm
                for doc, tx in texts.items():
                    f.write(json.dumps({"channel": "defpass", "aspect_id": mid,
                                        "datapoint_id": doc,
                                        "prompt": TEMPLATE.format(text=tx[:8000],
                                                                  rubric=rubric)}) + "\n")
                    n += 1
        print(f"{task}: defpass {n} prompts")


def s3_critic():
    """Critic prompts on PROBE texts (the objective substrate; disjoint from eval corpora):
    holistic construct score per probe, 150-probe hash subsample per task (quota cap)."""
    pools = json.load(open(OUT / "pools.json"))
    for task, kept in pools.items():
        con = construct_of(task)
        texts = {}
        for line in open(MD / f"{task}_probe_texts.jsonl"):
            r = json.loads(line)
            texts[r["probe_id"]] = r["text"]
        sub = sorted(texts, key=lambda d: hashlib.md5(f"crit:{task}:{d}".encode()).hexdigest())[:150]
        n = 0
        with open(OUT / f"critic_{task}_prompts.jsonl", "w") as f:
            for mid in kept:
                nm, desc = con.get(mid, (mid, ""))
                rubric = f"{nm}: {desc}" if desc else nm
                for doc in sub:
                    f.write(json.dumps({"channel": "critic", "aspect_id": mid,
                                        "datapoint_id": doc,
                                        "prompt": TEMPLATE.format(text=texts[doc][:8000],
                                                                  rubric=rubric)}) + "\n")
                    n += 1
        print(f"{task}: critic {n} prompts over {len(sub)} probes")


def _ibin(a, b):
    import math
    ab = (a >= np.median(a)).astype(int); bb = (b >= np.median(b)).astype(int)
    P = np.zeros((2, 2))
    for x, y in zip(ab, bb):
        P[x, y] += 1
    P /= P.sum()
    mi = 0.0
    for x in (0, 1):
        for y in (0, 1):
            if P[x, y] > 0:
                mi += P[x, y] * math.log2(P[x, y] / (P[x].sum() * P[:, y].sum()))
    return mi


def _rankagree(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return -2.0
    return float(np.corrcoef(ra, rb)[0, 1])


def select():
    """Arms per prereg. m_recon computable once ocdef probes exist; m_fb once critic lands
    (rows for a metric may be partial — require >=100 finite critic probes)."""
    pools = json.load(open(OUT / "pools.json"))
    crit = defaultdict(dict)
    cf = OUT / "critic_all_results.jsonl"
    if cf.exists():
        for line in open(cf):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("score") is not None:
                crit[r["aspect_id"]][r["datapoint_id"]] = float(r["score"])
    sel = {}
    for task, kept in pools.items():
        ids, per = form_scores(task)
        mi_d = json.load(open(MD / f"ocdef_{task}_probes_g4.json"))
        mi_ids = mi_d["post_ids"]; mi_S = mi_d["scores"]
        idx_of = {d: i for i, d in enumerate(ids)}
        for mid, formkeys in kept.items():
            mk = f"{mid}__-1"
            if mk not in mi_S:
                continue
            Mi = np.asarray(mi_S[mk], float)
            Mi_al = np.array([Mi[mi_ids.index(d)] if d in mi_ids else np.nan for d in ids])
            ok = np.isfinite(Mi_al)
            rec_best, fb_best = None, None
            cr = crit.get(mid, {})
            cr_al = np.array([cr.get(d, np.nan) for d in ids])
            okc = np.isfinite(cr_al)
            for fk in formkeys:
                v = per[mid][fk]
                if ok.sum() >= 100 and v[ok].std() > 0 and Mi_al[ok].std() > 0:
                    mi = _ibin(v[ok], Mi_al[ok])
                    if rec_best is None or mi > rec_best[1]:
                        rec_best = (fk, round(mi, 4))
                if okc.sum() >= 100 and v[okc].std() > 0 and cr_al[okc].std() > 0:
                    ra = _rankagree(v[okc], cr_al[okc])
                    if fb_best is None or ra > fb_best[1]:
                        fb_best = (fk, round(ra, 4))
            sel.setdefault(task, {})[mid] = {
                "m_recon": rec_best, "m_fb": fb_best,
                "m_desc": f"{mid}__-1", "n_critic_probes": int(okc.sum())}
    json.dump(sel, open(OUT / "selections.json", "w"), indent=1)
    for task, mm in sel.items():
        nrec = sum(1 for v in mm.values() if v["m_recon"])
        nfb = sum(1 for v in mm.values() if v["m_fb"])
        agree = sum(1 for v in mm.values()
                    if v["m_recon"] and v["m_fb"] and v["m_recon"][0] == v["m_fb"][0])
        print(f"{task}: {len(mm)} metrics | m_recon selected {nrec} | m_fb selected {nfb} "
              f"| same-form agreement {agree}")


if __name__ == "__main__":
    {"s1": s1, "s2_prompts": s2_prompts, "s3_critic": s3_critic,
     "select": select}[sys.argv[1]]()
