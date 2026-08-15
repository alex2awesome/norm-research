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


if __name__ == "__main__":
    {"s1": s1, "s2_prompts": s2_prompts, "s3_critic": s3_critic}[sys.argv[1]]()
