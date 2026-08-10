#!/usr/bin/env python3
"""N&C A-rubric scorer (Gemma-4-31B offline batch vLLM).

Scores every (comment, rubric) pair from a sample shard: 198 merged N&C rubrics
(rtc_extracted/rubrics.jsonl format), one token 1.0/0.5/0.0/NA each. Label-agnostic —
y_out/y_eng ride along as metadata so aggregation never needs a GPU re-score.

Three synthetic anchor comments (strong/mid/weak) are appended to every shard; their
mean scores are printed and saved for the judge-sanity check.

Usage (gemma4 env, 1 GPU):
  CUDA_VISIBLE_DEVICES=5 python score_va_gemma_nc.py --shard shard_0.jsonl \
      --rubrics nc_rubrics.jsonl --out nc_scores_shard0.npz --util 0.90
"""
import argparse, json, re
import numpy as np
from vllm import LLM, SamplingParams

SYS = ("You are an expert regulatory analyst reviewing PUBLIC COMMENTS submitted on proposed "
       "federal rules. You are given one public comment and ONE quality criterion. Decide how "
       "strongly the comment, on its own evidence, satisfies that criterion. Answer with "
       "EXACTLY ONE token:\n"
       "  1.0 = clearly satisfies the criterion\n"
       "  0.5 = partially / weakly / borderline\n"
       "  0.0 = fails / cuts against the criterion\n"
       "  NA = the comment gives no evidence bearing on this criterion\n"
       "Judge the comment's quality as a piece of regulatory input, not whether you agree with "
       "its position. Output only the token.")

ANCHORS = [
    dict(doc_id="__anchor_strong", agency="__ANCHOR", docket="", year=None, y_out=-1, y_eng=-1,
         text=("On behalf of the National Environmental Modeling Association (42 member "
               "institutions), we submit comments on the proposed revisions to 40 CFR Part 60. "
               "Section III.B of the proposal assumes an emissions baseline of 1.2 Mt/yr; our "
               "attached analysis of EPA's own 2019-2023 CEMS data (Appendix A, Table 2) shows "
               "the correct baseline is 0.94 Mt/yr, which changes the cost-benefit ratio in "
               "Section V from 1.8 to 1.1. We recommend a specific alternative: retain the "
               "subpart Da monitoring schedule but phase the new limit over 36 months, which "
               "our modeling (Appendix B) shows achieves 92% of the projected benefit at 60% "
               "of the compliance cost. Proposed regulatory text implementing this alternative "
               "is attached as Exhibit 1. We also note the RIA fails to quantify grid-reliability "
               "impacts as required by E.O. 12866; we provide a sensitivity analysis in Appendix C.")),
    dict(doc_id="__anchor_mid", agency="__ANCHOR", docket="", year=None, y_out=-1, y_eng=-1,
         text=("I am a registered nurse with 15 years of experience in rural hospitals. I am "
               "writing about the proposed staffing rule. I support the goal of the rule, but I "
               "am worried that small hospitals like mine will not be able to hire enough staff "
               "to meet the requirement. In my hospital we already have trouble filling night "
               "shifts. I think the agency should consider giving rural hospitals more time to "
               "comply, or some kind of exception. Thank you for considering my comment.")),
    dict(doc_id="__anchor_weak", agency="__ANCHOR", docket="", year=None, y_out=-1, y_eng=-1,
         text=("THIS IS A TERRIBLE IDEA AND EVERYONE INVOLVED SHOULD BE ASHAMED. THE GOVERNMENT "
               "IS ALWAYS TRYING TO CONTROL US AND I AM SICK OF IT!!! NOBODY ASKED FOR THIS. "
               "STOP WASTING OUR TAX DOLLARS ON GARBAGE LIKE THIS. I OPPOSE. EVERYONE I KNOW "
               "OPPOSES. DO YOUR JOBS FOR ONCE AND LISTEN TO THE PEOPLE!!!")),
]

def parse_tok(t):
    t = (t or "").strip().lower()
    if t.startswith("na") or "n/a" in t: return np.nan
    if "0.5" in t: return 0.5
    if re.search(r"\b1(\.0)?\b", t) or t.startswith("1"): return 1.0
    if re.search(r"\b0(\.0)?\b", t) or t.startswith("0"): return 0.0
    return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--rubrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    a = ap.parse_args()

    rubrics = []
    with open(a.rubrics) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("name"):
                    rubrics.append(r)
    blocks = [(f"CRITERION: {m['name']}\n"
               f"DESCRIPTION: {m.get('description','')}\n\nAnswer with one token:")
              for m in rubrics]
    a_names = [m["name"] for m in rubrics]

    rows = [json.loads(l) for l in open(a.shard) if l.strip()]
    rows = rows + ANCHORS
    print(f"[nc_va] {len(rows)} comments (incl. {len(ANCHORS)} anchors) x {len(rubrics)} rubrics "
          f"= {len(rows)*len(rubrics)} prompts", flush=True)

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    convs = []
    for r in rows:
        f = (r["text"] or "")[:4000]
        for b in blocks:
            convs.append([{"role": "user", "content": f"{SYS}\n\nPUBLIC COMMENT:\n{f}\n\n{b}"}])
    outs = llm.chat(convs, sp)
    vals = [parse_tok(o.outputs[0].text) for o in outs]
    X = np.array(vals, dtype=float).reshape(len(rows), len(rubrics))
    na = float(np.isnan(X).mean())
    print(f"[nc_va] NA rate {na:.3f}; score dist: "
          f"1.0={float(np.nanmean(X==1.0)):.3f} 0.5={float(np.nanmean(X==0.5)):.3f} "
          f"0.0={float(np.nanmean(X==0.0)):.3f}", flush=True)
    for i, r in enumerate(rows):
        if r["agency"] == "__ANCHOR":
            print(f"[anchor] {r['doc_id']}: mean={np.nanmean(X[i]):.3f} "
                  f"na={float(np.isnan(X[i]).mean()):.2f}", flush=True)

    meta = {k: np.array([r.get(k) if r.get(k) is not None else "" for r in rows], dtype=object)
            for k in ("doc_id", "agency", "docket", "org", "split")}
    np.savez_compressed(a.out, X=X,
                        y_out=np.array([r.get("y_out", -1) for r in rows]),
                        y_eng=np.array([r.get("y_eng", -1) for r in rows]),
                        year=np.array([r.get("year") or 0 for r in rows]),
                        a_names=np.array(a_names, dtype=object), na_rate=na, **meta)
    print(f"[nc_va] saved -> {a.out}", flush=True)
    print("SCORE_DONE", flush=True)

if __name__ == "__main__":
    main()
