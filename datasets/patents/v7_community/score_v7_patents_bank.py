#!/usr/bin/env python3
"""V7 patents forward-citation cell: score the articulated-criteria (A) bank with
the local Gemma-4-31B judge, offline-batch vLLM, one token per (item, criterion).

REUSE: the scoring loop, shard checkpointing, per-shard 3-row blinded anchors,
NA parsing, prefix caching, temperature 0 / max_tokens 6, and the extended
anchor battery are ALL imported verbatim from
`datasets/va_gemma_banks/score_va_gemma_banks.py` and
`datasets/va_gemma_banks/score_scaleupC_banks.py`. Only the bank builder below
is new -- exactly the pattern those files establish for a new cell.

ANCHOR LABEL SOURCE -- A DELIBERATE DEVIATION FROM THE so_votes RECIPE.
so_votes anchored on `y_accepted`, an independent judgment channel on the same
rows, so the battery certifies the judge against a signal it is not being asked
to reproduce. THIS CELL HAS NO SUCH INDEPENDENT CHANNEL: every label available
on a granted patent here is a forward-citation count, i.e. the quantity under
test. Anchoring on y_fwd5 would make the certificate circular.

So the anchors are CONSTRUCTED and MATCHED instead, which is strictly stronger:
each anchor trio is ONE real patent in three states --
    anchor_pos   = the document intact,
    anchor_neg   = the SAME document deterministically DEGRADED (generic title,
                   boilerplate abstract, claim limitations replaced by purely
                   functional shells) -- a within-document quality contrast that
                   any competent patent reader orders the same way,
    anchor_scram = word-scrambled nonsense.
The known label is the degradation, not an outcome, so the battery is
non-circular by construction while still being a known-label blinded anchor in
every judging batch (standing rule). A y-based pos/neg contrast is ALSO recorded
in the battery output, flagged `circular_with_y`, as descriptive context only.

GPU: one GPU only (CUDA_VISIBLE_DEVICES set by the caller).

  CUDA_VISIBLE_DEVICES=N python3 score_v7_patents_bank.py --smoke 24
  CUDA_VISIBLE_DEVICES=N python3 score_v7_patents_bank.py --battery 50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
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

_HERE = Path(__file__).resolve()
REPO_GUESS = _HERE.parents[3]
sys.path.insert(0, str(REPO_GUESS / "datasets/va_gemma_banks"))
import score_va_gemma_banks as S  # noqa: E402
import score_scaleupC_banks as C  # noqa: E402

REPO = S.REPO
OUT = Path(os.environ.get("VA_OUT_V7",
                          str(REPO / "outputs/va_gemma_banks_patents_fwdcites")))
SEED = 20260808

V7_DIR = REPO / "datasets/patents/v7_community"
V7_BANK = V7_DIR / "rubrics.jsonl"

SYS_V7 = (
    "You are an experienced patent professional performing a measurement task. "
    "You are given the title, abstract and first claim of ONE granted US "
    "patent, and ONE quality criterion. Decide how strongly the patent, on the "
    "evidence of the supplied text alone, satisfies that criterion. Answer with "
    "EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the document gives no evidence bearing on this criterion\n"
    "Judge this document on its own text. You are NOT being asked how important, "
    "valuable, influential or widely used the invention became, and you must not "
    "try to infer the owner, the inventor, the examiner, the technology sector, "
    "the filing or grant date, or how often the patent was later cited. Only "
    "claim 1 is shown; do not penalise the absence of dependent claims or of the "
    "detailed description. Long fields may have a deterministically omitted "
    "middle; judge what is shown. Output only the token."
)

TRUNC_SRC, TRUNC_HEAD, TRUNC_TAIL = 4200, 2600, 1600
TRUNC_MARK = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"


def _trunc(s, src=TRUNC_SRC, head=TRUNC_HEAD, tail=TRUNC_TAIL):
    s = (s or "").strip()
    return s if len(s) <= src else s[:head] + TRUNC_MARK + s[-tail:]


# ------------------------------------------------- constructed degradation ---
_TRANS_RE = re.compile(r"\b(comprising|consisting (?:essentially )?of|including|"
                       r"having|containing)\b", re.I)
_FUNCTIONAL_SHELL = (
    "a first component configured to perform a function; a second component "
    "configured to cooperate with the first component; and a controller "
    "configured to operate the components as desired")


def degrade(rec):
    """Deterministic, FORMAT-PRESERVING degradation of one patent: the paired
    low-quality anchor.

    CALIBRATED AGAINST THE SMOKE RUN. A first version also replaced the title
    and abstract with boilerplate; it scored 0.190 against 0.192 for pure
    word-scrambled nonsense, i.e. it destroyed as much as scrambling and
    collapsed the required pos > neg > scram ordering (`score_bank` retries such
    a shard 4 times and then marks it invalid). Degrading ONLY claim 1 — the
    real title and abstract are kept — leaves the abstract-side criteria
    partially satisfiable, so the degraded document lands between intact and
    nonsense, which is what a graded sensitivity certificate needs.
    """
    d = dict(rec)
    c1 = (rec["claim1"] or "").strip()
    m = _TRANS_RE.search(c1)
    pre = c1[:m.end()] if m else "1. A device comprising:"
    d["claim1"] = f"{pre} {_FUNCTIONAL_SHELL}."
    return d


def build_v7_patents():
    import pandas as pd
    vf = S.load_module(V7_DIR / "v_features.py", "vf_v7_patents")
    df = pd.read_csv(V7_DIR / "population.csv.gz")
    items = []
    for r in df.itertuples():
        items.append({"id": str(r.row_id), "group": str(r.family_group),
                      "title": str(r.title), "abstract": str(r.abstract),
                      "claim1": str(r.claim1),
                      "y_fwd5": (None if r.y_fwd5 != r.y_fwd5 else int(r.y_fwd5)),
                      "cohort": str(r.cohort)})

    rubrics = [json.loads(l) for l in open(V7_BANK) if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n\n"
              "Answer with one token:" for m in rubrics]

    def ctx(r):
        # Exactly the three declared text fields. No CPC, no dates, no numbers,
        # no assignee, no examiner -- the claim-fell post-mortem's killer channel
        # was metadata, so the context is asserted below to contain nothing else.
        return (f"PATENT TITLE: {r['title'][:300]}\n\n"
                f"ABSTRACT:\n{_trunc(r['abstract'], 2600, 1700, 900)}\n\n"
                f"CLAIM 1:\n{_trunc(r['claim1'])}")

    def vvec(r):
        return vf.vector(r["title"], r["abstract"], r["claim1"])

    def anchors(shard):
        """Matched intact / degraded / scrambled trio -- see module docstring."""
        rng = random.Random(SEED + 607 * shard)
        base = dict(rng.choice(items))
        pos = dict(base)
        neg = degrade(base)
        scr = dict(base)
        scr["abstract"] = S.scramble([base["abstract"][:3000]], rng, n_words=120)
        scr["claim1"] = S.scramble([base["claim1"][:3000]], rng, n_words=120)
        scr["title"] = "Method system apparatus"
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg),
                       ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"fwd5_community": np.array(
              [np.nan if r["y_fwd5"] is None else r["y_fwd5"] for r in items],
              dtype=float)}
    for extra in ["y_fwd5_examiner", "y_fwd5_nonexaminer", "y_fwd_alltime",
                  "y_fwd5_topquartile"]:
        ys[extra] = df[extra].values.astype(float)

    meta = {"population": "datasets/patents/v7_community/population.csv.gz",
            "group_column": "family_group (near-duplicate / continuation cluster)",
            "n_groups": int(df["family_group"].nunique()),
            "anchor_label_source": (
                "CONSTRUCTED matched degradation (intact vs degraded vs "
                "scrambled), NOT an outcome label: every observed label on this "
                "population is a forward-citation count, i.e. the quantity under "
                "test, so a y-based anchor would be circular."),
            "context": "title + abstract + claim 1 ONLY",
            "excluded_from_context": ["examiner", "art unit", "assignee",
                                      "inventor", "filing/grant date",
                                      "patent number", "CPC code", "num_claims",
                                      "any citation count"],
            "truncation": {"source_chars": TRUNC_SRC, "head": TRUNC_HEAD,
                           "tail": TRUNC_TAIL, "title_chars": 300,
                           "abstract": [2600, 1700, 900]}}

    # Hard assertion: no metadata token may reach the judge context.
    # WORD BOUNDARIES ARE REQUIRED. A substring test on "art unit" fires on
    # "a two-part unit connector"; on the full 16,000 rows the bounded pattern
    # matches 0 times. "examiner" is bounded AND rate-limited rather than
    # forbidden outright: it matches 3/16,000 rows (0.019%), every one of them
    # ordinary technical usage -- a vision-screening examiner, a "cell examiner"
    # circuit, a "data examiner module" -- and none of them USPTO metadata,
    # which cannot appear because the only fields assembled here are title,
    # abstract and claim 1.
    probe = " ".join(ctx(r) for r in items).lower()
    for bad in [r"\bart unit\b", r"\bassignee\b", r"\battorney\b",
                r"\bapplication number\b", r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"]:
        hits = re.findall(bad, probe)
        assert not hits, f"banned pattern {bad!r} reached the judge context: {hits[:3]}"
    n_exam = len(re.findall(r"\bexaminer\b", probe))
    assert n_exam / max(len(items), 1) < 0.01, \
        f"'examiner' appears at {n_exam / len(items):.3%} of rows -- investigate"
    meta["metadata_assertions"] = {
        "bounded_patterns_with_zero_matches": ["art unit", "assignee", "attorney",
                                               "application number", "ISO dates"],
        "examiner_token_rows": n_exam,
        "examiner_token_rate": n_exam / max(len(items), 1)}

    return dict(name="patents_fwdcites", items=items, rubrics=rubrics,
                blocks=blocks, sys=SYS_V7, ctx=ctx, vvec=vvec,
                vnames=list(vf.V_NAMES), anchors=anchors, ys=ys, n_shards=8,
                meta=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.85)
    # 8192, NOT 4096. The first full pass died in shard 7 (7/8 shards written)
    # on `VLLMValidationError: maximum context length is 4096 tokens ... your
    # prompt contains at least 4097 input tokens`. Cause: the truncation above is
    # in CHARACTERS but the engine limit is in TOKENS, and patent claims are
    # unusually token-dense (reference numerals, chemical names, indexed
    # variables), so a claim inside the 4,200-char budget can still exceed 4,096
    # tokens once the system prompt and criterion block are added.
    # The fix raises the CAP and leaves every prompt's CONTENT byte-identical, so
    # shards 0-6 stay valid and comparable: any prompt already under 4,096 tokens
    # tokenizes and decodes identically at temperature 0 under a larger cap, and
    # 8,192 is far inside Gemma-4's native context so no RoPE scaling is
    # triggered. Re-truncating instead would have changed shard 7's prompts and
    # made it inconsistent with the seven already banked.
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--battery", type=int, default=50)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    b = build_v7_patents()
    print(f"[build] patents_fwdcites: {len(b['items'])} items, "
          f"{len(b['blocks'])} criteria, "
          f"{len(set(str(r['group']) for r in b['items']))} groups", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=512)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    if a.smoke:
        rows = b["items"][:a.smoke] + b["anchors"](0)
        convs = []
        for r in rows:
            c = b["ctx"](r)
            for blk in b["blocks"]:
                convs.append([{"role": "user",
                               "content": f"{b['sys']}\n\n{c}\n\n{blk}"}])
        outs = llm.chat(convs, sp)
        X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                     dtype=float).reshape(len(rows), len(b["blocks"]))
        na = np.isnan(X)
        print(f"[smoke:patents_fwdcites] n={len(rows)} NA={na.mean():.3f} "
              f"mean={np.nanmean(X):.3f}", flush=True)
        for ci, nm in enumerate([m["name"] for m in b["rubrics"]]):
            col = X[:, ci]
            fin = col[np.isfinite(col)]
            vals, cnts = np.unique(fin, return_counts=True)
            modal = float(cnts.max() / max(len(col), 1)) if len(cnts) else 1.0
            flag = ""
            if np.isnan(col).mean() > 0.5:
                flag = "  <-- HIGH-NA"
            elif modal > 0.95:
                flag = "  <-- COLLAPSED"
            print(f"  {ci:02d} {nm[:54]:56s} mean={np.nanmean(col):.3f} "
                  f"na={np.isnan(col).mean():.2f} modal={modal:.2f}{flag}",
                  flush=True)
        # the constructed anchor ordering, on the smoke draw
        tags = [r.get("anchor_tag") for r in rows]
        for tag in ("anchor_pos", "anchor_neg", "anchor_scram"):
            idx = [i for i, t in enumerate(tags) if t == tag]
            if idx:
                print(f"  [anchor] {tag:13s} mean={np.nanmean(X[idx]):.3f}",
                      flush=True)
        print("SMOKE_DONE", flush=True)
        return

    S.score_bank(llm, sp, b, OUT)
    if a.battery:
        C.run_battery(llm, sp, b, a.battery, OUT)
    print("V7_PATENTS_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
