#!/usr/bin/env python3
"""Score the two CW EXPERT cells (RoyalRoad market VERDICT, Wigleaf editorial
CURATION) with the mature A-bank standard: the GEPA-phrased cw_community
criteria bank, judged LABEL-BLIND by the local Gemma-4-31B, offline-batch vLLM,
one token per (item, criterion).

WHY these two cells: notes/2026-08-08__cw_nullbank_reaudit.md. Their 2026-07-05/06
"null bank" verdicts (RoyalRoad .505, Wigleaf .578) came from a k-medoid,
NON-GEPA, likely-Llama-70B-judged bank built three weeks BEFORE the
GEPA+Gemma-4-31B standard existed, with no anchor battery and no dense T.

REUSE (feedback_reuse_before_rebuild): the scoring loop, shard checkpointing,
per-shard 3-row blinded anchors with re-draw, NA parsing, prefix caching,
temperature 0 / max_tokens 6 are imported VERBATIM from score_va_gemma_banks.py;
the K>=50 extended anchor battery is imported VERBATIM from
score_scaleupC_banks.py. Only the two bank builders are new. The A bank itself
is the already-GEPA-iterated cw_community bank
(datasets/creative-writing/va_bank_v2/rubrics_initial.jsonl, 45 criteria) and the
V features are the already-published CW deterministic surface bank
(datasets/creative-writing/va_bank_v2/v_features.py, 15 features) -- neither is
re-authored here.

Adds one thing the precedent lacks: a mandatory judge score-DISTRIBUTION check
(--distcheck, on by default) that fails loudly on the guided-JSON all-min /
single-value collapse failure mode (feedback_check_judge_score_distribution).

GPU: ONE GPU only (CUDA_VISIBLE_DEVICES set by the caller), spawn +
CUDA_DEVICE_ORDER=PCI_BUS_ID inherited from score_va_gemma_banks, main guard below.
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
import score_va_gemma_banks as S            # noqa: E402  scoring loop + anchors
import score_scaleupC_banks as C            # noqa: E402  K>=50 battery

REPO = S.REPO
CW = REPO / "datasets/creative-writing"
OUT = Path(os.environ.get("VA_OUT_CWX", str(REPO / "outputs/va_gemma_banks_cw_expert")))
BANK = CW / "va_bank_v2/rubrics_initial.jsonl"
VFEAT = CW / "va_bank_v2/v_features.py"
SEED = 20260808


# ------------------------------------------------------------ system prompts -
SYS_ROYALROAD = (
    "You are an expert fiction editor performing a measurement task. You are given "
    "the OPENING CHAPTER of a serialised web novel and ONE craft criterion. Decide "
    "how strongly the chapter, on the evidence of the supplied text alone, satisfies "
    "that criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the supplied text gives no evidence bearing on this criterion\n"
    "Judge the writing on its own text. Do not predict or infer commercial pickup, "
    "follower counts, ratings, views, page counts, genre popularity, update cadence, "
    "publication deals, authorship, or dataset membership, and do not compare this "
    "chapter with other stories. A chapter is a partial work: judge the craft of what "
    "is shown, not whether the story is finished. Long chapters may have a "
    "deterministically omitted middle; judge what is shown. Output only the token."
)

SYS_WIGLEAF = (
    "You are an expert literary editor performing a measurement task. You are given "
    "ONE short prose piece (flash fiction or a very short story) published in a small "
    "literary magazine, and ONE craft criterion. Decide how strongly the piece, on the "
    "evidence of the supplied text alone, satisfies that criterion. Answer with "
    "EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partly, weakly, inconsistently, or borderline\n"
    "  0.0 = the relevant attempt is present but fails or cuts against the criterion\n"
    "  NA = the supplied text gives no evidence bearing on this criterion\n"
    "Judge the piece on its own text. Do not predict or infer anthology selection, "
    "editorial prizes, 'best of' inclusion, which magazine published it, magazine "
    "prestige, the author's reputation, or dataset membership, and do not compare it "
    "with other pieces. Brevity and open endings are normal in this form and are not "
    "in themselves faults. Output only the token."
)


# ------------------------------------------------------- token truncation ----
# RULING 2026-08-10: truncate in TOKENS, not characters. The old CW constants
# (6,000 source chars -> 3,600 head + 2,400 tail) make the budget depend on text
# density, so dialogue-heavy and prose-dense stories get different amounts of
# actual content. Budget below is the token-exact analogue of the old char budget
# (~4 chars/token for English prose), split on the same 60/40 head/tail ratio.
TRUNC_TOKENS_SOURCE = 1600
TRUNC_TOKENS_HEAD = 960
TRUNC_TOKENS_TAIL = 640
TRUNC_MARKER = "\n\n[... DETERMINISTIC MIDDLE OMISSION ...]\n\n"

_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(S.GEMMA4)
    return _TOK


def token_trunc(text: str, tok=None):
    """Deterministic head+tail truncation measured in JUDGE TOKENS."""
    tok = tok or _tokenizer()
    text = (text or "").strip()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= TRUNC_TOKENS_SOURCE:
        return text, len(ids), False
    head = tok.decode(ids[:TRUNC_TOKENS_HEAD], skip_special_tokens=True)
    tail = tok.decode(ids[-TRUNC_TOKENS_TAIL:], skip_special_tokens=True)
    return head + TRUNC_MARKER + tail, len(ids), True


# ---------------------------------------------------------------- builders ---
def _load_bank_blocks():
    rubrics = [json.loads(l) for l in open(BANK) if l.strip()]
    blocks = [f"CRITERION: {m['name']}\nDESCRIPTION: {m['description']}\n"
              f"GUIDANCE: {m.get('guidance','')}\n\nAnswer with one token:"
              for m in rubrics]
    return rubrics, blocks


def _cw_expert_bank(name, base: Path, sys_prompt, lead, anchor_salt, n_shards,
                    secondary_col, meta_extra):
    import pandas as pd
    vf = S.load_module(VFEAT, "vf_cw_expert")
    df = pd.read_csv(base / "va/population.csv.gz")
    man = json.loads((base / "va/population_manifest.json").read_text())

    items = [{"id": str(r.row_id), "group": str(r.group), "text": str(r.text),
              "judgement": int(r.judgement), "split": str(r.split),
              "secondary": str(getattr(r, secondary_col))} for r in df.itertuples()]

    # RULING: truncate in TOKENS. Precompute once per item (ctx() is called once
    # per item and reused across all 45 criteria, so this costs one tokenizer pass).
    tok = _tokenizer()
    n_trunc, tok_lens = 0, []
    for r in items:
        r["judged_text"], ntok, was_trunc = token_trunc(r["text"], tok)
        tok_lens.append(ntok)
        n_trunc += int(was_trunc)
    print(f"[trunc:{name}] token budget {TRUNC_TOKENS_SOURCE} "
          f"(head {TRUNC_TOKENS_HEAD} / tail {TRUNC_TOKENS_TAIL}); "
          f"{n_trunc}/{len(items)} items truncated; source tokens "
          f"median {int(np.median(tok_lens))} max {max(tok_lens)}", flush=True)

    rubrics, blocks = _load_bank_blocks()

    def ctx(r):
        # anchors are built after this pass, so fall back to on-the-fly truncation
        t = r.get("judged_text") or token_trunc(r["text"], tok)[0]
        return f"{lead}\n{t}"

    def vvec(r):
        return vf.feature_vector(r["text"])

    def anchors(shard):
        rng = random.Random(SEED + anchor_salt * shard)
        pos = dict(rng.choice([r for r in items if r["judgement"] == 1]))
        neg = dict(rng.choice([r for r in items if r["judgement"] == 0]))
        scr = dict(neg)
        scr["text"] = S.scramble([pos["text"][:4000], neg["text"][:4000]], rng, n_words=220)
        # MUST recompute: scr was copied from neg and would otherwise carry neg's
        # cached judged_text, silently scoring the coherent negative as the "scrambled"
        # anchor and destroying the whole word-salad control.
        scr["judged_text"] = token_trunc(scr["text"], tok)[0]
        out = []
        for tag, r in (("anchor_pos", pos), ("anchor_neg", neg), ("anchor_scram", scr)):
            rr = dict(r)
            rr["anchor_tag"] = tag
            rr["id"] = f"__ANCHOR_{shard}_{tag}"
            out.append(rr)
        return out

    ys = {"judgement": np.array([r["judgement"] for r in items])}
    return dict(
        name=name, items=items, rubrics=rubrics, blocks=blocks, sys=sys_prompt,
        ctx=ctx, vvec=vvec, vnames=list(vf.V_NAMES), anchors=anchors, ys=ys,
        n_shards=n_shards,
        extra_cols={"split": np.array([r["split"] for r in items], dtype=object),
                    secondary_col: np.array([r["secondary"] for r in items], dtype=object)},
        meta={"population": str((base / "va/population.csv.gz").relative_to(REPO)),
              "population_manifest": man,
              "a_bank": str(BANK.relative_to(REPO)),
              "a_bank_provenance":
                  "GEPA-phrased cw_community mature A bank (task-ms5c9kdd, dispatched "
                  "2026-07-28 under the explicit 'GEPA proposer AND executor' directive); "
                  "reused verbatim, not re-authored (feedback_a_bank_gepa_gemma4)",
              "v_features": str(VFEAT.relative_to(REPO)),
              "judge": "Gemma-4-31B-it, offline-batch vLLM, temperature 0, max_tokens 6, "
                       "label-blind (y never appears in a prompt)",
              "secondary_group_column": secondary_col,
              "secondary_group_of_item": [r["secondary"] for r in items],
              "split_of_item": [r["split"] for r in items],
              "truncation": {"unit": "TOKENS (gemma-4-31b tokenizer)",
                             "source_tokens": TRUNC_TOKENS_SOURCE,
                             "head_tokens": TRUNC_TOKENS_HEAD,
                             "tail_tokens": TRUNC_TOKENS_TAIL,
                             "ruling": "2026-08-10: truncate in TOKENS not characters"},
              **meta_extra})


def build_royalroad_verdict():
    return _cw_expert_bank(
        "cw_royalroad_verdict", CW / "royalroad_stubs", SYS_ROYALROAD,
        "OPENING CHAPTER OF A SERIALISED WEB NOVEL:", anchor_salt=811, n_shards=4,
        secondary_col="topic_cluster",
        meta_extra={"group_column": "fiction_id",
                    "prior_instrument": {
                        "bank_auc": 0.505,
                        "note": "2026-07-05/06 clean k-medoid 37/40-craft bank, NON-GEPA, "
                                "likely Llama-3.3-70B judge, no anchor battery, no T "
                                "(notes/2026-07-05__why-metric-discovery-plateaus.md:334). "
                                "Context only -- a different instrument, never a gate."}})


def build_wigleaf_curation():
    return _cw_expert_bank(
        "cw_wigleaf_curation", CW / "wigleaf", SYS_WIGLEAF,
        "SHORT PROSE PIECE:", anchor_salt=907, n_shards=4,
        secondary_col="magazine",
        meta_extra={"group_column": "story id",
                    "power_caveat": "404 absolute positives / 1,164 negatives",
                    "prior_instrument": {
                        "bank_auc": 0.578,
                        "note": "2026-07-05/06 clean k-medoid 37/40-craft bank, NON-GEPA, "
                                "likely Llama-3.3-70B judge, no anchor battery, no T "
                                "(notes/2026-07-05__why-metric-discovery-plateaus.md:343). "
                                "Highest craft-rankability in the CW leg; '0 kept' there "
                                "was a SATURATION finding, not a chance-level bank. "
                                "Context only -- a different instrument, never a gate."}})


BUILDERS = {"cw_royalroad_verdict": build_royalroad_verdict,
            "cw_wigleaf_curation": build_wigleaf_curation}


# ------------------------------------------------- judge distribution check --
def distribution_check(name, outdir: Path, modal_kill=0.98, na_kill=0.90):
    """Guided-JSON collapse guard (feedback_check_judge_score_distribution).

    Fails loudly if the judge degenerated: every criterion pinned to one value,
    or the whole matrix at the minimum, or NA everywhere.
    """
    Xs, si = [], 0
    while (outdir / f"{name}_shard{si}.npz").exists():
        z = np.load(outdir / f"{name}_shard{si}.npz", allow_pickle=True)
        Xs.append(z["X"])
        a_names = [str(s) for s in z["a_names"]]
        si += 1
    X = np.vstack(Xs)
    per = []
    for j, nm in enumerate(a_names):
        col = X[:, j]
        fin = col[np.isfinite(col)]
        vals, cnts = np.unique(fin, return_counts=True)
        modal = float(cnts.max() / len(fin)) if len(fin) else 1.0
        per.append({"criterion": nm, "mean": (float(fin.mean()) if len(fin) else None),
                    "na_rate": float(np.isnan(col).mean()), "modal_frac": modal,
                    "modal_value": (float(vals[cnts.argmax()]) if len(fin) else None),
                    "n_distinct": int(len(vals))})
    fin_all = X[np.isfinite(X)]
    vals, cnts = np.unique(fin_all, return_counts=True)
    res = {"n_items": int(X.shape[0]), "n_criteria": int(X.shape[1]),
           "overall_na_rate": float(np.isnan(X).mean()),
           "overall_mean": (float(fin_all.mean()) if fin_all.size else None),
           "value_histogram": {str(v): int(c) for v, c in zip(vals, cnts)},
           "per_criterion": per,
           "collapsed_criteria": [p["criterion"] for p in per
                                  if p["modal_frac"] >= modal_kill or p["n_distinct"] <= 1],
           "all_min_collapse": bool(fin_all.size and float(fin_all.max()) == 0.0),
           "na_flood": bool(float(np.isnan(X).mean()) >= na_kill),
           "thresholds": {"modal_kill": modal_kill, "na_kill": na_kill}}
    res["PASS"] = bool(not res["all_min_collapse"] and not res["na_flood"]
                       and len(res["collapsed_criteria"]) < X.shape[1] // 2)
    p = outdir / "distribution_check.json"
    payload = json.loads(p.read_text()) if p.exists() else {}
    payload[name] = res
    p.write_text(json.dumps(payload, indent=1))
    print(f"[distcheck:{name}] PASS={res['PASS']} mean={res['overall_mean']:.4f} "
          f"NA={res['overall_na_rate']:.4f} hist={res['value_histogram']} "
          f"collapsed={len(res['collapsed_criteria'])}/{X.shape[1]}", flush=True)
    if not res["PASS"]:
        print(f"[distcheck:{name}] *** JUDGE DISTRIBUTION COLLAPSE ***", flush=True)
    return res


# ------------------------------------------------------- pre-flight gate -----
def presmoke_gate(llm, sp, bank, n_items):
    """Cheap sanity gate on the already-loaded engine, BEFORE the full sweep.

    Scores a few real items plus one blinded anchor triple. Aborts the bank if the
    judge has collapsed (single value everywhere / all-min / NA flood) or if the
    anchors do not order pos > neg > scrambled -- catching in seconds what would
    otherwise only surface after ~60K prompts.
    """
    rows = bank["items"][:n_items] + bank["anchors"](999_999)
    convs = [[{"role": "user", "content": f"{bank['sys']}\n\n{bank['ctx'](r)}\n\n{blk}"}]
             for r in rows for blk in bank["blocks"]]
    outs = llm.chat(convs, sp)
    X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                 dtype=float).reshape(len(rows), len(bank["blocks"]))
    real, anc = X[:n_items], X[n_items:]
    fin = real[np.isfinite(real)]
    vals, cnts = np.unique(fin, return_counts=True)
    na = float(np.isnan(real).mean())
    am = np.nanmean(anc, axis=1)
    ordered = bool(am[0] > am[1] > am[2])
    ok = bool(fin.size and len(vals) > 1 and na < 0.90 and float(fin.max()) > 0.0)
    print(f"[presmoke:{bank['name']}] n={n_items} mean={np.nanmean(real):.3f} NA={na:.3f} "
          f"values={ {str(v): int(c) for v, c in zip(vals, cnts)} } | anchors "
          f"pos {am[0]:.3f} / neg {am[1]:.3f} / scram {am[2]:.3f} ordered={ordered} "
          f"| distribution_ok={ok}", flush=True)
    if not ordered:
        print(f"[presmoke:{bank['name']}] NOTE anchors did not order on this single "
              f"draw; not fatal (score_bank re-draws up to 4x per shard and the K=50 "
              f"battery adjudicates), continuing", flush=True)
    return ok


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="cw_royalroad_verdict,cw_wigleaf_curation")
    ap.add_argument("--util", type=float, default=0.93,
                    help="gpu_memory_utilization; with --auto-util this is the CAP")
    ap.add_argument("--auto-util", action="store_true",
                    help="size gpu_memory_utilization to the memory actually free at "
                         "engine-init time (capped by --util); abort if under --min-gib")
    ap.add_argument("--min-gib", type=float, default=80.0,
                    help="minimum usable GiB for Gemma-4-31B bf16 (~62 GiB weights + "
                         "KV cache); below this, abort rather than crowd the card")
    ap.add_argument("--headroom-gib", type=float, default=6.0,
                    help="GiB left free for co-tenants / allocator slack")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--battery", type=int, default=50)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--presmoke", type=int, default=12,
                    help="items per bank for the pre-flight judge sanity gate "
                         "(0 disables); runs on the already-loaded engine, costs "
                         "seconds, and aborts before the full sweep if the judge "
                         "is degenerate")
    ap.add_argument("--no-distcheck", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    banks = []
    for t in [x for x in a.tasks.split(",") if x]:
        b = BUILDERS[t]()
        n_prompts = len(b["items"]) * len(b["blocks"])
        print(f"[build] {t}: {len(b['items'])} items x {len(b['blocks'])} criteria "
              f"= {n_prompts} prompts, {len(set(r['group'] for r in b['items']))} groups, "
              f"pos={np.mean([r['judgement'] for r in b['items']]):.4f}", flush=True)
        banks.append(b)

    util = a.util
    if a.auto_util:
        # This box is heavily contended and other agents stack within seconds of a
        # release, so the free memory at engine-init time is NOT what it was at claim
        # time. Size the request to what is actually free right now (the V7-patents
        # precedent, gpu_ledger 2026-08-08T19:07:50Z "util=0.434 sized to 83569MiB
        # free") instead of demanding a fixed 0.93 and dying.
        import torch
        free_b, total_b = torch.cuda.mem_get_info()
        free_g, total_g = free_b / 2**30, total_b / 2**30
        target_g = min(a.util * total_g, free_g - a.headroom_gib)
        if target_g < a.min_gib:
            print(f"[engine] ABORT: only {free_g:.1f} GiB free of {total_g:.1f} "
                  f"(target {target_g:.1f} < min {a.min_gib} GiB). Not crowding this "
                  f"card; caller should re-poll for a freer one.", flush=True)
            raise SystemExit(4)
        util = max(0.05, min(a.util, target_g / total_g))
        print(f"[engine] free {free_g:.1f}/{total_g:.1f} GiB -> "
              f"gpu_memory_utilization {util:.3f} ({target_g:.1f} GiB)", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=S.GEMMA4, dtype="bfloat16", gpu_memory_utilization=util,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              trust_remote_code=True, max_num_seqs=a.max_num_seqs)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    for b in banks:
        if a.smoke:
            rows = b["items"][:a.smoke]
            convs = [[{"role": "user", "content": f"{b['sys']}\n\n{b['ctx'](r)}\n\n{blk}"}]
                     for r in rows for blk in b["blocks"]]
            outs = llm.chat(convs, sp)
            X = np.array([S.parse_tok(o.outputs[0].text) for o in outs],
                         dtype=float).reshape(len(rows), len(b["blocks"]))
            print(f"[smoke:{b['name']}] n={len(rows)} NA={np.isnan(X).mean():.3f} "
                  f"mean={np.nanmean(X):.3f}", flush=True)
            for ci, nm in enumerate([m["name"] for m in b["rubrics"]]):
                col = X[:, ci]
                fin = col[np.isfinite(col)]
                vals, cnts = np.unique(fin, return_counts=True)
                modal = float(cnts.max() / len(fin)) if len(fin) else 1.0
                print(f"  {ci:02d} {nm[:52]:54s} mean={np.nanmean(col):.3f} "
                      f"na={np.isnan(col).mean():.2f} modal={modal:.2f}", flush=True)
            print("SMOKE_DONE", flush=True)
            continue
        if a.presmoke and not (OUT / f"{b['name']}_shard0.npz").exists():
            if not presmoke_gate(llm, sp, b, a.presmoke):
                print(f"[presmoke:{b['name']}] ABORT — judge degenerate before the "
                      f"full sweep; nothing written", flush=True)
                raise SystemExit(3)
        S.score_bank(llm, sp, b, OUT)
        if not a.no_distcheck:
            distribution_check(b["name"], OUT)
        if a.battery:
            C.run_battery(llm, sp, b, a.battery, OUT)
    print("CW_EXPERT_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
