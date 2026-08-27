"""Controlled test: does giving the disclosure verifier PARENT-CLAIM context recover the 27.3%
verification loss (misses even when localization landed on the examiner's paragraph)?

The localization battery found the verifier's single largest loss is context-blindness: for a
dependent-claim element like "wherein said audio profile is encrypted", the verifier never sees
what "said audio profile" is (defined in the parent claim). This reuses the ALREADY-LOCALIZED
spans (localize_results_scale_gemma.jsonl) and re-runs ONLY the verify step, blind, two ways:
  BASELINE  : (element, span)                       -- the current pipeline
  TREATMENT : (parent-claim context, element, span) -- element made self-contained
Same units, same spans, same model -> the disclosed-rate delta is the pure effect of context.
Negatives get the identical treatment (specificity guard: context must not just inflate yes).

Run ON sk3 (gemma4 env, 1 GPU):
  CUDA_VISIBLE_DEVICES=N python scripts/patents_verifier_parentctx.py [--n 1500]
"""
import argparse, json, sys
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts")
from claim_resolver import parse_claims, resolve_chain
from score_gold_mixture import DEP_RE, load_app_claims_cache

BASE = "/lfs/skampere3/0/alexspan/norm-research"
TB = f"{BASE}/datasets/patents/processed"
UNITS = f"{TB}/localize_units_scale.jsonl"
RESULTS = f"{TB}/localize_results_scale_gemma.jsonl"
OUT = f"{BASE}/outputs/claimverify_paper/verifier_parentctx.json"
AB = f"{BASE}/datasets/patents/processed/gold_mixture_testbed_v1/units_ab12k.jsonl"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"

SYS_V = ("You are a strict USPTO examiner judging anticipation. Decide whether the prior-art "
         "PASSAGE discloses the specific claim LIMITATION. Substance counts, not exact wording, "
         "but the passage must actually teach the limitation (not merely the general field). "
         "Answer with ONE JSON object.")
BASE_Q = ('LIMITATION:\n{el}\n\nPRIOR-ART PASSAGE:\n{span}\n\nDoes the passage disclose this '
          'limitation? Reply JSON: {{"discloses": true|false, "reason": "<=20 words"}}')
CTX_Q = ('CLAIM CONTEXT (parent limitations this element depends on — for interpreting terms like '
         '"said X"; do NOT require the passage to disclose these):\n{parent}\n\n'
         'LIMITATION (the element to check):\n{el}\n\nPRIOR-ART PASSAGE:\n{span}\n\nUsing the '
         'context only to resolve what the limitation refers to, does the passage disclose THIS '
         'limitation? Reply JSON: {{"discloses": true|false, "reason": "<=20 words"}}')

import re
_OBJ = re.compile(r"\{[\s\S]*\}")


def parse_disc(raw):
    m = _OBJ.search(raw or "")
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
    except Exception:
        try:
            o = json.loads(re.sub(r",\s*}", "}", m.group(0)))
        except Exception:
            return None
    v = o.get("discloses")
    if isinstance(v, str):
        v = v.strip().lower() in ("true", "yes", "1")
    return bool(v) if v is not None else None


def parent_text(u, app_claims):
    pg = app_claims.get(u["app_id"])
    if not pg:
        return None
    claims = parse_claims(pg)
    cn = u["claim_num"]
    if cn not in claims:
        return None
    chain, _f, _s = resolve_chain(cn, claims)
    ordered = [c for c in reversed(chain) if c in claims and c != cn]
    return "\n".join(claims[c] for c in ordered)[:2500] if ordered else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="max units per class")
    a = ap.parse_args()

    spans = {}
    for ln in open(RESULTS):
        r = json.loads(ln)
        sp = r.get("spans") or []
        if isinstance(sp, list):
            sp = " ".join(s for s in sp if isinstance(s, str))
        spans[r["uid"]] = {"span": (sp or "")[:1500], "old_disc": r.get("discloses")}

    app_claims = load_app_claims_cache(AB)
    pos, neg = [], []
    for ln in open(UNITS):
        u = json.loads(ln)
        uid = u["uid"]
        s = spans.get(uid)
        if not s or not s["span"].strip():
            continue
        pt = parent_text(u, app_claims)
        if not pt:  # only dependent claims with a resolvable parent chain
            continue
        rec = {"uid": uid, "label": u["label"], "element": u["element"][:600],
               "span": s["span"], "parent": pt, "old_disc": s["old_disc"]}
        (pos if u["label"] == "pos" else neg).append(rec)
    pos, neg = pos[:a.n], neg[:a.n]
    print(f"[vpc] dependent-claim units w/ span+parent: pos={len(pos)} neg={len(neg)}", flush=True)
    if not pos:
        print("[vpc] NO eligible units — abort"); return

    allu = pos + neg
    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=8192, enable_prefix_caching=True, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=90)

    base_msgs = [[{"role": "system", "content": SYS_V},
                  {"role": "user", "content": BASE_Q.format(el=r["element"], span=r["span"])}] for r in allu]
    ctx_msgs = [[{"role": "system", "content": SYS_V},
                 {"role": "user", "content": CTX_Q.format(parent=r["parent"], el=r["element"], span=r["span"])}]
                for r in allu]
    print("[vpc] verifying baseline ...", flush=True)
    base_out = llm.chat(base_msgs, sp)
    print("[vpc] verifying +parent-context ...", flush=True)
    ctx_out = llm.chat(ctx_msgs, sp)
    for r, b, c in zip(allu, base_out, ctx_out):
        r["base"] = parse_disc(b.outputs[0].text)
        r["ctx"] = parse_disc(c.outputs[0].text)

    def rate(rows, key):
        v = [r[key] for r in rows if r[key] is not None]
        return (sum(v) / len(v)) if v else float("nan"), len(v)

    res = {}
    for name, rows in (("pos", pos), ("neg", neg)):
        (b, nb), (c, nc) = rate(rows, "base"), rate(rows, "ctx")
        flips_yes = sum(1 for r in rows if r["base"] is False and r["ctx"] is True)
        flips_no = sum(1 for r in rows if r["base"] is True and r["ctx"] is False)
        res[name] = {"n": len(rows), "base_disc": round(b, 4), "ctx_disc": round(c, 4),
                     "delta": round(c - b, 4), "flips_no_to_yes": flips_yes, "flips_yes_to_no": flips_no}
        print(f"[vpc] {name}: base={b:.3f} (+parent)={c:.3f} delta={c-b:+.3f}  "
              f"flips no->yes {flips_yes}, yes->no {flips_no}  (n={len(rows)})", flush=True)
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": res,
               "examples": [{k: r[k] for k in ("uid", "label", "element", "base", "ctx")}
                            for r in pos if r["base"] is False and r["ctx"] is True][:15]},
              open(OUT, "w"), indent=1)
    pd, nd = res["pos"]["delta"], res["neg"]["delta"]
    print(f"[vpc] VERDICT: pos gain {pd:+.3f}, neg gain {nd:+.3f} -> "
          f"{'PARENT CONTEXT HELPS (pos up, neg flat)' if pd > 0.02 and nd < pd/2 else 'no clean gain'}", flush=True)
    print("VPC_DONE", flush=True)


if __name__ == "__main__":
    main()
