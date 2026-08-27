#!/usr/bin/env python3
"""Claim-unit fix A/B on the gold-mixture testbed (task: nail claim extraction).

History: the #61 OA-scale result (retrieved-only V=0.62) decomposed INDEPENDENT claims only
(DEP_RE skip inherited from v2_full_pipeline); dependent claims — the majority — were scored as
single undecomposed blobs. score_gold_mixture.py later gained FIX #3 (union_claim_text: full
parent-chain resolution + decompose ALL claims) but the elements cache was never regenerated
(3,970 rows = independents only). This driver runs the controlled comparison:

  Arm A (old unit):  independents decomposed; dependents = single full-scope-union blob
                     (load_elements fallback) — replicates #61 behavior.
  Arm B (fixed unit): ALL claims decomposed from full-scope union text (FIX #3 as intended).
  Arm C (delta unit): union-decomposed, then dependents keep only DELTA elements (those NOT
                     lexically covered by the parent chain). Motivated by the 2026-06-16
                     finding (project_patents_gold_evidence_misassignment): on dumped element
                     scores, delta-only 0.55 > union 0.519 > parent-only 0.485 within-AUC —
                     the examiner cites art against the ADDED limitation; parent elements
                     drag the softmin.

Both arms use the SAME decomposer (Gemma-4; Qwen-122B was removed from the shared cache
2026-07-07), the same balanced 12K sample, and the same v6a+xenc+softmin scoring — the delta
isolates the claim-unit representation.

Subcommands (run ON sk3):
  sample     build units_ab12k(_armA).jsonl — stable-hash balanced sample (3K pos-102 + 3K
             pos-103, both has_gold, + 6K neg), replicating the #61 spec
  check      validate a (smoke or full) decompose output: parse rate, elements/claim,
             prints dependent-claim examples for eyeballing
  derive-a   filter arm-B elements to independent claims only -> arm-A elements file
"""
import argparse, hashlib, json, os, re, sys

TB = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/gold_mixture_testbed_v1"
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts")
from score_gold_mixture import DEP_RE  # noqa: E402  (same regex as the pipeline)

UNITS = f"{TB}/units.jsonl"
AB = f"{TB}/units_ab12k.jsonl"
AB_A = f"{TB}/units_ab12k_armA.jsonl"
AB_C = f"{TB}/units_ab12k_armC.jsonl"

_TOK_RE = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in for with said wherein claim according comprising"
            " configured further at least one plurality method system apparatus device".split())


def _toks(s):
    return [w for w in _TOK_RE.findall((s or "").lower()) if w not in _STOP and len(w) > 2]


def skey(u):
    return hashlib.md5(f"{u['app_id']}|{u['ifw_number']}|{u['claim_num']}".encode()).hexdigest()


def cmd_sample(_):
    pos102, pos103, neg = [], [], []
    with open(UNITS) as f:
        for ln in f:
            u = json.loads(ln)
            if u["label_fell"] and u.get("has_gold"):
                t = str(u.get("oa_primary_rejection_type"))
                if t == "102":
                    pos102.append(u)
                elif t == "103":
                    pos103.append(u)
            elif not u["label_fell"]:
                neg.append(u)
    for lst in (pos102, pos103, neg):
        lst.sort(key=skey)
    sample = pos102[:3000] + pos103[:3000] + neg[:6000]
    with open(AB, "w") as fh:
        for u in sample:
            fh.write(json.dumps(u) + "\n")
    # byte-identical copy under the arm-A name so elements_path() resolves per-arm
    with open(AB_A, "w") as fh:
        for u in sample:
            fh.write(json.dumps(u) + "\n")
    dep = sum(1 for u in sample if DEP_RE.search(u["claim_text"][:300]))
    print(f"sample: {len(sample)} units (pos102={len(pos102[:3000])} pos103={len(pos103[:3000])} "
          f"neg={len(neg[:6000])}) dependent-claim units={dep} ({dep/len(sample):.1%})")
    print(f"wrote {AB} + arm-A copy")


def cmd_check(a):
    els_path = a.elements
    units = {}
    with open(AB) as f:
        for ln in f:
            u = json.loads(ln)
            units[(u["app_id"], u["ifw_number"], u["claim_num"])] = u
    rows = [json.loads(l) for l in open(els_path)]
    n = len(rows)
    empty = sum(1 for r in rows if not r["elements"])
    counts = sorted(len(r["elements"]) for r in rows if r["elements"])
    dep_rows = [r for r in rows if (k := (r["app_id"], r["ifw"], r["claim_num"])) in units
                and DEP_RE.search(units[k]["claim_text"][:300])]
    dep_counts = sorted(len(r["elements"]) for r in dep_rows if r["elements"])
    print(f"decomposed: {n} rows; empty(parse-fail): {empty} ({empty/max(1,n):.1%})")
    if counts:
        print(f"elements/claim: median {counts[len(counts)//2]}, p10 {counts[len(counts)//10]}, "
              f"p90 {counts[9*len(counts)//10]}")
    if dep_counts:
        print(f"DEPENDENT claims decomposed: {len(dep_rows)}; elements/claim median "
              f"{dep_counts[len(dep_counts)//2]}")
    print("\n--- dependent-claim examples (claim_text -> elements) ---")
    for r in dep_rows[:6]:
        u = units[(r["app_id"], r["ifw"], r["claim_num"])]
        print(f"\n[{r['app_id']} c{r['claim_num']}] {u['claim_text'][:180]}")
        for e in r["elements"][:8]:
            print(f"    - {e[:120]}")
    ok = n > 0 and empty / max(1, n) < 0.15 and (not dep_counts or dep_counts[len(dep_counts)//2] >= 2)
    print(f"\nCHECK_{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_derive_a(_):
    src = AB.replace(".jsonl", "_elements.jsonl")
    dst = AB_A.replace(".jsonl", "_elements.jsonl")
    units = {}
    with open(AB) as f:
        for ln in f:
            u = json.loads(ln)
            units[(u["app_id"], u["ifw_number"], u["claim_num"])] = u["claim_text"]
    kept = dropped = 0
    with open(src) as fi, open(dst, "w") as fo:
        for ln in fi:
            r = json.loads(ln)
            ct = units.get((r["app_id"], r["ifw"], r["claim_num"]), "")
            if DEP_RE.search(ct[:300]):
                dropped += 1  # dependent -> arm A falls back to single union blob
                continue
            fo.write(ln)
            kept += 1
    print(f"arm-A elements: kept {kept} independent-claim rows, dropped {dropped} dependents "
          f"(they fall back to union-blob) -> {dst}")


def cmd_derive_delta(_):
    """Arm C: keep only DELTA elements for dependent claims (elements not lexically covered
    by the parent chain). 2026-06-16 lesson: examiner cites art against the ADDED limitation;
    parent elements drag the softmin (delta .55 > union .519 > parent .485 within-AUC)."""
    from claim_resolver import parse_claims, resolve_chain
    from score_gold_mixture import load_app_claims_cache
    app_claims = load_app_claims_cache(AB)
    units = {}
    with open(AB) as f:
        for ln in f:
            u = json.loads(ln)
            units[(u["app_id"], u["ifw_number"], u["claim_num"])] = u
    # arm-C units file (byte-identical) so elements_path() resolves per-arm
    with open(AB) as fi, open(AB_C, "w") as fo:
        fo.write(fi.read())
    src = AB.replace(".jsonl", "_elements.jsonl")
    dst = AB_C.replace(".jsonl", "_elements.jsonl")
    kept_ind = filt = emptied = 0
    with open(src) as fi, open(dst, "w") as fo:
        for ln in fi:
            r = json.loads(ln)
            u = units.get((r["app_id"], r["ifw"], r["claim_num"]))
            if not u or not DEP_RE.search(u["claim_text"][:300]):
                fo.write(ln)
                kept_ind += 1
                continue
            parent_text = ""
            pg = app_claims.get(r["app_id"])
            if pg:
                claims = parse_claims(pg)
                if r["claim_num"] in claims:
                    chain, _f, _s = resolve_chain(r["claim_num"], claims)
                    parent_text = " ".join(claims[c] for c in chain if c != r["claim_num"] and c in claims)
            ptoks = set(_toks(parent_text))
            delta = []
            for e in r["elements"]:
                et = _toks(e)
                cov = (sum(1 for w in et if w in ptoks) / len(et)) if et else 1.0
                if cov < 0.7:
                    delta.append(e)
            if not delta:
                delta = r["elements"][-2:]  # trailing elements = the delta by claim convention
                emptied += 1
            filt += 1
            fo.write(json.dumps({"app_id": r["app_id"], "ifw": r["ifw"],
                                 "claim_num": r["claim_num"], "elements": delta}) + "\n")
    print(f"arm-C elements: {kept_ind} independent rows kept whole; {filt} dependent rows "
          f"delta-filtered ({emptied} fell back to trailing-2) -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    c = sub.add_parser("check")
    c.add_argument("--elements", default=AB.replace(".jsonl", "_elements.jsonl"))
    sub.add_parser("derive-a")
    sub.add_parser("derive-delta")
    a = ap.parse_args()
    rc = {"sample": cmd_sample, "check": cmd_check, "derive-a": cmd_derive_a,
          "derive-delta": cmd_derive_delta}[a.cmd](a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
