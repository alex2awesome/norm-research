#!/usr/bin/env python3
"""Build the certified-span core for the claim-matching testbed (label-cleaning levers 1+2). v3.

certified_uids — claims whose fetched GOLD span textually sits inside a paragraph the EXAMINER
  cited in the Office Action (span-level examiner testimony).
clean_uids — element assignment provenance-verified; superset of certified.

v3 fixes (Codex review 2026-07-12):
  #1 (critical) (app,claim) keys no longer collapse to one uid — every element uid is processed;
     duplicate Track-C unit rows are MERGED (gp/gold lists extended), not last-write-wins.
  #4 anchors come only from ELEMENT-COMPATIBLE provenance records (containment >= ELEM_T), span
     containment is tested PER anchored paragraph (max, not vocabulary-union of a concatenation),
     and the matched anchor + containment are persisted per uid.
  #7 element verification requires containment >= .6 AND >= MIN_SHARED shared tokens (3-of-5
     trivial passes no longer qualify).
  Cardinality asserts + membership provenance in the output JSON.

Primary sources (full coverage): gold_mixture_testbed_v1/units.jsonl (Track-C GP paragraph scrape),
option3_claims_gemma_scale.jsonl (gold ref doc per uid), pair_testbed_v2.jsonl (elements/spans).
No LLM judgment anywhere — joins + token containment only.

  python scripts/claim_matching_certify.py           # scored 800-claim probe
  python scripts/claim_matching_certify.py --full    # all 35,857 testbed claims -> _full output
"""
import argparse, json, re, collections

BASE = "/lfs/skampere3/0/alexspan/norm-research"
PROC = f"{BASE}/datasets/patents/processed"
SCORES = f"{BASE}/outputs/claim_matching/scores_gemma3_12b.jsonl"   # defines the scored probe
TESTBED = f"{BASE}/datasets/claim-matching/testbed/pair_testbed_v2.jsonl"
UNITS = f"{PROC}/gold_mixture_testbed_v1/units.jsonl"
SCALE = f"{PROC}/option3_claims_gemma_scale.jsonl"
OUT = f"{BASE}/outputs/claim_matching/certified_core.json"

WORD = re.compile(r"[a-z0-9]{3,}")
ANCHOR = re.compile(r"\[(\d{4})\](?:\s*-\s*\[?(\d{4})\]?)?")
APPHEAD = re.compile(r'"app_id":\s*"?(\d+)"?')
PLACEHOLDER = "(no per-element breakdown)"
BOILER_ONLY = re.compile(r"^the .{0,60} of claim \d+[.,]?$", re.I)
ELEM_T, SPAN_T, MIN_SHARED = 0.6, 0.7, 5


def toks(s):
    return set(WORD.findall((s or "").lower()))


def anchors_of(raw):
    out = set()
    for m in ANCHOR.finditer(raw or ""):
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
        lo, hi = min(a, b), max(a, b)
        if hi - lo <= 30:
            out.update(f"{x:04d}" for x in range(lo, hi + 1))
        else:
            out.add(f"{a:04d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="certify ALL testbed claims (not just the scored probe)")
    a = ap.parse_args()
    out_path = OUT.replace(".json", "_full.json") if a.full else OUT
    probe = None if a.full else {json.loads(l)["uid"] for l in open(SCORES)}
    pos = {}
    for ln in open(TESTBED):
        r = json.loads(ln)
        if r["y"] == 1 and (probe is None or r["uid"] in probe):
            pos[r["uid"]] = r
    if probe is None:
        probe = set(pos)
    golddoc = {}
    for ln in open(SCALE):
        r = json.loads(ln)
        uid = str(r["uid"])
        if uid not in pos:
            continue
        for ref in r["refs"]:
            if str(ref.get("is_gold")) == "True":
                golddoc[uid] = str(ref.get("doc_id")); break

    # (app,claim) -> ALL uids (Codex #1: no last-write-wins collapse)
    keys = collections.defaultdict(list)
    for u, r in pos.items():
        keys[(str(r["app_id"]), str(r["claim_num"]))].append(u)
    n_multi = sum(1 for v in keys.values() if len(v) > 1)
    print(f"[certify] probe {len(probe)} claims; pos joined {len(pos)}; golddoc {len(golddoc)}; "
          f"{len(keys)} (app,claim) keys ({n_multi} carry >1 uid)", flush=True)
    assert sum(len(v) for v in keys.values()) == len(pos), "uid/key cardinality mismatch"

    # stream Track-C units; MERGE duplicate rows per key (Codex #1)
    apps = {k[0] for k in keys}
    units = collections.defaultdict(lambda: {"gp": [], "gold": []})
    nrow = ndup = 0
    for ln in open(UNITS):
        nrow += 1
        m = APPHEAD.search(ln[:600])
        if m is not None and m.group(1) not in apps:
            continue
        r = json.loads(ln)
        k = (str(r["app_id"]), str(r["claim_num"]))
        if k not in keys:
            continue
        if not (r.get("gold_provenance") or r.get("gold")):
            continue
        if units[k]["gp"] or units[k]["gold"]:
            ndup += 1
        units[k]["gp"].extend(r.get("gold_provenance") or [])
        units[k]["gold"].extend(r.get("gold") or [])
    print(f"[certify] scanned {nrow} unit rows; matched {len(units)}/{len(keys)} keys "
          f"({ndup} duplicate unit rows merged)", flush=True)

    certified, clean = [], []
    detail = {}
    stats = collections.Counter()
    by_rt = collections.defaultdict(collections.Counter)
    for k, uids in keys.items():
        u = units.get(k)
        for uid in uids:
            r = pos[uid]
            rt = r.get("rejection_type") or "NA"
            el = r["element"].strip()
            if el == PLACEHOLDER or BOILER_ONLY.match(el):
                stats["dropped_placeholder_boiler"] += 1; by_rt[rt]["dropped"] += 1; continue
            gd = golddoc.get(uid)
            if u is None or gd is None:
                stats["no_trackc_coverage"] += 1; by_rt[rt]["no_cov"] += 1; continue
            gps = [g for g in u["gp"] if str(g.get("doc_id")) == gd]
            if not gps:
                stats["no_provenance_for_golddoc"] += 1; by_rt[rt]["no_prov"] += 1; continue
            # lever 2 (Codex #7): containment >= ELEM_T AND >= MIN_SHARED shared tokens
            et = toks(el)
            compat = []
            for g in gps:
                gt = toks(g.get("claim_element"))
                shared = len(gt & et)
                cont = shared / max(1, len(gt))
                if cont >= ELEM_T and shared >= MIN_SHARED:
                    compat.append((cont, shared, g))
            if not compat:
                stats["element_orphan"] += 1; by_rt[rt]["orphan"] += 1; continue
            clean.append(uid)
            # lever 1 (Codex #4): anchors ONLY from element-compatible records; containment tested
            # PER anchored paragraph, take the max
            anch = set()
            for _, _, g in compat:
                anch |= anchors_of(g.get("location_raw"))
            paras = {}
            for g in u["gold"]:
                if str(g.get("doc_id")) == gd:
                    for aa, txt in (g.get("paragraphs") or {}).items():
                        if aa in anch:
                            paras.setdefault(aa, txt)
            if not paras:
                stats["clean_no_anchor_text"] += 1; by_rt[rt]["no_anchor_text"] += 1; continue
            st = toks(r["span"])
            best_a, best_c = None, 0.0
            for aa, txt in paras.items():
                c = len(st & toks(txt)) / max(1, len(st))
                if c > best_c:
                    best_a, best_c = aa, c
            if best_c >= SPAN_T:
                certified.append(uid)
                detail[uid] = {"anchor": best_a, "span_contain": round(best_c, 3),
                               "elem_contain": round(max(c for c, _, _ in compat), 3)}
                stats["certified"] += 1; by_rt[rt]["certified"] += 1
            else:
                stats["clean_span_elsewhere"] += 1; by_rt[rt]["span_elsewhere"] += 1

    assert len(set(certified)) == len(certified) and len(set(clean)) == len(clean), "dup uids"
    assert set(certified) <= set(clean), "certified not subset of clean"
    print(f"[certify] certified (span=examiner paragraph) {len(certified)} "
          f"({len(certified)/max(1,len(pos)):.1%} of probe)", flush=True)
    print(f"[certify] clean (element provenance-verified) {len(clean)} "
          f"({len(clean)/max(1,len(pos)):.1%})", flush=True)
    for kk, v in sorted(stats.items()):
        print(f"  {kk}: {v}", flush=True)
    print("[by rejection type]", flush=True)
    for rt, c in sorted(by_rt.items()):
        print(f"  {rt}: {dict(c)}", flush=True)

    json.dump({"certified_uids": certified, "clean_uids": clean, "detail": detail,
               "elem_threshold": ELEM_T, "span_threshold": SPAN_T, "min_shared": MIN_SHARED,
               "version": 3, "stats": dict(stats)},
              open(out_path, "w"), indent=1)
    print(f"CERTIFY_DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
