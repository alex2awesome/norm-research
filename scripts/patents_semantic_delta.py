#!/usr/bin/env python3
"""Semantic-delta decomposition — the clean version of arm C.

Arm C filtered arm-B's full-union elements by LEXICAL parent-overlap (token containment < 0.7).
This instead asks the decomposer DIRECTLY, for each dependent claim, to list only the limitations
it ADDS beyond its parent chain — a semantic diff, not a lexical one. Independent claims are
decomposed normally (identical to arm B). Then score via score_gold_mixture (same v6a+xenc+softmin).

Writes units_semdelta.jsonl (copy of ab12k) + units_semdelta_elements.jsonl so score_gold_mixture
--units .../units_semdelta.jsonl picks up the semantic elements via elements_path().

Run ON sk3 (gemma4 env, 1 GPU):
  CUDA_VISIBLE_DEVICES=N python scripts/patents_semantic_delta.py
"""
import json, os, re, sys
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts")
from claim_resolver import parse_claims, resolve_chain
from score_gold_mixture import DEP_RE, load_app_claims_cache, parse_element_array, MAX_ELS, UNION_CAP

TB = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/gold_mixture_testbed_v1"
AB = f"{TB}/units_ab12k.jsonl"
SD = f"{TB}/units_semdelta.jsonl"
SD_ELS = f"{TB}/units_semdelta_elements.jsonl"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"

SYS_D = "You decompose US patent claims into their distinct limitations (elements)."
IND_PROMPT = ("CLAIM:\n{claim}\n\nList the claim's elements as a JSON array of short strings "
              "(one per distinct limitation, max {n}). Output ONLY the JSON array.")
# semantic delta: parent given as context, ask ONLY for what the dependent claim adds
DEP_PROMPT = ("PARENT CLAIM(S) (already known context — do NOT list these):\n{parent}\n\n"
              "DEPENDENT CLAIM {cn}:\n{target}\n\nList ONLY the NEW limitations that dependent "
              "claim {cn} ADDS beyond the parent claim(s) above — the narrowing features that are "
              "not already present in the parent. Give a JSON array of short strings (max {n}), "
              "one per added limitation. If the claim adds a single limitation, return one string. "
              "Output ONLY the JSON array.")


def parent_and_target(u, app_claims):
    """Return (parent_text, target_text) for a dependent claim, or (None, claim_text)."""
    pg = app_claims.get(u["app_id"])
    if not pg:
        return None, u["claim_text"][:UNION_CAP]
    claims = parse_claims(pg)
    cn = u["claim_num"]
    if cn not in claims:
        return None, u["claim_text"][:UNION_CAP]
    chain, _f, _s = resolve_chain(cn, claims)
    ordered = [c for c in reversed(chain) if c in claims]  # root .. target
    target = claims[cn]
    if len(ordered) <= 1:
        return None, target[:UNION_CAP]
    parent = "\n".join(claims[c] for c in ordered[:-1])
    return parent[:UNION_CAP], target[:2000]


def main():
    units, seen = [], set()
    with open(AB) as f:
        for ln in f:
            u = json.loads(ln)
            k = (u["app_id"], u["ifw_number"], u["claim_num"])
            if k in seen:
                continue
            seen.add(k)
            units.append(u)
    app_claims = load_app_claims_cache(AB)
    print(f"[semdelta] {len(units)} unique claims; app_claims {len(app_claims)}", flush=True)

    done = set()
    if os.path.exists(SD_ELS):
        for ln in open(SD_ELS):
            try:
                r = json.loads(ln); done.add((r["app_id"], r["ifw"], r["claim_num"]))
            except Exception:
                pass
    todo = [u for u in units if (u["app_id"], u["ifw_number"], u["claim_num"]) not in done]
    print(f"[semdelta] to decompose: {len(todo)} (done {len(done)})", flush=True)

    with open(SD, "w") as fo:
        for u in units:
            fo.write(json.dumps(u) + "\n")

    if todo:
        from vllm import LLM, SamplingParams
        llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.85,
                  max_model_len=8192, enable_prefix_caching=True, trust_remote_code=True)
        samp = SamplingParams(temperature=0.0, max_tokens=1200, top_p=1.0)
        arr_re = re.compile(r"\[.*\]", re.S)
        out_f = open(SD_ELS, "a", buffering=1)
        CH = 2000
        n_dep = n_ind = 0
        for s in range(0, len(todo), CH):
            chunk = todo[s:s + CH]
            msgs = []
            for u in chunk:
                if DEP_RE.search(u["claim_text"][:300]):
                    parent, target = parent_and_target(u, app_claims)
                    if parent:
                        n_dep += 1
                        content = DEP_PROMPT.format(parent=parent, target=target, cn=u["claim_num"], n=MAX_ELS)
                    else:  # dependent but no resolvable parent -> decompose whole
                        n_ind += 1
                        content = IND_PROMPT.format(claim=u["claim_text"][:UNION_CAP], n=MAX_ELS)
                else:
                    n_ind += 1
                    content = IND_PROMPT.format(claim=u["claim_text"][:UNION_CAP], n=MAX_ELS)
                msgs.append([{"role": "system", "content": SYS_D}, {"role": "user", "content": content}])
            outs = llm.chat(msgs, samp)
            for u, o in zip(chunk, outs):
                els = parse_element_array(o.outputs[0].text or "")
                out_f.write(json.dumps({"app_id": u["app_id"], "ifw": u["ifw_number"],
                                        "claim_num": u["claim_num"], "elements": els}) + "\n")
            os.fsync(out_f.fileno())
            print(f"[semdelta]   {min(s + CH, len(todo))}/{len(todo)} (dep-prompted {n_dep}, whole {n_ind})", flush=True)
        out_f.close()

    # quick stats
    rows = [json.loads(l) for l in open(SD_ELS)]
    counts = sorted(len(r["elements"]) for r in rows if r["elements"])
    empty = sum(1 for r in rows if not r["elements"])
    print(f"[semdelta] elements: {len(rows)} rows, empty {empty}, median els/claim "
          f"{counts[len(counts)//2] if counts else 0}", flush=True)
    print("SEMDELTA_DECOMPOSE_DONE", flush=True)


if __name__ == "__main__":
    main()
