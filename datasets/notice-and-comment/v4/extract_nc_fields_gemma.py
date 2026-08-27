#!/usr/bin/env python3
"""Gemma vLLM extraction pass for the notice-and-comment hybrid-code-metric aspects: 8 round-1
(_h0.py) aspects + 4 round-2 (_h1.py) aspects targeting properties BEYOND raw stance that
plausibly drive agency agreement (stance_alignment, ask_modesty, deference_tone,
technical_precision) + 2 round-3 (_h2.py) VERIFICATION-tier aspects that make correctness
decisions about the extracted content (numeric_consistency, authority_lookup).

For each (comment, aspect) pair, prompt Gemma to return JSON of that aspect's LLM_FIELDS
(loaded directly from the metric_seam modules in programs_notice_and_comment/ — single
source of truth). Aspects with LLM_FIELDS = {} (code-only, e.g. structure_org, deference_tone)
are skipped entirely — no prompts are issued for them. Output jsonl: {id: doc_id, fields:
{aspect: {field: value}}}. The downstream scorer (score_nc_code_metrics.py) computes
score(text, extracted, ops) for every aspect (code-only aspects get extracted={}).

Usage (vLLM env, 1 GPU):
  CUDA_VISIBLE_DEVICES=0 python extract_nc_fields_gemma.py \
      --shard nc_vat_sample.jsonl --out nc_fields.jsonl --model <gemma-path> --util 0.90
"""
import argparse, importlib.util, json, pathlib, re, sys

BASE = pathlib.Path(__file__).resolve().parents[3]  # repo root
PROG = BASE / "methods/metric_seam/hybrids/programs_notice_and_comment"

# round 1 (h0) aspects
ASPECTS_H0 = [
    "cba_rigor", "citation_validity", "redline_ask", "evidence_provenance",
    "alternatives_analysis", "structure_org", "legal_argument", "stake_specificity",
]

# round 2 (h1) aspects: properties BEYOND raw stance that plausibly drive agency agreement
ASPECTS_H1 = [
    "stance_alignment", "ask_modesty", "deference_tone", "technical_precision",
]

# round 3 (h2) aspects: VERIFICATION-tier -- code makes correctness decisions about the
# extracted content (does the quantitative argument add up? do the cited authorities exist?)
ASPECTS_H2 = [
    "numeric_consistency", "authority_lookup",
]

# (aspect_id, suffix) pairs, suffix in {"h0", "h1", "h2"} selects which *_h{n}.py file to load
ASPECTS = ([(aid, "h0") for aid in ASPECTS_H0] + [(aid, "h1") for aid in ASPECTS_H1]
           + [(aid, "h2") for aid in ASPECTS_H2])

ASPECT_LABELS = {
    "cba_rigor": "engagement with the rule's cost-benefit / economic analysis",
    "citation_validity": "legal citation precision (CFR/USC/Federal Register)",
    "redline_ask": "specificity of the requested change to the rule",
    "evidence_provenance": "named external evidence and firsthand credentials",
    "alternatives_analysis": "proposed policy alternatives and their trade-offs",
    "legal_argument": "the specific legal theory asserted against the rule",
    "stake_specificity": "specificity of who is affected and how",
    "stance_alignment": "how aligned the comment is with the rule as proposed "
                         "(support-with-refinements vs wholesale opposition vs demand to withdraw)",
    "ask_modesty": "how modest/implementable the comment's main requested change is",
    # deference_tone has LLM_FIELDS = {} (code-only) -> never looked up, no label needed
    "technical_precision": "engagement with the proposed rule's own technical apparatus "
                            "(preamble sections, docket exhibits, section numbers, defined terms)",
    "numeric_consistency": "whether the comment's quantitative argument holds together "
                            "arithmetically (components summing to totals, percentages "
                            "matching their part/whole pairs, no internal contradictions)",
    "authority_lookup": "whether the comment's cited legal authorities (CFR/USC/Federal "
                         "Register) actually exist",
}


def load_module(aid, suffix="h0"):
    spec = importlib.util.spec_from_file_location(f"nc_{aid}_{suffix}", PROG / f"{aid}_{suffix}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SYS = ("You extract structured fields from a PUBLIC COMMENT submitted on a proposed federal "
       "rule. For each question, give the shortest faithful answer, or NONE if the comment "
       "gives no evidence. Reply with ONE compact JSON object mapping each field name to its "
       "value. No prose.")


def build_prompt(text, aid, mod):
    fields = mod.LLM_FIELDS
    qs = "\n".join(f'"{k}": {v}' for k, v in fields.items())
    return (f"PUBLIC COMMENT:\n{text[:4000]}\n\n"
            f"We are assessing {ASPECT_LABELS[aid]}. Extract these fields:\n{qs}\n\n"
            f'Reply with JSON only, e.g. {{"field1": "...", "field2": "NONE"}}.')


_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(raw):
    if not raw:
        return {}
    m = _OBJ_RE.search(raw)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", m.group(0))))
        except Exception:
            return {}


def load_rows(shard_path):
    rows = []
    with open(shard_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_id = d.get("doc_id") or d.get("id")
            txt = (d.get("text") or "").strip()
            if not doc_id or len(txt) < 20:
                continue
            rows.append((doc_id, txt))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=str(pathlib.Path(__file__).resolve().parent / "nc_vat_sample.jsonl"))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "nc_fields.jsonl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    mods = {aid: load_module(aid, suffix) for aid, suffix in ASPECTS}
    llm_aspects = [aid for aid, _ in ASPECTS if mods[aid].LLM_FIELDS]
    skipped = [aid for aid, _ in ASPECTS if not mods[aid].LLM_FIELDS]
    print(f"[fields] LLM aspects: {llm_aspects}", flush=True)
    print(f"[fields] skipped (code-only, no LLM_FIELDS): {skipped}", flush=True)

    rows = load_rows(a.shard)
    print(f"[fields] {len(rows)} comments x {len(llm_aspects)} aspects = "
          f"{len(rows) * len(llm_aspects)} prompts", flush=True)

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=180)

    convs, key = [], []
    for doc_id, txt in rows:
        for aid in llm_aspects:
            convs.append([{"role": "user", "content": f"{SYS}\n\n{build_prompt(txt, aid, mods[aid])}"}])
            key.append((doc_id, aid))
    print(f"[fields] extracting {len(convs)} prompts ...", flush=True)
    outs = llm.chat(convs, sp)

    by_id = {}
    for (doc_id, aid), o in zip(key, outs):
        obj = parse_json(o.outputs[0].text)
        for f in mods[aid].LLM_FIELDS:
            obj.setdefault(f, "NONE")
        by_id.setdefault(doc_id, {})[aid] = obj

    with open(a.out, "w") as fh:
        for doc_id, _ in rows:
            fh.write(json.dumps({"id": doc_id, "fields": by_id.get(doc_id, {})}) + "\n")
    print(f"[fields] saved -> {a.out}  ({len(by_id)} ids)", flush=True)
    print("FIELDS_DONE", flush=True)


if __name__ == "__main__":
    main()
