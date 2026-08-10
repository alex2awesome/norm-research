#!/usr/bin/env python3
"""Parse gathered claim-matching guideline docs -> {name,description,guidance} metric bank (Gemma).

Cross-domain claim-matching thread (user 2026-07-10). Subagents saved cleaned markdown guideline
docs under datasets/claim-matching/guidelines/raw/{journalism,academic,patents}/*.md . This reads
each, and with Gemma-4 (offline vLLM batch) extracts the EXPLICIT, reusable evaluative CRITERIA the
document gives for JUDGING WHETHER A CLAIM MATCHES / IS SUPPORTED BY reference material — the
claim-comparison skill — into the same schema as the repo's other rubric banks
(extracted.rubrics_metrics[] = {name, description, guidance}).

Design per BEST-PRACTICES:
  - free-form generation + JSON-extract + retry (NEVER structured decoding: collapse risk).
  - verbatim-fidelity instruction (borrowed from extract_rubric_features_v5verbose) so a reader of
    description+guidance recovers what the source said, no inventions.
  - task-specific few-shot of a GOOD claim-matching metric (else the model emits generic themes).
  - keep only criteria about COMPARING a claim to a reference (match / support / disclosure /
    contradiction / degree-of-support), drop generic drafting/writing advice.

Run on sk3, gemma4 env:
  CUDA_VISIBLE_DEVICES=N python scripts/claim_matching_parse.py
"""
import json, os, re, glob, sys

BASE = "/lfs/skampere3/0/alexspan/norm-research"
RAW = f"{BASE}/datasets/claim-matching/guidelines/raw"
OUT = f"{BASE}/datasets/claim-matching/guidelines/gemma-parsed"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"

SYS = (
    "You extract explicit, reusable evaluative CRITERIA (metrics) from a professional guidance "
    "document. The parent task is CLAIM-MATCHING: judging whether a stated CLAIM (an assertion, a "
    "patent claim limitation, a sentence attributed to a source) is MATCHED, SUPPORTED, "
    "DISCLOSED, CONTRADICTED, or UNSUPPORTED by a REFERENCE (prior-art passage, cited source, body "
    "of evidence). Extract ONLY criteria a practitioner uses to compare a claim against a reference "
    "and rate the match. DROP generic writing/drafting/formatting/procedure advice that is not "
    "about the claim-vs-reference comparison itself."
)

FIDELITY = """
For every criterion you extract, name/description/guidance must be as VERBOSE and SPECIFIC as the
source supports while NEVER inventing content beyond what the document says.
- name: a short label (5-15 words) using the document's own terminology.
- description: near-verbatim from the source — quote the rule/test in full, keep specific numbers,
  statute/section references (e.g. "MPEP 2131", "35 U.S.C. 102", "GRADE downgrade domain"),
  thresholds, enumerated conditions, and any worked example the source gives. Do not paraphrase 4
  sentences of substance down to 1.
- guidance: surrounding context — examples, anti-patterns, exceptions, scoring notes, sub-rules.
  Empty string only if the source genuinely says nothing more.
Never generalize away source-specific detail; never smooth idiosyncratic wording into generic
phrasing. If the source is terse, be terse.
"""

FEWSHOT = """
EXAMPLE of a GOOD claim-matching criterion (patents, anticipation):
{"name": "All-elements rule (every limitation disclosed in a single reference)",
 "description": "For a claim to be anticipated under 35 U.S.C. 102, a single prior-art reference must disclose each and every element of the claim, arranged as in the claim. Missing even one limitation defeats anticipation.",
 "guidance": "Map each claim limitation to specific reference text; if any limitation is not found in that one reference, there is no anticipation (though it may still be obvious under 103)."}
EXAMPLE of a GOOD claim-matching criterion (journalism, corroboration):
{"name": "Independent corroboration of factual claims",
 "description": "A contested factual claim should be confirmed by at least two independent sources, or by primary documentary evidence, before it is stated as fact rather than attributed.",
 "guidance": "Single-source claims are attributed, not asserted; sources sharing an origin do not count as independent."}
BAD (drop this — generic drafting advice, not claim-vs-reference comparison):
{"name": "Use clear antecedent basis", ...}
"""

INSTR = (
    "Read the DOCUMENT. Output ONE JSON object, nothing else:\n"
    '{"domain": "<journalism|academic|patents>", "subtask_short": "<=8 words", '
    '"subtask_description": "1-2 sentences on what claim-comparison this doc governs", '
    '"rubrics_metrics": [{"name": "...", "description": "...", "guidance": "..."}, ...]}\n'
    "Extract every distinct claim-matching criterion the document actually states (typically "
    "3-15). If the document has NO claim-comparison criteria, return an empty rubrics_metrics list."
)

_OBJ = re.compile(r"\{[\s\S]*\}")


def parse_json(raw):
    m = _OBJ.search(raw or "")
    if not m:
        return None
    for fix in (lambda s: s, lambda s: re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            o = json.loads(fix(m.group(0)))
            if isinstance(o, dict) and "rubrics_metrics" in o:
                return o
        except Exception:
            continue
    return None


def build_prompt(domain, title, body):
    return (f"{FIDELITY}\n{FEWSHOT}\n{INSTR}\n\nDOMAIN: {domain}\nDOCUMENT TITLE: {title}\n\n"
            f"DOCUMENT:\n{body[:14000]}")


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{RAW}/*/*.md"))
    print(f"[parse] {len(files)} raw guideline docs", flush=True)
    if not files:
        print("no raw docs found — sync from laptop first", flush=True); sys.exit(1)

    from vllm import LLM, SamplingParams
    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=20000,
              enable_prefix_caching=True, trust_remote_code=True)

    jobs = []
    for fp in files:
        domain = os.path.basename(os.path.dirname(fp))
        txt = open(fp, errors="ignore").read()
        title = next((l[2:].strip() for l in txt.splitlines() if l.startswith("# ")), os.path.basename(fp))
        jobs.append({"fp": fp, "domain": domain, "title": title, "body": txt})

    # two seeds: retry parse failures with a different sampling seed (no structured decoding)
    results = {}
    for seed, temp in ((0, 0.0), (1, 0.4)):
        todo = [j for j in jobs if j["fp"] not in results]
        if not todo:
            break
        sp = SamplingParams(temperature=temp, max_tokens=3000, seed=seed)
        convs = [[{"role": "system", "content": SYS},
                  {"role": "user", "content": build_prompt(j["domain"], j["title"], j["body"])}]
                 for j in todo]
        outs = llm.chat(convs, sp)
        for j, o in zip(todo, outs):
            parsed = parse_json(o.outputs[0].text)
            if parsed is not None:
                results[j["fp"]] = (j, parsed)
        print(f"[parse] seed {seed}: {len(results)}/{len(jobs)} parsed", flush=True)

    n_metrics = 0
    by_domain = {}
    for fp, (j, parsed) in results.items():
        parsed["domain"] = parsed.get("domain") or j["domain"]
        rec = {"source_file": fp, "domain": j["domain"], "title": j["title"], "extracted": parsed}
        base = f"{j['domain']}__{os.path.splitext(os.path.basename(fp))[0]}.json"
        json.dump(rec, open(f"{OUT}/{base}", "w"), indent=1)
        k = len(parsed.get("rubrics_metrics") or [])
        n_metrics += k
        by_domain[j["domain"]] = by_domain.get(j["domain"], 0) + k
    print(f"[parse] wrote {len(results)} docs, {n_metrics} raw metrics -> {OUT}", flush=True)
    print(f"[parse] by domain: {by_domain}", flush=True)
    print(f"[parse] failed to parse: {[os.path.basename(j['fp']) for j in jobs if j['fp'] not in results]}",
          flush=True)
    print("CLAIM_MATCHING_PARSE_DONE", flush=True)


if __name__ == "__main__":
    main()
