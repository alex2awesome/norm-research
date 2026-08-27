"""
extract_claims.py — vLLM offline batch extractor for typed, checkable claims
in codereview.stackexchange answers (CR.SE V-layer, analog of the math.se
sympy claim-checker).

Inputs:
  CSV (gz) with columns: text, judgement, split, question_id, answer_id, ...
  where `text` is "Question: <Q>\n\nAnswer: <A>". We extract from <A>.

Pipeline:
  1) For each (answer_id) build a few-shot extraction prompt.
  2) Run vLLM offline batch chat() over the whole pool — never an HTTP server.
  3) Three sampling passes with DIFFERENT seeds (NEVER repetition_penalty):
       pass1: greedy   (T=0.0)
       pass2: T=0.3, seed=11
       pass3: T=0.6, seed=29
     pass i only operates on rows that failed validation in pass i-1.
  4) Strict JSON + schema + verbatim-quote round-trip validation.
  5) Per-row isolation fallback: if a batch crashes (poison prompt), recover
     by re-running the surviving rows one at a time (we run them in small
     sub-batches; on engine-level failure, fall back to size 1).

Crash traps baked in:
  * apply_chat_template(tokenize=False) → render TEXT, then tokenizer.encode()
    SEPARATELY when measuring length. (transformers >=5 returns BatchEncoding
    from tokenize=True, and len() silently lies.)
  * Per-row isolation fallback (try-except around each llm.chat call).
  * Never repetition_penalty. Detect & retry instead.

Usage (on sk3):
  CUDA_VISIBLE_DEVICES=6 /lfs/skampere3/0/alexspan/miniconda3/bin/python \
      scripts/crse_claims/extract_claims.py \
      --input  datasets/code-review/crse_balanced_v1/crse_v1_propensity_balanced.csv.gz \
      --output outputs/crse_claims_pilot/claims.jsonl \
      --pilot-n 300 --stratify-by judgement --seed 42
"""
from __future__ import annotations
import argparse, gzip, json, os, random, re, sys, time
from pathlib import Path

# Pin HOME early (AFS-token trap for nohup jobs on sk3 — see memory note
# feedback_sk3_afs_tokens).
os.environ.setdefault("HOME", "/lfs/skampere3/0/alexspan")
os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/alexspan/.cache/huggingface")
os.environ.setdefault("TRITON_CACHE_DIR", "/lfs/skampere3/0/alexspan/.cache/triton")
os.environ.setdefault("VLLM_CACHE_ROOT", "/lfs/skampere3/0/alexspan/.cache/vllm")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/lfs/skampere3/0/alexspan/.cache/torchinductor")
os.environ.setdefault("XDG_CACHE_HOME", "/lfs/skampere3/0/alexspan/.cache")
os.environ.setdefault("TMPDIR", "/lfs/skampere3/0/alexspan/tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


# -------------------- prompt --------------------------------------------------

SYSTEM_PROMPT = """You extract TYPED, CHECKABLE CLAIMS from code-review answers.

A claim is a CONCRETE, FALSIFIABLE assertion about the user's code or about a
proposed alternative. Skip generic style advice, opinions, encouragement, and
meta-commentary. Skip claims that depend on unstated context.

NOT A CLAIM — never extract these (they are subjective or unverifiable):
  - aesthetic/style judgements: "this is more pythonic", "this is more
    readable", "cleaner", "simpler", "more concise", "reads better",
    "more idiomatic", "nicer", "more elegant"
  - design-taste statements: "minimizes use of static functions",
    "better separation of concerns", "this naming is clearer"
  - external facts we cannot test locally: library sizes ("the promise
    library is ~30KB"), download counts, popularity, "most people use X"
  - vague performance talk with no testable core: "this should be faster"
    with no mechanism, input, or measurable quantity

RECAST RULE: if a style remark CONTAINS a concrete falsifiable core, extract
ONLY the core as a checkable claim. Example: "use sum(xs), it's cleaner and
avoids the quadratic loop" -> extract the complexity claim (the loop is
O(n^2) / sum is O(n)), NOT the "cleaner" part. If there is no falsifiable
core, extract nothing.

ALLOWED CLAIM TYPES (use these exact strings in `claim_type`):
  - "complexity"     : a Big-O / running-time claim about a function or loop
                        ("this is O(n^2)", "this is linear in n")
  - "behavior"       : a claim about runtime behavior on a specific input
                        ("breaks on empty list", "throws KeyError when k missing",
                        "returns None for negative input")
  - "equivalence"    : "X does the same as Y" / "this is equivalent to Z"
  - "improvement"    : "this is faster", "this uses less memory" (comparative)
  - "api_fact"       : a fact about a library/builtin API
                        ("dict.get returns None by default",
                         "list.sort is in-place")
  - "other"          : anything else you think is checkable — but PREFER one of
                        the above when possible. (We will drop "other" downstream.)

For EVERY claim, return a VERBATIM source quote (a contiguous substring of
the answer text), which we will round-trip verify. The quote must contain the
language that supports the claim — NOT the surrounding setup.

If the answer contains NO checkable claims, return claims: [].

OUTPUT FORMAT — return a SINGLE JSON object, no markdown fences, no prose:

{
  "claims": [
    {
      "claim_type": "complexity" | "behavior" | "equivalence" | "improvement" | "api_fact" | "other",
      "claim_text": "ONE-sentence paraphrase of the claim, in the third person.",
      "source_quote": "VERBATIM substring of the answer (must appear character-exact).",
      "target": "what the claim is about — a function name, line, or 'OP code' / 'proposed code'",
      "language": "python" | "javascript" | "typescript" | "java" | "c" | "c++" | "c#" | "go" | "rust" | "ruby" | "php" | "sql" | "haskell" | "bash" | "html" | "css" | "kotlin" | "swift" | "scala" | "perl" | "lua" | "vb.net" | "objective-c" | "other" | "unknown",
      "inline_code": "OPTIONAL — the smallest self-contained code snippet from the answer that the claim is about (verbatim). If no code snippet is referenced, set to null."
    }
  ]
}

RULES:
  - source_quote MUST be a contiguous substring of the answer. If you cannot
    quote verbatim, do not emit the claim.
  - source_quote SHOULD be ≤ 240 chars. Trim ellipses; do not use "...".
  - claim_text MUST be a clean third-person paraphrase, not a copy of the quote.
  - inline_code, if present, MUST be a verbatim substring of the answer.
  - DO NOT invent claim types. Use one of the six exact strings.
"""

# Few-shots illustrate every type + the verbatim quote requirement.
FEWSHOT_USER_1 = """ANSWER:
You could replace the for-loop with `sum(xs)` which is O(n) instead of O(n^2)
and runs in C. Also, your `find_max` breaks on an empty list — it will raise
ValueError because max() of an empty sequence is undefined.
"""
FEWSHOT_ASST_1 = """{
  "claims": [
    {
      "claim_type": "improvement",
      "claim_text": "Replacing the for-loop with sum(xs) is faster because it runs in C.",
      "source_quote": "replace the for-loop with `sum(xs)` which is O(n) instead of O(n^2) and runs in C",
      "target": "OP code (the for-loop)",
      "language": "python",
      "inline_code": "sum(xs)"
    },
    {
      "claim_type": "complexity",
      "claim_text": "The OP's for-loop is O(n^2).",
      "source_quote": "which is O(n) instead of O(n^2)",
      "target": "OP code (the for-loop)",
      "language": "python",
      "inline_code": null
    },
    {
      "claim_type": "behavior",
      "claim_text": "find_max raises ValueError when called with an empty list.",
      "source_quote": "your `find_max` breaks on an empty list — it will raise ValueError because max() of an empty sequence is undefined",
      "target": "find_max",
      "language": "python",
      "inline_code": null
    }
  ]
}"""

FEWSHOT_USER_2 = """ANSWER:
Nice work overall — the code reads well. I would name things more
consistently though.
"""
FEWSHOT_ASST_2 = """{"claims": []}"""

# Anti-pattern few-shot: an answer FULL of style/aesthetic remarks that must
# all be rejected (or recast). Mirrors the dominant pilot failure mode.
FEWSHOT_USER_4 = """ANSWER:
The refactored toString is simpler and reads much better. I'd also say your
design minimizes use of static functions, which is good practice. If you want
promises, the bluebird library is approximately 30KB minified. One real issue:
calling `parseInt(s)` without a radix means strings like "08" were parsed as
octal in older engines, so always pass 10.
"""
FEWSHOT_ASST_4 = """{
  "claims": [
    {
      "claim_type": "behavior",
      "claim_text": "parseInt without an explicit radix parsed strings with a leading zero like \\"08\\" as octal in older JavaScript engines.",
      "source_quote": "calling `parseInt(s)` without a radix means strings like \\"08\\" were parsed as octal in older engines",
      "target": "parseInt(s)",
      "language": "javascript",
      "inline_code": "parseInt(s)"
    }
  ]
}"""

# Anti-pattern few-shot: pure style/quality answer -> empty claims, even
# though it sounds confident and technical.
FEWSHOT_USER_5 = """ANSWER:
This is much more pythonic than the original. The list comprehension is more
concise and more readable, and the helper is more elegant. I think the
solution would be cleaner if you extracted a function here.
"""
FEWSHOT_ASST_5 = """{"claims": []}"""

FEWSHOT_USER_3 = """ANSWER:
Use `dict.get(k)` instead of the `if k in d` check — it returns None by
default if the key is missing, so you don't need the branch. Note that this
changes behavior if `d[k]` is legitimately set to None.
"""
FEWSHOT_ASST_3 = """{
  "claims": [
    {
      "claim_type": "api_fact",
      "claim_text": "dict.get(k) returns None by default when k is missing from the dict.",
      "source_quote": "`dict.get(k)` ... returns None by default if the key is missing",
      "target": "dict.get",
      "language": "python",
      "inline_code": "dict.get(k)"
    },
    {
      "claim_type": "equivalence",
      "claim_text": "Replacing `if k in d` with dict.get(k) is equivalent when d[k] is never None.",
      "source_quote": "Use `dict.get(k)` instead of the `if k in d` check ... so you don't need the branch",
      "target": "OP code (the `if k in d` check)",
      "language": "python",
      "inline_code": null
    }
  ]
}"""


def build_messages(answer_text: str) -> list[dict]:
    """Render the system prompt + 3 few-shots + the new answer."""
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": FEWSHOT_USER_1},
        {"role": "assistant", "content": FEWSHOT_ASST_1},
        {"role": "user",      "content": FEWSHOT_USER_2},
        {"role": "assistant", "content": FEWSHOT_ASST_2},
        {"role": "user",      "content": FEWSHOT_USER_3},
        {"role": "assistant", "content": FEWSHOT_ASST_3},
        {"role": "user",      "content": FEWSHOT_USER_4},
        {"role": "assistant", "content": FEWSHOT_ASST_4},
        {"role": "user",      "content": FEWSHOT_USER_5},
        {"role": "assistant", "content": FEWSHOT_ASST_5},
        {"role": "user",      "content": f"ANSWER:\n{answer_text}\n"},
    ]


# -------------------- data ----------------------------------------------------

def parse_answer_text(row_text: str) -> str:
    """Split 'Question: ...\\n\\nAnswer: ...' into the answer portion. If no
    'Answer:' marker, return empty string."""
    if not row_text:
        return ""
    m = re.search(r"\n\nAnswer:\s*", row_text)
    if not m:
        return ""
    return row_text[m.end():].strip()


def load_pool(input_path: Path, pilot_n: int, stratify_by: str | None, seed: int,
              restrict_ids: set[str] | None = None) -> list[dict]:
    """Load the balanced CSV and return up to pilot_n rows, stratified.
    If restrict_ids is given, return EXACTLY those answer_ids (sample reuse
    across pilot re-runs) and skip stratified sampling."""
    import csv
    csv.field_size_limit(10_000_000)
    opener = gzip.open if str(input_path).endswith(".gz") else open
    rows = []
    with opener(input_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if restrict_ids is not None and r.get("answer_id") not in restrict_ids:
                continue
            r["_answer_text"] = parse_answer_text(r.get("text", ""))
            if not r["_answer_text"] or len(r["_answer_text"]) < 60:
                continue
            rows.append(r)
    if restrict_ids is not None:
        print(f"  restrict-ids: matched {len(rows)} / {len(restrict_ids)} ids",
              flush=True)
        return rows
    rng = random.Random(seed)
    if stratify_by and rows and stratify_by in rows[0]:
        from collections import defaultdict
        groups = defaultdict(list)
        for r in rows:
            groups[r[stratify_by]].append(r)
        per_group = max(1, pilot_n // max(1, len(groups)))
        picked = []
        for k, vs in groups.items():
            rng.shuffle(vs)
            picked.extend(vs[:per_group])
        rng.shuffle(picked)
        return picked[:pilot_n]
    rng.shuffle(rows)
    return rows[:pilot_n]


# -------------------- JSON salvage + validation ------------------------------

def salvage_json(raw: str):
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return None
    for end in range(len(s), start, -1):
        if s[end - 1] == "}":
            try:
                return json.loads(s[start:end])
            except Exception:
                continue
    return None


VALID_TYPES = {"complexity", "behavior", "equivalence", "improvement", "api_fact", "other"}
VALID_LANGS = {"python", "javascript", "typescript", "java", "c", "c++", "c#",
               "go", "rust", "ruby", "php", "sql", "haskell", "bash", "html",
               "css", "kotlin", "swift", "scala", "perl", "lua", "vb.net",
               "objective-c", "other", "unknown"}


def normalize_for_quote(s: str) -> str:
    """Loose-match: collapse whitespace, lowercase. The quote round-trip uses
    this. Returns a canonicalized version of s."""
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_appears_in_answer(quote: str, answer: str) -> bool:
    """True iff the quote appears character-exact (or whitespace-canon equal)
    in the answer. We allow whitespace normalization so the model isn't
    punished for word-wrapping differences."""
    if not quote:
        return False
    if quote in answer:
        return True
    return normalize_for_quote(quote) in normalize_for_quote(answer)


def validate_extraction(parsed, answer_text: str) -> tuple[bool, str | None, dict]:
    """Return (ok, reason_if_not_ok, normalized_parsed). Normalized_parsed
    has per-claim `quote_ok` and `quote_loose_ok` flags so downstream can
    keep partial info even when we mark the whole row invalid."""
    if not isinstance(parsed, dict):
        return False, "not_a_dict", {}
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        return False, "claims_not_list", {}
    out_claims = []
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            return False, f"claim_{i}_not_dict", {}
        for k in ("claim_type", "claim_text", "source_quote", "target", "language"):
            if k not in c:
                return False, f"claim_{i}_missing_{k}", {}
        ct = c["claim_type"]
        if ct not in VALID_TYPES:
            return False, f"claim_{i}_invalid_type={ct}", {}
        lang = c.get("language")
        if lang not in VALID_LANGS:
            return False, f"claim_{i}_invalid_language={lang}", {}
        q = c["source_quote"]
        if not isinstance(q, str) or not q.strip():
            return False, f"claim_{i}_empty_quote", {}
        if len(q) > 600:
            return False, f"claim_{i}_quote_too_long", {}
        # Round-trip
        exact = q in answer_text
        loose = exact or normalize_for_quote(q) in normalize_for_quote(answer_text)
        c2 = dict(c)
        c2["quote_exact"] = exact
        c2["quote_loose_ok"] = loose
        if not loose:
            return False, f"claim_{i}_quote_not_in_answer", {}
        out_claims.append(c2)
    return True, None, {"claims": out_claims}


# -------------------- driver -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="JSONL path for claims")
    ap.add_argument("--pilot-n", type=int, default=300)
    ap.add_argument("--stratify-by", default="judgement")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restrict-ids", default=None,
                    help="Optional file with one answer_id per line; if set, "
                         "run on EXACTLY these ids (sample reuse).")
    ap.add_argument("--model", default="/lfs/skampere3/0/shared_hf_cache/"
                    "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
                    "0e9e39f249a16976918f6564b8830bc894c89659")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="Margin: prompts ~2-3K, output ~1K. 8K is comfortable.")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.55,
                    help="Stacked on a shared B200; do NOT take >0.55 of a "
                         "151GB-free GPU. 8B BF16 fits in ~16GB.")
    ap.add_argument("--batch-size", type=int, default=300,
                    help="Submit thousands per llm.chat call in production; "
                         "for a 300-pilot send all at once.")
    ap.add_argument("--validate-only-n", type=int, default=0,
                    help="If >0, run only the first N rows end-to-end (used "
                         "for the 20-row pre-flight validation).")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    restrict_ids = None
    if args.restrict_ids:
        restrict_ids = {L.strip() for L in open(args.restrict_ids) if L.strip()}
        print(f"restricting to {len(restrict_ids)} answer_ids from "
              f"{args.restrict_ids}", flush=True)

    print(f"loading pool from {args.input}", flush=True)
    pool = load_pool(Path(args.input), args.pilot_n, args.stratify_by, args.seed,
                     restrict_ids=restrict_ids)
    print(f"  loaded {len(pool)} answers", flush=True)
    if args.validate_only_n:
        pool = pool[: args.validate_only_n]
        print(f"  validate-only: trimmed to {len(pool)}", flush=True)

    from vllm import LLM, SamplingParams
    print(f"loading vLLM model: {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
        dtype="auto",
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    pass_specs = [
        ("pass1", SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=1)),
        ("pass2", SamplingParams(temperature=0.3, max_tokens=args.max_tokens, seed=11)),
        ("pass3", SamplingParams(temperature=0.6, max_tokens=args.max_tokens, seed=29)),
    ]

    # Build prompts (rendered text — NEVER tokenize=True, which returns
    # BatchEncoding and breaks len() guards in transformers >=5).
    def render_prompt(answer: str) -> str:
        msgs = build_messages(answer)
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)

    def n_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Token-length guard with separate encode() so we get a real list of ints.
    skipped = []
    work = []
    for r in pool:
        ans = r["_answer_text"]
        # Truncate very long answers (we still keep them but trimmed) — CR.SE
        # answers are typically < 2K chars.
        if len(ans) > 8000:
            ans = ans[:8000]
            r["_truncated"] = True
        prompt = render_prompt(ans)
        toks = n_tokens(prompt)
        if toks > args.max_model_len - args.max_tokens - 64:
            skipped.append({"answer_id": r.get("answer_id"), "reason": "too_long",
                            "prompt_tokens": toks})
            continue
        work.append({"row": r, "answer": ans, "prompt": prompt})
    print(f"  prompts built: {len(work)}; skipped (too long): {len(skipped)}", flush=True)
    if skipped:
        with (out_path.parent / "skipped.jsonl").open("w") as f:
            for s in skipped:
                f.write(json.dumps(s) + "\n")

    # State per row: best validated extraction + pass diagnostics
    results = {i: {"row": w["row"], "answer": w["answer"],
                   "prompt_tokens": n_tokens(w["prompt"]),
                   "validated": None, "passes": [], "final_pass": None,
                   "final_error": None}
               for i, w in enumerate(work)}

    pending_idxs = list(range(len(work)))

    def run_pass(label, params, idxs):
        if not idxs:
            return [], []
        # Per-row isolation fallback: try whole batch, on engine error fall
        # back to per-row.
        try:
            outputs = llm.generate(
                [work[i]["prompt"] for i in idxs], params, use_tqdm=False)
        except Exception as e:
            print(f"  [{label}] batch generate failed ({e!r}); "
                  f"falling back to per-row", flush=True)
            outputs = []
            for i in idxs:
                try:
                    o = llm.generate([work[i]["prompt"]], params, use_tqdm=False)
                    outputs.append(o[0])
                except Exception as ee:
                    print(f"  [{label}] row {i} failed: {ee!r}", flush=True)
                    outputs.append(None)
        survivors, failures = [], []
        for i, out in zip(idxs, outputs):
            raw = ""
            if out is not None and getattr(out, "outputs", None):
                raw = out.outputs[0].text or ""
            parsed = salvage_json(raw)
            if parsed is None:
                results[i]["passes"].append({"pass": label, "ok": False,
                                              "error": "json_salvage_failed",
                                              "raw_head": raw[:300]})
                failures.append(i)
                continue
            ok, why, norm = validate_extraction(parsed, work[i]["answer"])
            if not ok:
                results[i]["passes"].append({"pass": label, "ok": False,
                                              "error": why,
                                              "raw_head": raw[:300]})
                failures.append(i)
                continue
            results[i]["passes"].append({"pass": label, "ok": True,
                                          "n_claims": len(norm["claims"])})
            results[i]["validated"] = norm
            results[i]["final_pass"] = label
            survivors.append(i)
        return survivors, failures

    t0 = time.perf_counter()
    for label, params in pass_specs:
        print(f"\n[{label}] running over {len(pending_idxs)} rows "
              f"(T={params.temperature}, seed={params.seed})", flush=True)
        survivors, failures = run_pass(label, params, pending_idxs)
        print(f"  [{label}] ok={len(survivors)}  fail={len(failures)}  "
              f"elapsed={time.perf_counter() - t0:.1f}s", flush=True)
        pending_idxs = failures
        if not pending_idxs:
            break

    # Mark final errors for rows that never validated.
    for i in pending_idxs:
        results[i]["final_error"] = (
            results[i]["passes"][-1]["error"] if results[i]["passes"] else "no_passes"
        )

    # Emit JSONL — one row per answer.
    with out_path.open("w") as f:
        for i in sorted(results):
            r = results[i]
            row = r["row"]
            rec = {
                "answer_id":    row.get("answer_id"),
                "question_id":  row.get("question_id"),
                "judgement":    row.get("judgement"),
                "split":        row.get("split"),
                "primary_tag":  row.get("primary_tag"),
                "answer_text":  r["answer"],
                "prompt_tokens": r["prompt_tokens"],
                "extracted":    r["validated"],
                "final_pass":   r["final_pass"],
                "final_error":  r["final_error"],
                "passes":       r["passes"],
            }
            f.write(json.dumps(rec) + "\n")

    print(f"\nDONE — wrote {len(results)} rows to {out_path}", flush=True)
    n_ok = sum(1 for r in results.values() if r["validated"] is not None)
    n_fail = len(results) - n_ok
    print(f"  validated: {n_ok}  ({100 * n_ok / max(1, len(results)):.1f}%)", flush=True)
    print(f"  failed:    {n_fail}", flush=True)


if __name__ == "__main__":
    main()
