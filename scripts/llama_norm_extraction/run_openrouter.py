#!/usr/bin/env python3
"""
Run norm-adjacent extraction via OpenRouter (Llama-3.3-70B-Instruct) on a small batch.
Iteration harness for the Llama prompt — small batches, fast turnaround, diff vs Claude.

Usage:
    python run_openrouter.py --task peer_review --n 5 --run-name v1
"""
import argparse, json, os, pathlib, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

ROOT = pathlib.Path(__file__).parent
NORM_RESEARCH = ROOT.parent.parent

# Per-task data sources. Each entry maps a task name to:
#   input_path      — JSONL or CSV.GZ of input units (n_sample first rows after shuffle)
#   input_loader    — "jsonl" or "csv_gz"
#   id_field        — column/key holding the unit_id
#   text_field      — column/key holding the input text
#   meta_fields     — extra columns to pass through into results for analysis (e.g. "venue", "language")
#   rubrics_jsonl   — canonical rubric vocabulary
TASK_SOURCES = {
    "peer_review": {
        "input_path": str(NORM_RESEARCH / "datasets/peer-review/extracted/smoke_v1/input_20.jsonl"),
        "input_loader": "jsonl",
        "id_field": "review_id",
        "text_field": "review_text",
        "meta_fields": ["venue", "decision"],
        "rubrics_jsonl": str(NORM_RESEARCH / "datasets/peer-review/extracted/smoke_v2/rubrics.jsonl"),
    },
    "code_review": {
        "input_path": str(NORM_RESEARCH / "datasets/code-review/code_review_pr_aggregated.csv.gz"),
        "input_loader": "csv_gz",
        "id_field": "paper_id",
        "text_field": "text",
        "meta_fields": ["language", "judgement", "n_comments"],
        "rubrics_jsonl": str(NORM_RESEARCH / "datasets/code-review/extracted/smoke_v1/rubrics.jsonl"),
    },
    "notice_and_comment": {
        "input_path": str(NORM_RESEARCH / "datasets/notice-and-comment/rtc_extracted/rtc_sections.parquet"),
        "input_loader": "parquet",
        "id_field": "rule_id",
        "text_field": "rtc_text",
        "meta_fields": ["agency", "rule_title", "rtc_text_len"],
        "rubrics_jsonl": str(NORM_RESEARCH / "datasets/notice-and-comment/rtc_extracted/rubrics.jsonl"),
    },
    "humor": {
        # One row per Reddit StandUpWorkshop thread; text = post + threaded substantive comments.
        # Unit choice = option (a): post + concatenated comments. Rationale: extracting
        # ALL norm signals across a discussion is the goal; per-pair rows would lose
        # the thread context that disambiguates whose joke is being judged.
        "input_path": str(NORM_RESEARCH / "datasets/humor/standup_reddit/filtered_threads.jsonl"),
        "input_loader": "jsonl",
        "id_field": "thread_id",
        "text_field": "text",
        "meta_fields": ["post_title", "score", "n_comments_kept", "month", "permalink"],
        "rubrics_jsonl": str(NORM_RESEARCH / "datasets/humor/standup_reddit/rubrics.jsonl"),
    },
    "aops_forum": {
        # One row per AoPS forum post (solution or reply on a contest problem thread).
        # `text` is `post_canonical` from priority_posts.jsonl.gz, preserved byte-identically
        # including LaTeX and BBCode. unit_id = "aops_post_<post_id>".
        "input_path": str(ROOT / "_smoke_inputs/aops_forum_smoke10.jsonl"),
        "input_loader": "jsonl",
        "id_field": "unit_id",
        "text_field": "text",
        "meta_fields": ["post_type", "topic_title", "thanks_received", "is_edited", "post_number"],
        "rubrics_jsonl": str(NORM_RESEARCH / "datasets/math/aops/rubrics.jsonl"),
    },
    "press_releases": {
        # One row per (press_release, news_article) coverage pair. text = the PR text and
        # the covering news article joined with a `---` separator (PR first, article second).
        # Choosing article+PR (not article-only) lets the model surface BOTH (a) passages
        # where the article echoes/qualifies/contests a PR claim and (b) passages where
        # the article adds context the PR omitted — without forcing a second loader. The
        # speaker is the journalist/outlet; the target being judged is the press release.
        # Built from press_release_modeling_dataset/train.csv joined to news_articles.csv
        # — see datasets/press-releases/pr_article_pairs.jsonl (n=2366).
        "input_path": str(NORM_RESEARCH / "datasets/press-releases/pr_article_pairs.jsonl"),
        "input_loader": "jsonl",
        "id_field": "pair_id",
        "text_field": "text",
        "meta_fields": ["outlet", "article_source", "pr_company", "judgement",
                         "press_release_id", "article_id"],
        "rubrics_jsonl": str(NORM_RESEARCH / "datasets/press-releases/rubrics.jsonl"),
    },
}

MODEL = os.environ.get("OR_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

def load_key():
    return pathlib.Path("~/.openrouter-api-key.txt").expanduser().read_text().strip()

def load_template():
    return (ROOT / "template.txt").read_text()

def load_config(task):
    return json.load(open(ROOT / "configs" / f"{task}.json"))

def load_few_shots(task):
    return json.load(open(ROOT / "few_shots" / f"{task}_examples.json"))

def load_rubrics(task):
    src = TASK_SOURCES[task]
    return [json.loads(l) for l in open(src["rubrics_jsonl"])]

def load_inputs(task, n, seed, exclude_ids=None):
    """Load N input units (after shuffling with `seed`), excluding any unit_ids in `exclude_ids`.
    Returns list of dicts with keys: unit_id, text, plus any meta_fields.
    """
    src = TASK_SOURCES[task]
    exclude_ids = exclude_ids or set()
    loader = src["input_loader"]
    rows = []
    if loader == "jsonl":
        for line in open(src["input_path"]):
            r = json.loads(line)
            uid = r[src["id_field"]]
            if uid in exclude_ids:
                continue
            entry = {"unit_id": uid, "text": r[src["text_field"]]}
            for m in src["meta_fields"]:
                if m in r:
                    entry[m] = r[m]
            rows.append(entry)
    elif loader == "csv_gz":
        import pandas as pd
        cols = [src["id_field"], src["text_field"]] + list(src["meta_fields"])
        df = pd.read_csv(src["input_path"], usecols=cols)
        df = df[~df[src["id_field"]].isin(exclude_ids)]
        for _, r in df.iterrows():
            entry = {"unit_id": r[src["id_field"]], "text": r[src["text_field"]]}
            for m in src["meta_fields"]:
                entry[m] = r[m]
            rows.append(entry)
    elif loader == "parquet":
        import pandas as pd
        df = pd.read_parquet(src["input_path"])
        df = df[~df[src["id_field"]].isin(exclude_ids)]
        for _, r in df.iterrows():
            entry = {"unit_id": str(r[src["id_field"]]), "text": str(r[src["text_field"]])}
            for m in src["meta_fields"]:
                if m in r.index:
                    entry[m] = r[m]
            rows.append(entry)
    else:
        raise ValueError(f"unknown loader: {loader}")
    import random
    random.Random(seed).shuffle(rows)
    return rows[:n]

def fmt_rubric_block(rubrics):
    lines = []
    for r in rubrics:
        rid = r["rubric_id"]
        name = r.get("name") or r.get("merged_name") or "?"
        desc = (r.get("description") or "")[:160].replace("\n", " ")
        lines.append(f"[{rid}] {name} — {desc}")
    return "\n".join(lines)

def fmt_worked_examples(demos):
    """Format demos as INPUT → OUTPUT pairs Llama can imitate."""
    chunks = []
    for i, d in enumerate(demos, 1):
        in_text = d["input"]["text"]
        # truncate to keep prompt small
        if len(in_text) > 1500:
            in_text = in_text[:1500] + "... [truncated]"
        out_json = json.dumps(d["output"], ensure_ascii=False)
        # Header context can be venue (peer_review) or language (code_review)
        ctx_bits = []
        for k in ("venue", "language", "judgement"):
            if k in d:
                ctx_bits.append(f"{k}={d[k]}")
        ctx = ", ".join(ctx_bits)
        chunks.append(
            f"## Example {i} ({d['label']}{', ' + ctx if ctx else ''})\n\n"
            f"INPUT (unit_id={d['input']['unit_id']}):\n{in_text}\n\n"
            f"OUTPUT (one line of JSON):\n{out_json}"
        )
    return "\n\n".join(chunks)

def build_prompt(task, unit_id, input_text, template, config, rubric_block, worked):
    return template.format(
        role=config["role"],
        input_unit=config["input_unit"],
        speaker_subject=config["speaker_subject"],
        speaker_possessive=config["speaker_possessive"],
        target=config["target"],
        norm_term=config["norm_term"],
        norm_term_plural=config["norm_term_plural"],
        inline_evidence_example=config["inline_evidence_example"],
        polarity_hint=config.get("polarity_hint", ""),
        rubric_block=rubric_block,
        worked_examples=worked,
        unit_id=unit_id,
        input_text=input_text,
    )

def call_openrouter(api_key, prompt, max_tokens=16384, temperature=0.0):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "provider": {"require_parameters": True},
        # Disable thinking for Qwen3.5 / other reasoning models
        "reasoning": {"exclude": True, "max_tokens": 0},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://stanford-research.norm-research",
            "X-Title": "norm-research llama smoke",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())

def call_sk3(prompt, max_tokens=12000, temperature=0.0,
             url="http://localhost:8001/v1/chat/completions",
             model="qwen3.5-122b-a10b-fp8"):
    """Hit a local sk3 vLLM (OpenAI-compatible). Run on sk3 itself or via SSH tunnel.

    Per `reference_qwen35_vllm_sk3`: chat_template_kwargs at TOP-LEVEL, not extra_body.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())

def parse_output(text):
    """Llama may emit prose around the JSON. Try strict, then json-repair, then balanced extract+repair."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip("` \n")
    # Detect truncation: response should end with } (after stripping prose).
    # If the model ran out of tokens mid-output, we shouldn't trust json-repair's invented closure.
    s_stripped = s.rstrip()
    is_truncated = not s_stripped.endswith("}")
    try:
        return json.loads(s), None
    except Exception:
        pass
    if is_truncated:
        return None, "truncated (no closing brace)"
    # Try json-repair (handles LaTeX escape issues, trailing commas, etc.)
    try:
        import json_repair
        repaired = json_repair.loads(s)
        if isinstance(repaired, dict):
            return repaired, "repaired"
    except Exception:
        pass
    # find balanced top-level object, then try repair on the substring
    start = s.find("{")
    if start == -1:
        return None, "no opening brace"
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{": depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                substr = s[start:i+1]
                try:
                    return json.loads(substr), None
                except Exception:
                    try:
                        import json_repair
                        return json_repair.loads(substr), "repaired"
                    except Exception as e:
                        return None, f"parse failed: {e}"
    return None, "no balanced object"

import re as _re
_WS = _re.compile(r"\s+")
def _norm(s): return _WS.sub(" ", s).strip()

def snap_to_source(span, source, min_len=20):
    """Find best substring of source that matches span. Returns (snapped_text, match_type) or (span, 'no_match')."""
    if not span: return span, "empty"
    if span in source: return span, "exact"
    # Try whitespace-normalized match
    nspan = _norm(span)
    nsrc = _norm(source)
    if nspan in nsrc:
        # Find offset in normalized and map back to source (approximate)
        i = nsrc.find(nspan)
        # walk through source counting normalized chars
        src_idx, n_idx = 0, 0
        start = None
        while src_idx < len(source) and n_idx < i + len(nspan):
            if start is None and n_idx == i:
                start = src_idx
            c = source[src_idx]
            if c.isspace():
                # whitespace becomes single space in normalized
                # advance n_idx only on first whitespace of a run
                if src_idx == 0 or not source[src_idx-1].isspace():
                    n_idx += 1
            else:
                n_idx += 1
            src_idx += 1
        if start is not None:
            return source[start:src_idx].rstrip(), "ws_normalized"
    # Try anchor-based fuzzy: find the first 20 chars in source, then expand
    if len(span) >= min_len:
        anchor = span[:min_len]
        nanchor = _norm(anchor)
        idx = nsrc.find(nanchor)
        if idx != -1:
            # roughly map back
            return span, "anchor_only"  # leave as-is but mark
    return span, "no_match"

def validate_and_snap(parsed, source_text):
    """Validate + snap spans to source. Modifies `parsed` in place. Returns (issues, snap_stats)."""
    issues = []
    snaps = {"exact": 0, "ws_normalized": 0, "no_match": 0, "anchor_only": 0, "empty": 0}
    if not isinstance(parsed, dict):
        return ["not a dict"], snaps
    for required in ("unit_id", "passages"):
        if required not in parsed:
            issues.append(f"missing key: {required}")
    passages = parsed.get("passages", [])
    if not isinstance(passages, list):
        issues.append("passages not a list")
        return issues, snaps
    for pi, p in enumerate(passages):
        ptext_orig = p.get("passage_text", "")
        ptext, kind = snap_to_source(ptext_orig, source_text)
        snaps[kind] += 1
        p["passage_text"] = ptext
        p["_snap_kind"] = kind
        if kind in ("no_match", "anchor_only"):
            issues.append(f"passage[{pi}] could not snap to source (kind={kind})")
        # Now validate signals against (possibly snapped) passage
        for si, s in enumerate(p.get("signals", [])):
            stext_orig = s.get("signal_text", "")
            stext, skind = snap_to_source(stext_orig, ptext)
            snaps[skind] = snaps.get(skind, 0) + 1
            s["signal_text"] = stext
            s["_snap_kind"] = skind
            if skind in ("no_match", "anchor_only"):
                issues.append(f"passage[{pi}].signal[{si}] could not snap to passage (kind={skind})")
            if s.get("signal_type") not in ("complaint","praise","observation","suggestion"):
                issues.append(f"passage[{pi}].signal[{si}] bad signal_type={s.get('signal_type')}")
            if s.get("polarity") not in ("positive","negative","neutral"):
                issues.append(f"passage[{pi}].signal[{si}] bad polarity={s.get('polarity')}")
    return issues, snaps

# Keep old name for compat
def validate(parsed, source_text):
    return validate_and_snap(parsed, source_text)[0]

def process_one(api_key, prompt, source_text, unit_id, backend="openrouter", sk3_url=None):
    t0 = time.time()
    try:
        if backend == "sk3":
            resp = call_sk3(prompt, url=sk3_url or "http://localhost:8001/v1/chat/completions")
        else:
            resp = call_openrouter(api_key, prompt)
    except Exception as e:
        return {"unit_id": unit_id, "ok": False, "error": f"http: {e}", "elapsed": time.time()-t0}
    if "error" in resp:
        return {"unit_id": unit_id, "ok": False, "error": str(resp["error"]), "elapsed": time.time()-t0}
    try:
        msg = resp["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning") or ""
        if not raw:
            return {"unit_id": unit_id, "ok": False, "error": "empty content", "raw_response": resp, "elapsed": time.time()-t0}
    except Exception:
        return {"unit_id": unit_id, "ok": False, "error": "no choices", "raw_response": resp, "elapsed": time.time()-t0}
    parsed, perr = parse_output(raw)
    if parsed is None:
        return {"unit_id": unit_id, "ok": False, "error": f"parse: {perr}", "raw_text": raw, "elapsed": time.time()-t0}
    issues, snaps = validate_and_snap(parsed, source_text)
    return {
        "unit_id": unit_id,
        "ok": len(issues) == 0,
        "issues": issues,
        "snap_stats": snaps,
        "parsed": parsed,
        "raw_text": raw,
        "usage": resp.get("usage"),
        "elapsed": time.time()-t0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer_review")
    ap.add_argument("--n", type=int, default=5, help="number of reviews to process")
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-rubrics", type=int, default=0, help="if >0, sample this many rubrics (for prompt-size testing)")
    ap.add_argument("--backend", default="openrouter", choices=["openrouter", "sk3"], help="openrouter API or local sk3 vLLM")
    ap.add_argument("--sk3-url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--max-text-chars", type=int, default=20000, help="truncate input text to fit context budget")
    args = ap.parse_args()

    api_key = load_key() if args.backend == "openrouter" else None
    template = load_template()
    config = load_config(args.task)
    demos = load_few_shots(args.task)
    rubrics = load_rubrics(args.task)
    if args.limit_rubrics > 0:
        rubrics = rubrics[:args.limit_rubrics]
    rubric_block = fmt_rubric_block(rubrics)
    worked = fmt_worked_examples(demos)

    # Load test inputs — pick N NOT in the few-shot set
    fs_ids = {d["input"]["unit_id"] for d in demos}
    inputs = load_inputs(args.task, args.n, args.seed, exclude_ids=fs_ids)

    run_dir = ROOT / "runs" / f"{args.task}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt fragments for inspection
    (run_dir / "rubric_block.txt").write_text(rubric_block)
    (run_dir / "worked_examples.txt").write_text(worked)
    sample_prompt = build_prompt(args.task, inputs[0]["unit_id"], inputs[0]["text"][:2000],
                                  template, config, rubric_block, worked)
    (run_dir / "sample_prompt.txt").write_text(sample_prompt)
    print(f"=== sample prompt: {len(sample_prompt):,} chars ===")
    print(f"=== rubric block: {len(rubric_block):,} chars ({len(rubrics)} rubrics) ===")
    print(f"=== worked examples: {len(worked):,} chars ({len(demos)} demos) ===")
    print(f"=== processing {len(inputs)} units on {MODEL} via OpenRouter ===\n")

    # Save the set of unit_ids we'll process (for downstream Claude comparison)
    with open(run_dir / "input_units.jsonl", "w") as f:
        for r in inputs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for r in inputs:
            text = r["text"]
            if args.max_text_chars > 0 and len(text) > args.max_text_chars:
                text = text[:args.max_text_chars] + "\n\n... [truncated]"
            prompt = build_prompt(args.task, r["unit_id"], text,
                                   template, config, rubric_block, worked)
            futs[ex.submit(process_one, api_key, prompt, text, r["unit_id"], args.backend, args.sk3_url)] = r
        for fut in as_completed(futs):
            r = futs[fut]
            res = fut.result()
            # Pass through meta fields for analysis
            for k in TASK_SOURCES[args.task]["meta_fields"]:
                if k in r:
                    res[k] = r[k]
            results.append(res)
            # Write incrementally so a crash later doesn't lose work
            with open(run_dir / "results.jsonl", "w") as f:
                for x in results:
                    f.write(json.dumps(x, default=str) + "\n")
            ok = "✓" if res["ok"] else "✗"
            p = res.get("parsed")
            if isinstance(p, dict):
                n_pass = len(p.get("passages", []))
                n_sig = sum(len(pp.get("signals", [])) for pp in p.get("passages", []))
            elif isinstance(p, list):
                n_pass = sum(len(x.get("passages", [])) for x in p)
                n_sig = sum(len(pp.get("signals", [])) for x in p for pp in x.get("passages", []))
            else:
                n_pass = n_sig = 0
            # pick first meta field for display
            meta_keys = TASK_SOURCES[args.task]["meta_fields"]
            meta_val = r.get(meta_keys[0], "?") if meta_keys else "?"
            print(f"  {ok} {str(res['unit_id'])[:30]:>30} ({str(meta_val):>10})  pass={n_pass:>2} sig={n_sig:>3}  {res['elapsed']:.1f}s  issues={len(res.get('issues',[]))}")

    # Aggregates
    n_ok = sum(r["ok"] for r in results)
    n_parsed = sum(isinstance(r.get("parsed"), dict) for r in results)
    total_pass = sum(len(r["parsed"].get("passages", [])) for r in results if isinstance(r.get("parsed"), dict))
    total_sig = sum(len(p.get("signals", [])) for r in results if isinstance(r.get("parsed"), dict) for p in r["parsed"].get("passages", []))
    mean_sig_per_review = total_sig / max(1, n_parsed)

    print(f"\n=== SUMMARY ===")
    print(f"valid: {n_ok}/{len(results)}  parsed: {n_parsed}/{len(results)}")
    print(f"total passages: {total_pass}  total signals: {total_sig}  mean sig/review (parsed): {mean_sig_per_review:.1f}")
    print(f"output dir: {run_dir}")

if __name__ == "__main__":
    main()
