#!/usr/bin/env python3
"""Norm extraction over mathlib4 PR review threads via offline vLLM.

Unit of classification = THREAD: the LLM classifies the norm articulated by
the FIRST comment; replies are context (for author_response / escalation).
Taxonomy and prompt skeleton: notes/thread_norm_taxonomy_pilot.md §4
(18 categories + norm_statement / vat / has_suggestion_block /
author_response / escalated_to_zulip).

Conventions copied from aops/scripts/judge_approach_sk3.py (the canonical
sk3 Qwen3.5-122B-A10B-FP8 offline batch judge):
  * env pinned BEFORE importing torch/vllm (HOME on /lfs, shared HF cache,
    VLLM_USE_FLASHINFER_MOE_FP8=0)
  * offline LLM.chat over thousands of prompts per call, never HTTP
  * token-fit guard renders with tokenize=False then tok.encode()
    (transformers>=5 BatchEncoding trap)
  * JSON sanitization (curly quotes, bad backslash escapes), field-wise
    salvage with salvaged=true
  * retry with a different seed on invalid output — NEVER repetition_penalty
  * append-only plain JSONL (never gzip-append), fsync per chunk,
    resume-safe via done thread_ids

Modes:
  --pilot  run on every 4th row of the 600-thread hand-labeled sample
           (150 threads), write thread_norms_pilot.jsonl, and print
           primary-category agreement vs the hand labels.
  (default) full corpus run: flatten pr_thread_comments.jsonl into threads,
           pre-route bot-authored threads as AUTOMATED (no LLM), classify
           the rest.

Usage (on sk3, GPU 3 only):
  HOME=/lfs/skampere3/0/alexspan CUDA_VISIBLE_DEVICES=3 nohup \
    /lfs/skampere3/0/alexspan/miniconda3/bin/python3.11 \
    extract_thread_norms_sk3.py \
    > /lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/thread_norms_extract.log 2>&1 &
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

# --- env BEFORE torch/vllm ---------------------------------------------------
if os.path.isdir("/lfs/skampere3/0/alexspan"):
    os.environ.setdefault("HOME", "/lfs/skampere3/0/alexspan")
    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
    os.environ.setdefault("TRITON_CACHE_DIR",
                          "/lfs/skampere3/0/alexspan/.cache/triton")
    os.environ.setdefault("VLLM_CACHE_ROOT",
                          "/lfs/skampere3/0/alexspan/.cache/vllm")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                          "/lfs/skampere3/0/alexspan/.cache/torchinductor")
    os.environ.setdefault("TMPDIR", "/lfs/skampere3/0/alexspan/tmp")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")  # Qwen3.5 MoE FP8

DEFAULT_MODEL = ("/lfs/skampere3/0/shared_hf_cache/hub/"
                 "models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/"
                 "fb53b9f3bdaab287c597d4e943783153ec527e06")

BASE = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib")
DEFAULT_INPUT = BASE / "pr_thread_comments.jsonl"
DEFAULT_OUT = BASE / "thread_norms.jsonl"
DEFAULT_PILOT_OUT = BASE / "thread_norms_pilot.jsonl"
DEFAULT_SAMPLE = BASE / "notes" / "thread_norm_pilot_sample.jsonl.gz"
DEFAULT_LABELS = BASE / "notes" / "thread_norm_pilot_labels.json"

CATEGORIES = [
    "PROOF_STYLE", "DOCUMENTATION", "API_COMPLETENESS", "NAMING",
    "STATEMENT_FORM", "SIMP_ATTRS", "FORMATTING", "GENERALITY",
    "ORGANIZATION", "DEPRECATION", "METAPROGRAMMING", "CI_INFRA",
    "DIFF_VIGILANCE", "DUPLICATION", "MATH_DESIGN", "PERFORMANCE",
    "PR_PROCESS", "OTHER_SOCIAL",
]
CATEGORY_SET = set(CATEGORIES)
VAT_SET = {"V", "A", "T"}
RESPONSE_SET = {"accepted", "pushed_back", "negotiated", "no_reply"}

# pilot hand-label code -> prompt category
PILOT2PROMPT = {
    "GOLF": "PROOF_STYLE", "DOC": "DOCUMENTATION", "API": "API_COMPLETENESS",
    "NAME": "NAMING", "SIG": "STATEMENT_FORM", "SIMP": "SIMP_ATTRS",
    "STYLE": "FORMATTING", "GEN": "GENERALITY", "ORG": "ORGANIZATION",
    "DEPR": "DEPRECATION", "META": "METAPROGRAMMING", "INFRA": "CI_INFRA",
    "VIGIL": "DIFF_VIGILANCE", "DUP": "DUPLICATION", "MATH": "MATH_DESIGN",
    "PERF": "PERFORMANCE", "SCOPE": "PR_PROCESS", "MISC": "OTHER_SOCIAL",
}

BOT_LOGINS = {
    "github-actions", "dependabot", "mathlib4-dependent-issues-bot",
    "leanprover-community-bot", "leanprover-community-bot-assistant",
    "leanprover-community-mathlib4-bot", "leanprover-bot",
}

INFRA_PREFIXES = (".github/", "scripts/", "Cache/", "lakefile",
                  "lean-toolchain", "LongestPole/")
META_PREFIXES = ("Mathlib/Tactic/", "MathlibTest/", "test/", "Mathlib/Util/",
                 "Mathlib/Lean/")

FIRST_CHAR_CAP = 8000    # first comment kept whole up to this (flag if cut)
REPLY_CHAR_CAP = 1500    # per reply
MAX_REPLIES = 8

SYSTEM_PROMPT = """\
You are analyzing code-review threads from mathlib4, the Lean 4 mathematical library.
Correctness is machine-checked by the Lean compiler and CI, so review comments express
quality norms BEYOND correctness. Classify the norm articulated by the FIRST comment of
the thread. Replies are provided as context only.

Categories (choose exactly one primary; optionally one secondary):

- PROOF_STYLE: shorter/cleaner/more robust proof requested: golfing, term vs tactic
  mode, squeezing simp calls, avoiding erw/change/defeq abuse, avoiding duplicated
  subproofs, extracting sublemmas, preferring sturdy tactics.
- DOCUMENTATION: add/fix/relocate a docstring or comment; explain a magic constant;
  keep comments in sync with code; module-doc content.
- API_COMPLETENESS: add missing variants (symm/iff/ofNat/equiv/dual), instances,
  induction principles; OR push back that a lemma/def is not worth having.
- NAMING: declaration name choice: casing, dot-notation, naming-convention patterns
  (Foo.of_bar, _mono, primes), name<->statement consistency, to_additive names.
- STATEMENT_FORM: how the statement is spelled: implicit vs explicit binders, variable
  sections, hypothesis spelling (0 < k vs k != 0, Ne vs Not), iff/eq direction,
  canonical spelling of an expression, private/protected.
- SIMP_ATTRS: attribute hygiene: should/shouldn't be @[simp]; simp-normal form; simp
  loops; norm_cast/gcongr/fun_prop/nontriviality/reassoc/grind tagging.
- FORMATTING: whitespace, indentation, line breaks, English grammar/typos, notation
  spacing, style-guide mechanics.
- GENERALITY: weaken/strengthen hypotheses or typeclasses of a STATEMENT ("monoid
  suffices"), universe generality, fake-generality pushback.
- ORGANIZATION: which file/section/namespace a declaration belongs in; import
  minimization; ordering of declarations.
- DEPRECATION: deprecated aliases, #align (mathlib3 port), porting notes,
  don't-delete-just-deprecate.
- METAPROGRAMMING: review of tactic/linter/meta code internals: monads, Expr handling,
  caches, elaboration, test robustness of tactics.
- CI_INFRA: GitHub workflows, build scripts, cache system, toolchain — non-Lean
  infrastructure.
- DIFF_VIGILANCE: questioning whether a change is intended: bad merges, accidental
  deletions, unexplained or unrelated diff hunks.
- DUPLICATION: the result/definition already exists in mathlib or core, or is a special
  case of an existing one; duplicate PR.
- MATH_DESIGN: choice of mathematical definition itself: canonical notion, junk-value
  conventions, typeclass-diamond safety, whether a generalization is mathematically
  meaningful.
- PERFORMANCE: compile-time/elaboration performance, benchmarks, heartbeats, instance
  cost.
- PR_PROCESS: PR scoping (split unrelated changes), follow-up-PR etiquette, review
  workflow (bors, delegation).
- OTHER_SOCIAL: praise, thanks, pointer comments ("same as above", "ditto"), bot
  notifications, unclassifiable.

Also output:
- norm_statement: one sentence stating the norm as a general rule (not the instance).
- vat: "V" if a program/linter/benchmark could check compliance; "A" if it requires
  natural-language/mathematical judgment an LLM could render; "T" if the thread shows
  the norm is contested or case-by-case among experts.
- has_suggestion_block: bool (first comment contains ```suggestion).
- author_response: one of {accepted, pushed_back, negotiated, no_reply}.
  Use "no_reply" when there are no replies; "accepted" when the author agrees or
  applies the change; "pushed_back" when the author disagrees; "negotiated" when the
  thread goes back and forth toward a compromise.
- escalated_to_zulip: bool (the thread defers the question to Zulip or links a Zulip
  discussion).

Respond with STRICT JSON only — a single JSON object, no markdown fences, no
commentary — with exactly these keys:
{"primary": "<CATEGORY>", "secondary": "<CATEGORY>" or null,
 "norm_statement": "<one sentence>", "vat": "V"|"A"|"T",
 "has_suggestion_block": true|false,
 "author_response": "accepted"|"pushed_back"|"negotiated"|"no_reply",
 "escalated_to_zulip": true|false}

Examples (real mathlib4 first comments, hand-validated):

1. PR#16877: "Instead of using `<;>`, can you use a `wlog` or a `suffices` for the inequality? That way you produce one proof and use it twice, rather than producing the same proof twice."
{"primary": "PROOF_STYLE", "secondary": null, "norm_statement": "A proof should not construct the same subproof twice; prove it once (wlog/suffices) and reuse it.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

2. PR#6570: "Could we have a doc-string? At least telling me what `CS` stands for. :-)"
{"primary": "DOCUMENTATION", "secondary": null, "norm_statement": "Every definition needs a docstring that at minimum expands its abbreviations.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

3. PR#9925: "Can you add the equiv versions too, where `e ⁻¹' Icc x y = Icc (e.symm x) (e.symm y)` or maybe also `e.symm ⁻¹' Icc x y = Icc (e x) (e y)`?"
{"primary": "API_COMPLETENESS", "secondary": null, "norm_statement": "When adding lemmas about a construction, also provide the standard variant forms (equiv/symm versions).", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

4. PR#28708: "`pdist` is impenetrable. I had absolutely no idea what it meant until I read both the lemma *and* its docstring."
{"primary": "NAMING", "secondary": null, "norm_statement": "A declaration name should be understandable without reading the statement or docstring.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

5. PR#30484: "Why are these arguments implicit? The other ones are all explicit."
{"primary": "STATEMENT_FORM", "secondary": null, "norm_statement": "Binder explicitness should follow the usual rules and be consistent with neighboring declarations.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

6. PR#21160: "Usually the more complicated expression goes on the left hand side, because then you can simplify expressions by rewriting with this lemma from left to right. Could you please swap the two sides of the equation (and the name)?"
{"primary": "SIMP_ATTRS", "secondary": "NAMING", "norm_statement": "Simp lemmas must be stated in simp-normal form, with the more complex expression on the left-hand side.", "vat": "V", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

7. PR#7919: "When you have a multiline tactic like `simp` or `rw`, all lines but the first one are indented by two more spaces to help locate the beginning and the end of the tactic."
{"primary": "FORMATTING", "secondary": null, "norm_statement": "Continuation lines of a multiline tactic are indented two extra spaces.", "vat": "V", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

8. PR#9343: "It's enough to assume that the domain is an `AddZeroClass`. Please rewrite assuming `AddZeroClass`."
{"primary": "GENERALITY", "secondary": null, "norm_statement": "State every result under the weakest typeclass assumptions that suffice.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

9. PR#24804: "I would suggest moving this part to a new file `WithTerminal.Discrete`, so as to reduce imports."
{"primary": "ORGANIZATION", "secondary": null, "norm_statement": "Split material into separate files when doing so reduces the import graph.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

10. PR#6875: "You should not delete #aligns (same below)"
{"primary": "DEPRECATION", "secondary": null, "norm_statement": "Migration bookkeeping (#align entries, deprecated aliases) must be preserved, not deleted.", "vat": "V", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

11. PR#36841: "This function should be in `MetaM`, because otherwise it may misuse the `SimpM` cache. ... you can only share a cache safely if every use of the cache is computing the same thing."
{"primary": "METAPROGRAMMING", "secondary": null, "norm_statement": "Meta code must run in the weakest monad that is cache-safe for what it computes.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

12. PR#23868: "Do you also need to disable the `build`, `lint` and `test` steps explicitly? The lean-action README is not clear to me, can you check?"
{"primary": "CI_INFRA", "secondary": null, "norm_statement": "CI workflow configuration must be verified against the documented behavior of the actions it uses.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

13. PR#19506: "I'm guessing this is a bad merge. Could someone try reverting this file?"
{"primary": "DIFF_VIGILANCE", "secondary": null, "norm_statement": "Diff hunks that look like merge artifacts must be confirmed as intended or reverted.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

14. PR#38218: "Special case of `Measurable.of_discrete`."
{"primary": "DUPLICATION", "secondary": null, "norm_statement": "Do not add a lemma that is a special case of an existing one; use the existing lemma.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

15. PR#25042: "Do we really want to have the power to control the junk values?"
{"primary": "MATH_DESIGN", "secondary": null, "norm_statement": "Junk-value conventions in a definition are a design choice that must be deliberately justified.", "vat": "T", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

16. PR#22842: "I worry this instance and the `isDomain` one below might be a performance drag; could you split them (together) to a separate PR so that we can benchmark them in isolation?"
{"primary": "PERFORMANCE", "secondary": "PR_PROCESS", "norm_statement": "Potentially expensive instances should be benchmarked in isolation before merging.", "vat": "V", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

17. PR#9403: "For a `feat` PR, it's better practice to not mix refactoring into it, unless you can find some reviewer who thinks it's immediately a good idea ... If you limit the PR to just adding `equivPresentedGroup` you'll have a better shot at getting it quickly reviewed."
{"primary": "PR_PROCESS", "secondary": null, "norm_statement": "A feature PR should not mix in refactoring; split unrelated changes into separate PRs.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}

18. PR#31895: "Powergolf! 🏌️ ⛳"
{"primary": "OTHER_SOCIAL", "secondary": null, "norm_statement": "Praise/social comment; no norm articulated.", "vat": "A", "has_suggestion_block": false, "author_response": "no_reply", "escalated_to_zulip": false}
"""


def trunc(s: str, cap: int) -> str:
    s = (s or "").strip()
    if len(s) > cap:
        return s[:cap] + " [...truncated]"
    return s


def comment_author(c) -> str:
    a = c.get("author")
    if isinstance(a, dict):
        return a.get("login") or "[deleted]"
    if isinstance(a, str):
        return a or "[deleted]"
    return "[deleted]"


def is_bot(login: str) -> bool:
    lo = (login or "").lower()
    return (lo in BOT_LOGINS or lo.endswith("[bot]") or lo.endswith("-bot")
            or lo.startswith("github-actions"))


def routing_hint(path: str):
    p = path or ""
    if p.startswith(INFRA_PREFIXES):
        return ("Routing hint: the file path is infrastructure "
                "(workflows/scripts/cache/toolchain) — CI_INFRA is likely "
                "but use your judgment.")
    if p.startswith(META_PREFIXES):
        return ("Routing hint: the file path is tactic/meta/test code — "
                "METAPROGRAMMING is likely but use your judgment.")
    return None


def make_row(thread_id, pr, t, comments):
    first = comments[0]
    created = first.get("createdAt") or ""
    year = int(created[:4]) if created[:4].isdigit() else None
    return {
        "thread_id": thread_id,
        "pr": pr,
        "year": year,
        "path": t.get("path"),
        "line": t.get("line"),
        "isResolved": bool(t.get("isResolved")),
        "isOutdated": bool(t.get("isOutdated")),
        "n_comments": len(comments),
        "first_author": comment_author(first),
        "comments": [
            {"author": comment_author(c), "body": c.get("body") or ""}
            for c in comments
        ],
    }


def iter_corpus_threads(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pr = d.get("number")
            nodes = (d.get("reviewThreads") or {}).get("nodes") or []
            for i, t in enumerate(nodes):
                comments = ((t.get("comments") or {}).get("nodes")) or []
                if not comments:
                    continue
                yield make_row(f"{pr}:{i}", pr, t, comments)


def load_pilot_rows(sample_path: Path, every: int = 4):
    rows = []
    with gzip.open(sample_path, "rt") as fh:
        for idx, line in enumerate(fh):
            d = json.loads(line)
            comments = d.get("comments") or []
            if not comments:
                continue
            row = make_row(f"pilot:{idx}", d.get("pr"), d, comments)
            if d.get("year") is not None:
                row["year"] = d["year"]
            row["pilot_index"] = idx
            rows.append(row)
    return [r for r in rows if r["pilot_index"] % every == 0]


def build_user_prompt(row, first_cap, reply_cap, max_replies):
    head = (f"PR #{row['pr']} ({row['year']}), file: {row['path']}, "
            f"resolved: {row['isResolved']}")
    parts = [head]
    hint = routing_hint(row["path"] or "")
    if hint:
        parts.append(hint)
    comments = row["comments"]
    parts.append(f"[REVIEWER {comments[0]['author']}] "
                 f"{trunc(comments[0]['body'], first_cap)}")
    replies = comments[1:]
    for j, c in enumerate(replies[:max_replies], 1):
        parts.append(f"[REPLY {j} — {c['author']}] {trunc(c['body'], reply_cap)}")
    if len(replies) > max_replies:
        parts.append(f"[... {len(replies) - max_replies} more replies omitted]")
    parts.append("JSON:")
    return "\n".join(parts)


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
BAD_ESCAPE_RE = re.compile(r'\\(?![\\"/bfnrtu])')


def _salvage_fields(raw: str):
    """Field-wise extraction when the JSON never closes (the known Qwen-FP8
    degeneration usually loops inside the final string field)."""
    d = {}
    m = re.search(r'"primary"\s*:\s*"([A-Z_]{3,30})"', raw)
    if m:
        d["primary"] = m.group(1)
    m = re.search(r'"secondary"\s*:\s*(null|"([A-Z_]{3,30})")', raw)
    if m:
        d["secondary"] = m.group(2) if m.group(2) else None
    m = re.search(r'"norm_statement"\s*:\s*"([^"]{3,500})"', raw)
    if m:
        d["norm_statement"] = m.group(1)
    m = re.search(r'"vat"\s*:\s*"([VAT])"', raw)
    if m:
        d["vat"] = m.group(1)
    for key in ("has_suggestion_block", "escalated_to_zulip"):
        m = re.search(rf'"{key}"\s*:\s*"?(true|false)"?', raw, re.IGNORECASE)
        if m:
            d[key] = m.group(1).lower() == "true"
    m = re.search(r'"author_response"\s*:\s*"(accepted|pushed_back|negotiated'
                  r'|no_reply)"', raw)
    if m:
        d["author_response"] = m.group(1)
    if all(k in d for k in ("primary", "norm_statement", "vat")):
        d["_salvaged"] = True
        return d
    return None


def parse_verdict(raw: str, has_replies: bool):
    """Return (verdict_dict, None) or (None, error_string)."""
    if not raw or not raw.strip():
        return None, "empty_output"
    raw = (raw.replace("“", '"').replace("”", '"')
              .replace("‘", "'").replace("’", "'"))
    d = None
    m = JSON_RE.search(raw)
    if m:
        for cand in (m.group(0), BAD_ESCAPE_RE.sub(r"\\\\", m.group(0))):
            try:
                d = json.loads(cand)
                break
            except Exception:
                continue
    if not isinstance(d, dict):
        d = _salvage_fields(raw)
    if d is None:
        return None, "no_json_object" if not m else "json_unrecoverable"

    primary = d.get("primary")
    if not isinstance(primary, str) or primary.strip().upper() not in \
            CATEGORY_SET:
        return None, f"bad_primary:{str(primary)[:40]}"
    primary = primary.strip().upper()

    secondary = d.get("secondary")
    if isinstance(secondary, str):
        s = secondary.strip().upper()
        if s in ("", "NULL", "NONE"):
            secondary = None
        elif s in CATEGORY_SET:
            secondary = s
        else:
            return None, f"bad_secondary:{str(secondary)[:40]}"
    elif secondary is not None:
        return None, "bad_secondary_type"
    if secondary == primary:
        secondary = None

    norm = d.get("norm_statement")
    if not isinstance(norm, str) or not (3 <= len(norm.strip()) <= 600):
        return None, "bad_norm_statement"

    vat = d.get("vat")
    if not isinstance(vat, str) or vat.strip().upper() not in VAT_SET:
        return None, f"bad_vat:{str(vat)[:20]}"
    vat = vat.strip().upper()

    for key in ("has_suggestion_block", "escalated_to_zulip"):
        v = d.get(key)
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        if not isinstance(v, bool):
            return None, f"bad_{key}"
        d[key] = v

    resp = d.get("author_response")
    if not isinstance(resp, str) or resp.strip().lower() not in RESPONSE_SET:
        return None, f"bad_author_response:{str(resp)[:30]}"
    resp = resp.strip().lower()
    if not has_replies:
        resp = "no_reply"  # mechanical override: no replies exist

    return {
        "primary": primary,
        "secondary": secondary,
        "norm_statement": norm.strip()[:600],
        "vat": vat,
        "has_suggestion_block": d["has_suggestion_block"],
        "author_response": resp,
        "escalated_to_zulip": d["escalated_to_zulip"],
        "salvaged": bool(d.get("_salvaged", False)),
    }, None


def base_rec(row):
    rec = {"thread_id": row["thread_id"], "pr": row["pr"],
           "year": row["year"], "path": row["path"], "line": row["line"],
           "isResolved": row["isResolved"], "isOutdated": row["isOutdated"],
           "n_comments": row["n_comments"],
           "first_author": row["first_author"],
           "suggestion_block_mech":
               "```suggestion" in (row["comments"][0]["body"] or ""),
           "zulip_link_mech": any("zulip" in (c["body"] or "").lower()
                                  for c in row["comments"])}
    if "pilot_index" in row:
        rec["pilot_index"] = row["pilot_index"]
    return rec


def load_done_ids(out_path: Path, retry_failed: bool = False):
    done, failed = set(), set()
    if not out_path.exists():
        return done
    with out_path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            tid = d.get("thread_id")
            if tid is None:
                continue
            done.add(tid)
            if d.get("extract_failed"):
                failed.add(tid)
            else:
                failed.discard(tid)
    return done - failed if retry_failed else done


def report_pilot_agreement(out_path: Path, labels_path: Path):
    labels = json.loads(labels_path.read_text())
    gold = {int(k): PILOT2PROMPT.get(v, v) for k, v in labels.items()}
    preds = {}
    with out_path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "pilot_index" in d and d.get("primary"):
                preds[d["pilot_index"]] = d["primary"]
    pairs = [(gold[i], p) for i, p in preds.items() if i in gold]
    if not pairs:
        print("PILOT: no scored rows!?")
        return
    n_ok = sum(g == p for g, p in pairs)
    acc = n_ok / len(pairs)
    print(f"\n=== PILOT AGREEMENT: {n_ok}/{len(pairs)} = {acc:.3f} ===")
    per_cat = Counter(g for g, _ in pairs)
    per_cat_ok = Counter(g for g, p in pairs if g == p)
    recalls = []
    print(f"{'category':<18} {'n':>4} {'recall':>7}")
    for cat, n in per_cat.most_common():
        r = per_cat_ok[cat] / n
        recalls.append(r)
        print(f"{cat:<18} {n:>4} {r:>7.2f}")
    print(f"macro recall: {sum(recalls) / len(recalls):.3f}")
    conf = Counter((g, p) for g, p in pairs if g != p)
    print("top confusions (gold -> pred):")
    for (g, p), n in conf.most_common(12):
        print(f"  {g} -> {p}: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=None,
                    help="default: thread_norms.jsonl "
                         "(thread_norms_pilot.jsonl in --pilot mode)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pilot", action="store_true",
                    help="run on the hand-labeled sample and score agreement")
    ap.add_argument("--pilot-sample", default=str(DEFAULT_SAMPLE))
    ap.add_argument("--pilot-labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--pilot-every", type=int, default=4,
                    help="take every Nth row of the 600 sample (4 -> 150)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=2000,
                    help="prompts per llm.chat() call (submit thousands)")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.93)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--retries", type=int, default=4,
                    help="extra passes with a different seed on bad output")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt rows whose only output is an "
                         "extract_failed sentinel")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else (
        DEFAULT_PILOT_OUT if args.pilot else DEFAULT_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        rows = load_pilot_rows(Path(args.pilot_sample), args.pilot_every)
        print(f"=== PILOT mode: {len(rows)} threads from "
              f"{args.pilot_sample} ===", flush=True)
    else:
        rows = list(iter_corpus_threads(Path(args.input)))
        print(f"=== FULL mode: {len(rows):,} threads from "
              f"{args.input} ===", flush=True)
    if args.limit:
        rows = rows[:args.limit]

    done = load_done_ids(out_path, retry_failed=args.retry_failed)
    pending = [r for r in rows if r["thread_id"] not in done]
    print(f"=== {len(rows):,} threads, {len(done):,} already done, "
          f"{len(pending):,} pending ===", flush=True)

    fout = out_path.open("a")
    classifiable = []
    n_bot = 0
    bot_authors = Counter()
    for r in pending:
        if (not args.pilot) and is_bot(r["first_author"]):
            fout.write(json.dumps({**base_rec(r), "primary": "AUTOMATED",
                                   "routed": "bot_author"},
                                  ensure_ascii=False) + "\n")
            n_bot += 1
            bot_authors[r["first_author"]] += 1
            continue
        classifiable.append(r)
    fout.flush()
    os.fsync(fout.fileno())
    print(f"=== {n_bot:,} bot-authored threads pre-routed AUTOMATED "
          f"{dict(bot_authors.most_common(10))}, "
          f"{len(classifiable):,} to classify ===", flush=True)
    if not classifiable:
        print("nothing to do")
        fout.close()
        if args.pilot:
            report_pilot_agreement(out_path, Path(args.pilot_labels))
        return

    from vllm import LLM, SamplingParams
    print(f"=== loading vLLM model: {args.model} ===", flush=True)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        kv_cache_dtype="auto",
        dtype="auto",
        tensor_parallel_size=args.tp,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,  # long shared system prompt
    )

    tok = llm.get_tokenizer()
    input_limit = args.max_model_len - args.max_tokens

    def n_tokens(convo):
        # render to text, then encode (transformers>=5 BatchEncoding trap)
        s = tok.apply_chat_template(convo, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        return len(tok.encode(s, add_special_tokens=False))

    def build_convo_fitted(row):
        first_cap, reply_cap, max_replies = (FIRST_CHAR_CAP, REPLY_CHAR_CAP,
                                             MAX_REPLIES)
        truncated = False
        while True:
            user = build_user_prompt(row, first_cap, reply_cap, max_replies)
            convo = [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user}]
            if n_tokens(convo) <= input_limit:
                first_body = row["comments"][0]["body"] or ""
                truncated = (truncated or len(first_body) > first_cap
                             or any(len(c["body"] or "") > reply_cap
                                    for c in row["comments"][1:max_replies+1])
                             or len(row["comments"]) - 1 > max_replies)
                return convo, truncated
            if reply_cap > 300:
                reply_cap = max(300, reply_cap // 2)
            elif max_replies > 2:
                max_replies = max(2, max_replies // 2)
            elif first_cap > 500:
                first_cap = max(500, first_cap // 2)
            else:
                return None, True  # unfittable even at minimum caps
            truncated = True

    t_start = time.time()
    n_done = n_ok = n_failed = 0
    try:
        queue = classifiable
        for attempt in range(args.retries + 1):
            if not queue:
                break
            if attempt == 0:
                sp = SamplingParams(temperature=0.0,
                                    max_tokens=args.max_tokens)
            else:
                # different seed, never repetition_penalty
                sp = SamplingParams(temperature=0.7, top_p=0.95,
                                    seed=args.seed + attempt,
                                    max_tokens=args.max_tokens)
            print(f"\n=== pass {attempt}: {len(queue):,} rows "
                  f"(temperature={sp.temperature}, "
                  f"seed={getattr(sp, 'seed', None)}) ===", flush=True)
            next_queue = []
            for ci in range(0, len(queue), args.chunk_size):
                batch_all = queue[ci:ci + args.chunk_size]
                batch, convos, truncs = [], [], []
                for r in batch_all:
                    convo, truncated = build_convo_fitted(r)
                    if convo is None:
                        fout.write(json.dumps(
                            {**base_rec(r), "extract_failed": "overlong",
                             "attempts": attempt + 1}) + "\n")
                        n_done += 1
                        n_failed += 1
                        continue
                    batch.append(r)
                    convos.append(convo)
                    truncs.append(truncated)
                if not batch:
                    continue
                t0 = time.time()
                try:
                    outs = llm.chat(convos, sp, use_tqdm=False,
                                    chat_template_kwargs={
                                        "enable_thinking": False})
                except Exception as e:
                    print(f"  chunk chat failed ({type(e).__name__}: "
                          f"{str(e)[:160]}); isolating row-by-row",
                          flush=True)
                    outs = []
                    for c_one in convos:
                        try:
                            outs.extend(llm.chat(
                                [c_one], sp, use_tqdm=False,
                                chat_template_kwargs={
                                    "enable_thinking": False}))
                        except Exception:
                            outs.append(None)
                dt = time.time() - t0
                chunk_ok = chunk_retry = 0
                err_counter = Counter()
                err_samples = []
                for row, out, truncated in zip(batch, outs, truncs):
                    if out is None:  # failed even in isolation
                        fout.write(json.dumps(
                            {**base_rec(row),
                             "extract_failed": "render_failed",
                             "attempts": attempt + 1}) + "\n")
                        n_done += 1
                        n_failed += 1
                        continue
                    raw = out.outputs[0].text if out.outputs else ""
                    verdict, err = parse_verdict(
                        raw, has_replies=row["n_comments"] > 1)
                    final = (attempt == args.retries)
                    if verdict is None:
                        err_counter[err] += 1
                        if len(err_samples) < 3:
                            snip = raw if len(raw) <= 400 else \
                                raw[:200] + " <...> " + raw[-200:]
                            err_samples.append(
                                (row["thread_id"], err,
                                 snip.replace("\n", "\\n")))
                        if not final:
                            next_queue.append(row)
                            chunk_retry += 1
                            continue
                        fout.write(json.dumps(
                            {**base_rec(row),
                             "extract_failed": f"bad_output:{err}",
                             "attempts": attempt + 1,
                             "raw_head": raw[:300]}) + "\n")
                        n_failed += 1
                    else:
                        fout.write(json.dumps(
                            {**base_rec(row), **verdict,
                             "truncated": truncated,
                             "attempts": attempt + 1},
                            ensure_ascii=False) + "\n")
                        n_ok += 1
                    n_done += 1
                    chunk_ok += 1
                fout.flush()
                os.fsync(fout.fileno())  # checkpoint after every chunk
                elapsed = time.time() - t_start
                rate = max(n_done, 1) / max(elapsed, 1)
                print(f"  pass{attempt} chunk {ci // args.chunk_size + 1}: "
                      f"{len(batch)} prompts in {dt:.0f}s | "
                      f"accepted={chunk_ok} retry={chunk_retry} | "
                      f"total done={n_done:,} ok={n_ok:,} "
                      f"failed={n_failed:,} rate={rate * 60:.1f}/min",
                      flush=True)
                if err_counter:
                    print(f"    errors: {dict(err_counter.most_common(8))}",
                          flush=True)
                    for tid_, e_, s_ in err_samples:
                        print(f"    sample thread={tid_} err={e_} raw: {s_}",
                              flush=True)
            queue = next_queue
    finally:
        fout.flush()
        try:
            os.fsync(fout.fileno())
        except Exception:
            pass
        fout.close()

    wall = time.time() - t_start
    print(f"\n=== DONE in {wall / 3600:.2f}h: classified={n_done:,} "
          f"ok={n_ok:,} failed={n_failed:,} (+{n_bot:,} bot-routed) ===")
    print(f"output: {out_path}")
    if args.pilot:
        report_pilot_agreement(out_path, Path(args.pilot_labels))


if __name__ == "__main__":
    main()
