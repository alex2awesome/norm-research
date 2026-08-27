#!/usr/bin/env python3
"""LLM-judged articulable metrics for mathlib accept/reject.

Run on sk3, GLM/z.ai API:
  cd /lfs/skampere3/0/alexspan/norm-research
  HOME=/lfs/skampere3/0/alexspan \
    /lfs/skampere3/0/alexspan/envs/ai_usage/bin/python3 \
    scripts/mathlib_accept_reject_llm_judged_metrics.py --backend glm

Run on sk3, offline-batch Qwen/vLLM using LLM.generate, not an HTTP server:
  cd /lfs/skampere3/0/alexspan/norm-research
  HOME=/lfs/skampere3/0/alexspan \
    /lfs/skampere3/0/alexspan/envs/ai_usage/bin/python3 \
    scripts/mathlib_accept_reject_llm_judged_metrics.py --backend qwen-vllm

This script intentionally judges only diff_noauth. It never sends contributor
names, area, PR number, or size columns to the LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(os.environ.get("NORM_RESEARCH_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
BASE = Path(
    os.environ.get(
        "MATHLIB_BASE",
        "/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib",
    )
)
DEFAULT_DATA = BASE / "accept_reject_clean_deconf.parquet"
DEFAULT_V = BASE / "mathlib_diff_v_features.parquet"
DEFAULT_OUT = REPO / ".staging" / "mathlib_llm_judged_metrics_results.json"
DEFAULT_CACHE = REPO / ".staging" / "mathlib_llm_judged_metrics_cache.jsonl"
REFERENCE_VPP_AUC = 0.702
DEFAULT_QWEN_MODEL = (
    "/lfs/skampere3/0/shared_hf_cache/"
    "models--Qwen--Qwen3.5-122B-A10B-FP8/"
    "snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    name: str
    prompt: str


METRICS: list[MetricSpec] = [
    MetricSpec(
        "proof_elegance_conciseness",
        "Proof elegance/conciseness",
        """Judge the qualitative elegance and conciseness of the added or changed proofs.
Score 1-5:
1 = sprawling, brittle, noisy, or obscure proof construction; the core idea is hard to see.
2 = workable but awkward; avoidable tactic noise, detours, or low-level manipulation dominates.
3 = ordinary or mixed; acceptable proof style but no clear elegance signal, or insufficient evidence.
4 = clear and direct; proof steps expose the intended mathematical structure with little waste.
5 = unusually elegant; short or conceptually direct in a way that improves maintainability.
Do not score by line count alone: a longer structured proof can be elegant, and a short opaque proof can be poor.""",
    ),
    MetricSpec(
        "tactic_appropriateness",
        "Tactic appropriateness",
        """Judge whether the tactics are semantically appropriate for the local goals, not how many tactics appear.
Score 1-5:
1 = shotgun, fragile, or misleading tactic use; tactics appear to mask rather than solve the goal.
2 = mostly functional but poorly targeted; avoidable overpowered automation, brittle rewrites, or ad hoc simp sets.
3 = ordinary or mixed; tactics are plausible but not especially well matched, or evidence is limited.
4 = tactics are well chosen for the proof shape; rewrites, automation, and exact terms are used with clear purpose.
5 = excellent tactic fit; the proof uses the right library tactic or local argument in a precise, robust way.
Do not reward or punish the raw amount of tactic use.""",
    ),
    MetricSpec(
        "naming_quality",
        "Naming quality",
        """Judge names introduced or modified in the diff: declarations, lemmas, definitions, variables, hypotheses, namespaces.
Score 1-5:
1 = misleading, inconsistent, cryptic, or copy-pasted names that obscure meaning.
2 = names are partly informative but awkward, non-idiomatic, or likely to confuse later users.
3 = ordinary or mixed; names are serviceable, or there is too little naming evidence.
4 = clear, idiomatic mathlib-style names that communicate role and scope.
5 = names are especially precise and reusable; they make the API or proof structure easier to discover.
Do not score by name length alone.""",
    ),
    MetricSpec(
        "right_generality",
        "Right-generality",
        """Judge whether statements are pitched at the right mathematical/general API level.
Score 1-5:
1 = clearly over-specialized, over-generalized, or has unnecessary/missing hypotheses that harm reuse.
2 = likely not at the right abstraction level; some avoidable specialization or generality burden.
3 = ordinary or mixed; plausible generality, or insufficient evidence from the diff.
4 = good balance: hypotheses, typeclasses, variables, and statement form are reusable without being bloated.
5 = excellent abstraction boundary; the result is stated at the natural reusable level for mathlib.
Ignore topic/area prestige; judge only the statement and surrounding code visible in the diff.""",
    ),
    MetricSpec(
        "library_fit_api_design",
        "Library-fit/API-design",
        """Judge how well the change fits mathlib as a library/API contribution.
Score 1-5:
1 = duplicates existing API, fights local conventions, places concepts poorly, or exposes an awkward public surface.
2 = partially fits but has noticeable integration/API-design problems.
3 = ordinary or mixed; seems acceptable but with limited evidence of library-level fit.
4 = fits existing conventions, namespace/attribute choices, theorem shape, and downstream usability well.
5 = very strong library integration; the change looks like a natural, discoverable extension of existing API.
Do not use file path, area, or patch size as a proxy for fit.""",
    ),
    MetricSpec(
        "decomposition",
        "Decomposition",
        """Judge whether the diff decomposes the contribution into helpful lemmas/definitions/proof blocks.
Score 1-5:
1 = monolithic, tangled, or fragmented in a way that obscures dependencies and reuse.
2 = decomposition is weak; important ideas are buried, or there are excessive micro-lemmas without payoff.
3 = ordinary or mixed; structure is plausible, or the diff is too small to tell.
4 = clear decomposition; helper lemmas/sections/proof blocks make the main contribution easier to verify and reuse.
5 = excellent factoring; each component has a natural role and the whole change is easier to maintain because of it.
Do not score by the number of declarations alone.""",
    ),
    MetricSpec(
        "docstring_why",
        "Docstring-WHY",
        """Judge whether comments/docstrings explain the purpose, motivation, or non-obvious design choice where such explanation is expected.
Score 1-5:
1 = non-obvious public API or tricky proof with missing, misleading, or purely mechanical explanation.
2 = some explanation exists but mostly restates code or misses the reason a future user needs.
3 = ordinary or not applicable; no clear need for WHY-level documentation, or insufficient evidence.
4 = useful explanation of purpose/intuition/design tradeoff for the added API or proof.
5 = excellent WHY-level documentation that makes the contribution substantially easier to use or review.
Do not require docstrings for tiny obvious internal lemmas; use 3 when documentation is not clearly expected.""",
    ),
]

SYSTEM_PROMPT = """You are a careful mathlib reviewer judging qualitative properties of an author-stripped Lean/mathlib diff.

Important constraints:
- Use only the author-stripped diff text provided under DIFF_NOAUTH.
- Do not infer quality from contributor identity, PR number, topic area, file path prestige, or patch size.
- Do not predict whether the PR was accepted directly. Score the requested qualitative constructs.
- If the visible diff does not provide enough evidence for a metric, give score 3 for that metric.
- Return JSON only. No markdown, no prose outside JSON."""


def full_metric_prompt_template(spec: MetricSpec) -> str:
    return f"""SYSTEM:
{SYSTEM_PROMPT}

USER TEMPLATE:
Review an author-stripped Lean/mathlib diff under DIFF_NOAUTH. In the actual run all seven metrics are judged in one combined JSON response, but this is the complete rubric for `{spec.key}` ({spec.name}):

{spec.prompt}

The model sees only:
DIFF_NOAUTH:
--- BEGIN DIFF_NOAUTH ---
<author-stripped diff_noauth text>
--- END DIFF_NOAUTH ---

Return JSON field:
"{spec.key}": {{"score": <integer 1-5>, "reason": "<short reason grounded in the diff>"}}"""


def truncate_middle(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[...DIFF TRUNCATED...]\n\n" + text[-half:]


def build_user_prompt(diff_noauth: str, max_diff_chars: int) -> str:
    metric_text = "\n\n".join(
        f"METRIC `{m.key}` ({m.name}):\n{m.prompt}" for m in METRICS
    )
    diff_text = truncate_middle(diff_noauth, max_diff_chars)
    keys = ", ".join(f'"{m.key}"' for m in METRICS)
    return f"""Review this author-stripped diff and score every metric.

{metric_text}

Return exactly one JSON object with this schema:
{{
  "metrics": {{
    {keys}: {{"score": <integer 1-5>, "reason": "<short reason grounded in the diff>"}}
  }}
}}

The only input to judge is below. Do not use any missing metadata.

DIFF_NOAUTH:
--- BEGIN DIFF_NOAUTH ---
{diff_text}
--- END DIFF_NOAUTH ---"""


def prompt_hash(diff_noauth: str, max_diff_chars: int) -> str:
    material = json.dumps(
        {
            "system": SYSTEM_PROMPT,
            "user": build_user_prompt(diff_noauth, max_diff_chars),
            "metrics": [m.key for m in METRICS],
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def parse_jsonish(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    txt = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S | re.I)
    if fence:
        txt = fence.group(1).strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = JSON_OBJ_RE.search(txt)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def parse_scores(raw: str) -> dict[str, dict[str, Any]] | None:
    obj = parse_jsonish(raw)
    if not isinstance(obj, dict):
        return None
    metrics_obj = obj.get("metrics", obj)
    if not isinstance(metrics_obj, dict):
        return None
    out: dict[str, dict[str, Any]] = {}
    for spec in METRICS:
        val = metrics_obj.get(spec.key)
        if isinstance(val, dict):
            score = val.get("score")
            reason = val.get("reason", "")
        else:
            score = val
            reason = ""
        try:
            score_i = int(round(float(score)))
        except Exception:
            return None
        if score_i < 1 or score_i > 5:
            return None
        out[spec.key] = {"score": score_i, "reason": str(reason)[:500]}
    return out


def parse_diff_sections(diff: str) -> list[tuple[str, list[str]]]:
    if not isinstance(diff, str):
        return []
    sections: list[tuple[str, list[str]]] = []
    path = ""
    lines: list[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if path or lines:
                sections.append((path, lines))
            path = ""
            lines = []
            m = re.match(r"diff --git a/(.*?) b/(.*)", line)
            if m:
                path = m.group(2)
            continue
        if line.startswith("+++ b/"):
            path = line[6:]
            continue
        lines.append(line)
    if path or lines:
        sections.append((path, lines))
    return sections


def added_code_lines(diff: str, lean_only: bool = True) -> list[str]:
    lines: list[str] = []
    for path, sec in parse_diff_sections(diff):
        if lean_only and path and not path.endswith(".lean"):
            continue
        for line in sec:
            if line.startswith("+") and not line.startswith("+++") and not line.startswith("+--"):
                lines.append(line[1:])
    return lines


MANUAL_REF_RE = re.compile(
    r"\b(eq|Eq|congrArg|congrarg|le_trans|lt_of_le_of_lt|le_antisymm|Subtype\.ext|funext|propext|congr)\b"
)


def max_indent(lines: list[str]) -> int:
    vals = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    return max(vals) if vals else 0


def proof_complexity_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for number, diff in zip(df["number"], df["diff_noauth"]):
        added = added_code_lines(diff, lean_only=True)
        joined = "\n".join(added)
        have = len(re.findall(r"^\s*have\b", joined, flags=re.M))
        show = len(re.findall(r"^\s*show\b", joined, flags=re.M))
        calc = len(re.findall(r"^\s*calc\b", joined, flags=re.M))
        suffices = len(re.findall(r"^\s*suffices\b", joined, flags=re.M))
        let = len(re.findall(r"^\s*let\b", joined, flags=re.M))
        rows.append(
            {
                "number": number,
                "pc_proof_markers": len(re.findall(r"\bby\b|:=\s*by\b", joined)),
                "pc_have_lines": have,
                "pc_show_lines": show,
                "pc_calc_lines": calc,
                "pc_suffices_lines": suffices,
                "pc_let_lines": let,
                "pc_chain_lines": have + show + calc + suffices + let,
                "pc_nested_bullets": len(re.findall(r"^\s*[·.-]\s", joined, flags=re.M)),
                "pc_manual_lowlevel_refs": len(MANUAL_REF_RE.findall(joined)),
                "pc_max_indent": max_indent(added),
            }
        )
    return pd.DataFrame(rows)


def normalize_label(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    lower = s.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "true": 1,
        "accept": 1,
        "accepted": 1,
        "merged": 1,
        "yes": 1,
        "0": 0,
        "false": 0,
        "reject": 0,
        "rejected": 0,
        "closed": 0,
        "no": 0,
    }
    return lower.map(mapping).astype(int)


def load_real_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    df = pd.read_parquet(args.data_path)
    if "diff_noauth" not in df.columns:
        raise ValueError("Input parquet must contain diff_noauth; refusing to use diff.")
    if "additions" not in df.columns and "addn" in df.columns:
        df["additions"] = df["addn"]
    if "judgement" not in df.columns:
        raise ValueError("Input parquet must contain judgement.")
    df["judgement"] = normalize_label(df["judgement"])

    v_cols: list[str] = []
    if args.v_path and Path(args.v_path).exists():
        v = pd.read_parquet(args.v_path)
        if "number" not in v.columns:
            raise ValueError(f"{args.v_path} must contain number")
        v_cols = [c for c in v.columns if c != "number"]
        df = df.merge(v, on="number", how="left", suffixes=("", "_v"))
        v_cols = [c for c in v_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        candidate_v = [
            c
            for c in df.columns
            if (c.startswith("V") or c.startswith("v_"))
            and pd.api.types.is_numeric_dtype(df[c])
            and c not in {"number", "judgement"}
        ]
        v_cols = candidate_v

    tac_cols = sorted(
        c for c in df.columns if c.startswith("tac_") and pd.api.types.is_numeric_dtype(df[c])
    )

    existing_pc = [
        c
        for c in df.columns
        if (
            c.startswith("proof_")
            or c.startswith("pc_")
            or c
            in {
                "new_proof_markers",
                "new_have_lines",
                "new_show_lines",
                "new_calc_lines",
                "new_suffices_lines",
                "new_let_lines",
                "new_chain_lines",
                "new_nested_bullets",
                "new_manual_lowlevel_refs",
                "new_max_indent",
            }
        )
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    recomputed_pc_cols: list[str] = []
    if args.recompute_proof_complexity:
        pc = proof_complexity_features(df)
        recomputed_pc_cols = [c for c in pc.columns if c != "number"]
        for c in recomputed_pc_cols:
            if c in df.columns:
                df = df.drop(columns=[c])
        df = df.merge(pc, on="number", how="left")
    pc_cols = existing_pc + [c for c in recomputed_pc_cols if c not in existing_pc]

    vpp_cols = list(dict.fromkeys(v_cols + tac_cols + pc_cols))
    vpp_cols = [c for c in vpp_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    meta = {
        "data_path": str(args.data_path),
        "v_path": str(args.v_path),
        "df_shape": list(df.shape),
        "v_cols_n": len(v_cols),
        "tac_cols_n": len(tac_cols),
        "proof_complexity_cols_n": len(pc_cols),
        "vpp_cols_n": len(vpp_cols),
        "vpp_cols": vpp_cols,
        "split_counts": df["split"].value_counts(dropna=False).to_dict() if "split" in df.columns else {},
    }
    return df, vpp_cols, meta


def make_mock_data(seed: int) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    areas = ["t-algebra", "t-analysis", "t-topology", "t-data"]
    for i in range(120):
        y = int(i % 2 == 0)
        good = "simpa using h" if y else "simp_all"
        name = "map_mul_comp" if y else "foo_aux2"
        diff = f"""diff --git a/Mathlib/Mock/File{i % 5}.lean b/Mathlib/Mock/File{i % 5}.lean
--- a/Mathlib/Mock/File{i % 5}.lean
+++ b/Mathlib/Mock/File{i % 5}.lean
@@
+/-- Mock theorem for local smoke testing. -/
+lemma {name} (h : x = y) : x = y := by
+  {good}
"""
        rows.append(
            {
                "number": i,
                "diff_noauth": diff,
                "judgement": y,
                "split": "eval" if i < 80 else "train",
                "area": areas[i % len(areas)],
                "additions": int(rng.integers(3, 60)),
                "V_build_clean": float(y) + rng.normal(0, 0.8),
                "V_lint_clean": float(y) + rng.normal(0, 0.8),
                "tac_simp": int(rng.integers(0, 4)),
                "tac_exact": int(rng.integers(0, 3)),
            }
        )
    df = pd.DataFrame(rows)
    pc = proof_complexity_features(df)
    df = df.merge(pc, on="number", how="left")
    vpp_cols = ["V_build_clean", "V_lint_clean", "tac_simp", "tac_exact"] + [
        c for c in pc.columns if c != "number"
    ]
    return df, vpp_cols, {
        "mock": True,
        "df_shape": list(df.shape),
        "vpp_cols_n": len(vpp_cols),
        "vpp_cols": vpp_cols,
    }


def balanced_eval_sample(df: pd.DataFrame, max_sample: int, seed: int) -> pd.DataFrame:
    if "split" in df.columns:
        ev = df[df["split"].astype(str).eq("eval")].copy()
    else:
        ev = df.copy()
    if ev.empty:
        raise ValueError("No eval split rows found.")
    y = ev["judgement"].astype(int)
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError("Eval split has fewer than two judgement classes.")
    n_each = min(max_sample // 2, int(counts.min()))
    parts = []
    for cls in [0, 1]:
        parts.append(ev[y.eq(cls)].sample(n=n_each, random_state=seed))
    sample = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Drop author-bearing diff defensively. The judge path below has no access to it.
    sample = sample.drop(columns=["diff"], errors="ignore")
    return sample


def load_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path or not cache_path.exists():
        return cache
    with cache_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("key") and rec.get("scores"):
                cache[rec["key"]] = rec
    return cache


def append_cache(cache_path: Path | None, rec: dict[str, Any]) -> None:
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def prepared_tasks(sample: pd.DataFrame, max_diff_chars: int) -> list[dict[str, Any]]:
    tasks = []
    for idx, row in sample.iterrows():
        diff = str(row["diff_noauth"] or "")
        key = prompt_hash(diff, max_diff_chars)
        tasks.append(
            {
                "idx": int(idx),
                "number": int(row["number"]) if pd.notna(row.get("number")) else int(idx),
                "key": key,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(diff, max_diff_chars)},
                ],
            }
        )
    return tasks


def mock_scores_for_task(task: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    user = task["messages"][1]["content"].lower()
    base = 3
    if "simpa using" in user or "map_mul_comp" in user:
        base += 1
    if "foo_aux2" in user or "simp_all" in user:
        base -= 1
    h = int(hashlib.sha256(task["key"].encode()).hexdigest()[:8], 16)
    scores = {}
    for i, spec in enumerate(METRICS):
        val = max(1, min(5, base + ((h >> (i * 2)) & 1) - 0))
        scores[spec.key] = {"score": int(val), "reason": "mock score for smoke test"}
    raw = json.dumps({"metrics": scores})
    return raw, scores


def judge_mock(tasks: list[dict[str, Any]], cache: dict[str, Any], cache_path: Path | None) -> dict[int, dict[str, Any]]:
    out = {}
    for t in tasks:
        if t["key"] in cache:
            out[t["idx"]] = cache[t["key"]]
            continue
        raw, scores = mock_scores_for_task(t)
        rec = {
            "key": t["key"],
            "number": t["number"],
            "raw": raw,
            "scores": scores,
            "backend": "mock",
        }
        append_cache(cache_path, rec)
        out[t["idx"]] = rec
    return out


def judge_glm(
    tasks: list[dict[str, Any]],
    cache: dict[str, Any],
    cache_path: Path | None,
    args: argparse.Namespace,
) -> dict[int, dict[str, Any]]:
    import requests

    api_key = os.environ.get(args.glm_api_key_env) or os.environ.get("ZAI_API_KEY") or os.environ.get("BIGMODEL_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"No GLM API key found. Set {args.glm_api_key_env}, ZAI_API_KEY, or BIGMODEL_API_KEY."
        )
    # GLM via the FREE z.ai subscription (Anthropic-format) endpoint, not pay-per-token paas.
    base_url = args.glm_base_url
    if "/paas/" in base_url:
        base_url = "https://api.z.ai/api/anthropic/v1/messages"
    session = requests.Session()
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    out: dict[int, dict[str, Any]] = {}
    todo = [t for t in tasks if t["key"] not in cache]
    for t in tasks:
        if t["key"] in cache:
            out[t["idx"]] = cache[t["key"]]
    print(f"JUDGE_BACKEND glm(free-anthropic) cached={len(out)} todo={len(todo)}", flush=True)
    for n, t in enumerate(todo, 1):
        sys_msgs = [m["content"] for m in t["messages"] if m.get("role") == "system"]
        usr_msgs = [{"role": m["role"], "content": m["content"]} for m in t["messages"] if m.get("role") != "system"]
        body = {
            "model": args.glm_model,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "messages": usr_msgs,
        }
        if sys_msgs:
            body["system"] = "\n\n".join(str(s) for s in sys_msgs)
        raw = ""
        scores = None
        last_err = None
        for attempt in range(args.retries + 1):
            try:
                resp = session.post(base_url, headers=headers, json=body, timeout=args.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"http {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                raw = (data.get("content") or [{}])[0].get("text", "") or ""
                scores = parse_scores(raw)
                if scores is not None:
                    break
                last_err = "parse failed"
            except Exception as e:
                last_err = repr(e)
            if attempt < args.retries:
                time.sleep(args.retry_sleep * (2**attempt))
        if scores is None:
            raise RuntimeError(f"GLM failed for number={t['number']} key={t['key']}: {last_err}; raw={raw[:500]!r}")
        rec = {
            "key": t["key"],
            "number": t["number"],
            "raw": raw,
            "scores": scores,
            "backend": "glm",
            "model": args.glm_model,
        }
        append_cache(cache_path, rec)
        out[t["idx"]] = rec
        if n % 10 == 0 or n == len(todo):
            print(f"  glm judged {n}/{len(todo)}", flush=True)
    return out


def render_chat_prompts(llm: Any, messages: list[list[dict[str, str]]]) -> list[str]:
    tokenizer = llm.get_tokenizer()
    prompts: list[str] = []
    for msg in messages:
        try:
            prompts.append(
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            prompts.append(
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
    return prompts


def vllm_generate(llm: Any, messages: list[list[dict[str, str]]], sampling_params: Any) -> Any:
    prompts = render_chat_prompts(llm, messages)
    try:
        return llm.generate(prompts, sampling_params, use_tqdm=False)
    except TypeError:
        return llm.generate(prompts, sampling_params)


def judge_qwen_vllm(
    tasks: list[dict[str, Any]],
    cache: dict[str, Any],
    cache_path: Path | None,
    args: argparse.Namespace,
) -> dict[int, dict[str, Any]]:
    from vllm import LLM, SamplingParams

    out: dict[int, dict[str, Any]] = {}
    todo = [t for t in tasks if t["key"] not in cache]
    for t in tasks:
        if t["key"] in cache:
            out[t["idx"]] = cache[t["key"]]
    print(f"JUDGE_BACKEND qwen-vllm cached={len(out)} todo={len(todo)}", flush=True)
    if todo:
        llm = LLM(
            model=args.qwen_model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype="auto",
            trust_remote_code=True,
            enforce_eager=args.enforce_eager,
            max_num_seqs=args.batch_size,
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        )
        samp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, top_p=1.0)
        retry_samp = SamplingParams(temperature=0.2, max_tokens=args.max_tokens, top_p=0.95, seed=args.seed)
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start : start + args.batch_size]
            responses = vllm_generate(llm, [t["messages"] for t in chunk], samp)
            raws = [r.outputs[0].text for r in responses]
            parsed = [parse_scores(raw) for raw in raws]
            retry_idx = [i for i, p in enumerate(parsed) if p is None]
            if retry_idx and args.retries:
                retry_msgs = [chunk[i]["messages"] for i in retry_idx]
                retry_responses = vllm_generate(llm, retry_msgs, retry_samp)
                for local_i, resp in zip(retry_idx, retry_responses):
                    raws[local_i] = resp.outputs[0].text
                    parsed[local_i] = parse_scores(raws[local_i])
            for t, raw, scores in zip(chunk, raws, parsed):
                if scores is None:
                    raise RuntimeError(
                        f"Qwen parse failed for number={t['number']} key={t['key']}; raw={raw[:500]!r}"
                    )
                rec = {
                    "key": t["key"],
                    "number": t["number"],
                    "raw": raw,
                    "scores": scores,
                    "backend": "qwen-vllm",
                    "model": args.qwen_model,
                }
                append_cache(cache_path, rec)
                out[t["idx"]] = rec
            print(f"  qwen judged {min(start + len(chunk), len(todo))}/{len(todo)}", flush=True)
    return out


def attach_judgments(sample: pd.DataFrame, judgments: dict[int, dict[str, Any]]) -> pd.DataFrame:
    judged = sample.copy()
    for spec in METRICS:
        judged[f"judge_{spec.key}"] = np.nan
        judged[f"judge_{spec.key}_reason"] = ""
    for idx, rec in judgments.items():
        scores = rec.get("scores") or {}
        for spec in METRICS:
            judged.loc[idx, f"judge_{spec.key}"] = scores[spec.key]["score"]
            judged.loc[idx, f"judge_{spec.key}_reason"] = scores[spec.key].get("reason", "")
    missing = judged[[f"judge_{m.key}" for m in METRICS]].isna().any(axis=1).sum()
    if missing:
        raise RuntimeError(f"{missing} sample rows lack complete judgments")
    return judged


def safe_auc(y_true: Any, score: Any) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], s[mask]))


def numeric_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def oof_auc(df: pd.DataFrame, cols: list[str], seed: int, n_splits: int = 5) -> float:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return float("nan")
    y = df["judgement"].astype(int).to_numpy()
    min_class = int(pd.Series(y).value_counts().min())
    k = min(n_splits, min_class)
    if k < 2:
        return float("nan")
    pred = np.full(len(df), np.nan, dtype=float)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    x = df[cols].astype(float)
    for tr, te in skf.split(x, y):
        pipe = numeric_pipeline(seed)
        pipe.fit(x.iloc[tr], y[tr])
        pred[te] = pipe.predict_proba(x.iloc[te])[:, 1]
    return safe_auc(y, pred)


def spearman_numeric(x: Any, y: Any) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 4 or np.nanstd(x_arr[mask]) == 0 or np.nanstd(y_arr[mask]) == 0:
        return float("nan"), float("nan")
    rho, p = spearmanr(x_arr[mask], y_arr[mask])
    return float(rho), float(p)


def area_max_spearman(scores: pd.Series, area: pd.Series) -> dict[str, Any]:
    if area is None:
        return {"rho": float("nan"), "p": float("nan"), "area": None}
    best = {"rho": float("nan"), "p": float("nan"), "area": None}
    s = scores.astype(float)
    for val, mask in area.astype(str).fillna("NA").groupby(area.astype(str).fillna("NA")).groups.items():
        if len(mask) < 3 or len(mask) == len(area):
            continue
        indicator = pd.Series(0.0, index=area.index)
        indicator.loc[list(mask)] = 1.0
        rho, p = spearman_numeric(s, indicator)
        if math.isfinite(rho) and (not math.isfinite(best["rho"]) or abs(rho) > abs(best["rho"])):
            best = {"rho": rho, "p": p, "area": str(val)}
    return best


def residualize_on_area_size(df: pd.DataFrame, score_col: str) -> np.ndarray:
    y = df[score_col].astype(float).to_numpy()
    mats = []
    if "additions" in df.columns:
        mats.append(np.log1p(pd.to_numeric(df["additions"], errors="coerce").fillna(0).astype(float)).to_numpy()[:, None])
    if "area" in df.columns:
        mats.append(pd.get_dummies(df["area"].astype(str), prefix="area", dtype=float).to_numpy())
    if not mats:
        return y - np.nanmean(y)
    x = np.column_stack(mats)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    reg = LinearRegression()
    reg.fit(x, y)
    return y - reg.predict(x)


def compact_diff_snippet(diff: str, max_lines: int = 18, max_chars: int = 1800) -> str:
    if not isinstance(diff, str):
        return ""
    out = []
    current_path = ""
    for path, lines in parse_diff_sections(diff):
        current_path = path or current_path
        for line in lines:
            if line.startswith(("+++", "---", "@@")):
                continue
            if line.startswith(("+", "-")) and line.strip() not in {"+", "-"}:
                prefix = f"{current_path}: " if current_path else ""
                out.append(prefix + line)
            if len(out) >= max_lines:
                break
        if len(out) >= max_lines:
            break
    text = "\n".join(out)
    return truncate_middle(text, max_chars)


def aligned_examples(df: pd.DataFrame, metric_col: str, n: int = 3) -> list[dict[str, Any]]:
    tmp = df.copy()
    score = tmp[metric_col].astype(float)
    y = tmp["judgement"].astype(int)
    tmp["_align"] = np.where(y.eq(1), score, 6.0 - score)
    tmp = tmp.sort_values(["_align", metric_col], ascending=False)
    chosen = []
    seen_labels = set()
    # Try to include both labels when possible.
    for _, r in tmp.iterrows():
        lbl = int(r["judgement"])
        if lbl in seen_labels and len(chosen) < 2:
            continue
        chosen.append(r)
        seen_labels.add(lbl)
        if len(chosen) >= n:
            break
    for _, r in tmp.iterrows():
        if len(chosen) >= n:
            break
        if int(r["number"]) not in {int(x["number"]) for x in chosen}:
            chosen.append(r)
    examples = []
    reason_col = metric_col + "_reason"
    for r in chosen[:n]:
        examples.append(
            {
                "number": int(r["number"]) if pd.notna(r.get("number")) else None,
                "judgement": int(r["judgement"]),
                "score": float(r[metric_col]),
                "reason": str(r.get(reason_col, ""))[:300],
                "area": str(r.get("area", "")),
                "additions": int(r["additions"]) if "additions" in r and pd.notna(r["additions"]) else None,
                "snippet": compact_diff_snippet(str(r.get("diff_noauth", ""))),
            }
        )
    return examples


def evaluate(judged: pd.DataFrame, vpp_cols: list[str], seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_sample_auc = oof_auc(judged, vpp_cols, seed)
    results = []
    examples: dict[str, Any] = {}
    for spec in METRICS:
        col = f"judge_{spec.key}"
        resid_col = f"{col}_resid_area_additions"
        judged[resid_col] = residualize_on_area_size(judged, col)
        standalone_auc = safe_auc(judged["judgement"], judged[col])
        vpp_plus_auc = oof_auc(judged, vpp_cols + [col], seed)
        vpp_plus_resid_auc = oof_auc(judged, vpp_cols + [resid_col], seed)
        size_rho = size_p = float("nan")
        if "additions" in judged.columns:
            size_rho, size_p = spearman_numeric(
                judged[col],
                np.log1p(pd.to_numeric(judged["additions"], errors="coerce").fillna(0).astype(float)),
            )
        area = area_max_spearman(judged[col], judged["area"]) if "area" in judged.columns else {"rho": float("nan"), "p": float("nan"), "area": None}
        inc_lift = vpp_plus_auc - base_sample_auc if math.isfinite(vpp_plus_auc) and math.isfinite(base_sample_auc) else float("nan")
        confound_safe = (
            (not math.isfinite(size_rho) or abs(size_rho) <= 0.30)
            and (not math.isfinite(area["rho"]) or abs(area["rho"]) <= 0.30)
        )
        any_signal = (
            (math.isfinite(standalone_auc) and abs(standalone_auc - 0.5) >= 0.05)
            or (math.isfinite(inc_lift) and inc_lift > 0.005)
        )
        row = {
            "metric": spec.key,
            "metric_name": spec.name,
            "sample_n": int(len(judged)),
            "standalone_auc": standalone_auc,
            "sample_vpp_auc": base_sample_auc,
            "vpp_plus_metric_auc": vpp_plus_auc,
            "incremental_lift_vs_sample_vpp": inc_lift,
            "lift_vs_reference_0_702": vpp_plus_auc - REFERENCE_VPP_AUC if math.isfinite(vpp_plus_auc) else float("nan"),
            "beats_reference_0_702": bool(math.isfinite(vpp_plus_auc) and vpp_plus_auc > REFERENCE_VPP_AUC),
            "size_additions_rho": size_rho,
            "size_additions_p": size_p,
            "area_max_abs_rho": area["rho"],
            "area_max_abs_p": area["p"],
            "area_max_abs_area": area["area"],
            "confound_safe_rho_le_0_30": bool(confound_safe),
            "resid_area_additions_standalone_auc": safe_auc(judged["judgement"], judged[resid_col]),
            "vpp_plus_resid_metric_auc": vpp_plus_resid_auc,
            "prompt": spec.prompt,
            "full_judge_prompt": full_metric_prompt_template(spec),
        }
        results.append(row)
        if any_signal:
            examples[spec.key] = aligned_examples(judged, col, n=3)
    res = pd.DataFrame(results).sort_values(
        ["vpp_plus_metric_auc", "standalone_auc"], ascending=False
    )
    meta = {
        "sample_vpp_auc": base_sample_auc,
        "reference_vpp_full_cv_auc": REFERENCE_VPP_AUC,
        "examples": examples,
    }
    return res, meta


def print_results(results: pd.DataFrame, meta: dict[str, Any]) -> None:
    cols = [
        "metric_name",
        "standalone_auc",
        "sample_vpp_auc",
        "vpp_plus_metric_auc",
        "incremental_lift_vs_sample_vpp",
        "lift_vs_reference_0_702",
        "beats_reference_0_702",
        "size_additions_rho",
        "area_max_abs_rho",
        "confound_safe_rho_le_0_30",
    ]
    print("SUMMARY_VERDICT_TABLE")
    print(results[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nRANKED_METRICS_WITH_PROMPTS_JSON")
    ranked = results.to_dict(orient="records")
    print(json.dumps(ranked, indent=2, ensure_ascii=False, default=str))
    print("\nSIGNAL_EXAMPLES_JSON")
    print(json.dumps(meta["examples"], indent=2, ensure_ascii=False, default=str))


def save_json(path: Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["glm", "qwen-vllm", "mock"], default=os.environ.get("MATHLIB_JUDGE_BACKEND", "glm"))
    ap.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--v-path", type=Path, default=DEFAULT_V)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--mock", action="store_true", help="Use synthetic data for local smoke tests.")
    ap.add_argument("--max-sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-diff-chars", type=int, default=10000)
    ap.add_argument("--recompute-proof-complexity", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--glm-base-url", default=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/chat/completions"))
    ap.add_argument("--glm-model", default=os.environ.get("ZAI_MODEL", "glm-5.2"))
    ap.add_argument("--glm-api-key-env", default="ZAI_API_KEY")
    ap.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", DEFAULT_QWEN_MODEL))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    if args.max_sample > 300:
        raise ValueError("--max-sample must be <= 300 for this cost-constrained pass")
    cache_path = None if args.no_cache else args.cache

    if args.mock:
        df, vpp_cols, data_meta = make_mock_data(args.seed)
        if args.backend != "mock":
            print("--mock implies --backend mock", file=sys.stderr)
            args.backend = "mock"
    else:
        df, vpp_cols, data_meta = load_real_data(args)
    sample = balanced_eval_sample(df, args.max_sample, args.seed)
    print("DATA_AND_SAMPLE")
    print(
        json.dumps(
            {
                **data_meta,
                "sample_n": int(len(sample)),
                "sample_seed": args.seed,
                "sample_label_counts": sample["judgement"].value_counts().to_dict(),
                "judge_input_column": "diff_noauth",
                "diff_column_dropped_before_judging": "diff" not in sample.columns,
                "reference_vpp_full_cv_auc": REFERENCE_VPP_AUC,
            },
            indent=2,
            default=str,
        )
    )
    print("PROMPT_KEYS")
    print(
        json.dumps(
            {
                "common_system_prompt": SYSTEM_PROMPT,
                "judge_input_column": "diff_noauth",
                "combined_prompt_note": "Actual calls judge all seven metric rubrics in one JSON response per diff.",
                "metric_prompts": {m.key: full_metric_prompt_template(m) for m in METRICS},
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    tasks = prepared_tasks(sample, args.max_diff_chars)
    cache = load_cache(cache_path)
    if args.backend == "mock":
        judgments = judge_mock(tasks, cache, cache_path)
    elif args.backend == "glm":
        judgments = judge_glm(tasks, cache, cache_path, args)
    else:
        judgments = judge_qwen_vllm(tasks, cache, cache_path, args)

    judged = attach_judgments(sample, judgments)
    results, eval_meta = evaluate(judged, vpp_cols, args.seed)
    eval_meta.update(
        {
            "backend": args.backend,
            "sample_n": int(len(judged)),
            "sample_seed": args.seed,
            "judge_input_column": "diff_noauth",
            "reference_vpp_full_cv_auc": REFERENCE_VPP_AUC,
        }
    )
    print_results(results, eval_meta)
    save_json(
        args.out_json,
        {
            "data": data_meta,
            "eval": eval_meta,
            "results": results.to_dict(orient="records"),
            "judgments": [
                {
                    "number": int(r["number"]) if pd.notna(r.get("number")) else None,
                    "judgement": int(r["judgement"]),
                    "area": str(r.get("area", "")),
                    "additions": int(r["additions"]) if "additions" in r and pd.notna(r["additions"]) else None,
                    "scores": {
                        spec.key: {
                            "score": int(r[f"judge_{spec.key}"]),
                            "reason": str(r[f"judge_{spec.key}_reason"]),
                        }
                        for spec in METRICS
                    },
                }
                for _, r in judged.iterrows()
            ],
        },
    )
    print(f"\nWROTE_RESULTS_JSON {args.out_json}")


if __name__ == "__main__":
    main()
