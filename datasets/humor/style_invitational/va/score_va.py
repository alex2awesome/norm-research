#!/usr/bin/env python3
"""End-to-end label-blind V/A scoring and grouped readout.

The A judge is Codex ``gpt-5.6-sol``.  Each response cell is constrained to one
of {1.0, 0.5, 0.0, NA}; the natural-language prompt requests temperature-zero
behavior and the CLI is run ephemerally with fixed decoding instructions.

Stages:
  gepa     score a label-hidden pilot, inspect fidelity, and rewrite rubrics
  score    score a whole-week hash sample, with 3 blinded anchors every batch
  analyze  attach the two requested labels only after scoring and run grouped CV
  all      run all three stages in sequence

The main scoring path never includes ``tier`` in a model prompt or cache.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from v_features import V_NAMES, vector

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "style_invitational.jsonl"
RUBRICS = HERE / "rubrics.jsonl"
SEED_RUBRICS = HERE / "rubrics.seed.jsonl"
GEPA_AUDIT = HERE / "gepa_audit.json"
SCORE_DIR = HERE / "score_cache"
WORK_DIR = HERE / ".work"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "RESULTS.md"

MODEL = "gpt-5.6-sol"
SAMPLE_SALT = "style-va-v1"
PILOT_SALT = "style-va-gepa-pilot-v1"
ANCHOR_SEED = 20260728
ALLOWED = {"1.0", "0.5", "0.0", "NA"}
TOKEN_TO_FLOAT = {"1.0": 1.0, "0.5": 0.5, "0.0": 0.0, "NA": float("nan")}

JUDGE_INSTRUCTIONS = """You are the deterministic scoring executor for a
Washington Post Style Invitational humor instrument. Treat decoding as
temperature 0. For every (entry, criterion) pair, return exactly one token:
"1.0", "0.5", "0.0", or "NA".

1.0 means the entry clearly satisfies the criterion.
0.5 means it partly, weakly, or borderline satisfies it.
0.0 means the relevant attempt is present but fails or cuts against it.
NA means the mechanism genuinely does not apply or the specified evidence is
unavailable. Do not use NA merely because quality is low.

Judge only the contest prompt and entry text. Do not infer or predict editorial
tier, placement, author reputation, or popularity, and do not compare entries.
Parenthetical author names and locations may be archive bylines; ignore them
when assessing humor. Apply each criterion independently and preserve the
criterion order. Return only the schema-conforming object."""

GEPA_INSTRUCTIONS = """You are conducting one label-blind GEPA fidelity pass
for a humor rubric bank. You are given the proposed rubrics, a varied pilot of
contest prompts and entries, and the resulting single-token judgments. No
editorial tiers or outcome labels are present.

Inspect whether each rubric's construct was applied faithfully and
consistently across the pilot. Rewrite every description for higher scoring
fidelity: preserve its name and construct, make the 1.0/0.5/0.0/NA boundaries
mutually distinct, reserve NA for genuine non-applicability, and resolve
ambiguities exposed by the judgments. Do not add, remove, merge, reorder, or
rename criteria. Do not optimize discrimination, outcome prediction, or AUC.
Return only the schema-conforming object."""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_dataset() -> list[dict[str, Any]]:
    rows = _jsonl(DATASET)
    expected = {"week_id", "contest_prompt", "entry_text", "tier"}
    if not rows or any(set(r) != expected for r in rows):
        raise ValueError("unexpected Style Invitational schema")
    return rows


def load_rubrics(path: Path = RUBRICS) -> list[dict[str, str]]:
    rubrics = _jsonl(path)
    if not 25 <= len(rubrics) <= 40:
        raise ValueError(f"rubric bank must contain 25..40 criteria, found {len(rubrics)}")
    ids = [r.get("criterion_id") for r in rubrics]
    names = [r.get("name") for r in rubrics]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError("duplicate rubric id or name")
    for r in rubrics:
        if not all(r.get(k) for k in ("criterion_id", "name", "description")):
            raise ValueError(f"malformed rubric {r}")
        if not all(tok in r["description"] for tok in ("1.0", "0.5", "0.0", "NA")):
            raise ValueError(f"rubric lacks explicit token boundaries: {r['criterion_id']}")
    return rubrics


def doc_id(week_id: str, row_index: int, text: str) -> str:
    digest = sha256_text(f"{week_id}\0{row_index}\0{text}")[:12]
    return f"W{week_id}_{digest}"


def indexed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        rr = dict(r)
        rr["_row_index"] = i
        rr["_doc_id"] = doc_id(str(r["week_id"]), i, r["entry_text"])
        out.append(rr)
    return out


def sampled_rows(rows: list[dict[str, Any]], n_weeks: int) -> tuple[list[dict[str, Any]], list[str]]:
    by_week: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in indexed_rows(rows):
        by_week[str(r["week_id"])].append(r)
    ordered = sorted(
        by_week,
        key=lambda w: sha256_text(f"{SAMPLE_SALT}|{w}"),
    )
    chosen = ordered[:n_weeks]
    if len(chosen) != n_weeks:
        raise ValueError(f"requested {n_weeks} weeks, only {len(chosen)} available")
    selected = [r for w in chosen for r in by_week[w]]
    if n_weeks < 100 or len(selected) < 3000:
        raise ValueError(f"sample floor not met: {len(selected)} rows / {n_weeks} weeks")
    return selected, chosen


def blinded_item(r: dict[str, Any]) -> dict[str, str]:
    """Whitelist exactly the fields the model may see."""
    return {
        "contest_prompt": str(r["contest_prompt"]),
        "entry_text": str(r["entry_text"]),
    }


def choose_anchors(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Fixed winner plus a seeded-random HM from the same week, then nonsense.

    Tier is used only here to construct the required validation controls.  The
    returned prompt payload has no tier, and opaque IDs are assigned per batch.
    """
    week = "178"
    same = [r for r in rows if str(r["week_id"]) == week]
    winners = [r for r in same if r["tier"] == "winner"]
    hms = [r for r in same if r["tier"] == "honorable_mention"]
    winner_text = (
        "If we could just get everyone to close their eyes and visualize world peace "
        "for an hour, imagine how serene and quiet it would be, until the looting started. "
        "(Joseph Romm, Washington)"
    )
    winner = next((r for r in winners if r["entry_text"] == winner_text), None)
    if winner is None:
        raise ValueError("fixed anchor winner not found")
    hm = random.Random(ANCHOR_SEED).choice(hms)
    tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[.,!?;:]", winner["entry_text"] + " " + hm["entry_text"])
    random.Random(ANCHOR_SEED + 1).shuffle(tokens)
    scrambled = " ".join(tokens[:34])
    prompt = str(winner["contest_prompt"])
    return [
        {"anchor_tag": "winner", "contest_prompt": prompt, "entry_text": winner["entry_text"]},
        {"anchor_tag": "honorable_mention", "contest_prompt": prompt, "entry_text": hm["entry_text"]},
        {"anchor_tag": "scrambled", "contest_prompt": prompt, "entry_text": scrambled},
    ]


def score_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "scores": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(ALLOWED)},
                        },
                    },
                    "required": ["id", "scores"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    }


def gepa_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rubrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["criterion_id", "description"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rubrics"],
        "additionalProperties": False,
    }


def run_codex(
    prompt: str,
    schema: dict[str, Any],
    call_name: str,
    timeout: int = 1200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one schema-constrained, ephemeral gpt-5.6-sol call."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    call_dir = WORK_DIR / re.sub(r"[^A-Za-z0-9_.-]", "_", call_name)
    call_dir.mkdir(parents=True, exist_ok=True)
    state_dir = WORK_DIR / "codex_state"
    log_dir = WORK_DIR / "codex_log"
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    schema_path = call_dir / "schema.json"
    result_path = call_dir / "result.json"
    log_path = call_dir / "codex.log"
    _dump_json(schema_path, schema)
    if result_path.exists():
        result_path.unlink()
    cmd = [
        shutil.which("codex") or "codex",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--model",
        MODEL,
        "--ignore-user-config",
        "--ignore-rules",
        "-c",
        f'sqlite_home="{state_dir}"',
        "-c",
        f'log_dir="{log_dir}"',
        "-c",
        'cli_auth_credentials_store="file"',
        "-c",
        'model_reasoning_effort="low"',
        "-c",
        'model_verbosity="low"',
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(result_path),
        "-",
    ]
    started = time.time()
    completed = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=HERE,
    )
    log_path.write_text(
        f"returncode={completed.returncode}\n"
        f"elapsed_s={time.time() - started:.3f}\n\n"
        f"STDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}"
    )
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"Codex call {call_name} failed; see {log_path}")
    parsed = json.loads(result_path.read_text())
    meta = {
        "call_name": call_name,
        "model": MODEL,
        "temperature": 0,
        "backend": "codex_exec_ephemeral_schema",
        "prompt_sha256": sha256_text(prompt),
        "elapsed_s": round(time.time() - started, 3),
    }
    return parsed, meta


def build_score_prompt(
    rubrics: list[dict[str, str]],
    payload: list[dict[str, str]],
) -> str:
    return (
        JUDGE_INSTRUCTIONS
        + "\n\nCRITERIA IN REQUIRED OUTPUT ORDER:\n"
        + json.dumps(rubrics, ensure_ascii=False)
        + "\n\nBLINDED ITEMS:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nReturn one row for every item in the supplied order. Each row must repeat "
        "the item's id and contain exactly one score token per criterion."
    )


def validate_scores(
    response: dict[str, Any],
    payload: list[dict[str, str]],
    n_criteria: int,
) -> dict[str, list[str]]:
    rows = response.get("rows")
    if not isinstance(rows, list) or len(rows) != len(payload):
        raise ValueError(f"expected {len(payload)} response rows, got {len(rows or [])}")
    expected = [x["id"] for x in payload]
    returned = [x.get("id") for x in rows]
    if returned != expected:
        raise ValueError(f"response item order mismatch: {returned[:5]} vs {expected[:5]}")
    out = {}
    for row in rows:
        vals = row.get("scores")
        if not isinstance(vals, list) or len(vals) != n_criteria:
            raise ValueError(f"bad score count for {row.get('id')}: {len(vals or [])}")
        vals = [str(v) for v in vals]
        if any(v not in ALLOWED for v in vals):
            raise ValueError(f"bad score token for {row.get('id')}")
        out[row["id"]] = vals
    return out


def mean_tokens(tokens: Sequence[str]) -> float:
    values = [TOKEN_TO_FLOAT[x] for x in tokens]
    finite = [x for x in values if math.isfinite(x)]
    return sum(finite) / len(finite) if finite else float("nan")


def run_gepa(n_weeks: int) -> None:
    rows = load_dataset()
    selected, _ = sampled_rows(rows, n_weeks)
    rubrics = load_rubrics()
    if GEPA_AUDIT.exists():
        print(f"[gepa] {GEPA_AUDIT.name} exists; fidelity pass already frozen", flush=True)
        return
    if not SEED_RUBRICS.exists():
        SEED_RUBRICS.write_text(RUBRICS.read_text())

    # Exactly 18 label-hidden pilot items selected by hashes of their identity.
    pilot = sorted(
        selected,
        key=lambda r: sha256_text(f"{PILOT_SALT}|{r['_doc_id']}"),
    )[:18]
    anchors = choose_anchors(rows)
    raw_items: list[dict[str, str]] = []
    mapping: dict[str, str] = {}
    combined: list[tuple[str, dict[str, str]]] = [
        ("pilot", {**blinded_item(r), "source_doc_id": r["_doc_id"]}) for r in pilot
    ] + [
        (a["anchor_tag"], dict(a)) for a in anchors
    ]
    random.Random(ANCHOR_SEED + 2).shuffle(combined)
    for i, (tag, item) in enumerate(combined):
        bid = f"P{i:02d}"
        raw_items.append({"id": bid, **{k: item[k] for k in ("contest_prompt", "entry_text")}})
        mapping[bid] = tag

    score_prompt = build_score_prompt(rubrics, raw_items)
    scored, score_meta = run_codex(score_prompt, score_schema(), "gepa_pilot_score")
    scores = validate_scores(scored, raw_items, len(rubrics))
    pilot_packet = [
        {
            "id": item["id"],
            "contest_prompt": item["contest_prompt"],
            "entry_text": item["entry_text"],
            "scores": scores[item["id"]],
        }
        for item in raw_items
        if mapping[item["id"]] == "pilot"
    ]
    refine_prompt = (
        GEPA_INSTRUCTIONS
        + "\n\nPROPOSED RUBRICS:\n"
        + json.dumps(rubrics, ensure_ascii=False)
        + "\n\nPILOT ITEMS AND RESULTING JUDGMENTS:\n"
        + json.dumps(pilot_packet, ensure_ascii=False)
    )
    refined, refine_meta = run_codex(refine_prompt, gepa_schema(), "gepa_fidelity_rewrite")
    returned = refined.get("rubrics")
    if not isinstance(returned, list) or len(returned) != len(rubrics):
        raise ValueError("GEPA rewrite returned wrong bank size")
    if [x.get("criterion_id") for x in returned] != [r["criterion_id"] for r in rubrics]:
        raise ValueError("GEPA rewrite changed criterion identity or order")
    final = []
    for old, new in zip(rubrics, returned):
        description = re.sub(r"\s+", " ", str(new.get("description", ""))).strip()
        if not all(tok in description for tok in ("1.0", "0.5", "0.0", "NA")):
            raise ValueError(f"GEPA rewrite lost score boundary: {old['criterion_id']}")
        final.append({**old, "description": description})
    RUBRICS.write_text(
        "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in final)
    )
    load_rubrics()

    anchor_scores = {
        mapping[bid]: mean_tokens(vals)
        for bid, vals in scores.items()
        if mapping[bid] != "pilot"
    }
    audit = {
        "method": "label-blind GEPA propose-score-inspect-rewrite",
        "labels_exposed_to_proposal_or_fidelity_pass": False,
        "pilot_n": len(pilot),
        "pilot_fields": ["contest_prompt", "entry_text"],
        "seed_rubrics_sha256": sha256_text(SEED_RUBRICS.read_text()),
        "final_rubrics_sha256": sha256_text(RUBRICS.read_text()),
        "score_call": score_meta,
        "rewrite_call": refine_meta,
        "pilot_anchor_means": anchor_scores,
        "criterion_count": len(final),
    }
    _dump_json(GEPA_AUDIT, audit)
    print(
        "[gepa] final bank frozen; pilot anchors winner/HM/scrambled = "
        f"{anchor_scores['winner']:.3f}/{anchor_scores['honorable_mention']:.3f}/"
        f"{anchor_scores['scrambled']:.3f}",
        flush=True,
    )


def make_batches(
    selected: list[dict[str, Any]],
    max_rows: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    by_week: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_week[str(r["week_id"])].append(r)
    batches = []
    for week in sorted(by_week, key=lambda w: sha256_text(f"{SAMPLE_SALT}|{w}")):
        wr = by_week[week]
        for start in range(0, len(wr), max_rows):
            batches.append((f"week_{week}_part_{start // max_rows:02d}", wr[start : start + max_rows]))
    return batches


def score_one_batch(
    batch_name: str,
    batch_rows: list[dict[str, Any]],
    rubrics: list[dict[str, str]],
    anchors: list[dict[str, str]],
    retry: int = 0,
) -> dict[str, Any]:
    cached = SCORE_DIR / f"{batch_name}.json"
    if cached.exists() and retry == 0:
        return json.loads(cached.read_text())

    source: list[tuple[str, str, dict[str, str]]] = []
    for r in batch_rows:
        source.append(("main", r["_doc_id"], blinded_item(r)))
    for a in anchors:
        source.append(("anchor", a["anchor_tag"], {
            "contest_prompt": a["contest_prompt"],
            "entry_text": a["entry_text"],
        }))
    random.Random(sha256_text(f"{batch_name}|{retry}|blind")).shuffle(source)
    payload = []
    id_map = {}
    for i, (kind, source_id, item) in enumerate(source):
        bid = f"D{i:03d}"
        payload.append({"id": bid, **item})
        id_map[bid] = {"kind": kind, "source_id": source_id}

    prompt = build_score_prompt(rubrics, payload)
    if retry:
        prompt += (
            f"\n\nThis is independent deterministic rescore {retry}. Re-read every "
            "criterion boundary and item; do not copy or infer any earlier response."
        )
    response, call_meta = run_codex(
        prompt,
        score_schema(),
        f"{batch_name}_try_{retry}",
    )
    scores = validate_scores(response, payload, len(rubrics))
    main = {}
    anchor = {}
    for bid, vals in scores.items():
        info = id_map[bid]
        if info["kind"] == "main":
            main[info["source_id"]] = vals
        else:
            anchor[info["source_id"]] = vals
    anchor_means = {tag: mean_tokens(vals) for tag, vals in anchor.items()}
    passed = (
        anchor_means["winner"]
        > anchor_means["honorable_mention"]
        > anchor_means["scrambled"]
    )
    record = {
        "batch_id": batch_name,
        "week_id": str(batch_rows[0]["week_id"]),
        "n_main": len(batch_rows),
        "criterion_ids": [r["criterion_id"] for r in rubrics],
        "main_scores": main,
        "anchor_scores": anchor,
        "anchor_means": anchor_means,
        "anchor_order_passed": passed,
        "scoring": call_meta,
        "retry": retry,
    }
    if not passed:
        if retry < 2:
            return score_one_batch(batch_name, batch_rows, rubrics, anchors, retry + 1)
        failed = SCORE_DIR / "failed"
        failed.mkdir(parents=True, exist_ok=True)
        _dump_json(failed / f"{batch_name}.json", record)
        raise RuntimeError(
            f"anchor ordering failed after rescoring {batch_name}: {anchor_means}"
        )
    _dump_json(cached, record)
    return record


def run_scoring(n_weeks: int, max_rows: int, concurrency: int) -> None:
    if not GEPA_AUDIT.exists():
        raise RuntimeError("run the label-blind GEPA fidelity pass before production scoring")
    rows = load_dataset()
    selected, weeks = sampled_rows(rows, n_weeks)
    rubrics = load_rubrics()
    anchors = choose_anchors(rows)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    batches = make_batches(selected, max_rows)
    pending = [b for b in batches if not (SCORE_DIR / f"{b[0]}.json").exists()]
    print(
        f"[score] exact sample {len(selected)} entries / {len(weeks)} whole weeks; "
        f"{len(batches)} batches ({len(pending)} pending), {len(rubrics)} criteria",
        flush=True,
    )
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {
            pool.submit(score_one_batch, name, br, rubrics, anchors): name
            for name, br in pending
        }
        for fut in concurrent.futures.as_completed(futs):
            name = futs[fut]
            rec = fut.result()
            completed += 1
            am = rec["anchor_means"]
            print(
                f"[score] {completed}/{len(pending)} {name}: anchor "
                f"{am['winner']:.3f}>{am['honorable_mention']:.3f}>"
                f"{am['scrambled']:.3f}",
                flush=True,
            )

    # Fail closed on exact coverage and every-batch anchor validity.
    records = [_json_record(SCORE_DIR / f"{name}.json") for name, _ in batches]
    got = set()
    for rec in records:
        if not rec["anchor_order_passed"]:
            raise RuntimeError(f"cached invalid anchor batch: {rec['batch_id']}")
        got.update(rec["main_scores"])
    expected = {r["_doc_id"] for r in selected}
    if got != expected:
        raise RuntimeError(f"score coverage mismatch: got {len(got)}, expected {len(expected)}")
    print("[score] SCORE_DONE exact coverage and all anchor checks passed", flush=True)


def _json_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json_safe(value: Any) -> Any:
    """Convert numpy values and non-finite floats for strict JSON."""
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
        if isinstance(value, np.ndarray):
            return [_json_safe(x) for x in value.tolist()]
    except ImportError:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    return value


def grouped_oof_auc(X, y, groups, splits) -> float:
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred = np.full(len(y), np.nan)
    for train, test in splits:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                max_iter=5000,
                random_state=20260728,
            ),
        )
        model.fit(X[train], y[train])
        pred[test] = model.predict_proba(X[test])[:, 1]
    if np.isnan(pred).any():
        raise RuntimeError("incomplete OOF predictions")
    return float(roc_auc_score(y, pred))


def anchor_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    tags = ["winner", "honorable_mention", "scrambled"]
    means = {}
    for tag in tags:
        vals = [
            TOKEN_TO_FLOAT[tok]
            for rec in records
            for tok in rec["anchor_scores"][tag]
            if tok != "NA"
        ]
        means[tag] = sum(vals) / len(vals)
    return {
        "batches": len(records),
        "batches_passed": sum(bool(r["anchor_order_passed"]) for r in records),
        "all_batches_passed": all(bool(r["anchor_order_passed"]) for r in records),
        "mean_A": means,
        "ordering": "winner > honorable_mention > scrambled",
        "ordering_passed_on_aggregate": (
            means["winner"] > means["honorable_mention"] > means["scrambled"]
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Style Invitational V/A results",
        "",
        "## Design",
        "",
        "This is a **curation contrast**: every entry was already selected for print. "
        "The readout asks whether deterministic textual checks (V) and label-blind "
        "articulated humor criteria (A) discriminate the named editor's tiers among "
        "already-good entries, not good humor from bad humor.",
        "",
        f"The frozen hash sample contains **{result['sample']['n']:,} entries from "
        f"{result['sample']['weeks']} whole weekly contests**. All five CV folds are "
        "grouped by `week_id`; no week crosses folds. The shared `contest_prompt` is "
        "judge context and cannot by itself separate entries within a week.",
        "",
        "## Grouped-CV readouts",
        "",
        "| Outcome | n | pos | neg | V AUC | A AUC | V+A AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("top_tier", "winner_vs_rest"):
        x = result["outcomes"][key]
        lines.append(
            f"| {x['label']} | {x['n']:,} | {x['pos']:,} | {x['neg']:,} | "
            f"{x['auc']['V']:.3f} | {x['auc']['A']:.3f} | {x['auc']['V+A']:.3f} |"
        )

    a = result["anchors"]
    lines += [
        "",
        "## Mandatory anchor check",
        "",
        "| Anchor | Mean A | Rank |",
        "|---|---:|---:|",
    ]
    ordered = sorted(a["mean_A"].items(), key=lambda kv: -kv[1])
    for rank, (tag, value) in enumerate(ordered, 1):
        lines.append(f"| {tag.replace('_', ' ')} | {value:.3f} | {rank} |")
    lines += [
        "",
        f"Expected ordering passed in **{a['batches_passed']}/{a['batches']} batches**; "
        "failed batches would have been rejected and rescored.",
        "",
    ]

    for key in ("top_tier", "winner_vs_rest"):
        x = result["outcomes"][key]
        lines += [
            f"## Ranked univariate A criteria: {x['label']}",
            "",
            "| Rank | Criterion | Grouped-CV univariate AUC |",
            "|---:|---|---:|",
        ]
        for row in x["criterion_auc"]:
            lines.append(f"| {row['rank']} | {row['name']} | {row['auc']:.3f} |")
        lines.append("")

    lines += [
        "## Caveats",
        "",
        "- This is descriptive discrimination within a print-selected pool, not a causal "
        "model of humor quality or an estimate for unselected submissions.",
        "- The archive parser sometimes splits or merges entries, and author attributions "
        "remain inside `entry_text`; both can affect deterministic V checks. The A judge "
        "was explicitly told to ignore bylines.",
        "- Several prompts depend on omitted pictures, supplied lists, or 1990s topical "
        "knowledge. Criteria use `NA` where the archived evidence is genuinely unavailable.",
        "- A single model (`gpt-5.6-sol`) both proposed/refined the fidelity-optimized bank "
        "and executed it. The GEPA pass was label-blind and did not inspect AUC.",
        "- Results use the stated 110-week subsample, not a silent truncation of the full "
        "9,637-row archive.",
        "",
    ]
    return "\n".join(lines)


def run_analysis(n_weeks: int, max_rows: int) -> None:
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    rows = load_dataset()
    selected, weeks = sampled_rows(rows, n_weeks)
    rubrics = load_rubrics()
    batches = make_batches(selected, max_rows)
    records = []
    scores_by_id = {}
    for name, _ in batches:
        path = SCORE_DIR / f"{name}.json"
        if not path.exists():
            raise RuntimeError(f"missing score batch {path.name}")
        rec = _json_record(path)
        records.append(rec)
        scores_by_id.update(rec["main_scores"])
    if set(scores_by_id) != {r["_doc_id"] for r in selected}:
        raise RuntimeError("analysis score coverage mismatch")

    X_a = np.array(
        [[TOKEN_TO_FLOAT[t] for t in scores_by_id[r["_doc_id"]]] for r in selected],
        dtype=float,
    )
    X_v = np.array(
        [vector(r["entry_text"], r["contest_prompt"]) for r in selected],
        dtype=float,
    )
    X_va = np.concatenate([X_v, X_a], axis=1)
    groups = np.array([str(r["week_id"]) for r in selected], dtype=object)
    labels = {
        "top_tier": np.array(
            [r["tier"] in {"winner", "runnerup"} for r in selected], dtype=int
        ),
        "winner_vs_rest": np.array(
            [r["tier"] == "winner" for r in selected], dtype=int
        ),
    }
    display = {
        "top_tier": "top-tier (winner or runner-up vs HM)",
        "winner_vs_rest": "winner vs rest",
    }
    outcomes = {}
    for key, y in labels.items():
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260728)
        splits = list(sgkf.split(np.zeros(len(y)), y, groups))
        auc = {
            "V": grouped_oof_auc(X_v, y, groups, splits),
            "A": grouped_oof_auc(X_a, y, groups, splits),
            "V+A": grouped_oof_auc(X_va, y, groups, splits),
        }
        crit = []
        for j, r in enumerate(rubrics):
            score = grouped_oof_auc(X_a[:, [j]], y, groups, splits)
            crit.append({
                "criterion_id": r["criterion_id"],
                "name": r["name"],
                "auc": score,
            })
        crit.sort(key=lambda x: (-x["auc"], x["criterion_id"]))
        for rank, row in enumerate(crit, 1):
            row["rank"] = rank
        outcomes[key] = {
            "label": display[key],
            "n": int(len(y)),
            "pos": int(y.sum()),
            "neg": int(len(y) - y.sum()),
            "auc": auc,
            "criterion_auc": crit,
        }

    result = {
        "study": "Washington Post Style Invitational V/A curation contrast",
        "sample": {
            "method": f"first {n_weeks} weeks by SHA-256({SAMPLE_SALT}|week_id)",
            "n": len(selected),
            "weeks": len(weeks),
            "week_ids": weeks,
            "full_dataset_n": len(rows),
            "full_dataset_weeks": len({str(r["week_id"]) for r in rows}),
            "silent_truncation": False,
        },
        "instrument": {
            "v_feature_count": len(V_NAMES),
            "v_feature_names": V_NAMES,
            "a_criterion_count": len(rubrics),
            "a_criterion_ids": [r["criterion_id"] for r in rubrics],
            "gepa": json.loads(GEPA_AUDIT.read_text()),
            "judge_model": MODEL,
            "temperature": 0,
            "score_tokens": ["1.0", "0.5", "0.0", "NA"],
            "prompt_is_shared_within_week": True,
            "prompt_cannot_separate_entries_by_itself_within_week": True,
        },
        "cv": {
            "method": "5-fold StratifiedGroupKFold out-of-fold logistic readout",
            "group": "week_id",
            "folds": 5,
            "week_never_split": True,
            "imputation": "training-fold median",
            "scaling": "training-fold standardization",
            "logistic_C": 1.0,
        },
        "anchors": anchor_summary(records),
        "outcomes": outcomes,
    }
    strict = _json_safe(result)
    _dump_json(RESULTS_JSON, strict)
    RESULTS_MD.write_text(render_markdown(strict))
    print(json.dumps({
        key: value["auc"] for key, value in outcomes.items()
    }, indent=2), flush=True)
    print(f"[analyze] wrote {RESULTS_JSON.name} and {RESULTS_MD.name}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("gepa", "score", "analyze", "all"))
    ap.add_argument("--weeks", type=int, default=110)
    ap.add_argument("--max-rows-per-batch", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=6)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"gepa", "all"}:
        run_gepa(args.weeks)
    if args.stage in {"score", "all"}:
        run_scoring(args.weeks, args.max_rows_per_batch, args.concurrency)
    if args.stage in {"analyze", "all"}:
        run_analysis(args.weeks, args.max_rows_per_batch)


if __name__ == "__main__":
    main()
