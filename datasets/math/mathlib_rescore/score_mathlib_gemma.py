#!/usr/bin/env python3
"""Label-blind mathlib PR A-bank scorer using offline-batch Gemma-4-31B/vLLM.

The 1.0/0.5/0.0/NA token protocol and parser intentionally follow
datasets/peer-review/vat_3y/score_va_gemma_3y.py. Each output shard contains the
raw A-score matrix; downstream code may attach outcomes later by ``doc_id``.

Every shard also scores exactly three blinded sanity anchors. ``--anchors`` is a
two-row JSONL containing one ``merged_trivial_fix`` seed and one
``rejected_sorry`` seed; this script deterministically constructs the third
``scrambled_diff`` anchor and never consumes dataset outcome or partition fields.
"""

import os

# Set these before importing vLLM. They avoid the sk3 CUDA-after-fork wedge.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import re
from pathlib import Path

import numpy as np


BASE = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib")
HERE = Path(__file__).resolve().parent
GEMMA4 = (
    "/lfs/skampere3/0/shared_hf_cache/"
    "models--google--gemma-4-31b-it/snapshots/"
    "3548789868c5356dbf307c98e6f609007b82b3eb"
)
CONTEXT_RUBRIC_IDS = {
    "m07_nonduplication",
    "m09_library_fit",
    "m11_file_namespace_placement",
    "m12_existing_declaration_reuse",
}

SYS = (
    "You are an expert mathlib4 maintainer. You are given a pull request's TITLE "
    "and DIFF and ONE review criterion. Decide how strongly the code shown in the "
    "title+diff satisfies that criterion. Answer with EXACTLY ONE token:\n"
    "  1.0 = clearly satisfies the criterion\n"
    "  0.5 = partially / weakly / borderline\n"
    "  0.0 = fails / cuts against the criterion\n"
    "  NA = the title+diff gives no evidence bearing on this criterion\n"
    "Judge the shown code, not whether an external reviewer accepted it. Removed "
    "bad code is not a defect in the resulting change; distinguish added and "
    "removed lines.\n"
    "ANTI-COMPRESSION: discriminate among cases and avoid mode collapse. Use the "
    "full 1.0/0.5/0.0/NA range whenever the evidence warrants it; do not default "
    "to 0.5. Before answering, identify concrete supporting and contrary evidence "
    "and justify the score internally against the criterion's anchors. Do not "
    "output that reasoning. Output only the token."
)


def load_rubrics(path):
    metrics = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rubric = json.loads(line)
            missing = {"rubric_id", "name", "description", "guidance"} - set(rubric)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
            metrics.append(rubric)
    ids = [m["rubric_id"] for m in metrics]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate rubric_id in {path}")
    if not metrics:
        raise ValueError(f"no rubrics found in {path}")
    return metrics


def metric_block(metric):
    return (
        f"CRITERION: {metric['name']}\n"
        f"DESCRIPTION: {metric.get('description', '')}\n"
        f"SCORING GUIDANCE: {metric.get('guidance', '')}\n\n"
        "Answer with one token:"
    )


def parse_tok(text):
    text = (text or "").strip().lower()
    if text.startswith("na") or "n/a" in text or text == "na":
        return np.nan
    if "0.5" in text or text.startswith("0.5"):
        return 0.5
    if re.search(r"\b1(\.0)?\b", text) or text.startswith("1"):
        return 1.0
    if re.search(r"\b0(\.0)?\b", text) or text.startswith("0"):
        return 0.0
    return np.nan


def stable_doc_id(title, diff):
    payload = f"{title}\0{diff}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:24]


def load_rows(path):
    """Load only title, diff, and optional retrieval context."""
    path = Path(path)
    rows = []
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path, columns=["title", "diff"])
        iterator = frame.to_dict(orient="records")
    else:
        with open(path, encoding="utf-8") as fh:
            iterator = [json.loads(line) for line in fh if line.strip()]

    for record in iterator:
        title = str(record.get("title") or "")
        diff = str(record.get("diff") or "")
        row = {
            "title": title,
            "diff": diff,
            "doc_id": stable_doc_id(title, diff),
            "retrieval_context": record.get("retrieval_context") or [],
            "anchor_kind": "",
        }
        rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def load_anchor_seeds(path):
    """Load two already-blinded anchors and deterministically create the third."""
    seeds = {}
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("anchor_kind")
            if kind not in {"merged_trivial_fix", "rejected_sorry"}:
                raise ValueError(
                    f"{path}:{line_no}: anchor_kind must be merged_trivial_fix or rejected_sorry"
                )
            if kind in seeds:
                raise ValueError(f"{path}:{line_no}: duplicate anchor_kind {kind}")
            title = str(record.get("title") or "")
            diff = str(record.get("diff") or "")
            if not title or not diff:
                raise ValueError(f"{path}:{line_no}: anchor requires nonempty title and diff")
            seeds[kind] = {
                "title": title,
                "diff": diff,
                "doc_id": f"__ANCHOR_{kind}",
                "retrieval_context": record.get("retrieval_context") or [],
                "anchor_kind": kind,
            }

    expected = {"merged_trivial_fix", "rejected_sorry"}
    if set(seeds) != expected:
        raise ValueError(f"{path}: expected exactly the two anchor kinds {sorted(expected)}")
    if not re.search(r"(?mi)^\+.*\b(?:sorry|admit)\b", seeds["rejected_sorry"]["diff"]):
        raise ValueError("rejected_sorry anchor must visibly add `sorry` or `admit`")

    # Follow the caption scorer's pattern: fixed RNG, derive the nonsense anchor
    # from blinded source text, and append all three anchors to every shard.
    source = seeds["merged_trivial_fix"]["diff"]
    pieces = re.findall(r"\S+", source)
    if len(pieces) < 8:
        raise ValueError("merged_trivial_fix anchor diff is too short to scramble")
    random.Random(0).shuffle(pieces)
    scrambled = {
        "title": "[blinded sanity anchor] shuffled diff",
        "diff": " ".join(pieces),
        "doc_id": "__ANCHOR_scrambled_diff",
        "retrieval_context": [],
        "anchor_kind": "scrambled_diff",
    }
    return [seeds["merged_trivial_fix"], seeds["rejected_sorry"], scrambled]


def format_retrieval_context(row, rubric_id):
    if rubric_id not in CONTEXT_RUBRIC_IDS or not row["retrieval_context"]:
        return ""
    lines = ["RETRIEVED EXISTING DECLARATIONS (nearest lexical matches; evidence, not ground truth):"]
    for item in row["retrieval_context"][:3]:
        lines.append(
            f"- {item.get('file', '?')} | {item.get('kind', '?')} | "
            f"{item.get('decl', '')}"
        )
    return "\n".join(lines)


def render_chat(tokenizer, content, tokenize):
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=tokenize, add_generation_prompt=True
    )


def build_prompt(tokenizer, row, metric, max_model_len, max_output_tokens):
    """Truncate title+diff tokens so the rendered chat remains within 8192."""
    context = format_retrieval_context(row, metric["rubric_id"])
    prefix = f"{SYS}\n\nPULL REQUEST:\n"
    suffix = "\n\n"
    if context:
        suffix += f"{context}\n\n"
    suffix += metric_block(metric)
    document = f"TITLE:\n{row['title']}\n\nDIFF:\n{row['diff']}"

    max_input_tokens = max_model_len - max_output_tokens
    fixed = render_chat(tokenizer, prefix + suffix, tokenize=True)
    budget = max_input_tokens - len(fixed) - 8
    if budget <= 0:
        raise ValueError("rubric/system prompt leaves no token budget for title+diff")

    doc_tokens = tokenizer.encode(document, add_special_tokens=False)
    truncated = len(doc_tokens) > budget
    doc_tokens = doc_tokens[:budget]
    content = prefix + tokenizer.decode(doc_tokens) + suffix
    rendered = render_chat(tokenizer, content, tokenize=True)
    while len(rendered) > max_input_tokens and doc_tokens:
        excess = len(rendered) - max_input_tokens
        doc_tokens = doc_tokens[: max(0, len(doc_tokens) - excess - 4)]
        truncated = True
        content = prefix + tokenizer.decode(doc_tokens) + suffix
        rendered = render_chat(tokenizer, content, tokenize=True)
    if len(rendered) > max_input_tokens:
        raise ValueError("unable to fit prompt within max_model_len")
    return [{"role": "user", "content": content}], truncated


def shard_for(row, num_shards):
    return int(hashlib.sha1(row["doc_id"].encode("ascii")).hexdigest(), 16) % num_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BASE / "accept_reject_clean.parquet"))
    parser.add_argument("--rubrics", default=str(HERE / "rubrics_v2.jsonl"))
    parser.add_argument(
        "--anchors",
        required=True,
        help="Two-row blinded JSONL: merged_trivial_fix and rejected_sorry seeds",
    )
    parser.add_argument("--outdir", default=str(HERE / "scores_context_free"))
    parser.add_argument("--model", default=GEMMA4)
    parser.add_argument("--util", type=float, default=0.94)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.93 <= args.util <= 0.95:
        parser.error("--util must remain in the standing 0.93-0.95 range")
    if args.max_model_len != 8192:
        parser.error("--max-model-len must be 8192 for this pipeline")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if args.shard_id is not None and not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0, num-shards)")

    metrics = load_rubrics(args.rubrics)
    rows = load_rows(args.input)
    anchors = load_anchor_seeds(args.anchors)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shard_ids = [args.shard_id] if args.shard_id is not None else list(range(args.num_shards))
    work = []
    for shard_id in shard_ids:
        outpath = outdir / f"nc_scores_shard{shard_id:03d}.npz"
        if outpath.exists() and not args.overwrite:
            print(f"[mathlib] {outpath.name} exists; skip", flush=True)
            continue
        shard_rows = [row for row in rows if shard_for(row, args.num_shards) == shard_id]
        work.append((shard_id, shard_rows, outpath))
    if not work:
        print("SCORE_DONE", flush=True)
        return

    # Deliberately imported only after the spawn method and CUDA ordering are set.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.util,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=6)
    tokenizer = llm.get_tokenizer()

    for shard_id, shard_rows, outpath in work:
        batch = shard_rows + anchors
        conversations = []
        truncations = 0
        for row in batch:
            for metric in metrics:
                conversation, truncated = build_prompt(
                    tokenizer, row, metric, args.max_model_len, 6
                )
                conversations.append(conversation)
                truncations += int(truncated)

        print(
            f"[mathlib] shard {shard_id}: {len(shard_rows)} PRs + exactly 3 anchors "
            f"x {len(metrics)} rubrics = {len(conversations)} prompts",
            flush=True,
        )
        outputs = llm.chat(conversations, sampling)
        values = [parse_tok(output.outputs[0].text) for output in outputs]
        scores = np.array(values, dtype=float).reshape(len(batch), len(metrics))
        real_scores = scores[: len(shard_rows)]
        na_rate = float(np.isnan(real_scores).mean()) if len(shard_rows) else np.nan
        anchor_means = np.nanmean(scores[len(shard_rows) :], axis=1)
        print(
            f"[mathlib] shard {shard_id}: NA={na_rate:.3f}; "
            "anchors trivial/sorry/scrambled="
            + "/".join(f"{value:.3f}" for value in anchor_means)
            + f"; truncated pairs={truncations}",
            flush=True,
        )

        np.savez_compressed(
            outpath,
            X=scores,
            doc_id=np.array([row["doc_id"] for row in batch], dtype=object),
            title=np.array([row["title"] for row in batch], dtype=object),
            is_anchor=np.array([bool(row["anchor_kind"]) for row in batch], dtype=bool),
            anchor_kind=np.array([row["anchor_kind"] for row in batch], dtype=object),
            rubric_ids=np.array([metric["rubric_id"] for metric in metrics], dtype=object),
            a_names=np.array([metric["name"] for metric in metrics], dtype=object),
            na_rate=na_rate,
            truncated_pairs=truncations,
            has_retrieval_context=bool(any(row["retrieval_context"] for row in shard_rows)),
        )
        print(f"[mathlib] saved -> {outpath}", flush=True)

    print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
