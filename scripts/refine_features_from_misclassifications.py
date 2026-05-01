#!/usr/bin/env python
"""Iterative refinement: re-extract features from mispredicted training examples.

Label-agnostic re-extraction: for each doc the STaR-Local extractor predicted
wrong, ask the LLM to propose MORE SPECIFIC evaluative features it may have
missed, without being told which direction the true label is. The
misclassification signal is used only as a TARGETING signal (these docs are
where initial extraction was likely shallow). Features are self-classified
for/against by the LLM based on the same neutral semantics as the original
extraction prompt — no top-down bias from the true label.

Augment star_local_v1.jsonl with the new features. Re-clustering on the
augmented cache produces new canonical features shaped by filling coverage
gaps at misclassified examples.

Idempotent: writes to a NEW output_dir and copies the extraction cache there.
The existing clustering-only driver can be pointed at that new dir.

NOTE on --num-iterations: iteration 1 uses the LLM's original extraction-time
prediction as the misclassification signal. Iteration 2+ requires re-predicting
on the augmented cache, which this driver does NOT do (needs VLLM/Llama). So
for now --num-iterations=1 is the primary use case; >1 will repeat against the
same misclassified set each time and is mainly useful for stress-testing.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "methods" / "autometrics"))

from methods.local_explanations.config import TASK_METADATA_REGISTRY
from methods.local_explanations.parsing import parse_json_dict, parse_star_local_output
from scripts.run_local_explanations import DATASET_CONFIGS, load_splits

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


PROPOSE_PROMPT = """Re-read this {text_type} and look for MORE SPECIFIC evaluative \
features than the ones already listed. Each feature should reference concrete \
content in the text (a specific claim, result, methodological choice, etc.), \
not generic praise or criticism.

TEXT:
{text}

FEATURES ALREADY EXTRACTED:
FOR {positive_label}:
{features_for}

AGAINST {positive_label} (i.e. supporting {negative_label}):
{features_against}

Propose up to {max_new} ADDITIONAL evaluative features not already on either \
list. Each feature must:
  - Be grounded in specific language from the text
  - Be more specific than generic praise/criticism
  - Follow format: "[specific content from text] — [brief evaluation]"
  - Be classified as FOR (supports {positive_label}) or AGAINST (supports {negative_label})

{canonical_context}

Return JSON only:
{{"new_features": [
  {{"feature": "[content] — [evaluation]", "side": "for" | "against"}}
]}}"""


def _render_features_list(items: List[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {f}" for f in items)


def _render_canonical_context(canonical_features: Optional[List[Dict]], max_show: int = 15) -> str:
    if not canonical_features:
        return ""
    lines = ["For context, our current canonical concepts (propose features that are DIFFERENT from these):"]
    for cf in canonical_features[:max_show]:
        lines.append(f"  - {cf.get('name', '?')}: {cf.get('description', '')[:120]}")
    return "\n".join(lines) + "\n"


_REASONING_MODEL_TAGS = ("gpt-5", "gpt5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return any(tag in model.lower() for tag in _REASONING_MODEL_TAGS)


async def _propose_one(
    client, doc_id: str, text: str, result,
    positive_label: str, negative_label: str, text_type: str,
    canonical_context: str, max_new: int, model: str,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, List[Dict[str, str]]]:
    """Label-agnostic re-extraction: propose additional specific features.
    Targeting (which docs) uses misclassification signal, but the LLM is not
    told the true label — features are proposed neutrally and self-classified."""
    prompt = PROPOSE_PROMPT.format(
        text_type=text_type,
        text=text,
        features_for=_render_features_list(result.features_for),
        features_against=_render_features_list(result.features_against),
        positive_label=positive_label,
        negative_label=negative_label,
        max_new=max_new,
        canonical_context=canonical_context,
    )
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if _is_reasoning_model(model):
        kwargs["temperature"] = 1.0
        kwargs["max_completion_tokens"] = 8000
    else:
        kwargs["temperature"] = 0.3
        kwargs["max_tokens"] = 1000

    async with semaphore:
        try:
            resp = await client.chat.completions.create(**kwargs)
            response_text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Propose failed for %s: %s", doc_id, exc)
            return doc_id, []

    parsed = parse_json_dict(response_text)
    missed = parsed.get("new_features") or parsed.get("missed_features") or []
    out = []
    if isinstance(missed, list):
        for item in missed:
            if not isinstance(item, dict):
                continue
            feat = (item.get("feature") or "").strip()
            side = (item.get("side") or "").strip().lower()
            if feat and side in ("for", "against"):
                out.append({"feature": feat, "side": side})
    return doc_id, out


async def _run_propose_batch(
    client, tasks_data, model, concurrency, positive_label, negative_label,
    text_type, canonical_context, max_new,
):
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for (doc_id, text, result) in tasks_data:
        tasks.append(_propose_one(
            client, doc_id, text, result,
            positive_label, negative_label, text_type,
            canonical_context, max_new, model, sem,
        ))
    done = 0
    results: Dict[str, List[Dict[str, str]]] = {}
    for coro in asyncio.as_completed(tasks):
        doc_id, feats = await coro
        results[doc_id] = feats
        done += 1
        if done % 200 == 0 or done == len(tasks):
            n_proposed = sum(len(v) for v in results.values())
            logger.info("Proposed on %d/%d docs (%d features so far)", done, len(tasks), n_proposed)
    return results


def load_star_cache(cache_path: Path, positive_label: str, negative_label: str):
    """Load star_local_v1.jsonl → Dict[doc_id -> (raw, parsed_StarLocalResult)]."""
    records: Dict[str, Any] = {}
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                parsed = parse_star_local_output(rec["response"], positive_label, negative_label)
                records[rec["id"]] = {"raw": rec["response"], "parsed": parsed}
            except Exception:
                continue
    return records


def write_star_cache(records, out_path: Path):
    """Write augmented cache preserving existing raw responses (no reformat)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for doc_id, r in records.items():
            f.write(json.dumps({"id": doc_id, "response": r["raw"]}) + "\n")


def append_features_to_raw(raw: str, for_new: List[str], against_new: List[str]) -> str:
    """Append new features into the existing markdown-ish raw response so that
    parse_star_local_output picks them up on re-parse."""
    if not for_new and not against_new:
        return raw
    addendum = "\n\n# Refinement iteration appended features\n"
    if for_new:
        addendum += "FEATURES_FOR (refined):\n" + "\n".join(f"- {f}" for f in for_new) + "\n"
    if against_new:
        addendum += "FEATURES_AGAINST (refined):\n" + "\n".join(f"- {f}" for f in against_new) + "\n"
    return raw + addendum


def main():
    parser = argparse.ArgumentParser(description="Iterative misclassification-driven feature refinement")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--input-output-dir", type=str, required=True,
                        help="Existing output dir with explanation_cache/star_local_v1.jsonl")
    parser.add_argument("--new-output-dir", type=str, required=True,
                        help="New output dir to write augmented cache into")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--max-misclassified", type=int, default=2000,
                        help="Sample at most this many mispredicted docs per iteration")
    parser.add_argument("--max-new-features-per-doc", type=int, default=3)
    parser.add_argument("--canonical-features-path", type=str, default=None,
                        help="Optional clustering/canonical_features.json for prompt context")
    parser.add_argument("--agency", type=str, default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY env var must be set")

    dcfg = dict(DATASET_CONFIGS[args.dataset])
    if args.agency:
        dcfg["split_dir"] = (REPO_ROOT / "datasets" / "notice-and-comment" / "agencies" / args.agency)
    task_metadata = TASK_METADATA_REGISTRY[dcfg["dataset_name"]]

    train_df, _e, _t = load_splits(dcfg["split_dir"])
    id_col = dcfg["id_column"]
    if id_col not in train_df.columns:
        train_df[id_col] = [f"train_{i}" for i in range(len(train_df))]
    true_labels = dict(zip(
        train_df[id_col].astype(str),
        train_df[label_col := dcfg["label_column"]].astype(int),
    ))
    texts = dict(zip(
        train_df[id_col].astype(str),
        train_df[dcfg["text_column"]].astype(str),
    ))

    in_dir = Path(args.input_output_dir)
    out_dir = Path(args.new_output_dir)
    out_cache_dir = out_dir / "explanation_cache"
    out_cache_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clustering").mkdir(parents=True, exist_ok=True)
    (out_dir / "refinement").mkdir(parents=True, exist_ok=True)

    # Copy similarity_model + any other clustering artifacts to new dir so
    # re-clustering can start fresh without re-fine-tuning.
    src_cluster = in_dir / "clustering"
    dst_cluster = out_dir / "clustering"
    for sub in ("similarity_model", "feature_embeddings.npy"):
        src = src_cluster / sub
        dst = dst_cluster / sub
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            logger.info("Copied %s → %s", src, dst)

    in_cache = in_dir / "explanation_cache" / "star_local_v1.jsonl"
    out_cache = out_cache_dir / "star_local_v1.jsonl"
    if not in_cache.exists():
        raise SystemExit(f"Missing input cache: {in_cache}")

    canonical_features = None
    if args.canonical_features_path:
        with open(args.canonical_features_path) as f:
            canonical_features = json.load(f)
        logger.info("Loaded %d canonical features for prompt context", len(canonical_features))
    canonical_context = _render_canonical_context(canonical_features)

    # Start with input cache
    records = load_star_cache(
        in_cache, task_metadata.positive_label, task_metadata.negative_label,
    )
    logger.info("Loaded %d cached extractions from %s", len(records), in_cache)

    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    for it in range(1, args.num_iterations + 1):
        logger.info("=" * 60)
        logger.info("Refinement iteration %d/%d", it, args.num_iterations)
        logger.info("=" * 60)

        # Find mispredicted — used only as a TARGETING signal (which docs to
        # re-extract on). The LLM itself is not told the true label.
        mispredicted = []
        for doc_id, r in records.items():
            pred = r["parsed"].prediction
            y = true_labels.get(doc_id)
            if y is None or pred not in ("positive", "negative"):
                continue
            pred_label = 1 if pred == "positive" else 0
            if pred_label != y:
                mispredicted.append(doc_id)
        rng = random.Random(42 + it)
        if len(mispredicted) > args.max_misclassified:
            mispredicted = rng.sample(mispredicted, args.max_misclassified)
        logger.info("Mispredicted docs targeted this iteration: %d", len(mispredicted))

        tasks_data = [
            (doc_id, texts[doc_id], records[doc_id]["parsed"])
            for doc_id in mispredicted if doc_id in texts
        ]

        proposed = asyncio.run(_run_propose_batch(
            client, tasks_data, args.model, args.concurrency,
            task_metadata.positive_label, task_metadata.negative_label,
            task_metadata.text_type, canonical_context,
            args.max_new_features_per_doc,
        ))

        n_proposed = sum(len(v) for v in proposed.values())
        logger.info("Total proposed features this iter: %d", n_proposed)

        proposed_path = out_dir / "refinement" / f"iter{it}_proposed.json"
        with open(proposed_path, "w") as f:
            json.dump({k: v for k, v in proposed.items()}, f, indent=2)

        # Augment records (LLM self-classifies each new feature as for/against;
        # we trust its self-classification rather than verifying)
        n_docs_changed = 0
        for doc_id, feats in proposed.items():
            if not feats:
                continue
            for_new = [x["feature"] for x in feats if x["side"] == "for"]
            against_new = [x["feature"] for x in feats if x["side"] == "against"]
            if not for_new and not against_new:
                continue
            records[doc_id]["raw"] = append_features_to_raw(
                records[doc_id]["raw"], for_new, against_new,
            )
            records[doc_id]["parsed"].features_for = list(records[doc_id]["parsed"].features_for) + for_new
            records[doc_id]["parsed"].features_against = list(records[doc_id]["parsed"].features_against) + against_new
            n_docs_changed += 1
        logger.info("Augmented %d docs with new features", n_docs_changed)

        summary = {
            "iteration": it,
            "n_mispredicted": len(mispredicted),
            "n_proposed_features": n_proposed,
            "n_docs_augmented": n_docs_changed,
        }
        with open(out_dir / "refinement" / f"iter{it}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Iter %d summary: %s", it, summary)

    # Final write to augmented cache
    write_star_cache(records, out_cache)
    logger.info("Wrote augmented cache → %s", out_cache)

    final_summary = {
        "num_iterations": args.num_iterations,
        "input_output_dir": str(in_dir),
        "new_output_dir": str(out_dir),
        "model": args.model,
        "max_misclassified_per_iter": args.max_misclassified,
        "max_new_features_per_doc": args.max_new_features_per_doc,
        "label_agnostic_proposal": True,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "refinement" / "refinement_config.json", "w") as f:
        json.dump(final_summary, f, indent=2)


if __name__ == "__main__":
    main()
