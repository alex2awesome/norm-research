#!/usr/bin/env python
"""Score frozen item-adaptive ostensive prompts with the smaller Llama executor."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend


FORMS = ("canonical", "question", "boilerplate")
MODEL_JOB = "llama3_adaptive_ostensive_tf"


def _items(packet_root: str | Path, partition: str) -> dict[str, dict]:
    path = Path(packet_root) / "humor" / "items" / f"{partition}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["text_sha256"]: row for row in rows}


def adaptive_content(evaluation: dict, arm: dict, item_hash: str) -> str:
    retrieval = next(row for row in evaluation["retrievals"]
                     if row["id"] == arm["retrieval_id"])
    assignment = retrieval["assignments"][item_hash]
    examples = evaluation["teaching_examples"]
    blocks = []
    parent = evaluation["parent_texts"][arm["parent_id"]].strip()
    if parent:
        blocks.append(parent)
    blocks.append(
        "The following are explicit demonstrations of the fixed larger-Llama policy. "
        "They are not external ground truth. Reproduce the distinction they encode for the new "
        "item, including analogous mechanisms and boundary cases.")
    for polarity, label in (("positive", "YES"), ("negative", "NO")):
        values = []
        for example_hash in assignment[polarity]:
            example = examples[example_hash]
            score_note = f"target-policy score {example['target_score']:.2f}"
            if retrieval["pool"] == "residual":
                score_note += f"; smaller name-only score {example['small_name_score']:.2f}"
            values.append(
                f"[{label}; {score_note}]\n{example['text']}")
        blocks.append(f"Demonstrated {label} cases:\n" + "\n\n".join(values))
    blocks.append(
        "Use the demonstrations as local evidence about the criterion, not as topics or phrases "
        "that must literally recur.")
    return "\n\n".join(blocks)


def score_prompt(content: str, item_text: str, *, form: str) -> str:
    if form == "canonical":
        framing = f"Criterion and explicit guidance:\n{content}"
    elif form == "question":
        framing = f"Does the new item satisfy the criterion represented below?\n{content}"
    elif form == "boilerplate":
        framing = f"You are an expert evaluator. Apply this guidance strictly:\n{content}"
    else:
        raise ValueError(f"unknown form {form}")
    return (f"{framing}\n\nNEW ITEM:\n{item_text}\n\n"
            "Does the new item satisfy the represented criterion? Answer exactly YES or NO.")


def score_declared_binary(backend, prompts: list[str], *, pos: str = "YES",
                          neg: str = "NO", seed: int | Sequence[int] = 0,
                          expected_token_ids: dict[str, int] | None = None) -> np.ndarray:
    """Exact conditional P(pos) from teacher-forced pos/neg continuation likelihoods."""
    prompt_seeds = None
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
        prompt_seeds = [int(value) for value in seed]
        if len(prompt_seeds) != len(prompts):
            raise ValueError(
                f"got {len(prompt_seeds)} seeds for {len(prompts)} prompts"
            )
    if backend.__class__.__name__ == "FakeVLLM":
        values = np.asarray(backend.score_binary(
            prompts,
            pos=pos,
            neg=neg,
            seed=prompt_seeds if prompt_seeds is not None else seed,
        ), float)
        if values.shape != (len(prompts),):
            raise ValueError(
                "declared-binary backend returned an invalid output shape: "
                f"observed={values.shape}, expected={(len(prompts),)}"
            )
        return values
    from vllm import SamplingParams

    engine = backend._engine(backend.model, backend.cfg)
    tokenizer = engine.get_tokenizer()
    labels = [pos, neg]
    token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(values) != 1 for values in token_ids):
        raise ValueError(f"declared labels are not single tokens: {dict(zip(labels, token_ids))}")
    runtime_token_ids = [values[0] for values in token_ids]
    if expected_token_ids is not None:
        frozen = [int(expected_token_ids[label]) for label in labels]
        if runtime_token_ids != frozen:
            raise ValueError(
                "runtime declared-label token ids differ from the frozen manifest: "
                f"observed={dict(zip(labels, runtime_token_ids))} "
                f"expected={dict(zip(labels, frozen))}"
            )
    rendered_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        rendered_prompts.append(rendered)
    # Prompt logprobs always include the probability of the actual prompt token, even when that
    # token is outside a small top-k.  Scoring both one-token continuations therefore recovers the
    # two arbitrary logits without sampling, missing-label imputation, or a constrained-top-k bug.
    scoring_texts = [f"{rendered}{label}"
                     for rendered in rendered_prompts for label in labels]
    if prompt_seeds is None:
        params = SamplingParams(
            temperature=0.0, max_tokens=1, prompt_logprobs=0, seed=int(seed))
    else:
        # Each original prompt becomes two teacher-forced requests.  Repeating its frozen seed
        # across YES/NO preserves the exact per-prompt estimand when multiple arm/form rows are
        # flattened into one engine call.
        scoring_seeds = [
            prompt_seed
            for prompt_seed in prompt_seeds
            for _label in labels
        ]
        params = [
            SamplingParams(
                temperature=0.0, max_tokens=1, prompt_logprobs=0, seed=item_seed)
            for item_seed in scoring_seeds
        ]
    outputs = engine.generate(scoring_texts, params)
    if len(outputs) != len(scoring_texts):
        raise ValueError(
            "teacher-forced engine returned an invalid output count: "
            f"observed={len(outputs)}, expected={len(scoring_texts)}"
        )
    backend.stats.n_calls += 1
    backend.stats.n_prompts += len(scoring_texts)
    log_probabilities = []
    for output_index, output in enumerate(outputs):
        expected = runtime_token_ids[output_index % len(labels)]
        actual = int(output.prompt_token_ids[-1])
        if actual != expected:
            raise ValueError(
                f"continuation tokenization changed: expected {expected}, observed {actual}")
        prompt_logprobs = output.prompt_logprobs or []
        if not prompt_logprobs or not prompt_logprobs[-1]:
            raise ValueError("teacher-forced continuation logprob is missing")
        entry = prompt_logprobs[-1].get(actual)
        if entry is None:
            raise ValueError(
                f"teacher-forced logprob mapping omits actual continuation token {actual}"
            )
        log_probabilities.append(float(entry.logprob))
    log_probabilities = np.asarray(log_probabilities, float).reshape(len(prompts), 2)
    log_odds = np.clip(
        log_probabilities[:, 0] - log_probabilities[:, 1], -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-log_odds))


def run(*, bank_path: str, packet_root: str, out_root: str,
        model: str = "meta-llama/Llama-3.2-3B-Instruct", fake: bool = False) -> dict:
    bank = json.loads(Path(bank_path).read_text())
    if bank.get("status") != "frozen-before-adaptive-small-executor-scoring":
        raise ValueError("adaptive bank is not frozen")
    config = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
    if fake:
        config.vllm_fake = True
    backend = make_judge_backend(model, config, temperature=None)
    outputs = []
    for evaluation_index, evaluation in enumerate(bank["evaluations"]):
        partition = evaluation["evaluation_partition"]
        item_map = _items(packet_root, partition)
        hashes = evaluation["evaluation_item_sha256"]
        if not set(hashes) <= set(item_map):
            raise ValueError("frozen evaluation hashes and score packet differ")
        rows, meta = [], []
        for arm_index, arm in enumerate(evaluation["arms"]):
            for form_index, form in enumerate(FORMS):
                prompts = [score_prompt(
                    adaptive_content(evaluation, arm, item_hash),
                    item_map[item_hash]["text"][:config.max_text_chars], form=form)
                           for item_hash in hashes]
                scores = score_declared_binary(
                    backend, prompts, pos="YES", neg="NO",
                    seed=20260725 + 100_003 * evaluation_index
                    + 1009 * arm_index + form_index)
                if not np.isfinite(scores).all():
                    raise ValueError(f"non-finite binary readout for {partition}/{arm['id']}/{form}")
                rows.append(scores)
                meta.append({
                    "arm_id": arm["id"],
                    "cell_id": "N_humor_49",
                    "domain": "humor",
                    "gi": 49,
                    "construct": "Wordplay quality and clarity",
                    "form": form,
                    "channel": arm["channel"],
                    "provenance": arm["provenance"],
                    "source_partition": arm["source_partition"],
                    "parent_id": arm["parent_id"],
                    "retrieval_id": arm["retrieval_id"],
                    "semantic_content_word_count": arm["semantic_content_word_count"],
                })
        directory = Path(out_root) / partition / MODEL_JOB
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / "grid_humor_crossfit_Llama-3.2-3B-Instruct_rep0.npz"
        np.savez_compressed(
            out,
            scores=np.asarray(rows),
            meta=np.asarray([json.dumps(row, sort_keys=True) for row in meta], dtype=object),
            probe_sha256=np.asarray(hashes),
            probe_partition=np.asarray([partition] * len(hashes)),
            reader=model,
            model_job_id=MODEL_JOB,
            role="small",
            phase="crossfit",
            repetition=0,
            source_artifact_sha256=sha256_file(bank_path),
            isolated_partition=partition,
        )
        sidecar = out.with_suffix(".json")
        sidecar.write_text(json.dumps({
            "schema": "adaptive_ostensive_scores/v1",
            "status": "public-crossfit-development-only",
            "partition": partition,
            "teaching_partition": evaluation["teaching_partition"],
            "model": model,
            "model_family": "Llama",
            "n_items": len(hashes),
            "n_arms": len(evaluation["arms"]),
            "n_forms": len(FORMS),
            "bank_sha256": sha256_file(bank_path),
            "readout": ("teacher-forced first-token log P(YES) and log P(NO), normalized over "
                        "YES/NO; label vocabulary and conditional estimand match the fixed target"),
            "lockbox_status": "not read or authorized",
        }, indent=1))
        outputs.append({"partition": partition, "npz": str(out),
                        "sidecar": str(sidecar), "sha256": sha256_file(out)})
    return {"schema": "adaptive_ostensive_execution/v1",
            "bank_sha256": sha256_file(bank_path), "model": model,
            "model_job_id": MODEL_JOB, "outputs": outputs,
            "lockbox_status": "not read or authorized"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    result = run(bank_path=args.bank, packet_root=args.packet_root,
                 out_root=args.out_root, model=args.model, fake=args.fake)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
