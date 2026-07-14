"""Capacity-preserving behavioral decoder levers and release M-hat archival."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..batch_scoring import _YESNO_TEMPLATE
from .v14_behavioral_channel import normalized_rule
from .v14_panel_design import canonical_sha256


LEVER_ARMS = ("template_only", "best_of_4", "revise_once", "restate_then_induce")


def _generate(decoder, prompts: Sequence[str], seeds: Sequence[int], *, temperature: float) -> list[str]:
    output = decoder.generate_batch(
        list(prompts), system=None, max_tokens=160, temperature=float(temperature),
        seed=list(map(int, seeds)),
    )
    if len(output) != len(prompts):
        raise RuntimeError("behavioral lever decoder returned an incomplete batch")
    return [normalized_rule(value) for value in output]


def demo_fit_scores(
    executor, rules: Sequence[str], demo_texts: Sequence[str], labels: Sequence[int],
) -> np.ndarray:
    prompts = [
        _YESNO_TEMPLATE.format(rubric=rule, text=str(text))
        for rule in rules for text in demo_texts
    ]
    scores = np.asarray(executor.score_binary_constrained(
        prompts, system=None, pos="YES", neg="NO", seed=0,
    ), dtype=float).reshape(len(rules), len(demo_texts))
    target = np.asarray(labels, dtype=np.uint8)
    if scores.shape[1:] != target.shape or np.any(~np.isfinite(scores)):
        raise RuntimeError("demo-fit execution is incomplete")
    return np.mean((scores > 0.5) == target[None, :], axis=1)


def best_of_four(
    decoder, executor, *, induction_prompt: str, demo_texts: Sequence[str],
    labels: Sequence[int], seed: int,
) -> dict:
    rules = _generate(
        decoder, [induction_prompt] * 4, [int(seed) + index for index in range(4)],
        temperature=0.7,
    )
    fit = demo_fit_scores(executor, rules, demo_texts, labels)
    winner = min(
        range(4),
        key=lambda index: (-float(fit[index]), hashlib.sha256(rules[index].encode()).hexdigest()),
    )
    return {"rule": rules[winner], "demo_fit": float(fit[winner]), "candidates": rules}


def revise_once(
    decoder, executor, *, initial_rule: str, induction_prompt: str,
    demo_texts: Sequence[str], labels: Sequence[int], seed: int,
) -> dict:
    predictions = (np.asarray(executor.score_binary_constrained([
        _YESNO_TEMPLATE.format(rubric=initial_rule, text=str(text)) for text in demo_texts
    ], system=None, pos="YES", neg="NO", seed=0), dtype=float) > 0.5).astype(int)
    feedback = "\n".join(
        f"example {index + 1}: intended={int(gold)}, rule_predicted={int(predicted)}"
        for index, (gold, predicted) in enumerate(zip(labels, predictions))
    )
    prompt = (
        f"{induction_prompt}\n\nINITIAL RULE:\n{initial_rule}\n\n"
        f"EXECUTION ON THE TRAINING DEMOS ONLY:\n{feedback}\n\n"
        "Revise the rule once to fit the labeled demonstrations. Return only the revised rule."
    )
    rule = _generate(decoder, [prompt], [seed], temperature=0.0)[0]
    return {"rule": rule, "initial_rule": initial_rule, "demo_predictions": predictions.tolist()}


def restate_then_induce(decoder, *, induction_prompt: str, seed: int) -> dict:
    restatement = _generate(decoder, [
        f"{induction_prompt}\n\nDescribe only the contrastive label pattern in your own words; do not propose the final rule."
    ], [seed], temperature=0.0)[0]
    rule = _generate(decoder, [
        f"{induction_prompt}\n\nPATTERN RESTATEMENT:\n{restatement}\n\nReturn only the final general rule."
    ], [seed + 1], temperature=0.0)[0]
    return {"rule": rule, "restatement": restatement}


def archive_mhats(rows: Sequence[Mapping[str, object]], *, out_root: str | Path) -> dict:
    """Ship full content-addressed M-hat text plus an auditable identity index."""
    root = Path(out_root)
    objects = root / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    index_rows = []
    required = {
        "metric_key", "panel_sha256", "state", "arm", "decoder_family",
        "decoder_revision", "instrument", "lever", "rule",
    }
    for source in rows:
        if not required.issubset(source):
            raise ValueError(f"M-hat archive row lacks {sorted(required-set(source))}")
        rule = str(source["rule"]).strip()
        rule_sha = hashlib.sha256(rule.encode()).hexdigest()
        destination = objects / f"{rule_sha}.txt"
        if destination.exists() and destination.read_text(encoding="utf-8") != rule:
            raise RuntimeError("M-hat content-address collision")
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
            temporary.write_text(rule, encoding="utf-8")
            os.replace(temporary, destination)
        index_rows.append({**{key: source[key] for key in required if key != "rule"},
                           "rule_sha256": rule_sha, "rule_path": str(destination)})
    frame = pd.DataFrame(index_rows).sort_values([
        "metric_key", "panel_sha256", "state", "arm", "instrument", "lever",
    ])
    index_path = root / "index.parquet"
    frame.to_parquet(index_path, index=False)
    manifest = {
        "schema": "cr3-v14-mhat-archive-v1", "n_identity_rows": len(frame),
        "n_distinct_rules": int(frame.rule_sha256.nunique()),
        "index_path": str(index_path),
        "identity_sha256": canonical_sha256(index_rows),
    }
    manifest["sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
