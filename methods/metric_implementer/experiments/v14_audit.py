"""Pure never-absorbed v14 audit draws and fixed-executor scoring."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from .cr3_evidence_store import EvidenceCellStore
from .v14_behavioral_channel import execute_rule_probe_cells


AUDIT_SCHEMA = "cr3-v14-pure-audit-v1"
AUDIT_FAMILIES = {
    "phi4": "microsoft/phi-4",
    "qwen14": "Qwen/Qwen2.5-14B-Instruct",
    "llama8": "meta-llama/Llama-3.1-8B-Instruct",
}
ATOMIC_PROPOSE_TEMPLATE = """You are sampling candidate articulations of an evaluation metric.
Metric name: "{name}"
Metric definition: "{description}"
Propose ONE concrete, checkable yes/no criterion that a careful reader could verify about a text
under this metric. Cover substantive content, including edge cases; avoid generic quality words.
Output ONLY the criterion as one sentence ending in '?'."""


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def audit_quotas(metric_key: str, total: int = 400) -> dict[str, int]:
    families = sorted(AUDIT_FAMILIES)
    base, remainder = divmod(int(total), len(families))
    rotation = stable_seed("v14-audit-quota", metric_key) % len(families)
    extras = {families[(rotation + index) % len(families)] for index in range(remainder)}
    return {family: base + int(family in extras) for family in families}


def frozen_audit_schedule(metric_key: str, total: int = 400) -> list[dict]:
    slots = [
        (family, family_index)
        for family, count in audit_quotas(metric_key, total).items()
        for family_index in range(count)
    ]
    order = sorted(
        range(len(slots)),
        key=lambda index: hashlib.sha256(
            f"v14-audit-schedule\x1f{metric_key}\x1f{slots[index][0]}\x1f{slots[index][1]}".encode()
        ).hexdigest(),
    )
    return [
        {"pooled_index": pooled, "family": slots[index][0], "family_index": slots[index][1]}
        for pooled, index in enumerate(order)
    ]


def validate_atomic_criterion(text: str) -> bool:
    value = re.sub(r"\s+", " ", str(text).strip())
    return (
        15 <= len(value) <= 240
        and value.endswith("?") and value.count("?") == 1
        and "```" not in value and "\n" not in value
    )


def normalize_atomic_criterion(raw: str) -> str:
    value = str(raw or "").strip()
    value = re.sub(r"^```(?:text)?\s*|\s*```$", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def propose_family_audit(
    backend, *, out_root: str | Path, metric_key: str, metric_name: str,
    metric_description: str, family: str, model: str, model_revision: str,
    total_budget: int = 400, temperature: float = 0.9, max_attempts: int = 8,
) -> dict:
    """Fill one predeclared family quota using independent deterministic draw seeds."""
    if family not in AUDIT_FAMILIES or AUDIT_FAMILIES[family] != model:
        raise ValueError("audit family/model does not match the frozen proposer mixture")
    quota = audit_quotas(metric_key, total_budget)[family]
    root = Path(out_root) / "audit" / "proposals" / family / metric_key
    prompt = ATOMIC_PROPOSE_TEMPLATE.format(
        name=str(metric_name), description=str(metric_description),
    )
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    accepted = 0
    for family_index in range(quota):
        path = root / f"{family_index:04d}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if (row.get("schema") != AUDIT_SCHEMA or row.get("family") != family
                    or int(row.get("family_index", -1)) != family_index):
                raise RuntimeError(f"mutated audit draw {path}")
            accepted += 1
            continue
        row = None
        for attempt in range(int(max_attempts)):
            seed = stable_seed("v14-pure-audit", metric_key, family, family_index, attempt)
            output = backend.generate_batch(
                [prompt], system=None, max_tokens=80, temperature=float(temperature), seed=[seed],
            )
            if len(output) != 1:
                raise RuntimeError("audit proposer returned an incomplete draw")
            criterion = normalize_atomic_criterion(output[0])
            if not validate_atomic_criterion(criterion):
                continue
            row = {
                "schema": AUDIT_SCHEMA,
                "evidence_role": "never_absorbed_pure_audit",
                "metric_key": str(metric_key),
                "family": str(family),
                "family_index": int(family_index),
                "model": str(model),
                "model_revision": str(model_revision),
                "temperature": float(temperature),
                "seed": int(seed),
                "attempt_index": int(attempt),
                "prompt_sha256": prompt_sha,
                "criterion": criterion,
                "criterion_sha256": hashlib.sha256(criterion.encode("utf-8")).hexdigest(),
                "conditional_on_validator": "single_question_15_to_240_chars",
                "absorbed_into_adaptive_mining": False,
            }
            break
        if row is None:
            raise RuntimeError(
                f"audit family {family} failed to fill draw {family_index} within {max_attempts} attempts"
            )
        _atomic_json(path, row)
        accepted += 1
    return {
        "metric_key": metric_key, "family": family, "quota": quota,
        "accepted": accepted, "complete": accepted == quota,
    }

def assemble_audit_ledger(
    *, out_root: str | Path, metric_key: str, total_budget: int = 400,
) -> list[dict]:
    root = Path(out_root)
    rows = {}
    for family, quota in audit_quotas(metric_key, total_budget).items():
        for family_index in range(quota):
            path = root / "audit" / "proposals" / family / metric_key / f"{family_index:04d}.json"
            if not path.exists():
                raise RuntimeError(f"audit proposal quota incomplete: {path}")
            row = json.loads(path.read_text())
            if row.get("evidence_role") != "never_absorbed_pure_audit":
                raise RuntimeError("audit proposal has an invalid evidence role")
            rows[(family, family_index)] = row
    ledger = []
    for slot in frozen_audit_schedule(metric_key, total_budget):
        row = dict(rows[(slot["family"], slot["family_index"])])
        row["pooled_index"] = int(slot["pooled_index"])
        ledger.append(row)
    if len(ledger) != int(total_budget):
        raise RuntimeError("assembled audit ledger has the wrong budget")
    path = root / "audit" / "ledgers" / f"{metric_key}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in ledger)
    if path.exists():
        if path.read_text() != packed:
            raise RuntimeError("existing audit ledger disagrees with frozen proposal cells")
    else:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(packed)
        os.replace(temporary, path)
    return ledger


def score_audit_ledger(
    executor, *, out_root: str | Path, metric_key: str, probe_texts: Sequence[str],
    executor_revision: str, readout_id: str, store: EvidenceCellStore,
    total_budget: int = 400, max_chars: int = 600, query_batch_size: int = 4096,
) -> dict:
    ledger = assemble_audit_ledger(
        out_root=out_root, metric_key=metric_key, total_budget=total_budget,
    )
    rules = {str(row["criterion_sha256"]): str(row["criterion"]) for row in ledger}
    cells = execute_rule_probe_cells(
        executor, rules=rules, probe_texts=probe_texts,
        executor_revision=executor_revision, readout_id=readout_id, store=store,
        max_chars=max_chars, query_batch_size=query_batch_size,
    )
    signatures = np.asarray([
        [cells[(str(row["criterion_sha256"]), index)]["p_yes"] for index in range(len(probe_texts))]
        for row in ledger
    ], dtype=float)
    if signatures.shape != (int(total_budget), len(probe_texts)) or np.any(~np.isfinite(signatures)):
        raise RuntimeError("audit signature table is incomplete")
    path = Path(out_root) / "audit" / "signatures" / f"{metric_key}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        np.savez_compressed(
            temporary, sigs=signatures,
            prompts=np.asarray([row["criterion"] for row in ledger], dtype=object),
            families=np.asarray([row["family"] for row in ledger], dtype=object),
            seeds=np.asarray([row["seed"] for row in ledger], dtype=np.int64),
            pooled_indices=np.arange(int(total_budget), dtype=np.int64),
            evidence_role=np.asarray("never_absorbed_pure_audit"),
            executor_revision=np.asarray(executor_revision), readout_id=np.asarray(readout_id),
        )
        os.replace(temporary, path)
    else:
        with np.load(path, allow_pickle=True) as previous:
            if not np.array_equal(np.asarray(previous["sigs"], dtype=float), signatures):
                raise RuntimeError("existing audit signature artifact disagrees with cell store")
    return {
        "metric_key": metric_key, "n_draws": len(ledger), "n_unique_rules": len(rules),
        "signature_path": str(path),
    }
