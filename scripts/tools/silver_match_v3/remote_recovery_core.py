"""Pure contracts and classification for Humor CE recovery."""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "silver-match-v3-humor-ce-remote-recovery-v1"
REPORT_SCHEMA = "silver-match-v3-nemotron-bidirectional-cross-encoder-v1"
REMOTE_ROOT = Path(
    "/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_v2"
)
MODEL = Path(
    "/lfs/skampere2/0/alexspan/norm-research-silver-v3/models/llama-embed-nemotron-8b-aa3b43a495a9b280d1bdb716da37c54bb495d630-mirror-v1"
)
FORBIDDEN_SK3_GPUS = (1, 2, 3, 4)
MAX_FULL_BYTES = 4194304
TAIL_BYTES = 131072


@dataclass(frozen=True)
class PilotSpec:
    name: str
    rank: int
    alpha: int
    learning_rate: float
    expected_gpu: int

    def root(self, b: Path) -> Path:
        return b / "runs" / self.name


PILOTS = (
    PilotSpec("humor_ce_r16_a32_lr1e4_seed20260713_v2", 16, 32, 1e-4, 2),
    PilotSpec("humor_ce_r32_a64_lr5e5_seed20260713_v2", 32, 64, 5e-5, 3),
)


def canonical(x: Any) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def artifact_specs(b: Path, p: PilotSpec) -> list[dict[str, str]]:
    r = p.root(b)
    return [
        {"key": k, "path": str(x), "mode": m}
        for k, x, m in (
            ("run_config", r / "run_config.json", "full"),
            ("training_report", r / "training_report.json", "full"),
            ("reload_verification", r / "reload_verification.json", "full"),
            ("events", r / "events.jsonl", "tail"),
            ("split_assignments", r / "split_assignments.jsonl", "hash_only"),
            ("log", b / "runs" / "logs" / f"{p.name}.log", "tail"),
        )
    ]


def inspection_plan(b: Path, h: str, g: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "dry_run": True,
        "read_only": True,
        "pilot_host": h,
        "gpu_hosts": list(dict.fromkeys(g)),
        "remote_root": str(b),
        "pilots": [
            {
                "name": p.name,
                "expected_root": str(p.root(b)),
                "expected_gpu": p.expected_gpu,
                "artifacts": artifact_specs(b, p),
            }
            for p in PILOTS
        ],
        "forbidden": {"host": "sk3", "gpu_indices": list(FORBIDDEN_SK3_GPUS)},
        "mutation_commands": [],
        "checkpoints_read_or_copied": False,
    }


def read_local(i: Mapping[str, Any]) -> dict[str, Any]:
    p = Path(str(i["path"]))
    m = str(i["mode"])
    r = {"key": i["key"], "path": str(p), "mode": m, "exists": p.is_file()}
    if not p.is_file():
        return r
    d = p.read_bytes()
    r.update(size=len(d), sha256=sha(d))
    if m == "full":
        r["content" if len(d) <= MAX_FULL_BYTES else "error"] = (
            d if len(d) <= MAX_FULL_BYTES else "artifact exceeds limit"
        )
    elif m == "tail":
        t = d[-TAIL_BYTES:]
        r.update(content=t, content_sha256=sha(t), tail_bytes=len(t))
    return r


def local_probe(b: Path) -> dict[str, Any]:
    return {
        "roots": {p.name: p.root(b).is_dir() for p in PILOTS},
        "artifacts": {
            p.name: [read_local(x) for x in artifact_specs(b, p)] for p in PILOTS
        },
    }


def _load(rows: Mapping[str, Mapping[str, Any]], k: str) -> tuple[Any, str | None]:
    r = rows.get(k)
    if not r or not r.get("exists"):
        return None, None
    if not isinstance(r.get("content"), bytes):
        return None, str(r.get("error") or "content unavailable")
    try:
        return json.loads(r["content"]), None
    except Exception as e:
        return None, f"invalid JSON: {e}"


def _events(
    rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    r = rows.get("events")
    d = r.get("content") if r else None
    if not r or not r.get("exists"):
        return [], None
    if not isinstance(d, bytes):
        return [], "event tail unavailable"
    lines = d.decode(errors="replace").splitlines()
    if int(r.get("size", 0)) > len(d) and lines:
        lines = lines[1:]
    try:
        return [json.loads(x) for x in lines if x.strip()], None
    except Exception as e:
        return [], f"invalid event JSON: {e}"


def classify_pilot(
    p: PilotSpec, b: Path, root: bool, artifacts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = {str(x["key"]): x for x in artifacts}
    why = []
    cfg, ce = _load(rows, "run_config")
    rep, re = _load(rows, "training_report")
    reload, rle = _load(rows, "reload_verification")
    ev, ee = _events(rows)
    names = [str(x.get("event")) for x in ev]
    for label, error in (
        ("run_config", ce),
        ("training_report", re),
        ("reload_verification", rle),
        ("events", ee),
    ):
        if error:
            why.append(f"{label}: {error}")
    if "RUN_FAILED" in names:
        why.append("RUN_FAILED terminal event")
    if isinstance(cfg, Mapping):
        expected = {
            "schema_version": REPORT_SCHEMA,
            "seed": 20260713,
            "max_length": 1024,
            "exposure_budgets": [10000, 25000, 50000],
            "lora_learning_rate": p.learning_rate,
            "head_learning_rate": 1e-3,
            "attention": "eager",
        }
        why.extend(
            f"run_config {k} drift" for k, v in expected.items() if cfg.get(k) != v
        )
        lora = cfg.get("lora")
        if not isinstance(lora, Mapping) or (
            lora.get("rank"),
            lora.get("alpha"),
        ) != (
            p.rank,
            p.alpha,
        ):
            why.append("run_config LoRA recipe drift")
        if Path(str(cfg.get("model") or "")) != MODEL:
            why.append("run_config model path drift")
        s = rows.get("split_assignments", {})
        if s.get("exists") and cfg.get("split_assignments_sha256") != s.get("sha256"):
            why.append("split assignment hash drift")
    elif root or "RUN_STARTED" in names or rows.get("log", {}).get("exists"):
        why.append("run_config missing")
    complete = rep is not None or reload is not None or "RUN_COMPLETE" in names
    if complete:
        if not isinstance(rep, Mapping) or (
            rep.get("schema_version"),
            rep.get("status"),
        ) != (REPORT_SCHEMA, "COMPLETE"):
            why.append("complete report missing or invalid")
        if not isinstance(reload, Mapping) or reload.get("status") != "PASS":
            why.append("reload verification missing or failed")
        if isinstance(rep, Mapping) and rep.get("reload_verification") != reload:
            why.append("embedded reload verification differs")
        if "RUN_COMPLETE" not in names:
            why.append("RUN_COMPLETE terminal event missing")
        if isinstance(rep, Mapping):
            ledger = rep.get("input_sha256")
            if not isinstance(ledger, Mapping) or ledger.get("run_config") != rows.get(
                "run_config", {}
            ).get("sha256"):
                why.append("training report run_config hash drift")
            ends = [x for x in ev if x.get("event") == "RUN_COMPLETE"]
            if ends and ends[-1].get("training_report_sha256") != rows.get(
                "training_report", {}
            ).get("sha256"):
                why.append("RUN_COMPLETE report hash drift")
    absent = not root and not any(x.get("exists") for x in rows.values())
    if absent:
        status, why = "ABSENT", ["expected run root and log are absent"]
    elif why:
        status = "FAILED"
    elif complete:
        status = "COMPLETE"
    elif root and isinstance(cfg, Mapping) and "RUN_STARTED" in names:
        status, why = "RUNNING", ["RUN_STARTED observed with no terminal event"]
    else:
        status, why = "FAILED", ["inconsistent non-terminal artifact state"]
    return {
        "name": p.name,
        "status": status,
        "expected_root": str(p.root(b)),
        "expected_gpu": p.expected_gpu,
        "recipe": {"rank": p.rank, "alpha": p.alpha, "learning_rate": p.learning_rate},
        "events_observed": names,
        "reasons": why,
    }
