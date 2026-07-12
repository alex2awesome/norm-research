"""The record-keeping system: immutable versioned artifacts + append-only event ledger.

Layout (under ``outputs/metric_implementer/<task>/registry/``)::

    ledger.jsonl                                  # append-only event log, one JSON per line
    metrics/<metric_id>/
        versions/v000__prompt.json                # immutable version records
        versions/v001__prompt.json
        versions/v000__code.json
        scorecards/v001__prompt__<evalhash>.json  # scorecard per (version x eval config)
        disagreements/v000__<hash>.json           # code<->judge disagreement queues
        HEAD.json                                 # current accepted version per kind

Version records carry everything the scaling analysis needs, measured at creation time:

- ``budget``: auto-measured complexity (instruction_tokens / n_fewshots / clause_count for
  prompts; code_loc / code_ast_nodes / code_cyclomatic for code), the data budget (count +
  content hash of the item ids visible to the optimizer when this version was produced),
  the caps in force, and the models used.
- ``lineage``: parent version, the operator that produced it (INIT / CLARIFY / FEWSHOT+ /
  ANCHOR / EDGE / PRUNE / DECOMPOSE / CODE_REVISE / manual), run id, optimizer round.

Ledger events: METRIC_REGISTERED, VERSION_CREATED, VERSION_REJECTED_BUDGET,
SCORECARD_COMPUTED, ACCEPTED, RUN_STARTED, RUN_FINISHED (with cost accounting).
Everything is append-only; nothing is ever rewritten — re-running with the same inputs adds
new events rather than mutating old ones.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .artifact import MetricArtifact


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _hash_ids(ids: List[str]) -> str:
    return hashlib.sha256("|".join(sorted(map(str, ids))).encode()).hexdigest()[:12]


class Registry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ledger = self.root / "ledger.jsonl"

    # ---- ledger --------------------------------------------------------------------

    def log(self, event: str, **details) -> None:
        rec = {"ts": _now(), "event": event, **details}
        with open(self._ledger, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def ledger(self) -> List[dict]:
        if not self._ledger.exists():
            return []
        return [json.loads(ln) for ln in self._ledger.read_text().splitlines() if ln.strip()]

    # ---- metric dirs ------------------------------------------------------------------

    def _mdir(self, metric_id: str) -> Path:
        d = self.root / "metrics" / metric_id
        (d / "versions").mkdir(parents=True, exist_ok=True)
        (d / "scorecards").mkdir(exist_ok=True)
        (d / "disagreements").mkdir(exist_ok=True)
        return d

    def register_metric(self, metric_id: str, name: str, description: str) -> None:
        d = self._mdir(metric_id)
        meta = d / "metric.json"
        if not meta.exists():
            meta.write_text(json.dumps(
                {"metric_id": metric_id, "name": name, "description": description,
                 "created_at": _now()}, indent=2))
            self.log("METRIC_REGISTERED", metric_id=metric_id, name=name)

    # ---- versions -----------------------------------------------------------------

    def _next_version_id(self, metric_id: str, kind: str) -> str:
        vdir = self._mdir(metric_id) / "versions"
        existing = sorted(vdir.glob(f"v*__{kind}.json"))
        n = int(existing[-1].name[1:4]) + 1 if existing else 0
        return f"v{n:03d}"

    def create_version(
        self, artifact: MetricArtifact, *,
        operator: str, parent_version: Optional[str], run_id: Optional[str],
        optimizer_round: Optional[int], data_budget_ids: List[str],
        caps_in_force: Optional[dict], models: Dict[str, str],
        notes: str = "", optimizer: str = "gepa",
    ) -> str:
        """Persist an immutable version record; returns its version_id."""
        vid = self._next_version_id(artifact.metric_id, artifact.kind)
        rec = {
            "metric_id": artifact.metric_id,
            "version_id": vid,
            "kind": artifact.kind,
            "name": artifact.name,
            "description": artifact.description,
            "body": artifact.body,
            "body_hash": artifact.body_hash,
            "invariances": artifact.invariances,
            "budget": {
                **artifact.complexity(),
                "data_budget_n": len(data_budget_ids),
                "data_budget_hash": _hash_ids(data_budget_ids),
                "caps_in_force": caps_in_force,
                "models": models,
            },
            "lineage": {
                "parent_version": parent_version,
                "operator": operator,
                "optimizer": optimizer,
                "run_id": run_id,
                "optimizer_round": optimizer_round,
            },
            "created_at": _now(),
            "notes": notes,
        }
        path = self._mdir(artifact.metric_id) / "versions" / f"{vid}__{artifact.kind}.json"
        path.write_text(json.dumps(rec, indent=2))
        self.log("VERSION_CREATED", metric_id=artifact.metric_id, version_id=vid,
                 kind=artifact.kind, operator=operator, parent=parent_version,
                 body_hash=artifact.body_hash, run_id=run_id,
                 complexity=artifact.complexity())
        return vid

    def get_version(self, metric_id: str, version_id: str, kind: str) -> dict:
        p = self._mdir(metric_id) / "versions" / f"{version_id}__{kind}.json"
        return json.loads(p.read_text())

    def versions(self, metric_id: str, kind: Optional[str] = None) -> List[dict]:
        vdir = self._mdir(metric_id) / "versions"
        pat = f"v*__{kind}.json" if kind else "v*.json"
        return [json.loads(p.read_text()) for p in sorted(vdir.glob(pat))]

    def artifact_from_version(self, rec: dict) -> MetricArtifact:
        return MetricArtifact(
            metric_id=rec["metric_id"], kind=rec["kind"], body=rec["body"],
            name=rec.get("name", ""), description=rec.get("description", ""),
            invariances=rec.get("invariances", []))

    # ---- HEAD (current accepted version per kind) -----------------------------------

    def set_head(self, metric_id: str, kind: str, version_id: str, run_id: str,
                 judge_tier: Optional[str] = None) -> None:
        """HEAD is keyed by (kind, judge_tier): prompts are optimized FOR a specific
        judge, so each tier has its own current-best lineage. ``judge_tier=None`` keeps
        the legacy un-tiered pointer (code kind, pre-tier records)."""
        d = self._mdir(metric_id)
        head_path = d / "HEAD.json"
        head = json.loads(head_path.read_text()) if head_path.exists() else {}
        key = f"{kind}@{judge_tier}" if judge_tier else kind
        head[key] = {"version_id": version_id, "accepted_at": _now(), "run_id": run_id}
        head_path.write_text(json.dumps(head, indent=2))
        self.log("ACCEPTED", metric_id=metric_id, kind=kind, version_id=version_id,
                 run_id=run_id, judge_tier=judge_tier)

    def head(self, metric_id: str, kind: str,
             judge_tier: Optional[str] = None) -> Optional[str]:
        p = self._mdir(metric_id) / "HEAD.json"
        if not p.exists():
            return None
        heads = json.loads(p.read_text())
        key = f"{kind}@{judge_tier}" if judge_tier else kind
        return (heads.get(key) or {}).get("version_id")

    # ---- scorecards / disagreement queues --------------------------------------------

    def save_scorecard(self, metric_id: str, version_id: str, kind: str,
                       scorecard: dict, eval_config_hash: str) -> Path:
        p = self._mdir(metric_id) / "scorecards" / \
            f"{version_id}__{kind}__{eval_config_hash}.json"
        p.write_text(json.dumps(scorecard, indent=2, default=str))
        self.log("SCORECARD_COMPUTED", metric_id=metric_id, version_id=version_id,
                 kind=kind, eval_config_hash=eval_config_hash,
                 fidelity=scorecard.get("fidelity_scalar"))
        return p

    def scorecards(self, metric_id: str) -> List[dict]:
        sdir = self._mdir(metric_id) / "scorecards"
        return [json.loads(p.read_text()) for p in sorted(sdir.glob("*.json"))]

    def save_disagreements(self, metric_id: str, version_id: str, rows: List[dict],
                           tag: str) -> None:
        p = self._mdir(metric_id) / "disagreements" / f"{version_id}__{tag}.json"
        p.write_text(json.dumps(rows, indent=2, default=str))

    # ---- triad runs (CrowdTruth x G-theory panel analyses) ----------------------------

    def save_triad(self, metric_id: str, version_id: str, payload: dict,
                   long_table) -> Path:
        """Persist one triad run: the full long score table (judge x pass x item) as CSV
        next to the analysis JSON, timestamped so runs accumulate over time."""
        d = self._mdir(metric_id) / "triad"
        d.mkdir(exist_ok=True)
        stamp = _now().replace(":", "-")
        csv_path = d / f"{version_id}__{stamp}__long.csv"
        long_table.to_csv(csv_path, index=False)
        p = d / f"{version_id}__{stamp}__triad.json"
        p.write_text(json.dumps(payload, indent=2, default=str))
        self.log("TRIAD_COMPUTED", metric_id=metric_id, version_id=version_id,
                 aqs=(payload.get("crowdtruth") or {}).get("aqs"),
                 words_share=(payload.get("variance_components") or {}).get("words_share"),
                 cost_usd=payload.get("cost_usd"), path=str(p))
        return p
