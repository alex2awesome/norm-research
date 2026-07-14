#!/usr/bin/env python3
"""Build a lineage-preserving R1/R2/R3 similarity distillation dataset.

The lexicon pipeline intentionally persists terse score-only votes separately
from the payloads shown to judges.  This builder joins those two artifact
families, records every source, deduplicates repeated shards, balances teacher
families, and freezes leakage-safe splits.  Ambiguous rows are quarantined;
they are never silently assigned a teacher, level, task, or displayed text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


LABELS = ("DIFFERENT", "RELATED", "SAME")
LEVELS = ("R1", "R2", "R3")
PRIMARY_TEACHERS = frozenset({"sonnet", "gpt5"})
AUXILIARY_TEACHERS = frozenset({"opus", "glm"})
TASKS = (
    "math-stackexchange",
    "creative-writing",
    "code-review",
    "news-homepages",
    "peer-review",
    "notice-and-comment",
    "legal-outcome-prediction",
    "press-releases",
    "grant-funding",
    "patents",
    "humor",
)
TASK_ALIASES = {
    "math": "math-stackexchange",
    "math-stackexchange": "math-stackexchange",
    "cw": "creative-writing",
    "creative-writing": "creative-writing",
    "cr": "code-review",
    "code-review": "code-review",
    "news": "news-homepages",
    "news-homepages": "news-homepages",
    "peer": "peer-review",
    "peer-review": "peer-review",
    "notice": "notice-and-comment",
    "notice-and-comment": "notice-and-comment",
    "legal": "legal-outcome-prediction",
    "legal-outcome-prediction": "legal-outcome-prediction",
    "press": "press-releases",
    "press-releases": "press-releases",
    "grant": "grant-funding",
    "grant-funding": "grant-funding",
    "patent": "patents",
    "patents": "patents",
    "humor": "humor",
}

PROTOCOL_IDS = {
    "R1": "r1-narrow-construct-v1",
    "R2": "r2-legacy-theme-v1",
    "R2_V2": "r2-operational-theme-v2",
    "R2_V2_1": "r2-focused-operational-family-v2.1",
    "R3": "r3-top-level-category-v1",
}

PROTOCOL_RELATIVE_PATHS = {
    PROTOCOL_IDS["R1"]: "ARBITER_PROTOCOL_R1.txt",
    PROTOCOL_IDS["R2"]: "ARBITER_PROTOCOL_R2.txt",
    PROTOCOL_IDS["R2_V2"]: "r2_recluster_v2/R2_V2_PROTOCOL.md",
    PROTOCOL_IDS["R2_V2_1"]: "r2_recluster_v2/R2_V2_1_PROTOCOL.md",
    PROTOCOL_IDS["R3"]: "ARBITER_PROTOCOL_R3.txt",
}

RESERVED_PARTS = frozenset(
    {
        "arbiter_votes",
        "codex_val",
        "r1_truth_reaudit",
        "upper_precision_audit",
        "variant_eval",
        "versioned_level_eval",
        "large_group_cert",
        "l0_precision_audit",
    }
)


@dataclass(frozen=True)
class SourceSpec:
    teacher_family: str
    role: str
    provenance_strength: str
    label_kind: str = "independent"
    weight: float = 1.0


@dataclass
class Presentation:
    pair_id: str
    task: str | None
    level: str | None
    text_a: str
    text_b: str
    aliases_a: tuple[str, ...]
    aliases_b: tuple[str, ...]
    source_path: str


@dataclass
class Vote:
    pair_id: str
    score: int
    task: str
    level: str
    protocol_id: str
    teacher_family: str
    role: str
    provenance_strength: str
    label_kind: str
    source_path: str
    source_sha256: str
    source_line: int | None
    base_weight: float


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fraction(*parts: object) -> float:
    value = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) / float(16**16)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def identity_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _text_alias(text: str) -> str:
    return "text:" + sha256_bytes(identity_text(text).encode())[:24]


def _qualified_alias(task: str | None, level: str | None, kind: str, value: object) -> str:
    return f"{task or '?'}|{level or '?'}|{kind}:{normalize_text(value)}"


def _group_text(group: Mapping[str, Any]) -> str:
    rows = group.get("representative_members") or group.get("all_members") or []
    rendered: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = normalize_text(row.get("name") or row.get("text") or row.get("canonical"))
        gloss = normalize_text(row.get("gloss") or row.get("description"))
        value = ". ".join(part for part in (name, gloss) if part)
        if value and value not in rendered:
            rendered.append(value)
    heading = normalize_text(group.get("name") or group.get("label"))
    if heading:
        rendered.insert(0, heading)
    return "\n".join(rendered[:12])


def presentation_from_row(row: Mapping[str, Any], source_path: str) -> Presentation | None:
    pair_id = normalize_text(row.get("pair_id"))
    if not pair_id:
        return None
    task = normalize_task(row.get("task")) or infer_task(source_path)
    level = normalize_level(row.get("level")) or infer_level(source_path)
    if "concept_a" in row or "canonical_a" in row or "text_a" in row:
        text_a = normalize_text(row.get("concept_a") or row.get("canonical_a") or row.get("text_a"))
        text_b = normalize_text(row.get("concept_b") or row.get("canonical_b") or row.get("text_b"))
    elif isinstance(row.get("group_a"), Mapping) and isinstance(row.get("group_b"), Mapping):
        text_a = _group_text(row["group_a"])
        text_b = _group_text(row["group_b"])
    else:
        return None
    if not text_a or not text_b or identity_text(text_a) == identity_text(text_b):
        return None
    aliases_a = {_text_alias(text_a)}
    aliases_b = {_text_alias(text_b)}
    for side, aliases in (("a", aliases_a), ("b", aliases_b)):
        for field in (f"node_{side}", f"key_{side}", f"cluster_{side}"):
            if row.get(field) is not None:
                aliases.add(_qualified_alias(task, level, field[:-2], row[field]))
        group = row.get(f"group_{side}")
        if isinstance(group, Mapping) and group.get("group_id") is not None:
            aliases.add(_qualified_alias(task, level, "group", group["group_id"]))
    return Presentation(
        pair_id=pair_id,
        task=task,
        level=level,
        text_a=text_a,
        text_b=text_b,
        aliases_a=tuple(sorted(aliases_a)),
        aliases_b=tuple(sorted(aliases_b)),
        source_path=source_path,
    )


def normalize_task(value: object) -> str | None:
    token = normalize_text(value).lower().replace("_", "-")
    return TASK_ALIASES.get(token)


def normalize_level(value: object) -> str | None:
    token = normalize_text(value).upper()
    return token if token in LEVELS else None


def infer_task(path: str) -> str | None:
    value = path.lower().replace("_", "-")
    for task in sorted(TASKS, key=len, reverse=True):
        if task in value:
            return task
    name = Path(path).name.lower()
    for alias in ("cw", "cr", "math", "news", "peer", "notice", "legal", "press", "grant"):
        if re.search(rf"(^|[_-]){re.escape(alias)}r?[123]?(?:[_-]|\d|$)", name):
            return TASK_ALIASES[alias]
    return None


def infer_level(path: str) -> str | None:
    match = re.search(r"(?:^|[_-])(R[123])(?:[_\-.]|$)", path, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(?:cw|cr)(R[123])", Path(path).name, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def protocol_for(path: str, level: str) -> str:
    lower = path.lower()
    if level == "R2":
        if "v2_1" in lower or "v2-1" in lower or "r2v21" in lower:
            return PROTOCOL_IDS["R2_V2_1"]
        if "r2_recluster_v2" in lower:
            return PROTOCOL_IDS["R2_V2"]
        return PROTOCOL_IDS["R2"]
    return PROTOCOL_IDS[level]


def classify_score_source(relative_path: str) -> SourceSpec | None:
    """Return a conservative teacher/role assignment for a score artifact.

    The rules encode documented workflow provenance, not semantic guesses.
    Unknown paths remain visible in quarantine inventories.
    """

    path = relative_path.replace(os.sep, "/")
    lower = path.lower()
    parts = set(Path(lower).parts)
    name = Path(lower).name
    if "archive" in parts or "similarity_distill_v1" in parts:
        return None
    if infer_level(path) not in LEVELS:
        return None
    if any(part in RESERVED_PARTS for part in parts):
        if "codex_val" in parts:
            return SourceSpec("gpt5", "reserved", "workflow_inferred")
        if "r1_truth_reaudit" in parts:
            kind = "consensus" if "final_votes" in parts else "independent"
            return SourceSpec("sonnet", "reserved", "workflow_inferred", kind)
        return SourceSpec("unknown", "reserved", "unresolved")
    if "r2_recluster_v2" in parts and "comparison_votes" in name:
        return SourceSpec("gpt5", "reserved", "workflow_inferred")
    if "level_votes" in parts:
        if name.startswith("vrf_"):
            return SourceSpec("sonnet", "train", "workflow_inferred")
        if name.startswith("arbopus_"):
            return SourceSpec("opus", "reserved", "workflow_inferred", weight=0.25)
        if name.startswith("arb_"):
            return SourceSpec("sonnet", "reserved", "workflow_inferred")
        if name.startswith("cfm_"):
            return SourceSpec("opus", "auxiliary", "workflow_inferred", weight=0.25)
        return None
    if "codex_build" in parts:
        if "son" in name:
            return SourceSpec("sonnet", "train", "filename_inferred")
        return SourceSpec("gpt5", "train", "workflow_inferred")
    if "selective_vote_reaudit" in parts:
        if "audit_votes" in parts:
            return SourceSpec("gpt5", "train", "workflow_inferred")
        # Consolidated files are copies of original vrf labels; the original
        # level_votes rows already provide their independent observation.
        return None
    if "semantic_group_merge" in parts:
        if "confirm_votes" in lower:
            return SourceSpec("opus", "auxiliary", "workflow_inferred", weight=0.25)
        if "screen_votes" in lower:
            return SourceSpec("sonnet", "train", "workflow_inferred")
        if "clique9k_votes_c" in lower or name.startswith(("codex", "root", "harvey")):
            return SourceSpec("gpt5", "train", "filename_inferred")
        return None
    if "r2a_opus" in parts or "opus_t1" in parts:
        return SourceSpec("opus", "auxiliary", "filename_inferred", weight=0.25)
    return None


def _iter_objects(path: Path) -> Iterator[tuple[Any, int | None]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line), line_number
                except json.JSONDecodeError:
                    continue
        return
    try:
        yield json.loads(path.read_text(encoding="utf-8")), None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return


def _walk_pair_rows(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("pair_id") is not None:
            yield value
        for child in value.values():
            if isinstance(child, (Mapping, list)):
                yield from _walk_pair_rows(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (Mapping, list)):
                yield from _walk_pair_rows(child)


def _score_rows(value: Any) -> Iterator[tuple[str, int]]:
    if isinstance(value, Mapping) and value and all(
        isinstance(key, str) and type(score) is int and score in (0, 1, 2)
        for key, score in value.items()
    ):
        yield from ((key, score) for key, score in value.items())
        return
    for row in _walk_pair_rows(value):
        score = row.get("score")
        if type(score) is int and score in (0, 1, 2):
            yield normalize_text(row["pair_id"]), score


def discover_json_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".json", ".jsonl"}
        and "archive" not in path.relative_to(root).parts
        and "similarity_distill_v1" not in path.relative_to(root).parts
    )


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union_all(self, values: Sequence[str]) -> None:
        if not values:
            return
        root = self.find(values[0])
        for value in values[1:]:
            other = self.find(value)
            if other != root:
                smaller, larger = sorted((root, other))
                self.parent[larger] = smaller
                root = smaller


def _select_presentation(
    candidates: Sequence[Presentation], task: str | None, level: str | None
) -> Presentation | None:
    matches = [
        row
        for row in candidates
        if (not task or not row.task or row.task == task)
        and (not level or not row.level or row.level == level)
    ]
    signatures: dict[tuple[str, str], Presentation] = {}
    for row in matches:
        signature = tuple(sorted((identity_text(row.text_a), identity_text(row.text_b))))
        signatures.setdefault(signature, row)
    if len(signatures) != 1:
        return None
    return next(iter(signatures.values()))


def _family_distribution(votes: Sequence[Vote]) -> list[float]:
    selected = [vote for vote in votes if vote.label_kind == "consensus"] or list(votes)
    # Exact duplicate labels copied into consolidation directories are not
    # independent evidence.  Source path + line identifies a persisted call.
    unique = {(vote.source_sha256, vote.source_line, vote.score): vote for vote in selected}
    counts = Counter(vote.score for vote in unique.values())
    total = sum(counts.values())
    return [counts[index] / total for index in range(3)]


def balanced_target(votes: Sequence[Vote]) -> tuple[list[float], float, dict[str, list[float]]]:
    by_family: dict[str, list[Vote]] = defaultdict(list)
    for vote in votes:
        by_family[vote.teacher_family].append(vote)
    distributions = {
        family: _family_distribution(rows) for family, rows in sorted(by_family.items())
    }
    primary = [family for family in ("sonnet", "gpt5") if family in distributions]
    if primary:
        target = [sum(distributions[f][i] for f in primary) / len(primary) for i in range(3)]
        confidence = 1.0 if len(primary) == 2 else 0.5
    else:
        auxiliary = [family for family in ("opus", "glm") if family in distributions]
        if not auxiliary:
            raise ValueError("no attributable teacher family")
        target = [sum(distributions[f][i] for f in auxiliary) / len(auxiliary) for i in range(3)]
        confidence = 0.25
    return target, confidence, distributions


def render_prompt(protocol_text: str, task: str, text_a: str, text_b: str) -> str:
    return (
        "Classify the similarity relation using the frozen protocol below.\n\n"
        f"PROTOCOL\n{protocol_text.strip()}\n\n"
        f"DOMAIN: {task}\n\nCONCEPT A\n{text_a.strip()}\n\n"
        f"CONCEPT B\n{text_b.strip()}\n\n"
        "Return exactly one label: DIFFERENT, RELATED, or SAME."
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _protocol_bundle(root: Path) -> dict[str, dict[str, str]]:
    bundle: dict[str, dict[str, str]] = {}
    for protocol_id, relative in PROTOCOL_RELATIVE_PATHS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        bundle[protocol_id] = {
            "path": str(path.resolve()),
            "sha256": sha256_bytes(text.encode()),
            "text": text,
        }
    return bundle


def _split_rows(rows: list[dict[str, Any]], seed: int) -> None:
    identities: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["task"], row["level"])
        identities[key].update((row["identity_a"], row["identity_b"]))
    cold_test: set[str] = set()
    cold_dev: set[str] = set()
    for (task, level), values in identities.items():
        for identity in values:
            fraction = stable_fraction(seed, "identity", task, level, identity)
            qualified = f"{task}|{level}|{identity}"
            if fraction < 0.08:
                cold_test.add(qualified)
            elif fraction < 0.13:
                cold_dev.add(qualified)
    for row in rows:
        qa = f"{row['task']}|{row['level']}|{row['identity_a']}"
        qb = f"{row['task']}|{row['level']}|{row['identity_b']}"
        if row["reserved"]:
            row["split"] = "external_test"
        elif qa in cold_test or qb in cold_test:
            row["split"] = "cold_test_both" if qa in cold_test and qb in cold_test else "cold_test_any"
        elif qa in cold_dev or qb in cold_dev:
            row["split"] = "cold_dev_both" if qa in cold_dev and qb in cold_dev else "cold_dev_any"
        else:
            fraction = stable_fraction(seed, "pair", row["level"], row["task"], row["example_id"])
            if fraction < 0.10:
                row["split"] = "pair_test"
            elif fraction < 0.15:
                row["split"] = "pair_dev"
            else:
                row["split"] = "train"


def validate_splits(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    by_example: dict[str, set[str]] = defaultdict(set)
    train_identities: set[tuple[str, str, str]] = set()
    cold_identities: set[tuple[str, str, str]] = set()
    for row in rows:
        by_example[str(row["example_id"])].add(str(row["split"]))
        identities = {
            (str(row["task"]), str(row["level"]), str(row["identity_a"])),
            (str(row["task"]), str(row["level"]), str(row["identity_b"])),
        }
        if row["split"] == "train":
            train_identities.update(identities)
        if str(row["split"]).startswith("cold_"):
            # Only the actually held-out identity is guaranteed unseen.  The
            # counterpart in the `any` view may intentionally be familiar.
            for identity in identities:
                fraction = stable_fraction(94137, "identity", *identity)
                if fraction < 0.13:
                    cold_identities.add(identity)
    if any(len(splits) != 1 for splits in by_example.values()):
        raise AssertionError("an unordered pair crossed split boundaries")
    overlap = train_identities & cold_identities
    if overlap:
        raise AssertionError(f"cold identities leaked into train: {len(overlap)}")
    return {
        "unique_examples": len(by_example),
        "train_identities": len(train_identities),
        "cold_identities": len(cold_identities),
        "cold_train_identity_overlap": 0,
    }


def build_dataset(
    lexicon_root: Path,
    output: Path,
    *,
    seed: int = 94137,
    replace: bool = False,
) -> dict[str, Any]:
    lexicon_root = lexicon_root.resolve()
    output = output.resolve()
    if output.exists() and not replace:
        raise FileExistsError(output)
    protocols = _protocol_bundle(lexicon_root)
    files = discover_json_files(lexicon_root)

    presentations: dict[str, list[Presentation]] = defaultdict(list)
    score_candidates: list[tuple[Path, str, SourceSpec, Any, int | None, str]] = []
    source_refs: dict[str, dict[str, Any]] = {}
    quarantine: list[dict[str, Any]] = []

    # Pass 1 reads only explicitly classified score sources.  Persisted
    # payload banks are much larger than the selected vote set, so retaining
    # all of them before knowing the required pair IDs is needlessly costly.
    for path in files:
        relative = str(path.relative_to(lexicon_root))
        spec = classify_score_source(relative)
        if spec is None:
            continue
        file_sha = sha256_file(path)
        found_score = False
        for value, line_number in _iter_objects(path):
            scores = list(_score_rows(value))
            if not scores:
                continue
            found_score = True
            for pair_id, score in scores:
                score_candidates.append((path, relative, spec, (pair_id, score), line_number, file_sha))
        if found_score:
            source_refs[relative] = {
                "path": str(path),
                "sha256": file_sha,
                "bytes": path.stat().st_size,
            }

    # Pass 2 retains only presentations referenced by a classified score.
    needed_pair_ids = {str(value[0]) for _path, _relative, _spec, value, _line, _sha in score_candidates}
    for path in files:
        relative = str(path.relative_to(lexicon_root))
        for value, _line_number in _iter_objects(path):
            for row in _walk_pair_rows(value):
                pair_id = normalize_text(row.get("pair_id"))
                if pair_id not in needed_pair_ids:
                    continue
                presentation = presentation_from_row(row, relative)
                if presentation:
                    presentations[pair_id].append(presentation)

    votes_with_presentations: list[tuple[Vote, Presentation]] = []
    uf = UnionFind()
    for _path, relative, spec, value, line_number, file_sha in score_candidates:
        pair_id, score = value
        task = infer_task(relative)
        level = infer_level(relative)
        presentation = _select_presentation(presentations.get(pair_id, []), task, level)
        if not presentation:
            quarantine.append(
                {
                    "source_path": relative,
                    "source_line": line_number,
                    "pair_id": pair_id,
                    "reason": "missing_or_ambiguous_presentation",
                }
            )
            continue
        task = task or presentation.task
        level = level or presentation.level
        if task not in TASKS or level not in LEVELS:
            quarantine.append(
                {
                    "source_path": relative,
                    "pair_id": pair_id,
                    "reason": "missing_task_or_level",
                    "task": task,
                    "level": level,
                }
            )
            continue
        uf.union_all(presentation.aliases_a)
        uf.union_all(presentation.aliases_b)
        votes_with_presentations.append(
            (
                Vote(
                    pair_id=pair_id,
                    score=score,
                    task=task,
                    level=level,
                    protocol_id=protocol_for(relative, level),
                    teacher_family=spec.teacher_family,
                    role=spec.role,
                    provenance_strength=spec.provenance_strength,
                    label_kind=spec.label_kind,
                    source_path=relative,
                    source_sha256=file_sha,
                    source_line=line_number,
                    base_weight=spec.weight,
                ),
                presentation,
            )
        )

    grouped: dict[tuple[str, str, str, str, str], list[tuple[Vote, Presentation]]] = defaultdict(list)
    for vote, presentation in votes_with_presentations:
        identity_a = uf.find(presentation.aliases_a[0])
        identity_b = uf.find(presentation.aliases_b[0])
        if identity_a == identity_b:
            quarantine.append(
                {"pair_id": vote.pair_id, "source_path": vote.source_path, "reason": "self_pair"}
            )
            continue
        left, right = sorted((identity_a, identity_b))
        grouped[(vote.task, vote.level, vote.protocol_id, left, right)].append((vote, presentation))

    reserved_pair_ids = {
        vote.pair_id
        for entries in grouped.values()
        for vote, _presentation in entries
        if vote.role == "reserved"
    }
    # Reservation crosses protocol variants.  A concept pair used to measure
    # R2-v2.1 must not become training data merely because it also received a
    # legacy-R2 build vote.
    reserved_identity_pairs = {
        (task, level, identity_a, identity_b)
        for (task, level, _protocol_id, identity_a, identity_b), entries in grouped.items()
        if any(vote.role == "reserved" for vote, _presentation in entries)
    }
    rows: list[dict[str, Any]] = []
    for key, entries in sorted(grouped.items()):
        task, level, protocol_id, identity_a, identity_b = key
        all_attributable_votes = [
            vote for vote, _presentation in entries if vote.teacher_family != "unknown"
        ]
        is_reserved = (
            (task, level, identity_a, identity_b) in reserved_identity_pairs
            or any(vote.pair_id in reserved_pair_ids for vote, _presentation in entries)
        )
        reserved_votes = [vote for vote in all_attributable_votes if vote.role == "reserved"]
        if is_reserved and not reserved_votes:
            quarantine.append(
                {
                    "task": task, "level": level, "protocol_id": protocol_id,
                    "reason": "reserved_pair_without_attributable_reserved_teacher",
                    "pair_key": list(key),
                }
            )
            continue
        votes = reserved_votes if is_reserved else [
            vote for vote in all_attributable_votes if vote.role != "reserved"
        ]
        if not votes:
            quarantine.append(
                {"task": task, "level": level, "reason": "no_attributable_teacher", "pair_key": list(key)}
            )
            continue
        target, confidence, family_distributions = balanced_target(votes)
        presentation = entries[0][1]
        # Reorient the displayed text to the canonical identity ordering.
        first_identity = uf.find(presentation.aliases_a[0])
        if first_identity == identity_a:
            text_a, text_b = presentation.text_a, presentation.text_b
        else:
            text_a, text_b = presentation.text_b, presentation.text_a
        example_id = sha256_bytes("||".join(key).encode())[:24]
        rows.append(
            {
                "schema_version": "similarity-distill-v1",
                "example_id": example_id,
                "source_pair_ids": sorted({vote.pair_id for vote in votes}),
                "task": task,
                "level": level,
                "protocol_id": protocol_id,
                "protocol_sha256": protocols[protocol_id]["sha256"],
                "identity_a": identity_a,
                "identity_b": identity_b,
                "text_a": text_a,
                "text_b": text_b,
                "target_probs": target,
                "hard_label": LABELS[max(range(3), key=lambda index: target[index])],
                "example_weight": confidence,
                "family_distributions": family_distributions,
                "has_both_primary_families": all(family in family_distributions for family in PRIMARY_TEACHERS),
                "reserved": is_reserved,
                # Preserve all lineage while making `family_distributions`
                # explicitly reflect only the labels that define the target.
                "sources": [
                    {
                        "pair_id": vote.pair_id,
                        "score": vote.score,
                        "teacher_family": vote.teacher_family,
                        "role": vote.role,
                        "provenance_strength": vote.provenance_strength,
                        "label_kind": vote.label_kind,
                        "source_path": vote.source_path,
                        "source_sha256": vote.source_sha256,
                        "source_line": vote.source_line,
                        "used_in_target": vote in votes,
                    }
                    for vote, _presentation in sorted(
                        entries,
                        key=lambda item: (
                            item[0].teacher_family, item[0].source_path, item[0].source_line or 0
                        ),
                    )
                ],
            }
        )

    _split_rows(rows, seed)
    # validate_splits historically used the declared default seed; keep its
    # check exact for custom seeds by performing the same invariant here.
    train_identities = {
        (row["task"], row["level"], identity)
        for row in rows if row["split"] == "train"
        for identity in (row["identity_a"], row["identity_b"])
    }
    heldout_identities = {
        (row["task"], row["level"], identity)
        for row in rows
        for identity in (row["identity_a"], row["identity_b"])
        if stable_fraction(seed, "identity", row["task"], row["level"], identity) < 0.13
    }
    if train_identities & heldout_identities:
        raise AssertionError("cold identity leakage")

    temp_parent = output.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + ".tmp.", dir=temp_parent))
    try:
        protocol_path = temp / "protocols.json"
        protocol_path.write_text(json.dumps(protocols, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_jsonl(temp / "all.jsonl", rows)
        _write_jsonl(temp / "quarantine.jsonl", quarantine)
        _write_jsonl(temp / "source_files.jsonl", source_refs.values())
        counts: dict[str, Any] = {
            "rows": len(rows),
            "quarantined_records": len(quarantine),
            "by_level": Counter(row["level"] for row in rows),
            "by_task": Counter(row["task"] for row in rows),
            "by_split": Counter(row["split"] for row in rows),
            "by_hard_label": Counter(row["hard_label"] for row in rows),
            "by_protocol": Counter(row["protocol_id"] for row in rows),
        }
        for level in LEVELS:
            for split in (
                "train", "pair_dev", "pair_test", "cold_dev_any", "cold_dev_both",
                "cold_test_any", "cold_test_both", "external_test",
            ):
                selected = [row for row in rows if row["level"] == level and row["split"] == split]
                _write_jsonl(temp / f"{level}_{split}.jsonl", selected)
            _write_jsonl(
                temp / f"{level}_eval.jsonl",
                (
                    row for row in rows
                    if row["level"] == level
                    and row["split"] in {
                        "pair_test", "cold_test_any", "cold_test_both", "external_test"
                    }
                ),
            )
        powered: list[dict[str, Any]] = []
        for task in TASKS:
            for level in LEVELS:
                train_rows = [row for row in rows if row["task"] == task and row["level"] == level and row["split"] == "train"]
                test_rows = [row for row in rows if row["task"] == task and row["level"] == level and row["split"] in {"pair_test", "external_test"}]
                same = sum(row["hard_label"] == "SAME" for row in test_rows)
                powered.append(
                    {
                        "task": task,
                        "level": level,
                        "weighted_train_pairs": sum(float(row["example_weight"]) for row in train_rows),
                        "test_pairs": len(test_rows),
                        "test_same": same,
                        "powered": sum(float(row["example_weight"]) for row in train_rows) >= 1000 and len(test_rows) >= 100 and same >= 20,
                    }
                )
        inventory = {
            "schema_version": "similarity-distill-inventory-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "lexicon_root": str(lexicon_root),
            "counts": {key: dict(value) if isinstance(value, Counter) else value for key, value in counts.items()},
            "powered_cells": powered,
            "source_file_count": len(source_refs),
            "split_invariants": {
                "unordered_pair_cross_split": 0,
                "cold_identity_train_overlap": 0,
                "reserved_rows_in_train": 0,
            },
        }
        (temp / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact_refs = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(temp.iterdir()) if path.is_file()
        }
        manifest = {
            "schema_version": "similarity-distill-freeze-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "lexicon_root": str(lexicon_root),
            "source_files": source_refs,
            "artifacts": artifact_refs,
            "protocols": {key: {k: v for k, v in value.items() if k != "text"} for key, value in protocols.items()},
            "teacher_policy": {
                "primary": ["sonnet", "gpt5"],
                "within_family": "mean independent score distribution",
                "between_primary_families": "equal weight",
                "single_primary_weight": 0.5,
                "auxiliary_weight": 0.25,
                "consensus_replaces_raw_for_target_when_present": True,
            },
            "split_policy": {
                "cold_test_identity_fraction": 0.08,
                "cold_dev_identity_fraction": 0.05,
                "remaining_pair_test_fraction": 0.10,
                "remaining_pair_dev_fraction": 0.05,
            },
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        temp.rename(output)
        return inventory
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon-root", default="outputs/lexicon")
    parser.add_argument("--output", default="outputs/lexicon/similarity_distill_v1")
    parser.add_argument("--seed", type=int, default=94137)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_dataset(Path(args.lexicon_root), Path(args.output), seed=args.seed, replace=args.replace)
    print(json.dumps(inventory, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
