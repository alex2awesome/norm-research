"""JSONL-backed storage for generated programs and library snapshots."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GeneratedProgram:
    """A single LLM-generated prediction program."""

    program_id: str
    example_id: str
    round_idx: int
    source_code: str
    hypothesis: str
    generated_at: str = ""
    # V2: extraction-aware programs
    input_schema: Optional[List[dict]] = None  # list of field definitions
    tools_needed: Optional[List[str]] = None  # external tools that would help
    version: str = "v1"  # "v1", "v2", "approach1", "approach2"
    # Populated after execution
    self_correct: Optional[bool] = None
    test_accuracy: Optional[float] = None
    test_auc: Optional[float] = None
    test_correlation: Optional[float] = None
    execution_success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()


class ProgramStore:
    """Append-only JSONL store for generated programs."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.programs_path = self.store_dir / "programs.jsonl"
        self._programs: Dict[str, GeneratedProgram] = {}

    def load(self) -> None:
        if self.programs_path.exists():
            with open(self.programs_path) as f:
                for line in f:
                    obj = json.loads(line)
                    prog = GeneratedProgram(**obj)
                    self._programs[prog.program_id] = prog
            logger.info("Loaded %d programs from %s", len(self._programs), self.programs_path)

    def add(self, program: GeneratedProgram) -> None:
        self._programs[program.program_id] = program
        self.store_dir.mkdir(parents=True, exist_ok=True)
        with open(self.programs_path, "a") as f:
            f.write(json.dumps(asdict(program)) + "\n")

    def update_scores(self, program_id: str, **kwargs) -> None:
        if program_id in self._programs:
            for k, v in kwargs.items():
                setattr(self._programs[program_id], k, v)
            # Rewrite full file to persist updates
            self._rewrite()

    def _rewrite(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        with open(self.programs_path, "w") as f:
            for prog in self._programs.values():
                f.write(json.dumps(asdict(prog)) + "\n")

    def get_programs_for_round(self, round_idx: int) -> List[GeneratedProgram]:
        return [p for p in self._programs.values() if p.round_idx == round_idx]

    def get_all_programs(self) -> List[GeneratedProgram]:
        return list(self._programs.values())

    def get_successful_programs(self) -> List[GeneratedProgram]:
        return [p for p in self._programs.values() if p.execution_success]

    def get_programs_since_round(self, since_round: int) -> List[GeneratedProgram]:
        return [p for p in self._programs.values() if p.round_idx >= since_round]

    @property
    def count(self) -> int:
        return len(self._programs)

    @property
    def max_round(self) -> int:
        if not self._programs:
            return -1
        return max(p.round_idx for p in self._programs.values())


class LibraryStore:
    """Tracks library snapshots across refactoring rounds."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.snapshots_dir = self.store_dir / "library_snapshots"
        self.metadata_path = self.store_dir / "library_metadata.jsonl"

    def save_snapshot(self, round_idx: int, library_code: str, metrics: dict) -> None:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        # Save library source
        snapshot_path = self.snapshots_dir / f"library_round_{round_idx:03d}.py"
        snapshot_path.write_text(library_code)
        # Append metrics
        record = {"round_idx": round_idx, "timestamp": datetime.now().isoformat(), **metrics}
        with open(self.metadata_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Saved library snapshot round %d (%d lines)", round_idx, library_code.count("\n"))

    def load_latest(self) -> Optional[str]:
        if not self.snapshots_dir.exists():
            return None
        snapshots = sorted(self.snapshots_dir.glob("library_round_*.py"))
        if not snapshots:
            return None
        return snapshots[-1].read_text()

    def get_snapshot(self, round_idx: int) -> Optional[str]:
        path = self.snapshots_dir / f"library_round_{round_idx:03d}.py"
        if path.exists():
            return path.read_text()
        return None

    def get_all_metrics(self) -> List[dict]:
        if not self.metadata_path.exists():
            return []
        metrics = []
        with open(self.metadata_path) as f:
            for line in f:
                metrics.append(json.loads(line))
        return metrics

    @property
    def latest_round(self) -> int:
        metrics = self.get_all_metrics()
        if not metrics:
            return -1
        return max(m["round_idx"] for m in metrics)
