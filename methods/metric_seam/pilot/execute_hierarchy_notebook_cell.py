"""Execute and persist an allowlisted CPU-only summary cell in the seam notebook."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks/2026-07-02__metric-seam-certificates-and-overnight-report.ipynb"
DEFAULT_CELL_ID = "seam-20260713-hierarchy-funnel"
ALLOWED_CELL_IDS = frozenset({
    DEFAULT_CELL_ID,
    "seam-20260713-science-relations",
    "seam-20260713-technical",
})

PRELUDE = """\
import sys
import json
from pathlib import Path
import pandas as pd
from IPython.display import display
ROOT = Path({root!r})
sys.path.insert(0, str(ROOT))
from methods.metric_seam.pilot import metric_seam_notebook_stats as seam_stats
"""

EXTRA_PRELUDE_BY_CELL_ID = {
    "seam-20260713-technical": """\
a12_projection = seam_stats.math_a12_pair_projection_depth()
code_depth = seam_stats.active_code_depth_retrospective()
patent_family = seam_stats.patent_ws3_family_retrospective()
""",
}


def execute(*, notebook_path: Path = NOTEBOOK, cell_id: str = DEFAULT_CELL_ID) -> None:
    if cell_id not in ALLOWED_CELL_IDS:
        allowed = ", ".join(sorted(ALLOWED_CELL_IDS))
        raise ValueError(f"cell {cell_id!r} is not allowlisted; expected one of: {allowed}")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    targets = [cell for cell in payload["cells"] if cell.get("id") == cell_id]
    if len(targets) != 1 or targets[0].get("cell_type") != "code":
        raise ValueError(f"expected one code cell {cell_id!r}")

    target_source = targets[0].get("source", "")
    if isinstance(target_source, list):
        target_source = "".join(target_source)
    if not isinstance(target_source, str) or not target_source.strip():
        raise ValueError(f"notebook cell {cell_id!r} source is empty")
    prelude = PRELUDE.format(root=str(ROOT)) + EXTRA_PRELUDE_BY_CELL_ID.get(
        cell_id, ""
    )
    temporary = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell(prelude),
        nbformat.v4.new_code_cell(target_source),
    ])
    client = NotebookClient(
        temporary,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    executed = client.execute()
    result = executed.cells[1]
    if not result.get("outputs"):
        raise RuntimeError(f"notebook cell {cell_id!r} produced no display/stream output")
    errors = [output for output in result.outputs if output.get("output_type") == "error"]
    if errors:
        raise RuntimeError(f"notebook cell {cell_id!r} failed: {errors[0].get('evalue')}")

    maximum = max(
        (
            int(cell["execution_count"])
            for cell in payload["cells"]
            if isinstance(cell.get("execution_count"), int)
        ),
        default=0,
    )
    targets[0]["execution_count"] = maximum + 1
    targets[0]["outputs"] = json.loads(json.dumps(result.outputs))
    notebook_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--cell-id", choices=sorted(ALLOWED_CELL_IDS), default=DEFAULT_CELL_ID)
    args = parser.parse_args()
    # Defense in depth: the parent command also sanitizes these variables.
    secret_fragments = (
        "API_KEY",
        "AUTH_TOKEN",
        "ACCESS_TOKEN",
        "CREDENTIAL",
        "KEY_FILE",
    )
    secret_names = {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "OPENAI_ORG_ID"}
    for key in list(os.environ):
        upper = key.upper()
        if upper in secret_names or any(value in upper for value in secret_fragments):
            os.environ.pop(key, None)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "HIP_VISIBLE_DEVICES": "-1",
        "ROCR_VISIBLE_DEVICES": "-1",
    })
    execute(notebook_path=args.notebook, cell_id=args.cell_id)


if __name__ == "__main__":
    main()
