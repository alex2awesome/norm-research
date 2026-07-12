"""Export optimized prompt artifacts from the registry to a rubrics dir loadable by
``metrics_tree_infilling``.

Bridges Phase 1 (``run_distillation`` -> registry) to Phase 3 (the tree's ``--rubrics-dir``):
``registry.head(metric_id, "prompt")`` -> version body (the optimized judge rubric) ->
``{"extracted": {"rubrics_metrics": [{"name","description","guidance"}, ...]}}``, which
``metrics_tree_infilling.io_metrics.load_rubric_metrics_from_dir`` reads directly.

Phase 4 (later) adds ``export_code`` -> ``score(text)`` ``.py`` modules for ``load_code_metrics``.

Example
-------
    PYTHONPATH=methods python -m metric_implementer.export \\
        --task peer-review --out outputs/metrics_tree_infilling/distilled/peer-review
    # then:
    PYTHONPATH=methods python -m metrics_tree_infilling.run \\
        --task peer-review --metrics rubric --rubrics-dir <out> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

from .config import ImplementerConfig
from .registry import Registry

GEMMA4 = "google/gemma-4-31b-it"


def _resolve_head(registry: Registry, metric_id: str, kind: str,
                  judge_model: Optional[str]) -> Optional[str]:
    """Find the current prompt version: tiered HEAD (prompt@judge_model) first, else the legacy
    un-tiered HEAD, else the latest version of that kind (so export still works if HEAD pointers
    were not written). Prompt HEADs are per judge-tier, so prefer the tiered one."""
    if judge_model:
        vid = registry.head(metric_id, kind, judge_tier=judge_model)
        if vid:
            return vid
    vid = registry.head(metric_id, kind)
    if vid:
        return vid
    vs = registry.versions(metric_id, kind)
    return vs[-1]["version_id"] if vs else None


def export_prompts(
    registry: Registry, metric_ids: List[str], out_dir: Path, *,
    kind: str = "prompt", judge_model: Optional[str] = GEMMA4,
    mode: str = "head",
) -> Tuple[Path, List[dict], List[Tuple[str, str]]]:
    """Write one ``distilled_rubrics.json`` under ``out_dir`` with every metric's prompt.

    ``mode="head"`` (default) exports the current accepted HEAD (GEPA-improved) prompt;
    ``mode="seed"`` exports the INIT/seed prompt (v000) — for a clean seed-vs-HEAD power
    comparison on identical metric_ids. Returns ``(path, exported_entries, skipped)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: List[dict] = []
    skipped: List[Tuple[str, str]] = []
    for mid in metric_ids:
        if mode == "seed":
            vs = registry.versions(mid, kind)
            vid = vs[0]["version_id"] if vs else None
        else:
            vid = _resolve_head(registry, mid, kind, judge_model)
        if not vid:
            skipped.append((mid, "no prompt version"))
            continue
        rec = registry.get_version(mid, vid, kind)
        body = (rec.get("body") or "").strip()
        name = (rec.get("name") or mid).strip()
        desc = (rec.get("description") or name).strip()
        if not body:
            skipped.append((mid, f"empty body ({vid})"))
            continue
        exported.append({
            "name": name, "description": desc, "guidance": body,
            "metric_id": mid, "version_id": vid,
        })
    doc = {"extracted": {"rubrics_metrics": exported}}
    fp = out_dir / "distilled_rubrics.json"
    fp.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return fp, exported, skipped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="dataset/task name (scopes the registry)")
    p.add_argument("--out", required=True, help="output rubrics dir (for --rubrics-dir)")
    p.add_argument("--registry", default=None, help="override registry dir (defaults to cfg)")
    p.add_argument("--kind", default="prompt")
    p.add_argument("--judge-model", default=GEMMA4, help="judge tier for HEAD fallback")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ImplementerConfig()
    cfg.task = args.task
    root = Path(args.registry) if args.registry else cfg.registry_dir()
    registry = Registry(root)

    metric_ids = sorted(p.name for p in (registry.root / "metrics").iterdir()
                        if p.is_dir()) if (registry.root / "metrics").exists() else []
    if not metric_ids:
        print(f"[export] no metrics under {registry.root / 'metrics'}")
        return 1

    fp, exported, skipped = export_prompts(
        registry, metric_ids, Path(args.out), kind=args.kind, judge_model=args.judge_model)
    print(f"[export] task={args.task} metrics={len(metric_ids)} exported={len(exported)} "
          f"skipped={len(skipped)} -> {fp}")
    for mid, reason in skipped:
        print(f"   skip {mid}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
