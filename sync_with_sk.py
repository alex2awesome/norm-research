#!/usr/bin/env python3
"""Sync local norm-research directory with sk hosts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    candidates = [start] + list(start.parents)
    # First pass: look for the sync_with_sk package (dir with __init__.py).
    for candidate in candidates:
        pkg = candidate / "sync_with_sk"
        if pkg.is_dir() and (pkg / "__init__.py").exists():
            return candidate
    # Fallback: nearest .git root.
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Unable to locate repository root (missing .git directory).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync local norm-research tree to sk hosts via ssh/rsync/scp."
    )
    parser.add_argument("--upload", action="store_true", help="Sync local files to remote hosts.")
    parser.add_argument("--download", action="store_true", help="Sync remote files to local.")
    parser.add_argument(
        "--pull-remote-only",
        action="store_true",
        help="Deprecated alias for --download (now includes changed files).",
    )
    parser.add_argument(
        "--hosts",
        nargs="*",
        default=None,
        help="Hosts to sync. If omitted (or passed with no values), sync all default hosts.",
    )
    parser.add_argument(
        "--local-root",
        default=None,
        help="Local norm-research root. Defaults to directory containing this script.",
    )
    parser.add_argument(
        "--remote-root",
        default=None,
        help="Optional remote root override (applies to all hosts).",
    )
    parser.add_argument("--code-only", action="store_true", help="Sync code-only files.")
    parser.add_argument("--data-only", action="store_true", help="Sync data-only files.")
    parser.add_argument("--csv-only", action="store_true", help="Sync CSV-only files.")
    parser.add_argument(
        "--all-text-only",
        action="store_true",
        help="Only consider files ending with all_text.csv (data-only filter).",
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        help="Case-insensitive substring filter on relative paths. Repeatable.",
    )
    parser.add_argument(
        "--speed-download",
        action="store_true",
        help="Fast data mode: tar-stream grouped folders when enabled in config.",
    )
    parser.add_argument(
        "--delete-remote-only",
        action="store_true",
        help="Delete remote-only files after confirmation.",
    )
    parser.add_argument("--yes", action="store_true", help="Auto-confirm destructive actions.")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without executing.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full SSH scan of remote (slow). Default: use state file as remote proxy.",
    )
    parser.add_argument(
        "--max-hash-bytes",
        type=int,
        default=None,
        help="Hash files <= this size unless treated as data files.",
    )
    parser.add_argument(
        "--max-download-bytes",
        type=int,
        default=None,
        help="Skip downloading files larger than this size. Set to 0 to disable.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to state cache JSON file. Defaults to <local-root>/.sync_with_sk_state.json",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries for transfer operations."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pull_remote_only:
        args.download = True

    repo_root = find_repo_root(Path(__file__).resolve())
    sys.path.insert(0, str(repo_root))

    from sync_with_sk.core import ProjectConfig, run_sync  # noqa: WPS433

    local_root = Path(args.local_root).resolve() if args.local_root else Path(__file__).resolve().parent
    config = ProjectConfig(
        name="norm-research",
        local_root=local_root,
        scope_root=local_root,
        default_hosts=["sk1", "sk2", "sk3"],
        host_roots={
            "sk1": "/lfs/skampere1/0/alexspan/norm-research",
            "skampere1": "/lfs/skampere1/0/alexspan/norm-research",
            "sk2": "/lfs/skampere2/0/alexspan/norm-research",
            "sk3": "/lfs/skampere3/0/alexspan/norm-research",
        },
        data_dirs={"bulk_data", "data", "datasets", "outputs", "checkpoints"},
        data_suffixes={
            ".csv",
            ".bz2",
            ".parquet",
            ".jsonl",
            ".npy",
            ".npz",
            ".pt",
            ".bin",
            ".zip",
            ".tar",
            ".gz",
            ".xz",
        },
        data_allowlist={".csv", ".csv.gz", ".py"},
        code_suffixes=set(),
        include_unknown=True,
        ignore_paths=[local_root / ".sync-ignore"],
        max_hash_bytes=64 * 1024 * 1024,
        max_download_bytes=100 * 1024 * 1024,
        state_file_path=local_root / ".sync_with_sk_state.json",
        speed_grouping=None,
    )

    return run_sync(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
