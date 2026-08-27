#!/usr/bin/env python3
"""Relocate one strict task manifest to hash-identical runtime bank/norm mirrors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--norms", required=True)
    parser.add_argument("--expected-bank-sha256")
    parser.add_argument("--expected-norms-sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_path = Path(args.source_manifest).resolve()
    bank_path = Path(args.bank).resolve()
    norms_path = Path(args.norms).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    bank_meta = (source.get("banks") or {}).get(args.task)
    corpus_meta = (source.get("corpora") or {}).get(args.corpus)
    if not isinstance(bank_meta, dict) or not isinstance(corpus_meta, dict):
        raise ValueError("source manifest lacks requested task/corpus")
    expected_bank = bank_meta.get("sha256") or args.expected_bank_sha256
    expected_norms = corpus_meta.get("sha256") or args.expected_norms_sha256
    if not expected_bank or not expected_norms:
        raise ValueError(
            "manifest omits runtime artifact hashes; pass both explicit expected hashes"
        )
    if expected_bank != sha256_file(bank_path):
        raise ValueError("runtime bank bytes differ from source manifest")
    if expected_norms != sha256_file(norms_path):
        raise ValueError("runtime norm bytes differ from source manifest")
    relocated = json.loads(json.dumps(source))
    relocated["banks"][args.task]["path"] = str(bank_path)
    relocated["corpora"][args.corpus]["path"] = str(norms_path)
    if isinstance(relocated.get("merged_norms"), dict):
        relocated["merged_norms"]["path"] = str(norms_path)
    relocated["runtime_relocation"] = {
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "bank_bytes_changed": False,
        "norm_bytes_changed": False,
        "path_fields_only": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(relocated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "bank_sha256": sha256_file(bank_path),
                "norms_sha256": sha256_file(norms_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
