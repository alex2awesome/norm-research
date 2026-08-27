#!/usr/bin/env python3
"""Independent re-verification of the built V3-grid dataset dirs.

Shares NO state with build_v3_cell.py: it re-reads every emitted CSV from disk
and re-checks it against the ORIGINAL dense-standard split CSV that the cell's
own manifest names.  Checks, per split:

  1. the original split file's sha256 still matches what the build recorded
     (i.e. the upstream cell has not moved under us);
  2. n matches the manifest's n / n_orig;
  3. the emitted rows are an IN-ORDER SUBSEQUENCE of the original split, matched
     on (group, judgement) -- so row order and row identity are preserved and
     nothing was reordered or invented;
  4. each emitted text carries the ORIGINAL text byte-for-byte: it ENDS WITH
     "\\n" + original (PREPEND cells) or STARTS WITH
     "full text:\\n    " + original + "\\nVA metrics:" (APPEND cells);
  5. data.csv equals the concatenation of split/{train,eval,test}.csv.

CPU only.  Usage:
  python3 verify_v3_cell.py            # all v3grid_* dirs
  python3 verify_v3_cell.py nc_agree   # one or more slugs
Exit code 1 if any cell fails.
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DD = HERE.parent / "dense_data"
SPLITS = ("train", "eval", "test")


def verify(manifest_path: Path) -> bool:
    m = json.loads(manifest_path.read_text())
    d = manifest_path.parent
    orig = Path(m["dense_standard_split_dir"])
    prepend = m["block_placement"] == "PREPEND"
    parts, allok = [f"{m['arm']:36s}"], True
    for s in SPLITS:
        h = hashlib.sha256((orig / "split" / f"{s}.csv").read_bytes()).hexdigest()
        new = pd.read_csv(d / "split" / f"{s}.csv")
        old = pd.read_csv(orig / "split" / f"{s}.csv")
        ok = (h == m["orig_split_sha256"][s]
              and len(new) == m["n"][s] and len(old) == m["n_orig"][s])
        j = 0
        for a, g, y in zip(new.text.astype(str), new.group.astype(str), new.judgement):
            while j < len(old) and (str(old.group.iloc[j]) != g or old.judgement.iloc[j] != y):
                j += 1
            if j >= len(old):
                ok = False
                break
            t = str(old.text.iloc[j])
            carried = (a.endswith("\n" + t) if prepend
                       else a.startswith("full text:\n    " + t + "\nVA metrics:"))
            if not carried:
                ok = False
                break
            j += 1
        allok &= ok
        parts.append(f"{s}:{'OK' if ok else 'FAIL'}")
    data = pd.read_csv(d / "data.csv")
    cat = pd.concat([pd.read_csv(d / "split" / f"{s}.csv") for s in SPLITS],
                    ignore_index=True)
    dok = data.equals(cat)
    allok &= dok
    print(" ".join(parts), f"data.csv==concat:{dok}", "=>",
          "PASS" if allok else "FAIL", flush=True)
    return allok


def main():
    slugs = sys.argv[1:]
    paths = ([DD / f"v3grid_{s}" / "manifest.json" for s in slugs] if slugs
             else [Path(p) for p in sorted(glob.glob(str(DD / "v3grid_*" / "manifest.json")))])
    ok = all(verify(p) for p in paths)
    print("ALL_PASS" if ok else "SOME_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
