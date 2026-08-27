#!/usr/bin/env python3
"""Fix the longlist first-author parse bug + canonicalize magazine casing.

Bug: for years 2008-2017, the parser merged the section header line ("Others from
the List of 200" / "Longlist: The Wigleaf Top 50 (YYYY)") into the FIRST longlist
entry's author field, e.g. author = "Daniel Others from the List of 200 Alarcón"
(should be "Daniel Alarcón"). 10 rows affected. We strip the header phrase.

Also: canonicalize magazine names (SmokeLong/Smokelong, elimae/Elimae, etc.) so
the parse pipeline doesn't leak via casing, and strip a few stray translator-note
artifacts. Writes wigleaf_labels_fixed.csv (NEVER overwrites the original labels).
"""
import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wig_textproc import normalize_magazine

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/wigleaf"

HEADER_PAT = re.compile(
    r"\s*(Others from the List of 200|Longlist:\s*The Wigleaf Top 50(\s*\(\d{4}\))?)\s*",
    re.I)


def fix_author(a):
    if not isinstance(a, str):
        return a
    a2 = HEADER_PAT.sub(" ", a)
    a2 = re.sub(r"\s+", " ", a2).strip()
    return a2


def main():
    df = pd.read_csv(os.path.join(BASE, "wigleaf_labels.csv"))
    n_before = len(df)
    affected = df["author"].astype(str).str.contains(
        "Others from the List|Longlist: The Wigleaf", case=False, na=False)
    print(f"author-header rows to fix: {int(affected.sum())}")
    for i in df.index[affected]:
        old = df.at[i, "author"]
        df.at[i, "author"] = fix_author(old)
        print(f"  {df.at[i,'year']} {df.at[i,'tier']}: {old!r} -> {df.at[i,'author']!r}")
    # canonicalize magazine casing across ALL rows
    df["magazine"] = df["magazine"].apply(normalize_magazine)
    assert len(df) == n_before
    out = os.path.join(BASE, "wigleaf_labels_fixed.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} rows; original untouched)")
    # report how many distinct magazine spellings collapsed
    raw = pd.read_csv(os.path.join(BASE, "wigleaf_labels.csv"))
    print(f"distinct magazines: {raw.magazine.nunique()} -> {df.magazine.nunique()}")


if __name__ == "__main__":
    main()
