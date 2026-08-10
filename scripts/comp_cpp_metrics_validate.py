"""Validation pass for the new a600-a614 C++ metrics.

Loads 20 diverse C++ candidates from CF + AC + CC pools, prints a brief
snippet from each, and the score of every new metric. Used for manual
verification BEFORE scoring at scale.

Diversity strategy: sample evenly from short / medium / long candidates;
include CF and AC and CC pool C++ rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(REPO))

from methods.existing_metrics_runner.coded.metrics import load_all  # noqa: E402

NEW_IDS = [f"a{i}" for i in range(600, 615)]

LANG_TO_EXT = {"cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cpp14": "cpp"}


def to_synthetic_diff(file_path: str, file_text: str) -> str:
    if not file_text:
        return ""
    lines = file_text.split("\n")
    body = "\n".join("+" + ln for ln in lines)
    n = len(lines)
    return (
        f"diff --git a/{file_path} b/{file_path}\n"
        f"--- /dev/null\n"
        f"+++ b/{file_path}\n"
        f"@@ -0,0 +1,{n} @@\n"
        f"{body}\n"
    )


def main():
    samples = []

    # CF cpp
    cf = pd.read_parquet(REPO / "outputs/v2_analysis/comp_fourplatform_cells/cf_bank_input.parquet")
    cf_cpp = cf[cf["candidate_lang"].isin(["cpp", "c++", "cxx", "cpp14"])].copy()
    cf_cpp["text_len"] = cf_cpp["candidate_text"].str.len()
    print(f"CF cpp n={len(cf_cpp)}, len range "
          f"{cf_cpp['text_len'].min()}..{cf_cpp['text_len'].max()}")
    # sample short/med/long
    cf_cpp = cf_cpp.sort_values("text_len").reset_index(drop=True)
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    for q in quantiles:
        i = int(q * (len(cf_cpp) - 1))
        r = cf_cpp.iloc[i]
        samples.append(("CF", r["pair_id"], "cpp", r["candidate_text"]))

    # AC cpp
    ac = pd.read_parquet(REPO / "outputs/v2_analysis/comp_fourplatform_cells/ac_bank_input.parquet")
    ac_cpp = ac[ac["candidate_lang"].isin(["cpp", "c++", "cxx", "cpp14"])].copy()
    ac_cpp["text_len"] = ac_cpp["candidate_text"].str.len()
    print(f"AC cpp n={len(ac_cpp)}, len range "
          f"{ac_cpp['text_len'].min()}..{ac_cpp['text_len'].max()}")
    ac_cpp = ac_cpp.sort_values("text_len").reset_index(drop=True)
    for q in quantiles:
        i = int(q * (len(ac_cpp) - 1))
        r = ac_cpp.iloc[i]
        samples.append(("AC", r["pair_id"], "cpp", r["candidate_text"]))

    # CC cpp from unified pool
    cc_pool = pd.read_parquet(REPO / "outputs/v2_analysis/comp_fourplatform_cells/cc_pool.parquet")
    cc_cpp = cc_pool[cc_pool["candidate_lang"].isin(
        ["cpp", "c++", "cxx", "cpp14"])].copy()
    cc_cpp["text_len"] = cc_cpp["candidate_text"].str.len()
    print(f"CC cpp pool n={len(cc_cpp)}, len range "
          f"{cc_cpp['text_len'].min()}..{cc_cpp['text_len'].max()}")
    cc_cpp = cc_cpp.sort_values("text_len").reset_index(drop=True)
    qs2 = [0.2, 0.5, 0.7, 0.9]
    for q in qs2:
        i = int(q * (len(cc_cpp) - 1))
        r = cc_cpp.iloc[i]
        samples.append(("CC", r["pair_id"], "cpp", r["candidate_text"]))

    # Negative cases: Python and Java to verify applied=False
    cf_py = cf[cf["candidate_lang"] == "python"].head(2)
    for _, r in cf_py.iterrows():
        samples.append(("CF-py", r["pair_id"], "python", r["candidate_text"]))
    cf_java = cf[cf["candidate_lang"] == "java"].head(2)
    for _, r in cf_java.iterrows():
        samples.append(("CF-java", r["pair_id"], "java", r["candidate_text"]))

    print(f"\nTotal samples: {len(samples)}")

    # Load metrics
    all_m = load_all()
    new_m = {m.ASPECT_ID: m for m in all_m if m.ASPECT_ID in NEW_IDS}
    print(f"Loaded {len(new_m)}/{len(NEW_IDS)} new metrics: "
          f"{sorted(new_m)}")

    for i, (src, pid, lang, text) in enumerate(samples):
        ext = LANG_TO_EXT.get(lang.lower(), lang)
        fp = f"candidate.{ext}"
        diff = to_synthetic_diff(fp, text)
        print("\n" + "=" * 80)
        print(f"#{i:02d} src={src} pair_id={pid} lang={lang} "
              f"len={len(text)}")
        snippet = "\n".join(text.split("\n")[:18])
        if len(text.split("\n")) > 18:
            snippet += "\n  ... (truncated)"
        print(snippet)
        print("-")
        for aid in NEW_IDS:
            m = new_m.get(aid)
            if not m:
                print(f"  {aid}: <MISSING>")
                continue
            try:
                applied = int(bool(m.applies(diff)))
            except Exception as e:
                print(f"  {aid}: applies ERR {type(e).__name__}: {e}")
                continue
            score = None
            if applied:
                try:
                    score = m.score(diff)
                except Exception as e:
                    print(f"  {aid}: score ERR {type(e).__name__}: {e}")
                    continue
            score_s = f"{score:.3f}" if isinstance(score, float) else str(score)
            print(f"  {aid:5s} applied={applied} score={score_s}  "
                  f"({m.ASPECT_NAME})")


if __name__ == "__main__":
    main()
