#!/usr/bin/env python3
"""Score the notice-and-comment agentic-hybrid code-metrics over extracted fields: round-1
(programs_notice_and_comment/<aspect>_h0.py) + round-2 (<aspect>_h1.py) + round-3
VERIFICATION-tier (<aspect>_h2.py) aspects, discovered by globbing all three suffixes (see
discover_aspects()) so new *_h2.py (or future *_h3.py) programs are picked up with no edits
here.

Loads each metric_seam module, the Gemma-extracted fields (nc_fields.jsonl from
extract_nc_fields_gemma.py), builds an Ops TF-IDF retrieval corpus over the sample comment
texts (mirrors datasets/peer-review/score_code_metrics.py), and computes
score(text, extracted, ops) for every (comment, aspect) pair. Code-only aspects (empty
LLM_FIELDS, e.g. structure_org, deference_tone) get extracted={} since no fields were ever
requested for them.

NO label usage anywhere (y_out/y_eng are never read) — this is a pure code-metric scoring
pass, not an evaluation against labels.

Usage: python score_nc_code_metrics.py --sample nc_vat_sample.jsonl --fields nc_fields.jsonl
"""
import argparse, importlib.util, json, pathlib, sys
import numpy as np

BASE = pathlib.Path(__file__).resolve().parents[3]  # repo root
PROG = BASE / "methods/metric_seam/hybrids/programs_notice_and_comment"
sys.path.insert(0, str(BASE / "methods/metric_seam/hybrids"))
from ops import Ops  # noqa: E402


def discover_aspects():
    """Glob programs_notice_and_comment/*_h0.py, *_h1.py, *_h2.py -> [(aspect_id, suffix), ...],
    h0 (round 1) first, then h1 (round 2), then h2 (round 3, VERIFICATION-tier), each
    alphabetical by aspect id. Cleanly picks up any future _h3.py etc. without further edits
    here."""
    out = []
    for suffix in ("h0", "h1", "h2"):
        for f in sorted(PROG.glob(f"*_{suffix}.py")):
            out.append((f.stem[: -len(f"_{suffix}")], suffix))
    return out


ASPECTS_FULL = discover_aspects()          # [(aid, suffix), ...]
ASPECTS = [aid for aid, _ in ASPECTS_FULL]  # plain aspect ids, in discovery order (unique)


def load_module(aid, suffix="h0"):
    spec = importlib.util.spec_from_file_location(f"nc_{aid}_{suffix}", PROG / f"{aid}_{suffix}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_sample(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_id = d.get("doc_id") or d.get("id")
            txt = (d.get("text") or "").strip()
            if not doc_id or len(txt) < 20:
                continue
            rows.append({"doc_id": doc_id, "text": txt})
    return rows


def main():
    ap = argparse.ArgumentParser()
    default_sample = pathlib.Path(__file__).resolve().parent / "nc_vat_sample.jsonl"
    default_fields = pathlib.Path(__file__).resolve().parent / "nc_fields.jsonl"
    default_out = pathlib.Path(__file__).resolve().parent / "nc_code_scores.npz"
    ap.add_argument("--sample", default=str(default_sample))
    ap.add_argument("--fields", default=str(default_fields))
    ap.add_argument("--out", default=str(default_out))
    a = ap.parse_args()

    mods = {aid: load_module(aid, suffix) for aid, suffix in ASPECTS_FULL}
    rows = load_sample(a.sample)
    text_by_id = {r["doc_id"]: r["text"] for r in rows}

    fields_by_id = {}
    if pathlib.Path(a.fields).exists():
        for line in open(a.fields):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fields_by_id[r["id"]] = r.get("fields") or {}
    else:
        print(f"[score] WARNING: fields file {a.fields} not found; "
              f"all aspects will be scored with extracted={{}}", flush=True)

    # Ops needs a corpus json: [{datapoint_id, text}]; build it for TF-IDF retrieve_similar.
    corpus_path = str(pathlib.Path(a.out).with_suffix(".corpus.json"))
    json.dump([{"datapoint_id": r["doc_id"], "text": r["text"]} for r in rows], open(corpus_path, "w"))
    ops = Ops(corpus_path=corpus_path)

    ids = [r["doc_id"] for r in rows]
    X = np.full((len(rows), len(ASPECTS)), np.nan)
    for j, aid in enumerate(ASPECTS):
        fn = mods[aid].score
        is_code_only = not mods[aid].LLM_FIELDS
        for i, r in enumerate(rows):
            doc_id, text = r["doc_id"], r["text"]
            extracted = {} if is_code_only else (fields_by_id.get(doc_id, {}).get(aid, {}) or {})
            try:
                X[i, j] = float(fn(text, extracted, ops))
            except Exception:
                X[i, j] = np.nan

    print(f"[score] {len(rows)} comments x {len(ASPECTS)} code-metrics; "
          f"na={np.isnan(X).mean():.3f}", flush=True)
    print(f"{'aspect':22s} {'mean':>6s} {'std':>6s} {'n':>6s}")
    for j, aid in enumerate(ASPECTS):
        col = X[:, j]
        mask = ~np.isnan(col)
        mean = col[mask].mean() if mask.sum() else float("nan")
        std = col[mask].std() if mask.sum() else float("nan")
        print(f"{aid:22s} {mean:6.3f} {std:6.3f} {int(mask.sum()):6d}")

    np.savez_compressed(a.out, X=X, doc_id=np.array(ids, dtype=object),
                        names=np.array(ASPECTS, dtype=object))
    print(f"[score] saved -> {a.out}", flush=True)
    print("CODE_DONE", flush=True)


if __name__ == "__main__":
    main()
