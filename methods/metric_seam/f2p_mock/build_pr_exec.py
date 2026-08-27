"""Build the pr_exec seam task: PRs that ALREADY have transplant F2P/P2F measurements.

Items = rows of transplant_consolidated_2026_07_02_canonical.parquet joined to their stored
single-file diffs (datasets/code-review/diffs/<owner>__<repo>__<pr>.diff). NOTHING is built or
executed — the runner outcomes become the mocked evidence op's payload (exec_features.json).

Emits ONE prompts.jsonl: 6 judge aspects x 2 passes + scope + LLM fields for the 3 exec-op
hybrids (channel="field"). Acceptance labels (judgement, days_open) go into items.json meta
ONLY (anchors — never into exec_features).

Usage (sk3): python3.11 build_pr_exec.py
"""
import importlib.util, json, pathlib, re

R = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
PTE = R / "datasets/code-review/pr_test_execution"
DIFFS = R / "datasets/code-review/diffs"
OUT = R / "outputs/metric_seam_pilot/tasks/pr_exec"
PROGS = pathlib.Path(__file__).parent / "programs"
HEAD, TAIL = 5000, 2500

ASPECTS = ["a67", "a104", "a128", "a9", "a87", "a131"]
ROLE = ("an experienced senior software engineer reviewing pull requests",
        "pull request (title and code diff)")

T1 = """You are {role} grading ONE {doctype} against ONE criterion.

Criterion: {name}
Definition: {description}

Document (may be truncated; '[...]' marks an elided middle):
<document>
{text}
</document>

Grade how well the document satisfies this criterion on a 0-10 integer scale \
(0 = completely fails, 10 = exemplary). If the criterion genuinely does not apply to this \
kind of document, answer NA. Reply with exactly one line in the form:
SCORE: <integer 0-10 or NA>"""

T2 = """As {role}, evaluate the document below on a single quality criterion, ignoring all \
other qualities.

The criterion — {name}: {description}

Document ('[...]' = elided middle):
<document>
{text}
</document>

Give an integer 0-10 (0 = criterion badly violated or absent, 10 = criterion fully \
exemplified). If this criterion simply cannot apply to a document of this kind, answer NA. \
Your entire reply must be one line:
SCORE: <integer 0-10 or NA>"""

TSCOPE = """Look at the following text scraped from the web.

<text>
{text}
</text>

Is this a genuine, substantive {doctype} (as opposed to navigation chrome, an empty stub, \
an unrelated page, or a different document type)? Reply with exactly one line:
SCORE: <integer 0-10>  (0 = clearly NOT, 10 = clearly a substantive {doctype})"""

TFIELD = """From the document below, {instruction}

<document>
{text}
</document>

Reply with ONE short line (max 20 words): the answer only, or NONE if absent."""


def canonical(text):
    if len(text) <= HEAD + TAIL + 500:
        return text
    return text[:HEAD] + "\n[...]\n" + text[-TAIL:]


def main():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PTE / "outputs/transplant_consolidated_2026_07_02_canonical.parquet")

    # index the diffs dir once: (repo_lower, pr) -> path
    idx = {}
    for p in DIFFS.iterdir():
        parts = p.name[:-5].split("__") if p.name.endswith(".diff") else None
        if parts and len(parts) == 3:
            idx[(parts[1].lower(), parts[2])] = p

    # batch name -> diffs-corpus repo name where they differ (tektoncd/pipeline)
    ALIAS = {"tekton-pipeline": "pipeline"}
    items, feats, miss = [], {}, 0
    for row in df.itertuples(index=False):
        b = str(row.batch).lower()
        key = (ALIAS.get(b, b), str(row.paper_id))
        p = idx.get(key)
        if p is None:
            miss += 1
            continue
        try:
            dtext = p.read_text(errors="replace")
        except Exception:
            miss += 1
            continue
        if len(dtext) < 200:
            continue
        dpid = f"pe_{row.batch}_{row.paper_id}"
        title = row.title if isinstance(row.title, str) else ""
        text = f"PR TITLE: {title}\n\nCODE DIFF:\n{dtext}"
        items.append({
            "datapoint_id": dpid, "text": text, "ctext": canonical(text),
            "batch": row.batch, "paper_id": str(row.paper_id),
            "language": row.language,
            "judgement": 1 if row.judgement == "accepted" else
                         0 if row.judgement == "rejected" else None,
            "days_open": None if pd.isna(row.days_open) else float(row.days_open)})
        feats[dpid] = {
            "label": row.transplant_pr_label, "language": row.language,
            **{k: (None if pd.isna(getattr(row, k)) else float(getattr(row, k)))
               for k in ("n_assertion_fail", "n_vacuous_pass", "n_compile_fail",
                         "n_setup_fail", "n_uncollected", "test_byte_ratio",
                         "test_only_ratio", "days_open")},
            **{k: int(getattr(row, k)) for k in
               ("n_files_total", "n_files_test", "n_lines_added", "n_lines_deleted")}}
        feats[dpid].pop("days_open", None)   # anchor, not evidence
    print(f"items: {len(items)} (diff missing/short for {miss})")
    json.dump(items, open(OUT / "items.json", "w"))
    json.dump(feats, open(OUT / "exec_features.json", "w"))
    json.dump(ASPECTS, open(OUT / "aspects_used.json", "w"))

    aspects = {a["aspect_id"]: a for a in
               json.load(open(R / "runs/validity_full/v2/code_review/aspects.json"))}
    role, doctype = ROLE
    n = 0
    with open(OUT / "prompts.jsonl", "w") as f:
        def emit(ch, aid, dpid, prompt):
            nonlocal n
            f.write(json.dumps({"channel": ch, "aspect_id": aid,
                                "datapoint_id": dpid, "prompt": prompt}) + "\n")
            n += 1
        for aid in ASPECTS:
            a = aspects[aid]
            for it in items:
                emit("pass1", aid, it["datapoint_id"],
                     T1.format(role=role, doctype=doctype, name=a["name"],
                               description=a["description"], text=it["ctext"]))
                emit("pass2", aid, it["datapoint_id"],
                     T2.format(role=role, doctype=doctype, name=a["name"],
                               description=a["description"], text=it["ctext"]))
        for it in items:
            emit("scope", "scope", it["datapoint_id"],
                 TSCOPE.format(doctype=doctype, text=it["ctext"]))
        for prog in sorted(PROGS.glob("*_h0.py")):
            aid = prog.stem.split("_")[0]
            spec = importlib.util.spec_from_file_location(prog.stem, prog)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for field, instruction in list((getattr(mod, "LLM_FIELDS", {}) or {}).items())[:2]:
                for it in items:
                    emit("field", f"{aid}__{field}", it["datapoint_id"],
                         TFIELD.format(instruction=instruction, text=it["ctext"]))
    print(f"{len(ASPECTS)} aspects + scope + fields: {n} prompts -> {OUT}")


if __name__ == "__main__":
    main()
