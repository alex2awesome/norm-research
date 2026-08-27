"""Code-review seam re-survey on FULL-PR corpora (runs on sk3; data lives there).

Builds two new survey task dirs with the SAME 40 code_review aspects and the same
certified-survey protocol (2 judge passes + scope, canonical head5000+tail2500), so the
result is a direct A/B against the comments-only run:

  code_review_diffs  — datasets/code-review/code_review_dense_4096tok_with_reasoning.csv.gz
                       (PR title + description + unified diff; diff-bearing rows only)
  code_competition   — datasets/competition_unified/candidates.parquet x problems.parquet
                       (problem title/difficulty/tags + submitted solution code; verdict is
                       kept in items.json meta as a later external V anchor, NOT shown to
                       the judge)

Usage (sk3): python3.11 build_cr_rerun.py
Then queue:  echo "<dir>/prompts.jsonl <dir>/results.jsonl" > queue/NN_name.job
"""
import gzip, json, pathlib, random

R = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_BASE = R / "outputs/metric_seam_pilot/tasks"
HEAD, TAIL, SEED, N_ITEMS = 5000, 2500, 0, 300

ASPECTS_USED = ["a0", "a9", "a18", "a27", "a36", "a45", "a54", "a63", "a72", "a81",
                "a90", "a99", "a108", "a117", "a126", "a135", "a144", "a153", "a162",
                "a171", "a180", "a189", "a198", "a207", "a216", "a225", "a234", "a243",
                "a252", "a261", "a270", "a279", "a288", "a297", "a306", "a315", "a324",
                "a333", "a342", "a351"]   # same 40 as the comments-only survey

ROLE = {
    "code_review_diffs": ("an experienced senior software engineer reviewing pull requests",
                          "pull request (title, description, and code diff)"),
    "code_competition": ("an experienced competitive-programming coach reviewing submitted "
                         "solutions", "competitive-programming solution (problem title + "
                         "submitted code)"),
}

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


def canonical(text):
    if len(text) <= HEAD + TAIL + 500:
        return text
    return text[:HEAD] + "\n[...]\n" + text[-TAIL:]


def sample_cr_diffs():
    """Reservoir-sample diff-bearing PRs from the 4096tok csv (chunked; deterministic)."""
    import pandas as pd
    rng = random.Random(SEED)
    reservoir, seen = [], 0
    src = R / "datasets/code-review/code_review_dense_4096tok_with_reasoning.csv.gz"
    for chunk in pd.read_csv(src, chunksize=20000):
        for idx, row in zip(chunk.index, chunk.itertuples(index=False)):
            t = row.text if isinstance(row.text, str) else ""
            if "diff --git" not in t or len(t) < 1200:
                continue
            item = {"datapoint_id": f"crd{idx}", "text": t,
                    "judgement": int(row.judgement) if row.judgement == row.judgement else None}
            if len(reservoir) < N_ITEMS:
                reservoir.append(item)
            else:
                j = rng.randrange(seen + 1)
                if j < N_ITEMS:
                    reservoir[j] = item
            seen += 1
    print(f"cr_diffs: {seen} qualifying rows, sampled {len(reservoir)}")
    return reservoir


def sample_competition():
    import pandas as pd
    p = R / "datasets/competition_unified"
    ca = pd.read_parquet(p / "candidates.parquet",
                         columns=["platform", "problem_id", "code", "code_lang", "verdict",
                                  "runtime_ms", "memory_kb", "canonical_pid"])
    ca = ca[ca.code.str.len().fillna(0).between(400, 50000)]
    strata = {"AC": 120, "WA": 60, "TLE": 30, "RE": 30, "CE": 15, "unknown": 45}
    parts = []
    for v, n in strata.items():
        g = ca[ca.verdict == v]
        parts.append(g.sample(min(n, len(g)), random_state=SEED))
    df = pd.concat(parts)
    pr = pd.read_parquet(p / "problems.parquet",
                         columns=["canonical_pid", "title", "difficulty", "tags"])
    pr = pr.drop_duplicates("canonical_pid").set_index("canonical_pid")
    items = []
    for i, row in enumerate(df.itertuples(index=False)):
        meta = pr.loc[row.canonical_pid] if row.canonical_pid in pr.index else None
        title = (meta.title if meta is not None and isinstance(meta.title, str)
                 else str(row.problem_id))
        diff = meta.difficulty if meta is not None else ""
        tags = ", ".join(list(meta.tags)[:6]) if meta is not None and meta.tags is not None \
            else ""
        text = (f"PROBLEM: {title}\nPlatform: {row.platform} | difficulty: {diff} | "
                f"tags: {tags}\n\nSUBMITTED SOLUTION ({row.code_lang}):\n{row.code}")
        items.append({"datapoint_id": f"cc{i}", "text": text,
                      "verdict": row.verdict,
                      "runtime_ms": None if pd.isna(row.runtime_ms) else float(row.runtime_ms),
                      "memory_kb": None if pd.isna(row.memory_kb) else float(row.memory_kb),
                      "canonical_pid": row.canonical_pid})
    print(f"competition: sampled {len(items)} "
          f"({df.verdict.value_counts().to_dict()})")
    return items


def build(task, items):
    role, doctype = ROLE[task]
    out = OUT_BASE / task
    out.mkdir(parents=True, exist_ok=True)
    aspects = {x["aspect_id"]: x for x in
               json.load(open(R / "runs/validity_full/v2/code_review/aspects.json"))}
    for it in items:
        it["ctext"] = canonical(it["text"])
    json.dump(items, open(out / "items.json", "w"))
    json.dump(ASPECTS_USED, open(out / "aspects_used.json", "w"))
    n = 0
    with open(out / "prompts.jsonl", "w") as f:
        def emit(ch, aid, dpid, prompt):
            nonlocal n
            f.write(json.dumps({"channel": ch, "aspect_id": aid,
                                "datapoint_id": dpid, "prompt": prompt}) + "\n")
            n += 1
        for aid in ASPECTS_USED:
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
    print(f"{task}: {len(ASPECTS_USED)} aspects, {len(items)} items, {n} prompts -> {out}")


if __name__ == "__main__":
    build("code_review_diffs", sample_cr_diffs())
    build("code_competition", sample_competition())
