"""Generalized certified-seam-survey builder: any v2 task, N aspects, 2 judge passes + scope.

Usage: python3 build_task.py <task> <n_aspects> [extra_aspect_ids...]
Writes outputs/metric_seam_pilot/tasks/<task>/{items.json, prompts.jsonl, aspects_used.json}
Aspect selection: evenly-strided over aspects.json (reproducible spread across the bank),
keeping only aspects whose codegen v0 flavor exists; optional explicit extras prepended.
"""
import json, pathlib, random, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
HEAD, TAIL, SEED, N_ITEMS = 5000, 2500, 0, 250

ROLE = {
    "press_releases": ("an experienced newswire editor", "press release"),
    "math": ("an experienced mathematics educator and Math StackExchange moderator",
             "mathematical answer/solution"),
    "code_review": ("an experienced senior software engineer reviewing pull requests",
                    "code-review discussion (PR title, description, and comments)"),
    "patents": ("an experienced patent examiner", "patent application document"),
    "creative_writing": ("an experienced fiction editor and creative-writing workshop "
                         "instructor", "piece of short fiction (a story or story excerpt)"),
    "humor": ("an experienced comedy writer and humor editor",
              "attempted-humor text (a joke, humorous story, or comedic piece)"),
    "legal_title_vii": ("an experienced federal employment-discrimination litigator",
                        "Title VII case-facts narrative (ex-ante pleaded facts)"),
    "peer_review": ("an experienced peer reviewer for a machine-learning conference",
                    "research-paper submission (or an excerpt of one)"),
    "legal_ss_disability": ("an experienced Social Security disability appeals attorney",
                            "Social Security disability appeal case-facts narrative "
                            "(ex-ante facts of an ALJ denial under review)"),
}

# per-task minimum text length; default 600 targets web-scrape stubs in document tasks,
# but humor items are legitimately short (median ~480 chars — a joke is a complete document)
MIN_LEN = {"humor": 150}

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


def main():
    task, n_aspects = sys.argv[1], int(sys.argv[2])
    extras = sys.argv[3:]
    role, doctype = ROLE[task]
    V2 = ROOT / "runs/validity_full/v2" / task
    OUT = ROOT / "outputs/metric_seam_pilot/tasks" / task
    OUT.mkdir(parents=True, exist_ok=True)

    all_aspects = json.load(open(V2 / "aspects.json"))
    have_code = {p.name.split("_v0")[0] for p in (V2 / "codegen_claude").glob("*_v0_*.py")}
    eligible = [x for x in all_aspects if x["aspect_id"] in have_code
                and x["aspect_id"] not in extras]
    stride = max(1, len(eligible) // max(1, n_aspects - len(extras)))
    picked = extras + [x["aspect_id"] for x in eligible[::stride]][:n_aspects - len(extras)]
    aspects = {x["aspect_id"]: x for x in all_aspects}

    data = json.load(open(V2 / "datapoints.json"))
    pool = [d for d in data if len(d.get("text", "")) >= MIN_LEN.get(task, 600)]
    random.seed(SEED)
    items = random.sample(pool, min(N_ITEMS, len(pool)))
    for it in items:
        it["ctext"] = canonical(it["text"])
    json.dump(items, open(OUT / "items.json", "w"))
    json.dump(picked, open(OUT / "aspects_used.json", "w"))

    n = 0
    with open(OUT / "prompts.jsonl", "w") as f:
        def emit(ch, aid, dpid, prompt):
            nonlocal n
            f.write(json.dumps({"channel": ch, "aspect_id": aid,
                                "datapoint_id": dpid, "prompt": prompt}) + "\n")
            n += 1
        for aid in picked:
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
    print(f"{task}: {len(picked)} aspects, {len(items)} items, {n} prompts -> {OUT}")


if __name__ == "__main__":
    main()
