"""STAGE 1 judge-prompt builder for the refreshed code_review task.

Mirrors methods/metric_seam/pilot/build_task.py's T1/T2 two-pass template
VERBATIM (same wording, same "SCORE: <integer 0-10 or NA>" output convention
so gemma_score_v1.py scores it unchanged) -- only role/doctype and the
aspect list differ, because this task's items are PR DIFFS (no title/
description/comments), not the "code-review discussion" doctype used by
the original (comments-only) code_review build.

Reads outputs/metric_seam_pilot/tasks/code_review/{items.json,aspects_candidates.json}
Writes outputs/metric_seam_pilot/tasks/code_review/prompts.jsonl
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/tasks/code_review"

ROLE = "an experienced senior software engineer reviewing pull requests"
DOCTYPE = ("pull-request code diff (the full unified diff of the change; "
           "no PR title, description, or review comments are shown -- judge "
           "the change purely on the code itself)")

# VERBATIM from build_task.py -- do not edit independently of that file.
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


def main():
    items = json.load(open(OUT / "items.json"))
    aspects = json.load(open(OUT / "aspects_candidates.json"))
    picked = [a["aspect_id"] for a in aspects]
    json.dump(picked, open(OUT / "aspects_used.json", "w"))

    n = 0
    with open(OUT / "prompts.jsonl", "w") as f:
        def emit(ch, aid, dpid, prompt):
            nonlocal n
            f.write(json.dumps({"channel": ch, "aspect_id": aid,
                                "datapoint_id": dpid, "prompt": prompt}) + "\n")
            n += 1
        for a in aspects:
            aid = a["aspect_id"]
            for it in items:
                emit("pass1", aid, it["datapoint_id"],
                     T1.format(role=ROLE, doctype=DOCTYPE, name=a["name"],
                               description=a["description"], text=it["ctext"]))
                emit("pass2", aid, it["datapoint_id"],
                     T2.format(role=ROLE, doctype=DOCTYPE, name=a["name"],
                               description=a["description"], text=it["ctext"]))
        for it in items:
            emit("scope", "scope", it["datapoint_id"],
                 TSCOPE.format(doctype=DOCTYPE, text=it["ctext"]))

    print(f"code_review: {len(picked)} aspects, {len(items)} items, {n} prompts -> {OUT}")
    print(f"expected = {len(picked)}*{len(items)}*2 + {len(items)} = "
          f"{len(picked)*len(items)*2 + len(items)}")


if __name__ == "__main__":
    main()
