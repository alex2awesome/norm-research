"""Hygiene sweep v2: find substring/\\w*-suffix matching bugs across fleet programs.

Bug classes (confirmed live in humor a135/a351/a153, CW a90/a72 h0s):
  A. regex stems with a \\w* suffix that swallow unintended host words
     (spic\\w* -> "spice", punch\\w* -> "punchline", descend\\w* -> "descendants")
  B. bare-substring membership: lexicon terms checked via `term in text` that occur
     EMBEDDED in longer words (cock -> "cockpit")

Precision over recall: candidates come from AST context (string-const `in` comparisons +
string-list/set/tuple literals of >=3 short items = lexicons), and hosts come from the
TASK'S OWN CORPUS tokens (not a dictionary) — a flag means "this term actually appears
embedded inside a different word in real docs of this task."

Usage: python3 detect_substring_bugs.py
-> outputs/metric_seam_pilot/battery/substring_bug_report.json
"""
import ast, json, pathlib, re, sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import BASE, ROOT  # noqa: E402

HYB = ROOT / "methods/metric_seam/hybrids"
PROGDIRS = {"press_releases": ["programs_v2", "programs"],
            "creative_writing": ["programs_cw"],
            "math": ["programs_math"],
            "humor": ["programs_humor"],
            "legal_title_vii": ["programs_legal"]}
AGENTIC = {"press_releases": ["a119_agentic.py", "a115_agentic.py", "a87_agentic.py"],
           "creative_writing": ["a90cw_agentic.py", "a72cw_agentic.py",
                                "a99cw_agentic.py", "a342cw_agentic.py"],
           "math": ["a198math_agentic.py", "a132math_agentic.py", "a42math_agentic.py"],
           "humor": ["a351humor_agentic.py", "a135humor_agentic.py",
                     "a153humor_agentic.py"]}

STEM_RE = re.compile(r"([a-z][a-z]{2,15})\\w\*")
TOKEN_RE = re.compile(r"[a-z][a-z\-']+")
OK = re.compile(r"^[a-z][a-z ']{2,15}$")


def items_for(task):
    if task == "press_releases":
        p = BASE / "v1/items_v1.json"
        return [x["ctext"] for x in json.load(open(p))][:500] if p.exists() else []
    p = BASE / "tasks" / task / "items.json"
    return [x["ctext"] for x in json.load(open(p))][:500] if p.exists() else []


def candidates_from_ast(src):
    """(term, context) pairs: 'in'-comparison left operands + lexicon-list members."""
    out = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 \
                and isinstance(node.ops[0], ast.In) \
                and isinstance(node.left, ast.Constant) \
                and isinstance(node.left.value, str):
            v = node.left.value.strip().lower()
            if OK.match(v):
                out.add((v, "in_compare"))
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            vals = [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if len(vals) >= 3 and all(OK.match(v.strip().lower()) for v in vals if v):
                for v in vals:
                    out.add((v.strip().lower(), "lexicon_list"))
    return out


def main():
    report = {}
    for task, dirs in PROGDIRS.items():
        docs = [d.lower() for d in items_for(task)]
        # corpus token inventory: token -> doc count
        tok_docs = Counter()
        for d in docs:
            for t in set(TOKEN_RE.findall(d)):
                tok_docs[t] += 1
        toks = list(tok_docs)
        files = []
        for dd in dirs:
            files += sorted((HYB / dd).glob("*_h*.py"))
        files += [HYB / "programs_agentic" / f for f in AGENTIC.get(task, [])
                  if (HYB / "programs_agentic" / f).exists()]
        for f in files:
            src = f.read_text()
            flags = []
            # class A: stem\w*
            for stem in sorted(set(STEM_RE.findall(src))):
                hosts = [t for t in toks if t.startswith(stem) and t != stem
                         and not re.match(re.escape(stem) + r"(s|es|ed|ing|ly|er)$", t)]
                if not hosts:
                    continue
                n = sum(tok_docs[h] for h in hosts)
                flags.append({"class": "A_stem_w*", "term": stem,
                              "corpus_hosts": sorted(hosts, key=lambda h: -tok_docs[h])[:8],
                              "docs_hit_est": n})
            # class B: substring membership terms embedded in corpus tokens
            for term, ctxt in sorted(candidates_from_ast(src)):
                if " " in term:      # multiword phrases: embedding much less likely
                    continue
                hosts = [t for t in toks if term in t and t != term
                         and not re.match(re.escape(term) + r"(s|es|ed|ing|ly|er)$", t)
                         and not t.endswith(term)]   # suffix-host = often same morpheme
                if not hosts:
                    continue
                emb = 0
                stand = re.compile(r"\b" + re.escape(term) + r"\b")
                for d in docs:
                    if term in d and not stand.search(d):
                        emb += 1
                if emb >= 3:
                    flags.append({"class": "B_bare_substring", "term": term,
                                  "context": ctxt,
                                  "corpus_hosts": sorted(hosts, key=lambda h: -tok_docs[h])[:8],
                                  "docs_embedded_only": emb})
            if flags:
                report[str(f.relative_to(HYB))] = sorted(
                    flags, key=lambda x: -(x.get("docs_hit_est", 0) +
                                           x.get("docs_embedded_only", 0)))
    out = BASE / "battery/substring_bug_report.json"
    json.dump(report, open(out, "w"), indent=1)
    tot = sum(len(v) for v in report.values())
    print(f"{len(report)} programs flagged, {tot} (program, term) flags -> {out}")
    ranked = sorted(((k, fl) for k, v in report.items() for fl in v),
                    key=lambda x: -(x[1].get("docs_hit_est", 0) +
                                    x[1].get("docs_embedded_only", 0)))
    for k, fl in ranked[:30]:
        n = fl.get("docs_hit_est", fl.get("docs_embedded_only", 0))
        print(f"{k:44s} {fl['class']:16s} {fl['term']:14s} n={n:4d} "
              f"hosts={','.join(fl['corpus_hosts'][:4])}")


if __name__ == "__main__":
    main()
