"""Corpus-NATIVE code programs for the two most mechanical competition aspects — the direct
test of "this corpus should go V": the existing code rung is comments-era (OOD twice over);
if executor-matched programs jump toward the judge ceiling, codability was there all along.

a180 One statement per line and post-conditional line breaks
a135 Import organization and hygiene

Programs read the SAME canonical text the judge saw (apples-to-apples guard).
"""
import json, pathlib, re, statistics as st, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/metric_seam_pilot/tasks/code_competition"
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman, attenuation_ceiling


def get_code(ctext):
    m = re.search(r"SUBMITTED SOLUTION \(([^)]*)\):\n", ctext)
    if not m:
        return "", "?"
    return ctext[m.end():], m.group(1).strip().lower()


def lang_fam(lang):
    return "python" if "py" in lang else ("cpp" if lang in ("cpp", "c++", "gnu c++", "c")
                                          else "other")


_STR = re.compile(r"('([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\")")


def _strip_strings(line):
    return _STR.sub("''", line)


def a180_native(ctext):
    """Fraction of code lines honoring one-statement-per-line + post-conditional breaks."""
    code, lang = get_code(ctext)
    fam = lang_fam(lang)
    lines = [l for l in code.split("\n") if l.strip() and not l.strip().startswith(("#", "//"))]
    if len(lines) < 3:
        return None
    viol = 0
    for raw in lines:
        l = _strip_strings(raw.strip())
        if fam == "python":
            if ";" in l:
                viol += 1
            elif re.match(r"^(if|elif|else|for|while|try|except|finally|with|def|class)\b[^:]*:\s*\S", l) \
                    and not re.match(r".*:\s*(#|$)", l):
                viol += 1                       # one-line suite after the colon
            elif re.search(r"\blambda\b", l) is None and l.count(" if ") and l.count(" else "):
                pass                            # ternary expression: acceptable, not a suite
        elif fam == "cpp":
            semis = l.count(";")
            if semis >= 2 and not re.match(r"^\s*for\s*\(", l):
                viol += 1                       # 2+ statements on a line (for-headers exempt)
            elif re.match(r"^(if|while|else if)\s*\(.*\)\s*\S+.*;", l):
                viol += 1                       # conditional with body on same line
        else:
            if l.count(";") >= 2:
                viol += 1
    return max(0.0, 1.0 - viol / max(4.0, 0.3 * len(lines)))


def a135_native(ctext):
    """Import organization/hygiene: imports at top, no wildcard, no duplicates, used."""
    code, lang = get_code(ctext)
    fam = lang_fam(lang)
    lines = [l for l in code.split("\n") if l.strip()]
    if len(lines) < 3:
        return None
    s = 1.0
    if fam == "python":
        imports, seen, first_code = [], set(), None
        for i, raw in enumerate(lines):
            l = raw.strip()
            if re.match(r"^(import|from)\s+\w", l):
                imports.append((i, l))
                if l in seen:
                    s -= 0.15                   # duplicate import
                seen.add(l)
                if "import *" in l:
                    s -= 0.25                   # wildcard
                if first_code is not None:
                    s -= 0.1                    # import buried below code
            elif first_code is None and not l.startswith(("#", '"', "'")):
                first_code = i
        if not imports:
            return 0.6 if len(lines) > 6 else None   # no imports at all: neutral-ish
        body = "\n".join(l for i, l in enumerate(lines) if i not in {j for j, _ in imports})
        for _, imp in imports:
            m = re.match(r"^import\s+([\w.]+)(?:\s+as\s+(\w+))?", imp)
            f = re.match(r"^from\s+[\w.]+\s+import\s+(.+)", imp)
            name = None
            if m:
                name = (m.group(2) or m.group(1).split(".")[0])
            elif f and "*" not in f.group(1):
                name = f.group(1).split(",")[0].split(" as ")[-1].strip()
            if name and not re.search(r"\b" + re.escape(name) + r"\b", body):
                s -= 0.12                       # unused import
    elif fam == "cpp":
        incs, seen = [], set()
        first_code = None
        for i, raw in enumerate(lines):
            l = raw.strip()
            if l.startswith("#include"):
                if l in seen:
                    s -= 0.15
                seen.add(l)
                if first_code is not None:
                    s -= 0.1
                if "bits/stdc++" in l:
                    s -= 0.2                    # kitchen-sink include = hygiene violation
                incs.append(l)
            elif first_code is None and not l.startswith(("//", "/*", "using")):
                first_code = i
        if not incs:
            return 0.5
        if any("using namespace std" in l for l in lines):
            s -= 0.1
    else:
        return None
    return max(0.0, min(1.0, s))


def main():
    items = {x["datapoint_id"]: x["ctext"] for x in json.load(open(OUT / "items.json"))}
    p1, p2 = {}, {}
    for line in open(OUT / "results.jsonl"):
        r = json.loads(line)
        if not isinstance(r["score"], int) or r["aspect_id"] == "scope":
            continue
        d = p1 if r["channel"] == "pass1" else p2 if r["channel"] == "pass2" else None
        if d is not None:
            d.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = r["score"]

    for aid, fn, old in (("a180", a180_native, 0.30), ("a135", a135_native, -0.33)):
        judge = {}
        for dp in set(p1.get(aid, {})) | set(p2.get(aid, {})):
            vals = [m[aid][dp] for m in (p1, p2) if dp in m.get(aid, {})]
            judge[dp] = sum(vals) / len(vals)
        both = [d for d in p1.get(aid, {}) if d in p2.get(aid, {})]
        rel1 = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        ceil = attenuation_ceiling(min(max(rel1, 0), 1), 2)
        nat = {d: fn(items[d]) for d in judge if d in items}
        sel = [d for d in nat if nat[d] is not None]
        rho = spearman([nat[d] for d in sel], [judge[d] for d in sel])
        # per language family too
        by = {}
        for d in sel:
            _, lang = get_code(items[d])
            by.setdefault(lang_fam(lang), []).append(d)
        per = {f: round(spearman([nat[d] for d in g], [judge[d] for d in g]), 3)
               for f, g in by.items() if len(g) >= 30}
        print(f"{aid}: NATIVE rho={rho:.3f} (n={len(sel)})  vs comments-era {old}  "
              f"ceiling={ceil:.2f}  rho/ceil={rho/ceil:.2f}  per-lang={per}")


if __name__ == "__main__":
    main()
