import sys
def sub1(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, f"match count {s.count(old)} != 1 in {path} for: {old[:60]!r}"
    open(path, "w").write(s.replace(old, new))
    print(f"patched {path}: {old[:50]!r}...")

A = "/lfs/skampere3/0/alexspan/outputs/osl_multi/author_fam_arms.py"
B = "/lfs/skampere3/0/alexspan/outputs/osl_multi/build_zxa_freeze_fam.py"

# --- A1: prompt emphasis (qwen undershoots length) ---
sub1(A, 'operational, not promotional. STRICT LENGTH: between 135 and 175 words." + verb +',
        'operational, not promotional. STRICT LENGTH: between 135 and 175 words — anything '
        'under 135 words will be REJECTED, so aim for about 155 words." + verb +')
sub1(A, '"STRICT TOTAL LENGTH: between 370 and 440 words across all sections." + verb +',
        '"STRICT TOTAL LENGTH: between 370 and 440 words across all sections — aim for about '
        '405 words; under 370 will be REJECTED." + verb +')

# --- A2: targeted repair pass after main generation ---
old = """            res = ex.generate_batch([prompts[i] for i in idxs], system=None,
                                    max_tokens=1100 if is_doss else 450, validate=f)
            for j, i in enumerate(idxs):
                outs[i] = clean(res[j])
"""
new = old + """
        # repair pass: word-count misses get targeted expand/trim seeded with own text
        # (fresh resampling alone fails: qwen25-72b lands ~110-125w vs 130w floor 64/72 times)
        for _round in range(2):
            bad_by_rule, rp = {}, {}
            for i in range(len(todo)):
                if outs[i] is None:
                    continue
                cur = clean(outs[i])
                if (ok_doss(cur, rules[i]) if is_doss else ok_expl(cur, rules[i])):
                    continue
                if is_doss:
                    pos = [cur.find(L) for L in LABELS]
                    if any(p < 0 for p in pos) or pos != sorted(pos):
                        continue  # structural miss: repair has no anchor, leave to rerun
                n = wc(cur)
                verb = (f"\\nCRITICAL: the following rule sentence must appear VERBATIM "
                        f"(character-for-character) somewhere in your text:\\n\\"{rules[i]}\\"\\n"
                        ) if rules[i] else ""
                if is_doss:
                    tgt = ("expand it to between 380 and 430 words by adding concrete markers "
                           "in WHAT COUNTS and one more contrast exemplar per side"
                           if n < 360 else
                           "shorten it to between 380 and 430 words without dropping any section")
                    rp[i] = (f"The judging dossier below is {n} words, which is the wrong "
                             f"length. Rewrite it: {tgt}. Keep EXACTLY these four section "
                             "labels, in this order, each on its own line: DEFINITION, "
                             "WHAT COUNTS, CONTRAST EXEMPLARS, BOUNDARY CASES." + verb +
                             "\\n\\nDossier:\\n" + cur +
                             "\\n\\nOutput ONLY the rewritten dossier starting with the line "
                             "DEFINITION, no preamble.")
                else:
                    tgt = ("expand it to between 140 and 170 words by adding concrete "
                           "operational detail: what markers to look for and how to decide "
                           "borderline cases. Keep the existing content"
                           if n < 130 else "shorten it to between 140 and 170 words")
                    rp[i] = (f"The evaluator explanation below is {n} words, which is the "
                             f"wrong length. Rewrite it: {tgt}." + verb +
                             "\\n\\nExplanation:\\n" + cur +
                             "\\n\\nOutput ONLY the rewritten single paragraph, no title, "
                             "no preamble.")
                bad_by_rule.setdefault(rules[i], []).append(i)
            if not rp:
                break
            print(f"[fam-author] repair round {_round + 1} "
                  f"({'doss' if is_doss else 'expl'}): {len(rp)} rows")
            for rule, idxs in bad_by_rule.items():
                f = (lambda o, _r=rule: ok_doss(o, _r)) if is_doss else \\
                    (lambda o, _r=rule: ok_expl(o, _r))
                res = ex.generate_batch([rp[i] for i in idxs], system=None,
                                        max_tokens=1100 if is_doss else 500, validate=f)
                for j, i in enumerate(idxs):
                    cand = clean(res[j])
                    if (ok_doss(cand, rule) if is_doss else ok_expl(cand, rule)):
                        outs[i] = cand
"""
sub1(A, old, new)

# --- B: freeze builder drops bad bases instead of hard-failing ---
sub1(B, """    errs = []
    by_task = defaultdict(list)
    for m in slate:
        by_task[m["task"]].append(m)
        for fam in AUTHORS:
            a = authored.get((fam, m["task"], m["name"]))
            if a is None or not a.get("valid", False):
                errs.append(f"MISSING/invalid {fam} authored: {m['task']} / {m['name'][:50]}")
                continue""",
        """    errs = []
    bad_bases = set()
    by_task = defaultdict(list)
    for m in slate:
        by_task[m["task"]].append(m)
        for fam in AUTHORS:
            n_err0 = len(errs)
            a = authored.get((fam, m["task"], m["name"]))
            if a is None or not a.get("valid", False):
                errs.append(f"MISSING/invalid {fam} authored: {m['task']} / {m['name'][:50]}")
                bad_bases.add((m["task"], m["name"]))
                continue""")
sub1(B, """                    if rule not in a[fld]:
                        errs.append(f"{fam} planted rule NOT verbatim in {fld}: {m['name'][:50]}")
    if errs:
        print(f"VALIDATION FAILED ({len(errs)}):")
        for e in errs[:30]:
            print(" -", e)
        sys.exit(1)
""",
        """                    if rule not in a[fld]:
                        errs.append(f"{fam} planted rule NOT verbatim in {fld}: {m['name'][:50]}")
            if len(errs) > n_err0:
                bad_bases.add((m["task"], m["name"]))
    if errs:
        print(f"WARN {len(errs)} gate failures -> dropping {len(bad_bases)} bases "
              f"(2x2 kept only where BOTH authors fully valid):")
        for e in errs[:40]:
            print(" -", e)
    kept = {t: [m for m in ms if (t, m["name"]) not in bad_bases]
            for t, ms in by_task.items()}
    for t in kept:
        print(f"[fam-freeze] {t}: kept {len(kept[t])}/{len(by_task[t])} bases")
        if len(kept[t]) < 12:
            print(f"FATAL: {t} has <12 bases with both authors valid; fix authoring first")
            sys.exit(1)
    by_task = kept
""")
print("ALL PATCHES OK")
