#!/usr/bin/env python3
"""Builds + executes notebooks/2026-07-11__sibling-metric-sharing-tree.ipynb.
Data: notebooks/data/lattice-2026-07-10/*.json (fetched from sk3 outputs/ctree/lattice)."""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
C, M = [], nb.cells


def md(s): M.append(nbf.v4.new_markdown_cell(s))
def code(s): M.append(nbf.v4.new_code_cell(s))


md("""# Do sibling sub-communities share metrics? A tree/venn view of the lattice

**Task #66 (sibling metric lattice), data through 2026-07-10.**

For 12 sets of *sibling subtasks* (math topics, humor topics, creative-writing genres, ICLR
subfields, peer-review venues, code-review languages, notice-and-comment topics, press-release
topics) we asked two separable questions:

1. **Shared *explicit* vocabulary** — of a fixed, coverage-selected bank of ~40 real, human-written
   rubric metrics per task, which are *live* (univariate AUC ≥ .55 against the community's own
   accept/reject label) in each sibling? How often is the same explicit metric live in ≥2 siblings?
2. **Invented metrics** — metrics the GLM proposer invented per sibling that survived the
   discovery gate. Do the *same inventions* recur across siblings, or is each sibling's invention
   local?

**Method guardrails** (see BEST-PRACTICES `[metric inference / discovery]`): proposer never sees
labels; a proposal must beat the existing bank (residual gate); only strictly-disjoint stage-2
replication counts as a confirmed keep; stage-1 in-run survivors are candidates, not evidence.

> **Headline**: siblings *reuse* explicit vocabulary at wildly different rates by domain
> (code-review languages share most of their bank; peer-review venues share almost nothing), but
> **confirmed invented metrics never recur across siblings** — every stage-2 keep is local to one
> sub-community. Apparent cross-sibling recurrence (peer "theory-rigor") died 0/13 in disjoint
> replication.""")

code("""import json, glob, os, itertools, html
from IPython.display import HTML, display

DATA = "data/lattice-2026-07-10"
TAU = 0.55

liveness = {}          # setname -> {"siblings": [...], "per_leg": {sib: {"base":, "aucs": {m: auc}}}, "rec": set-record}
for fp in sorted(glob.glob(f"{DATA}/liveness_*.json")):
    d = json.load(open(fp))
    for setname, rec in d.get("sets", {}).items():
        per_leg = {}
        for sib in rec["siblings"]:
            leg = d["liveness"].get(sib) or {}
            per_leg[sib] = {"base": leg.get("base"), "n": leg.get("n"),
                            "aucs": {m: a for m, a in (leg.get("aucs") or {}).items() if a is not None}}
        liveness[setname] = {"siblings": rec["siblings"], "per_leg": per_leg, "rec": rec}

lat = json.load(open(f"{DATA}/sibling_lattice.json"))
invented = lat["sets"]          # setname -> theme_shared_clusters / theme_unique_examples / per_sibling_invented
pooled = lat["pooled_general"]
stage2 = json.load(open(f"{DATA}/stage2_audit.json"))   # list of {dir, name, leg, stage2_status, rep_p_auc, rep_p_bits, rep_bits_gain}

def _short_leg(leg):
    s = leg
    for p_ in ["math-", "humor-topic-", "cw-genre-", "peer-iclr-", "peer-venue-", "code-lang-",
               "notice-topic-", "press-topic-"]:
        if s.startswith(p_): s = s[len(p_):]
    return s.removesuffix("-glmprop")

kept2_pairs = {(r["name"], _short_leg(r["leg"])) for r in stage2 if r.get("stage2_status") == "KEPT"}
repl2_pairs = {(r["name"], _short_leg(r["leg"])) for r in stage2
               if (r.get("rep_p_auc") or 1) < .05 or (r.get("rep_p_bits") or 1) < .05}

ORDER = ["math-analysis", "math-algebra", "math-discrete-geom", "humor-scenario",
         "humor-wordplay", "cw-scifi", "cw-fate-deal", "peer-iclr",
         "peer-venue", "code-lang", "notice-topic", "press-topic"]
SETS = [s for s in ORDER if s in liveness] + [s for s in liveness if s not in ORDER]
print(f"{len(SETS)} sibling sets loaded: {SETS}")
print(f"stage-2 audited: {len(stage2)} rows, {len(kept2_pairs)} unique formal keeps, {len(repl2_pairs)} nominally-replicated (name, sibling) pairs")""")

md("""## 1&nbsp;·&nbsp;How often do siblings share real, explicit metrics?

For every sibling we mark a bank metric **live** if its univariate AUC ≥ .55 on that sibling's own
labels. Sharing is then just set overlap between siblings' live sets. Two readouts:

- **shared-of-union**: of all bank metrics live *anywhere* in the set, what fraction are live in ≥2
  siblings (the "common vocabulary" share);
- **pairwise Jaccard**: for a random *pair* of siblings, |live(A) ∩ live(B)| / |live(A) ∪ live(B)|.""")

code("""def live_sets(setname):
    L = liveness[setname]
    return {sib: {m for m, a in L["per_leg"][sib]["aucs"].items() if a >= TAU}
            for sib in L["siblings"]}

rows = []
for s in SETS:
    ls = live_sets(s)
    sibs = list(ls)
    union = set().union(*ls.values()) if ls else set()
    counts = {m: sum(m in ls[x] for x in sibs) for m in union}
    ge2 = {m for m, c in counts.items() if c >= 2}
    all_ = {m for m, c in counts.items() if c == len(sibs)}
    pj, pint = [], []
    for a, b in itertools.combinations(sibs, 2):
        u = ls[a] | ls[b]
        pj.append(len(ls[a] & ls[b]) / len(u) if u else 0.0)
        pint.append(len(ls[a] & ls[b]))
    rows.append(dict(set=s, sib=len(sibs), union=len(union), ge2=len(ge2), all_=len(all_),
                     frac_ge2=len(ge2) / len(union) if union else 0.0,
                     jacc=sum(pj) / len(pj) if pj else 0.0,
                     mean_int=sum(pint) / len(pint) if pint else 0.0,
                     nonempty=sum(1 for x in pint if x > 0) / len(pint) if pint else 0.0))

def bar(frac, color, w=120):
    return (f'<div style="background:#eee;border-radius:3px;width:{w}px;height:12px;display:inline-block">'
            f'<div style="background:{color};width:{max(2, frac * w):.0f}px;height:12px;border-radius:3px"></div></div>')

h = ['<table style="border-collapse:collapse;font:13px sans-serif">',
     '<tr style="border-bottom:2px solid #333"><th style="text-align:left;padding:4px 10px">sibling set</th>'
     '<th>sibs</th><th>live anywhere /40</th><th>live in &ge;2</th><th>live in ALL</th>'
     '<th style="text-align:left;padding:4px 10px">shared-of-union</th>'
     '<th style="text-align:left;padding:4px 10px">pairwise Jaccard</th>'
     '<th>P(pair shares &ge;1)</th></tr>']
for r in sorted(rows, key=lambda r: -r["frac_ge2"]):
    h.append(f'<tr style="border-bottom:1px solid #ddd"><td style="text-align:left;padding:3px 10px">'
             f'<b>{r["set"]}</b></td><td align=center>{r["sib"]}</td><td align=center>{r["union"]}</td>'
             f'<td align=center>{r["ge2"]}</td><td align=center>{r["all_"]}</td>'
             f'<td style="padding:3px 10px">{bar(r["frac_ge2"], "#2a9d5c")} {r["frac_ge2"]:.0%}</td>'
             f'<td style="padding:3px 10px">{bar(r["jacc"], "#4a7fd4")} {r["jacc"]:.2f}</td>'
             f'<td align=center>{r["nonempty"]:.0%}</td></tr>')
h.append('</table>')
display(HTML("".join(h)))

import statistics as st
print(f"\\nACROSS ALL {len(rows)} SETS:")
print(f"  shared-of-union: median {st.median(r['frac_ge2'] for r in rows):.0%}  "
      f"(range {min(r['frac_ge2'] for r in rows):.0%} - {max(r['frac_ge2'] for r in rows):.0%})")
print(f"  pairwise Jaccard: median {st.median(r['jacc'] for r in rows):.2f}  "
      f"(range {min(r['jacc'] for r in rows):.2f} - {max(r['jacc'] for r in rows):.2f})")
print(f"  P(a random sibling pair shares >=1 live explicit metric): "
      f"median {st.median(r['nonempty'] for r in rows):.0%}")""")

md("""## 2&nbsp;·&nbsp;The venn-tree, per sibling set

Each card is one sibling set. **Root** = the parent task (with its pooled-control verdict).
The **green band** is the intersection zone — explicit bank metrics live in ≥2 siblings (chips show
*which* siblings; solid chip = live there). **Leaves** are the siblings; each lists what is *only
theirs*:

- <span style="background:#dbeafe;border-radius:8px;padding:0 6px">blue pills</span> — explicit bank
  metrics live **only** in this sibling (shared vocabulary that failed to generalize);
- <span style="background:#fde68a;border-radius:8px;padding:0 6px">★ gold pills</span> — **invented,
  stage-2 confirmed** on disjoint items (the real discoveries);
- <span style="background:#f3f4f6;border-radius:8px;padding:0 6px;color:#888">gray pills</span> —
  invented, stage-1 only (candidates; most will not survive replication — shown faded for honesty;
  capped at ~12 examples per set).""")

code(r"""PREF = ["math-", "humor-topic-", "cw-genre-", "peer-iclr-", "peer-venue-", "code-lang-",
        "notice-topic-", "press-topic-"]
def short(leg):
    s = leg
    for p in PREF:
        if s.startswith(p): s = s[len(p):]
    return s.removesuffix("-glmprop")

def esc(x): return html.escape(str(x))

def pill(text, bg, fg="#222", title="", star=False):
    t = f' title="{esc(title)}"' if title else ""
    star_s = "★ " if star else ""
    return (f'<span{t} style="background:{bg};color:{fg};border-radius:9px;padding:1px 8px;'
            f'margin:2px;display:inline-block;font-size:11.5px;line-height:1.7">{star_s}{esc(text)}</span>')

def chips(live_in, sibs):
    out = []
    for s in sibs:
        on = s in live_in
        out.append(f'<span style="display:inline-block;width:11px;height:11px;border-radius:3px;'
                   f'margin:0 1px;background:{"#2a9d5c" if on else "#e5e7eb"}" title="{esc(short(s))}"></span>')
    return "".join(out)

def render_set(setname):
    L = liveness[setname]; sibs = L["siblings"]; ls = live_sets(setname)
    union = set().union(*ls.values()) if ls else set()
    shared = sorted((m for m in union if sum(m in ls[x] for x in sibs) >= 2),
                    key=lambda m: -sum(m in ls[x] for x in sibs))
    inv = invented.get(setname, {})
    inv_by_leg = {}
    for u in inv.get("theme_unique_examples", []):
        for leg in u["legs"]:
            for nm in u["names"]:
                sl = short(leg)
                inv_by_leg.setdefault(sl, []).append(
                    dict(name=nm, kept=(nm, sl) in kept2_pairs, repl=(nm, sl) in repl2_pairs))
    for c in inv.get("theme_shared_clusters", []):        # stage-1 shared clusters (all died at stage-2)
        for leg in c["legs"]:
            for nm in c["names"]:
                sl = short(leg)
                inv_by_leg.setdefault(sl, []).append(
                    dict(name=nm + "  (recurred at stage-1)", kept=(nm, sl) in kept2_pairs,
                         repl=(nm, sl) in repl2_pairs))
    cands = [k for k in pooled if k == setname or setname.startswith(k + "-")]
    pool = pooled[max(cands, key=len)] if cands else None
    pool_html = ""
    if pool is not None:
        n_inv = pool["invented_stage1"]
        note = ("pooled control: <b>0 invented</b> — general bank suffices (dilution)" if n_inv == 0 else
                f'pooled control: {n_inv} stage-1 "inventions" — <b>balanced-pool base-rate confound</b>, see §4')
        pool_html = f'<div style="font-size:11.5px;color:#666;margin-top:2px">{note}</div>'

    shared_html = ("".join(pill(m, "#d1f2df", title=f"live in {sum(m in ls[x] for x in sibs)}/{len(sibs)} siblings")
                           + chips({x for x in sibs if m in ls[x]}, sibs) for m in shared)
                   or '<span style="color:#999;font-size:12px">&empty; &nbsp;no explicit metric live in &ge;2 siblings</span>')

    kids = []
    for s in sibs:
        only = sorted(ls[s] - set().union(*(ls[x] for x in sibs if x != s)))
        base = L["per_leg"][s].get("base")
        items = [pill(m, "#dbeafe", title="bank metric live only here") for m in only]
        for iv in inv_by_leg.get(short(s), []):
            if iv["kept"]:
                items.append(pill(iv["name"], "#fde68a", title="invented, stage-2 KEPT (disjoint replication)", star=True))
            elif iv["repl"]:
                items.append(pill(iv["name"], "#fef3c7", "#7a5b00", title="invented, replicated nominally (p<.05, not Bonferroni)"))
            else:
                items.append(pill(iv["name"], "#f3f4f6", "#999", title="invented, stage-1 only"))
        kids.append(
            f'<div style="flex:1;min-width:150px;background:#fff;border:1px solid #d7dbe0;border-radius:8px;'
            f'padding:7px 9px;margin:4px"><div style="font-weight:600;font-size:13px">{esc(short(s))}'
            f'<span style="color:#999;font-weight:400;font-size:11px"> &nbsp;base AUC {base if base is not None else "?"}</span></div>'
            f'<div style="margin-top:4px">{"".join(items) or "<span style=color:#bbb;font-size:11px>nothing local</span>"}</div></div>')

    return (f'<div style="font-family:sans-serif;background:#f8fafc;border:1px solid #cbd5e1;border-radius:12px;'
            f'padding:12px 14px;margin:18px 0">'
            f'<div style="font-size:16px;font-weight:700">{esc(setname)}'
            f'<span style="color:#888;font-weight:400;font-size:12px"> &nbsp;{len(sibs)} siblings &middot; '
            f'{len(union)}/{L["rec"]["bank_size"]} bank metrics live anywhere</span></div>{pool_html}'
            f'<div style="margin:10px 0 4px;padding:8px 10px;background:#eafaf0;border:1px dashed #2a9d5c;'
            f'border-radius:8px"><span style="font-size:11px;font-weight:700;color:#1e7a46;letter-spacing:.05em">'
            f'SHARED (live in &ge;2 siblings)</span><div style="margin-top:4px">{shared_html}</div></div>'
            f'<div style="text-align:center;color:#94a3b8;font-size:14px;line-height:.8">&#9474;<br>'
            f'&#9484;{"&#9472;" * 30}&#9516;{"&#9472;" * 30}&#9488;</div>'
            f'<div style="display:flex;flex-wrap:wrap">{"".join(kids)}</div></div>')

for s in SETS:
    display(HTML(render_set(s)))""")

md("""## 3&nbsp;·&nbsp;The confirmed discoveries are all sibling-local

The gold pills above are the only invented metrics that survived **strictly-disjoint stage-2
replication** (fresh items, Bonferroni). Every one of them belongs to exactly one sub-community —
none is shared. These are the "invented keeps that are different per sibling":""")

code("""h = ['<table style="border-collapse:collapse;font:13px sans-serif">',
     '<tr style="border-bottom:2px solid #333"><th style="text-align:left;padding:4px 10px">invented metric (stage-2 KEPT)</th>'
     '<th style="text-align:left">sub-community</th><th>rep p_auc</th><th>rep bits</th></tr>']
seen = set()
keep_rows = []
for r in sorted((r for r in stage2 if r.get("stage2_status") == "KEPT"), key=lambda r: r.get("rep_p_auc") or 1):
    if (r["name"], r["leg"]) in seen: continue
    seen.add((r["name"], r["leg"])); keep_rows.append(r)
for r in keep_rows:
    h.append(f'<tr style="border-bottom:1px solid #ddd"><td style="text-align:left;padding:3px 10px">'
             f'&#9733; <b>{html.escape(r["name"])}</b></td>'
             f'<td style="text-align:left;padding:3px 10px">{html.escape(r["leg"])}</td>'
             f'<td align=center>{r["rep_p_auc"]:.2g}</td><td align=center>{r["rep_bits_gain"]:+.3f}</td></tr>')
h.append('</table>')
display(HTML("".join(h)))

print("\\nAnd the mirage for contrast — invented themes that RECURRED across siblings at stage-1:")
for s in SETS:
    for c in invented.get(s, {}).get("theme_shared_clusters", []):
        legs = [short(l) for l in c["legs"]]
        kept_in = sorted({sl for nm in c["names"] for sl in legs if (nm, sl) in kept2_pairs})
        if kept_in:
            verdict = f"KEPT in {', '.join(kept_in)} ONLY (single sibling) -- recurrence itself not confirmed"
        elif any((nm, sl) in repl2_pairs for nm in c["names"] for sl in legs):
            verdict = "replicated nominally in one sibling"
        else:
            verdict = "DIED at stage-2"
        print(f"  [{s}] across ({', '.join(legs)}): {c['names'][0][:60]}  ->  {verdict}")
print("\\npeer theory-rigor disjoint replication: 0/13 KEPT (3 nominal p<.05, all fail Bonferroni + bits)")""")

md("""## 4&nbsp;·&nbsp;Takeaways

1. **Explicit-metric sharing is a *domain property*, not a constant.** Code-review languages share
   most of their live vocabulary (any two languages overlap heavily — "static code analysis" is live
   in all five); press-release topics share widely but nothing is universal; peer-review
   (venues *and* subfields) and notice-and-comment topics share almost nothing because the bank is
   barely live there at all. Where the bank is alive, it is largely *shared*; where it is dead,
   siblings have nothing explicit in common.
2. **Confirmed invention is always local.** Four stage-2 keeps across the entire program — a
   counterexample-norm in topology, an alternative-method norm in series, a tone-subversion norm in
   meta-experimental CW, a punchline-structure norm in observational humor — and each lives in
   exactly one sub-community. Apparent cross-sibling recurrence (peer theory-rigor, 3 thematic
   families) was killed by disjoint replication (0/13).
3. **So the lattice is: shared explicit vocabulary (whose size varies by domain) + a thin layer of
   genuinely local invented norms.** "What do siblings have in common?" — reused *existing* metrics.
   "What must be invented?" — community-local norms, one per community, not transferable.
4. *Caveats*: liveness τ=.55 at n=600 is a lenient univariate screen; new axes (venue/language/
   topic) are class-balanced 50/50 while the older sets ran at natural rates; pooled-general
   "inventions" on the balanced axes are a manufactured base-rate confound (proven by within-venue
   collapse of the artifact-release keep), so pooled controls are only quotable for the
   natural-rate sets, where they invent 0.""")

import os
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT = "/Users/spangher/Projects/stanford-research/norm-research/notebooks/2026-07-11__sibling-metric-sharing-tree.ipynb"
client = NotebookClient(nb, timeout=300, resources={"metadata": {"path": os.path.dirname(OUT)}})
client.execute()
nbf.write(nb, OUT)
print("executed + written:", OUT)
