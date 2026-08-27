#!/usr/bin/env python3
"""Assemble the distributable N&C explicit-metrics packet.

Collects the three explicit-metric families from the 2026-07 VAT campaign into one
self-contained directory, with provenance (which agency documents each rubric was
mined from) and univariate performance stats. Re-runnable; overwrites its own outputs.

Inputs (repo-relative): v4/nc_rubrics.jsonl, v4/gepa_nc/bank_best.jsonl,
v4/nc_scores_shard{0..4}.npz, v4/nc_deepv3_scores.npz, v4/nc_vat_sample.jsonl,
online-rubrics/urls-visited.csv, outputs/hierarchy/notice-and-comment_{general,specific}_r2_expanded.json,
methods/metric_seam/hybrids/{ops.py,programs_notice_and_comment/}.
"""
import json, csv, re, shutil, collections
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

PKT = Path(__file__).resolve().parent
NC = PKT.parent                      # datasets/notice-and-comment
ROOT = NC.parent.parent              # repo root

# ---------------- provenance: leaf -> source doc -> agency ----------------
fn2url = {r["filename"]: r["url"] for r in csv.DictReader(open(NC/"online-rubrics/urls-visited.csv"))}

def real_url(u):
    m = re.search(r"web\.archive\.org/web/\d+/(.*)", u)
    return m.group(1) if m else u

DOM2AG = {"epa.gov":"EPA","faa.gov":"FAA","fda.gov":"FDA","cms.gov":"CMS","fws.gov":"FWS",
 "aphis.usda.gov":"APHIS","blm.gov":"BLM","noaa.gov":"NOAA","fisheries.noaa.gov":"NOAA",
 "nhtsa.gov":"NHTSA","transportation.gov":"DOT","irs.gov":"IRS","cdc.gov":"CDC","ed.gov":"ED",
 "fema.gov":"FEMA","uscis.gov":"USCIS","cbp.gov":"USCBP","hhs.gov":"HHS","acf.hhs.gov":"HHS",
 "fcc.gov":"FCC","sec.gov":"SEC","ferc.gov":"FERC","osha.gov":"OSHA","ftc.gov":"FTC","bia.gov":"BIA"}
CROSSCUT = {"whitehouse.gov":"OMB","obamawhitehouse.archives.gov":"OMB",
 "bidenwhitehouse.archives.gov":"OMB","reginfo.gov":"OMB","acus.gov":"ACUS",
 "advocacy.sba.gov":"SBA-Advocacy","regulations.gov":"regs.gov","archives.gov":"NARA",
 "ecfr.gov":"eCFR","gao.gov":"GAO"}
SLUG2AG = {"environmental-protection-agency":"EPA","federal-aviation-administration":"FAA",
 "food-and-drug-administration":"FDA","fish-and-wildlife-service":"FWS",
 "national-oceanic-and-atmospheric-administration":"NOAA",
 "national-highway-traffic-safety-administration":"NHTSA",
 "securities-and-exchange-commission":"SEC","federal-communications-commission":"FCC",
 "health-and-human-services-department":"HHS","homeland-security-department":"DHS",
 "transportation-department":"DOT","labor-department":"DOL","education-department":"ED",
 "federal-trade-commission":"FTC","nuclear-regulatory-commission":"NRC",
 "energy-department":"DOE","federal-energy-regulatory-commission":"FERC",
 "agriculture-department":"USDA","executive-office-of-the-president":"OMB",
 "consumer-financial-protection-bureau":"CFPB","justice-department":"DOJ",
 "housing-and-urban-development-department":"HUD",
 "federal-motor-carrier-safety-administration":"FMCSA","international-trade-commission":"ITC",
 "centers-for-medicare-medicaid-services":"CMS","us-citizenship-and-immigration-services":"USCIS",
 "internal-revenue-service":"IRS","forest-service":"FS","land-management-bureau":"BLM",
 "animal-and-plant-health-inspection-service":"APHIS",
 "food-safety-and-inspection-service":"FSIS","agricultural-marketing-service":"AMS",
 "mine-safety-and-health-administration":"MSHA","federal-emergency-management-agency":"FEMA",
 "centers-for-disease-control-and-prevention":"CDC"}

_fr_slug_cache = {}
FR_SLUG_PAT = re.compile(r'federalregister\.gov/agencies/([a-z0-9-]+)|href="/agencies/([a-z0-9-]+)"')

def fr_agency_slug(fn):
    if fn in _fr_slug_cache:
        return _fr_slug_cache[fn]
    p = NC/"online-rubrics/raw"/fn
    slug = None
    if p.exists():
        m = FR_SLUG_PAT.search(p.read_text(errors="ignore"))
        if m:
            slug = m.group(1) or m.group(2)
    _fr_slug_cache[fn] = slug
    return slug

_md_url_cache = {}
def md_frontmatter_url(fn):
    if fn not in _md_url_cache:
        _md_url_cache[fn] = None
        p = NC/"online-rubrics/claude-parsed"/fn
        if p.exists():
            m = re.search(r'^(?:source_)?url:\s*"?([^"\n]+)"?', p.read_text(errors="ignore")[:2000], re.M)
            if m:
                _md_url_cache[fn] = m.group(1).strip()
    return _md_url_cache[fn]

def leaf_source(fn):
    url = fn2url.get(fn) or (md_frontmatter_url(fn) if fn.endswith(".md") else None)
    if url is None:
        return dict(file=fn, url=None, kind="unknown", agency=None)
    u = real_url(url)
    # archive.org-unwrapped targets can be scheme-less ("whitehouse.gov/...")
    m = re.match(r"(?:https?://)?([^/]+)", u.lower())
    host = (m.group(1) if m else u.lower()).replace("www.", "")
    if "federalregister.gov" in host:
        slug = fr_agency_slug(fn)
        ag = SLUG2AG.get(slug, ("FR:" + slug) if slug else None)
        return dict(file=fn, url=u, kind="agency-FR-doc" if ag else "FR-unattributed", agency=ag)
    for d, a in sorted(DOM2AG.items(), key=lambda x: -len(x[0])):
        if host == d or host.endswith("." + d):
            return dict(file=fn, url=u, kind="agency-site", agency=a)
    for d, a in sorted(CROSSCUT.items(), key=lambda x: -len(x[0])):
        if host == d or host.endswith("." + d):
            return dict(file=fn, url=u, kind="crosscut-gov", agency=a)
    if host.endswith(".gov"):
        return dict(file=fn, url=u, kind="crosscut-gov", agency="othergov:" + host)
    kind = "academic" if (host.endswith(".edu") or "ssrn" in host or "jstor" in host) else "nongov"
    return dict(file=fn, url=u, kind=kind, agency=None)

def load_groups(bucket):
    d = json.load(open(ROOT/f"outputs/hierarchy/notice-and-comment_{bucket}_r2_expanded.json"))
    return {g["merged_name"]: g for g in d["merged_groups"]}

groups = {"general_r2_expanded": load_groups("general"),
          "specific_r2_expanded": load_groups("specific")}

# ---------------- univariate stats (pre-GEPA Gemma scores, outcome-y) ----------------
mats, doc_ids = [], []
for i in range(5):
    z = np.load(NC/f"v4/nc_scores_shard{i}.npz", allow_pickle=True)
    mats.append(z["X"]); doc_ids.extend([str(x) for x in z["doc_id"]])
    a_names = [str(x) for x in z["a_names"]]
A = np.vstack(mats).astype(float)
id2row = {d: i for i, d in enumerate(doc_ids)}
sample = [json.loads(l) for l in open(NC/"v4/nc_vat_sample.jsonl")]
rows, ys = zip(*[(id2row[str(s["doc_id"])], int(s["y_out"])) for s in sample
                 if s.get("y_out") is not None and str(s["doc_id"]) in id2row])
ys = np.array(ys); M = A[np.array(rows), :]
name2col = {n: i for i, n in enumerate(a_names)}

def uni_stats(col):
    x = M[:, col]
    m = ~np.isnan(x)
    st = dict(na_rate=round(float(1 - m.mean()), 3), n_applicable=int(m.sum()))
    if m.sum() >= 200 and len(set(ys[m])) >= 2:
        st["univariate_auc_outcome_y"] = round(float(roc_auc_score(ys[m], x[m])), 3)
    else:
        st["univariate_auc_outcome_y"] = None
    return st

# ---------------- A-bank: enriched rubrics.jsonl ----------------
bank = [json.loads(l) for l in open(NC/"v4/nc_rubrics.jsonl")]
gepa = {r["rubric_id"]: r for r in (json.loads(l) for l in open(NC/"v4/gepa_nc/bank_best.jsonl"))}

(PKT/"a_rubrics").mkdir(parents=True, exist_ok=True)
used_files = {}
out_rub, out_gepa = [], []
for r in bank:
    g = groups[r["name_source"]][r["name"]]
    srcs, ags = [], collections.Counter()
    seen = set()
    for leaf in g["all_leaves"]:
        fn = leaf["key"].split("::")[2]
        if fn in seen:
            continue
        seen.add(fn)
        s = leaf_source(fn)
        used_files[fn] = s
        srcs.append(s)
        if s["kind"].startswith("agency") and s["agency"]:
            ags[s["agency"]] += 1
    n_ag = len(ags)
    rec = dict(rubric_id=r["rubric_id"], name=r["name"], description=r["description"],
               bucket=("general" if r["name_source"].startswith("general") else "specific"),
               n_leaf_metrics=g["total_leaf_rubrics"],
               n_source_documents=len(srcs),
               agencies=sorted(ags),
               n_distinct_agencies=n_ag,
               provenance_class=("multi-agency" if n_ag >= 2 else
                                 "single-agency" if n_ag == 1 else "no-agency-doc"),
               source_documents=srcs)
    rec.update(uni_stats(name2col[r["name"]]))
    out_rub.append(rec)
    gr = gepa.get(r["rubric_id"])
    if gr:
        out_gepa.append(dict(rubric_id=r["rubric_id"], name=gr.get("name", r["name"]),
                             description=gr["description"],
                             note="GEPA fidelity-optimized variant; see README quoting rule"))

# ---------------- leaf criteria: the original per-source statements ----------------
_parse_cache = {}
def parsed_metrics(fn):
    if fn not in _parse_cache:
        base = NC/"online-rubrics/gpt-parsed/gpt-5-mini"
        _parse_cache[fn] = None
        for cand in (base/f"raw__{fn}.json", base/f"claude-parsed__{fn}.json"):
            try:
                _parse_cache[fn] = json.load(open(cand))["extracted"]["rubrics_metrics"]
                break
            except Exception:
                continue
    return _parse_cache[fn]

n_leaf_ok = n_leaf_miss = 0
with open(PKT/"a_rubrics/leaf_criteria.jsonl", "w") as f:
    for r in bank:
        g = groups[r["name_source"]][r["name"]]
        leaves = []
        for leaf in g["all_leaves"]:
            parts = leaf["key"].split("::")
            fn, idx = parts[2], int(parts[3])
            src = used_files.get(fn) or leaf_source(fn)
            rm = parsed_metrics(fn)
            item = dict(source_file=fn, url=src["url"], kind=src["kind"], agency=src["agency"],
                        name=leaf["name"])
            if rm is not None and idx < len(rm):
                item["original_description"] = rm[idx].get("description")
                if rm[idx].get("guidance"):
                    item["original_guidance"] = rm[idx]["guidance"]
                n_leaf_ok += 1
            else:
                n_leaf_miss += 1
            leaves.append(item)
        f.write(json.dumps(dict(rubric_id=r["rubric_id"], name=r["name"], leaves=leaves),
                           ensure_ascii=False) + "\n")
print(f"leaf criteria: {n_leaf_ok} with original text, {n_leaf_miss} name-only")

with open(PKT/"a_rubrics/rubrics.jsonl", "w") as f:
    for r in out_rub:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(PKT/"a_rubrics/rubrics_gepa_optimized.jsonl", "w") as f:
    for r in out_gepa:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(PKT/"a_rubrics/sources.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "kind", "agency", "url"])
    for fn, s in sorted(used_files.items()):
        w.writerow([fn, s["kind"], s["agency"] or "", s["url"] or ""])

# ---------------- V_deep programs ----------------
prog_src = ROOT/"methods/metric_seam/hybrids/programs_notice_and_comment"
prog_dst = PKT/"v_deep_programs"
prog_dst.mkdir(exist_ok=True)
shutil.copy(ROOT/"methods/metric_seam/hybrids/ops.py", prog_dst/"ops.py")
progs = sorted(p for p in prog_src.glob("*_h[012].py"))
for p in progs:
    shutil.copy(p, prog_dst/p.name)
shutil.copy(NC/"v4/cfr_parts_index.json.gz", prog_dst/"cfr_parts_index.json.gz")

# make authority_lookup find the eCFR index next to itself when distributed
al = prog_dst/"authority_lookup_h2.py"
src = al.read_text()
old_path = ('_INDEX_PATH = (pathlib.Path(__file__).resolve().parents[4]\n'
            '               / "datasets" / "notice-and-comment" / "v4" / "cfr_parts_index.json.gz")')
new_path = ('_INDEX_PATH = pathlib.Path(__file__).resolve().parent / "cfr_parts_index.json.gz"\n'
            'if not _INDEX_PATH.exists():  # fall back to the source-repo location\n'
            '    _INDEX_PATH = (pathlib.Path(__file__).resolve().parents[4]\n'
            '                   / "datasets" / "notice-and-comment" / "v4" / "cfr_parts_index.json.gz")')
if old_path in src:
    al.write_text(src.replace(old_path, new_path))

# deep-V univariate stats
z = np.load(NC/"v4/nc_deepv3_scores.npz", allow_pickle=True)
dX = z["X"].astype(float)
dnames = [str(x) for x in z["names"]]
did2row = {str(d): i for i, d in enumerate([str(x) for x in z["doc_id"]])}
drows, dys = zip(*[(did2row[str(s["doc_id"])], int(s["y_out"])) for s in sample
                   if s.get("y_out") is not None and str(s["doc_id"]) in did2row])
dys = np.array(dys); dM = dX[np.array(drows), :]
deep_auc = {}
for j, n in enumerate(dnames):
    x = dM[:, j]; m = ~np.isnan(x)
    deep_auc[n] = round(float(roc_auc_score(dys[m], x[m])), 3) if m.sum() >= 200 and len(set(dys[m])) >= 2 else None

# ---------------- V regex extractor (standalone copy) ----------------
vdir = PKT/"v_regex"; vdir.mkdir(exist_ok=True)
agg = (NC/"v4/aggregate_vat_nc.py").read_text()
mstart = agg.index("SENT_RE") if "SENT_RE" in agg else agg.index("KW = {")
mend = agg.index("# ---------------- grouped CV AUC")
header = ('#!/usr/bin/env python3\n"""Standalone copy of the 27 N&C V (verifiability) regex features.\n'
          'Extracted verbatim from v4/aggregate_vat_nc.py (2026-07 VAT campaign).\n'
          'Usage: from v_features import v_features, V_NAMES; feats = v_features(comment_text)\n"""\n'
          "import re\n\n")
pre = agg[:mstart]
imports = "\n".join(l for l in pre.splitlines() if l.startswith("NUMTOK") or l.startswith("SENT_RE"))
(vdir/"v_features.py").write_text(header + imports + "\n" + agg[mstart:mend])

# V regex univariate AUCs
import importlib.util
spec = importlib.util.spec_from_file_location("vfeat", vdir/"v_features.py")
vf = importlib.util.module_from_spec(spec); spec.loader.exec_module(vf)
vmat = np.array([[f[n] for n in vf.V_NAMES]
                 for f in (vf.v_features(s["text"]) for s in sample if s.get("y_out") is not None)])
vys = np.array([int(s["y_out"]) for s in sample if s.get("y_out") is not None])
vreg_auc = {n: round(float(roc_auc_score(vys, vmat[:, j])), 3) for j, n in enumerate(vf.V_NAMES)}

# ---------------- performance_summary.csv ----------------
with open(PKT/"performance_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "metric", "univariate_auc_outcome_y", "na_rate", "n_applicable",
                "bucket", "provenance_class", "agencies"])
    for r in out_rub:
        w.writerow(["A_rubric", r["name"], r["univariate_auc_outcome_y"], r["na_rate"],
                    r["n_applicable"], r["bucket"], r["provenance_class"], ";".join(r["agencies"])])
    for n, a in vreg_auc.items():
        w.writerow(["V_regex", n, a, 0.0, len(vys), "", "", ""])
    for n, a in deep_auc.items():
        w.writerow(["V_deep_program", n, a, "", int((~np.isnan(dM[:, dnames.index(n)])).sum()), "", "", ""])

print(f"rubrics: {len(out_rub)}  gepa variants: {len(out_gepa)}  source docs: {len(used_files)}")
print(f"programs copied: {len(progs)}  V regex feats: {len(vf.V_NAMES)}  deep names: {len(dnames)}")
