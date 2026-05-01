"""
Download all funded ERC (European Research Council) projects from CORDIS
under FP7 and Horizon 2020, enrich with ERC evaluation panel codes, and
produce a clean merged dataset suitable for panel-stratified clustering.

Data sources:
  - CORDIS bulk open data:    https://cordis.europa.eu/data/
  - ERC Datahub (panel codes): https://erc.easme-web.eu/json/list-N.json
    (snapshot of all signed ERC grants up to 28/3/2023; cert is expired,
    so we fetch with unverified SSL)

ERC funding schemes covered:
  FP7:   ERC-SG, ERC-CG, ERC-AG, ERC-SyG
  H2020: ERC-STG, ERC-COG, ERC-ADG, ERC-SyG, ERC-POC, ERC-POC-LS, ERC-LVG

Panel recovery strategy:
  1. FP7 exposes the panel directly in CORDIS topics.csv via codes like
     "ERC-SG-PE5", "ERC-AG-LS1". This is the primary FP7 source.
  2. H2020 does NOT expose the panel in CORDIS — only the call year and
     scheme. We pull it from the ERC Datahub's per-project `topic` field,
     which contains the panel code directly (e.g. "PE5", "LS1", "SH4").
  3. For consistency we also join the Datahub to FP7 and prefer the
     Datahub panel when both are present (usually identical).

Output:
  - erc_projects.csv       — one row per funded ERC project, with panel + domain
  - erc_euroscivoc.csv     — project_id → European Science Vocabulary codes/paths
  - raw/                   — original downloaded zips and Datahub JSON for provenance
"""

import argparse
import csv
import io
import json
import re
import shutil
import ssl
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

SOURCES = {
    "H2020": "https://cordis.europa.eu/data/cordis-h2020projects-csv.zip",
    "FP7":   "https://cordis.europa.eu/data/cordis-fp7projects-csv.zip",
}

# ERC Datahub list pages — 6 pages of beneficiaries, each containing
# projects[] arrays with panel info in the `topic` field.
DATAHUB_LIST_URL = "https://erc.easme-web.eu/json/list-{page}.json"
DATAHUB_PAGES = range(1, 7)  # 1..6; empirically verified 2026-04-10

# FP7 topics codes look like "ERC-SG-PE5", "ERC-AG-LS1", "ERC-CG-SH2", "ERC-SyG-LS1"
FP7_PANEL_RE = re.compile(r"^ERC-(?:SG|CG|AG|SyG|PoC)-(LS|PE|SH)(\d+)$")
# Datahub panel codes are bare: "PE5", "LS1", "SH4"
DH_PANEL_RE = re.compile(r"^(LS|PE|SH)(\d+)$")

# Stable, programme-agnostic column order for the merged output.
PROJECT_COLS = [
    "framework_programme",  # FP7 | H2020
    "id",                   # project id (grant agreement number)
    "acronym",
    "title",
    "status",
    "start_date",
    "end_date",
    "total_cost",
    "ec_max_contribution",
    "legal_basis",
    "topics",
    "framework",            # from the CSV's own frameworkProgramme (sanity check)
    "master_call",
    "sub_call",
    "funding_scheme",       # ERC-STG, ERC-COG, ...
    "objective",            # <-- the abstract we will cluster
    "grant_doi",
    "keywords",
    "panel",                # ERC evaluation panel (e.g. PE5, LS1, SH4); "" if unknown
    "domain",               # LS | PE | SH; "" if unknown
    "panel_source",         # cordis_topics | datahub | ""
]

_SSL_NOVERIFY = ssl._create_unverified_context()

def download(url: str, dest: Path, verify: bool = True) -> None:
    """Download via curl when available (robust to flaky SSL), else urllib.

    Verifies Content-Length when served; retries on partial transfers.
    """
    print(f"[download] {url}  ->  {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    if shutil.which("curl"):
        cmd = [
            "curl", "-fsS", "-L",
            "--retry", "5", "--retry-delay", "2",
            "--retry-connrefused", "--retry-all-errors",
            "--max-time", "300",
            "-A", "norm-research/erc-downloader",
            "-o", str(tmp), url,
        ]
        if not verify:
            cmd.insert(1, "-k")
        subprocess.run(cmd, check=True)
    else:
        req = Request(url, headers={"User-Agent": "norm-research/erc-downloader"})
        ctx = None if verify else _SSL_NOVERIFY
        with urlopen(req, context=ctx) as resp:
            expected = resp.headers.get("Content-Length")
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        if expected is not None and tmp.stat().st_size != int(expected):
            raise RuntimeError(
                f"short read: {tmp.stat().st_size} != {expected} for {url}"
            )

    tmp.replace(dest)
    print(f"[download] done: {dest.stat().st_size / 1e6:.1f} MB")

def iter_csv_from_zip(zip_path: Path, member: str):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
            yield from csv.DictReader(text, delimiter=";")

def parse_float_eu(s):
    """CORDIS uses European decimal commas: '1465603,75' -> 1465603.75"""
    if s is None or s == "":
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None

def project_row(programme: str, row: dict) -> dict:
    return {
        "framework_programme": programme,
        "id": row.get("id", ""),
        "acronym": row.get("acronym", ""),
        "title": row.get("title", ""),
        "status": row.get("status", ""),
        "start_date": row.get("startDate", ""),
        "end_date": row.get("endDate", ""),
        "total_cost": parse_float_eu(row.get("totalCost", "")),
        "ec_max_contribution": parse_float_eu(row.get("ecMaxContribution", "")),
        "legal_basis": row.get("legalBasis", ""),
        "topics": row.get("topics", ""),
        "framework": row.get("frameworkProgramme", ""),
        "master_call": row.get("masterCall", ""),
        "sub_call": row.get("subCall", ""),
        "funding_scheme": row.get("fundingScheme", ""),
        "objective": row.get("objective", ""),
        "grant_doi": row.get("grantDoi", ""),
        "keywords": row.get("keywords", ""),
        "panel": "",
        "domain": "",
        "panel_source": "",
    }

def is_erc(funding_scheme: str) -> bool:
    return bool(funding_scheme) and funding_scheme.startswith("ERC")

def load_fp7_panels(fp7_zip: Path) -> dict[str, str]:
    """Return {project_id: panel_code} from FP7 topics.csv ERC-XX-YYN codes."""
    pid_to_panel: dict[str, str] = {}
    with zipfile.ZipFile(fp7_zip) as zf:
        if "topics.csv" not in zf.namelist():
            return pid_to_panel
    for row in iter_csv_from_zip(fp7_zip, "topics.csv"):
        m = FP7_PANEL_RE.match(row.get("topic", "") or "")
        if m:
            pid_to_panel[row["projectID"]] = f"{m.group(1)}{m.group(2)}"
    return pid_to_panel

def fetch_datahub_panels(raw_dir: Path, skip_download: bool = False) -> dict[str, str]:
    """Return {project_number: panel_code} from the ERC Datahub list-N.json pages."""
    pid_to_panel: dict[str, str] = {}
    for page in DATAHUB_PAGES:
        dest = raw_dir / f"datahub_list_{page}.json"
        if skip_download and dest.exists():
            print(f"[datahub] skip (exists): {dest}")
        else:
            url = DATAHUB_LIST_URL.format(page=page)
            download(url, dest, verify=False)
        with open(dest, "rb") as f:
            data = json.load(f)
        for beneficiary in data.get("items", []):
            for proj in beneficiary.get("projects", []):
                num = proj.get("number")
                if not num:
                    continue
                if num in pid_to_panel:
                    continue
                m = DH_PANEL_RE.match(proj.get("topic", "") or "")
                if m:
                    pid_to_panel[num] = f"{m.group(1)}{m.group(2)}"
    return pid_to_panel

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/grant-funding/erc",
        help="Output directory (default: sk3 path)",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing zips in <out-dir>/raw if present",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download both programme zips
    zips: dict[str, Path] = {}
    for programme, url in SOURCES.items():
        dest = raw_dir / f"cordis-{programme.lower()}projects-csv.zip"
        if args.skip_download and dest.exists():
            print(f"[download] skip (exists): {dest}")
        else:
            download(url, dest)
        zips[programme] = dest

    # 2. Extract + filter projects to ERC only, merging programmes
    merged_rows: list[dict] = []
    erc_ids_by_prog: dict[str, set[str]] = {}

    for programme, zp in zips.items():
        print(f"\n[{programme}] scanning project.csv ...")
        total = 0
        erc_ids: set[str] = set()
        for row in iter_csv_from_zip(zp, "project.csv"):
            total += 1
            if is_erc(row.get("fundingScheme", "")):
                erc_ids.add(row["id"])
                merged_rows.append(project_row(programme, row))
        erc_ids_by_prog[programme] = erc_ids
        print(f"[{programme}] total projects: {total:,}   ERC: {len(erc_ids):,}")

    # Sanity: objectives non-empty
    with_obj = sum(1 for r in merged_rows if (r["objective"] or "").strip())
    print(f"\n[merged] total ERC projects: {len(merged_rows):,}")
    print(f"[merged] with non-empty objective: {with_obj:,}")

    # 2b. Enrich with panel codes from FP7 CORDIS topics.csv + ERC Datahub
    print("\n[panel] loading FP7 panel codes from CORDIS topics.csv ...")
    fp7_panels = load_fp7_panels(zips["FP7"])
    print(f"[panel] FP7 panels from CORDIS topics.csv: {len(fp7_panels):,}")

    print("[panel] loading panels from ERC Datahub ...")
    dh_panels = fetch_datahub_panels(raw_dir, skip_download=args.skip_download)
    print(f"[panel] Datahub panels (LS/PE/SH): {len(dh_panels):,}")

    panel_from = {"datahub": 0, "cordis_topics": 0, "none": 0}
    for r in merged_rows:
        pid = r["id"]
        if pid in dh_panels:
            r["panel"] = dh_panels[pid]
            r["panel_source"] = "datahub"
            panel_from["datahub"] += 1
        elif pid in fp7_panels:
            r["panel"] = fp7_panels[pid]
            r["panel_source"] = "cordis_topics"
            panel_from["cordis_topics"] += 1
        else:
            panel_from["none"] += 1
        if r["panel"]:
            r["domain"] = r["panel"][:2]  # LS | PE | SH

    total = len(merged_rows)
    covered = panel_from["datahub"] + panel_from["cordis_topics"]
    print(f"[panel] coverage: {covered:,}/{total:,} ({100*covered/total:.1f}%)")
    print(f"[panel]   from datahub:       {panel_from['datahub']:,}")
    print(f"[panel]   from cordis_topics: {panel_from['cordis_topics']:,}")
    print(f"[panel]   no panel assigned:  {panel_from['none']:,}")

    # 3. Write merged project-level CSV
    out_projects = out_dir / "erc_projects.csv"
    with open(out_projects, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROJECT_COLS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in merged_rows:
            w.writerow(r)
    print(f"[write]  {out_projects}  ({out_projects.stat().st_size / 1e6:.1f} MB)")

    # 4. Join euroSciVoc classifications for the ERC projects
    #    (useful as ground truth for cluster validation)
    out_sv = out_dir / "erc_euroscivoc.csv"
    all_erc_ids = set().union(*erc_ids_by_prog.values())
    sv_rows: list[dict] = []
    for programme, zp in zips.items():
        with zipfile.ZipFile(zp) as zf:
            if "euroSciVoc.csv" not in zf.namelist():
                continue
            erc_ids = erc_ids_by_prog[programme]
            for row in iter_csv_from_zip(zp, "euroSciVoc.csv"):
                pid = row.get("projectID") or row.get("projectId") or row.get("id", "")
                if pid in erc_ids:
                    sv_rows.append({
                        "framework_programme": programme,
                        "project_id": pid,
                        "eurosci_voc_code": row.get("euroSciVocCode", ""),
                        "eurosci_voc_path": row.get("euroSciVocPath", ""),
                        "eurosci_voc_title": row.get("euroSciVocTitle", ""),
                        "eurosci_voc_description": row.get("euroSciVocDescription", ""),
                    })
    with open(out_sv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "framework_programme", "project_id",
                "eurosci_voc_code", "eurosci_voc_path",
                "eurosci_voc_title", "eurosci_voc_description",
            ],
            quoting=csv.QUOTE_ALL,
        )
        w.writeheader()
        for r in sv_rows:
            w.writerow(r)
    print(f"[write]  {out_sv}  ({len(sv_rows):,} classification rows)")

    # 5. Per-scheme summary
    print("\n[summary] ERC funding schemes:")
    by_scheme: dict[tuple[str, str], int] = {}
    for r in merged_rows:
        key = (r["framework_programme"], r["funding_scheme"])
        by_scheme[key] = by_scheme.get(key, 0) + 1
    for (prog, scheme), n in sorted(by_scheme.items()):
        print(f"  {prog:6s} {scheme:12s} {n:6,}")

    # 6. Panel-level summary
    print("\n[summary] ERC panels (all programmes + schemes):")
    by_panel: dict[str, int] = {}
    for r in merged_rows:
        if r["panel"]:
            by_panel[r["panel"]] = by_panel.get(r["panel"], 0) + 1
    for domain in ("LS", "PE", "SH"):
        panels = sorted(
            (p for p in by_panel if p.startswith(domain)),
            key=lambda p: int(p[2:]),
        )
        if panels:
            total_dom = sum(by_panel[p] for p in panels)
            print(f"  {domain} ({total_dom:,} total):")
            for p in panels:
                print(f"    {p}: {by_panel[p]:,}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
