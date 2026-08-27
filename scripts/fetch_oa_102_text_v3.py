"""v3 §102 OA text fetcher — rotating uncompressed JSONL.

Differences from v2:
  - Writes to UNCOMPRESSED .jsonl files (no gzip = no concatenated-member corruption surface)
  - Rotates output every --rotate-apps apps (default 2000) → multiple small files
  - load_done() scans ALL parts in the output directory
  - Per-write fsync option (--strict) for crash durability

Output layout:
  {OUT_DIR}/office_actions_part_001.jsonl
  {OUT_DIR}/office_actions_part_002.jsonl
  ...

These files can be gzipped at rest by an external job after they're complete;
they're never appended to once full.
"""
import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
APP_IDS_TXT = f"{BASE}/processed/oa_102_app_ids_todo.txt"
KEY_FILE = "/lfs/skampere3/0/alexspan/.uspto-open-data-api-key.txt"
OUT_DIR = f"{BASE}/processed/office_actions_v3"

WANT_CODES = {"CTNF", "CTFR"}
PRINTABLE_RE = re.compile(rb"[\x20-\x7e\n\t]{20,}")
MIN_VALID_TEXT_LEN = 200
OCR_FALLBACK_THRESHOLD = 500


def get_key():
    return open(KEY_FILE).read().strip()


# ---- HTTP plumbing (same as v2) ----

def list_documents(session, app_num, key):
    url = f"https://api.uspto.gov/api/v1/patent/applications/{app_num}/documents"
    for attempt in range(5):
        try:
            r = session.get(url, headers={"X-API-KEY": key}, timeout=30)
            if r.status_code == 200:
                return r.json().get("documentBag", [])
            if r.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            return None
        except Exception:
            return None
    return None


def fetch_bytes(session, url, key):
    for attempt in range(5):
        try:
            r = session.get(url, headers={"X-API-KEY": key}, timeout=60, allow_redirects=False)
            if r.status_code == 200:
                if r.headers.get("content-type", "").startswith("application/json") and len(r.content) < 2000:
                    m = re.search(rb'https://[^ "\']+', r.content)
                    if m:
                        redir = m.group(0).decode(errors="ignore").rstrip(".")
                        for a2 in range(3):
                            try:
                                r2 = session.get(redir, timeout=60)
                                if r2.status_code == 200: return r2.content
                                if r2.status_code == 429:
                                    time.sleep(2 ** a2 + 1)
                                    continue
                                return None
                            except Exception:
                                return None
                        return None
                return r.content
            if r.status_code == 302:
                m = re.search(rb'https://[^ "\']+', r.content)
                if not m: return None
                redir = m.group(0).decode(errors="ignore").rstrip(".")
                for a2 in range(3):
                    try:
                        r2 = session.get(redir, timeout=60)
                        if r2.status_code == 200: return r2.content
                        if r2.status_code == 429:
                            time.sleep(2 ** a2 + 1)
                            continue
                        return None
                    except Exception:
                        return None
                return None
            if r.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            return None
        except Exception:
            return None
    return None


# ---- File-format extractors (same as v2) ----

def extract_docx(content):
    import docx2txt
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(content); path = f.name
    try:
        return docx2txt.process(path) or ""
    except Exception:
        try:
            from docx import Document
            d = Document(io.BytesIO(content))
            return "\n".join(p.text for p in d.paragraphs)
        except Exception:
            return ""
    finally:
        try: os.unlink(path)
        except: pass


def extract_doc(content):
    try:
        import olefile
        if olefile.isOleFile(io.BytesIO(content)):
            ole = olefile.OleFileIO(io.BytesIO(content))
            if ole.exists("WordDocument"):
                stream = ole.openstream("WordDocument").read()
            else:
                stream = content
            ole.close()
        else:
            stream = content
    except Exception:
        stream = content
    parts = []
    for m in PRINTABLE_RE.finditer(stream):
        try:
            s = m.group(0).decode("ascii", errors="ignore").strip()
            if len(s) > 20: parts.append(s)
        except Exception:
            pass
    return "\n".join(parts)


def extract_pdf(content, do_ocr=False):
    text = ""
    try:
        import fitz
        with fitz.open(stream=content, filetype="pdf") as pdf:
            text = "\n".join(page.get_text() for page in pdf)
    except Exception:
        return ""
    if len(text.strip()) >= OCR_FALLBACK_THRESHOLD or not do_ocr:
        return text
    try:
        import pytesseract, fitz
        ocr_parts = []
        with fitz.open(stream=content, filetype="pdf") as pdf:
            for page in pdf:
                pix = page.get_pixmap(dpi=200)
                from PIL import Image
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_parts.append(pytesseract.image_to_string(img))
        ocr_text = "\n".join(ocr_parts)
        if len(ocr_text.strip()) > len(text.strip()):
            return ocr_text
        return text
    except Exception:
        return text


def download_and_extract(session, doc, key, do_ocr=False):
    opts = {o["mimeTypeIdentifier"]: o for o in doc.get("downloadOptionBag", [])}
    text = ""
    if "MS_WORD" in opts:
        url = opts["MS_WORD"]["downloadUrl"]
        content = fetch_bytes(session, url, key)
        if content:
            is_docx = url.lower().endswith(".docx") or content[:4] == b"PK\x03\x04"
            text = extract_docx(content) if is_docx else extract_doc(content)
            if len(text.strip()) >= MIN_VALID_TEXT_LEN: return text
    if "PDF" in opts:
        url = opts["PDF"]["downloadUrl"]
        content = fetch_bytes(session, url, key)
        if content:
            text = extract_pdf(content, do_ocr=do_ocr)
            if len(text.strip()) >= MIN_VALID_TEXT_LEN: return text
    return text


def process_app(app_num, session, key, do_ocr=False):
    docs = list_documents(session, app_num, key)
    if not docs: return []
    out = []
    for d in docs:
        if d.get("documentCode") not in WANT_CODES:
            continue
        text = download_and_extract(session, d, key, do_ocr=do_ocr)
        if not text or len(text.strip()) < MIN_VALID_TEXT_LEN:
            continue
        out.append({
            "app_id": app_num,
            "ifw_number": d.get("documentIdentifier"),
            "document_code": d.get("documentCode"),
            "document_date": (d.get("officialDate") or "")[:10],
            "page_count": d.get("downloadOptionBag", [{}])[0].get("pageTotalQuantity"),
            "text": text[:50000],
        })
    return out


# ---- Multi-part output management ----

def load_done(out_dir):
    """Scan ALL part files (jsonl OR jsonl.gz) under out_dir; return set of app_ids."""
    done = set()
    paths = sorted(glob.glob(os.path.join(out_dir, "office_actions_part_*.jsonl"))) + \
            sorted(glob.glob(os.path.join(out_dir, "office_actions_part_*.jsonl.gz")))
    for p in paths:
        opener = (lambda x: __import__("gzip").open(x, "rt")) if p.endswith(".gz") else (lambda x: open(x))
        try:
            with opener(p) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if d.get("app_id"):
                            done.add(str(d["app_id"]))
                    except Exception:
                        pass
        except Exception as e:
            print(f"  (resume scan stopped on {os.path.basename(p)}: {type(e).__name__}: {e})",
                  flush=True)
    return done


def next_part_path(out_dir, max_apps_per_file, current_app_count_per_part,
                   prefix="office_actions_part_"):
    """Return path to write to. If current part has hit max_apps_per_file, roll over.

    prefix lets sharded processes (--shard-tag) keep disjoint part sequences
    in the SAME dir; the snapshot glob office_actions_part_*.jsonl matches all.
    """
    existing = sorted(glob.glob(os.path.join(out_dir, prefix + "[0-9]*.jsonl")))
    if not existing:
        return os.path.join(out_dir, f"{prefix}001.jsonl")
    last = existing[-1]
    # Count app_ids in last part (cheap: line count is a proxy)
    n = current_app_count_per_part.get(last, 0)
    if n >= max_apps_per_file:
        idx = int(re.search(r"_(\d+)\.jsonl$", last).group(1)) + 1
        return os.path.join(out_dir, f"{prefix}{idx:03d}.jsonl")
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ocr-fallback", action="store_true")
    p.add_argument("--rotate-apps", type=int, default=2000,
                   help="Roll over to new part file every N apps (default 2000)")
    p.add_argument("--strict", action="store_true",
                   help="fsync after every batch (crash-durable, slower)")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--todo", default=APP_IDS_TXT,
                   help="app_id list to fetch (one per line)")
    p.add_argument("--shard-tag", default=None,
                   help="e.g. s1: write office_actions_part_s1_NNN.jsonl so "
                        "multiple processes can share --out-dir (GIL workaround)")
    p.add_argument("--key-file", default=KEY_FILE,
                   help="USPTO ODP API key file. The /download endpoint is "
                        "rate-limited PER KEY (~4-5 req/s, silent 429+backoff), "
                        "so >1 process only helps with >1 key: one key per shard.")
    args = p.parse_args()
    part_prefix = ("office_actions_part_" if not args.shard_tag
                   else f"office_actions_part_{args.shard_tag}_")

    os.makedirs(args.out_dir, exist_ok=True)
    key = open(args.key_file).read().strip()
    print(f"Key length: {len(key)}", file=sys.stderr)

    print(f"Loading targets from {args.todo} ...", flush=True)
    with open(args.todo) as f:
        targets = [line.strip() for line in f if line.strip()]
    print(f"  {len(targets):,} app_ids in todo list", flush=True)

    print(f"Loading already-done app_ids from {args.out_dir} ...", flush=True)
    done = load_done(args.out_dir)
    print(f"  {len(done):,} already done", flush=True)
    todo = [a for a in targets if a not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"  {len(todo):,} to fetch", flush=True)

    n_ok = 0
    n_records = 0
    n_apps_in_part = {}  # path → app count
    t0 = time.time()
    session = requests.Session()

    cur_path = next_part_path(args.out_dir, args.rotate_apps, n_apps_in_part,
                                      prefix=part_prefix)
    cur_fh = open(cur_path, "a", buffering=1)  # line-buffered
    n_apps_in_part[cur_path] = 0
    print(f"Writing to: {cur_path}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(process_app, a, session, key, args.ocr_fallback): a for a in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                recs = fut.result()
            except Exception:
                recs = []
            for r in recs:
                cur_fh.write(json.dumps(r) + "\n")
                n_records += 1
            if recs:
                n_ok += 1
            n_apps_in_part[cur_path] = n_apps_in_part.get(cur_path, 0) + 1

            # Rotate when full
            if n_apps_in_part[cur_path] >= args.rotate_apps:
                cur_fh.flush()
                if args.strict:
                    os.fsync(cur_fh.fileno())
                cur_fh.close()
                cur_path = next_part_path(args.out_dir, args.rotate_apps, n_apps_in_part,
                                      prefix=part_prefix)
                cur_fh = open(cur_path, "a", buffering=1)
                n_apps_in_part[cur_path] = 0
                print(f"  ROTATED to: {cur_path}", flush=True)

            if i % 50 == 0:
                cur_fh.flush()
                if args.strict:
                    os.fsync(cur_fh.fileno())
                rate = i / max(1, time.time() - t0)
                eta_h = (len(todo) - i) / max(rate, 1e-3) / 3600
                pct_ok = 100 * n_ok / i
                print(f"  {i:,}/{len(todo):,}  apps_ok={n_ok:,} ({pct_ok:.0f}%)  "
                      f"oa_recs={n_records:,}  {rate:.1f} app/s  ETA {eta_h:.1f}h  "
                      f"part={os.path.basename(cur_path)}", flush=True)

    cur_fh.flush()
    if args.strict:
        os.fsync(cur_fh.fileno())
    cur_fh.close()
    print(f"\nDone. {n_ok:,} apps, {n_records:,} OA records.")


if __name__ == "__main__":
    main()
