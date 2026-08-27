"""
Tier 2 static V-layer: deterministic features parsed from unified diff text.
No checkout, no tools, no environment. Runs on sk3 over datasets/code-review/diffs/.

Usage:
  python pr_tier2_diff_features.py validate          # 30-diff validation sample w/ cross-checks
  python pr_tier2_diff_features.py run [--pool PATH] # full pool extraction -> parquet
  python pr_tier2_diff_features.py full              # full 140K corpus

Features per PR (all deterministic):
  file-level:   n_files, n_new_files, n_deleted_files, n_renamed_files, n_binary_files,
                new_file_ratio, n_generated_files, generated_file_ratio, n_distinct_exts,
                top_ext_frac
  loc by type:  loc_{added,removed}_{source,test,docs,config,generated,other}
  hunks:        n_hunks, hunks_per_file, max_hunk_added, mean_hunk_added
  comments:     comment_lines_added, comment_lines_removed, comment_delta,
                comment_added_ratio (comment_added / loc_added)
  todo:         todo_added, todo_removed, todo_delta
  imports:      import_lines_added, import_lines_removed, import_delta
  totals:       loc_added, loc_removed, loc_total, add_del_ratio
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
DIFF_DIR = ROOT / "datasets/code-review/diffs"
OUT = ROOT / "outputs/v2_analysis"

# ---------------- file classification ----------------

GENERATED_BASENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "go.sum", "cargo.lock",
    "poetry.lock", "pipfile.lock", "composer.lock", "gemfile.lock", "mix.lock",
    "packages.lock.json", "gradle.lockfile", "flake.lock", "uv.lock",
}
GENERATED_PATTERNS = [
    re.compile(r"\.min\.(js|css)$"),
    re.compile(r"_pb2(_grpc)?\.py$"),
    re.compile(r"\.pb\.(go|cc|h)$"),
    re.compile(r"\.g\.(cs|dart)$"),
    re.compile(r"(^|/)(vendor|node_modules|third_party|thirdparty)/"),
    re.compile(r"^dist/"),  # only repo-root dist/; nested .../dist/... is often a source module (e.g. hadoop-ozone/dist)
    re.compile(r"\.snap$"),
    re.compile(r"(^|/)__snapshots__/"),
    re.compile(r"\.generated\.\w+$"),
    re.compile(r"(^|/)generated/"),
]
TEST_PATTERNS = [
    re.compile(r"(^|/)tests?/"),
    re.compile(r"(^|/)testing/"),
    re.compile(r"(^|/)spec/"),
    re.compile(r"(^|/)__tests__/"),
    re.compile(r"(^|/)test_[^/]+$"),
    re.compile(r"[^/]+_test\.\w+$"),
    re.compile(r"[^/]+\.test\.\w+$"),
    re.compile(r"[^/]+\.spec\.\w+$"),
    re.compile(r"[^/]+Test\.(java|kt|scala|groovy)$"),
    re.compile(r"[^/]+Tests\.(java|kt|scala|cs)$"),
    re.compile(r"[^/]+IT\.java$"),
    re.compile(r"(^|/)conftest\.py$"),
]
DOCS_EXTS = {".md", ".rst", ".txt", ".adoc", ".tex", ".rdoc"}
DOCS_PATTERNS = [re.compile(r"(^|/)docs?/"), re.compile(r"(^|/)documentation/"),
                 re.compile(r"(^|/)(README|CHANGELOG|CONTRIBUTING|LICENSE|AUTHORS|NOTICE|HISTORY)[^/]*$", re.I)]
CONFIG_EXTS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".properties",
               ".xml", ".gradle", ".env", ".editorconfig", ".lock"}
CONFIG_BASENAMES = {"makefile", "dockerfile", "setup.py", "setup.cfg", "pyproject.toml",
                    "pom.xml", "build.gradle", "package.json", "tsconfig.json", ".gitignore",
                    ".gitattributes", "requirements.txt", "go.mod", "cmakelists.txt"}
SOURCE_EXTS = {".py", ".java", ".go", ".js", ".jsx", ".ts", ".tsx", ".c", ".h", ".cc",
               ".cpp", ".hpp", ".cs", ".rb", ".php", ".rs", ".kt", ".scala", ".swift",
               ".m", ".mm", ".sh", ".bash", ".pl", ".lua", ".sql", ".r", ".jl", ".groovy",
               ".dart", ".ex", ".exs", ".erl", ".hs", ".clj", ".vue", ".svelte", ".proto"}

# comment syntax by extension: (line_prefixes, block_markers)
HASH = ("#",)
SLASH = ("//",)
COMMENT_SYNTAX = {
    ".py": HASH, ".sh": HASH, ".bash": HASH, ".rb": HASH, ".pl": HASH, ".r": HASH,
    ".yaml": HASH, ".yml": HASH, ".toml": HASH, ".cfg": HASH, ".ini": HASH,
    ".ex": HASH, ".exs": HASH, ".jl": HASH, ".properties": HASH, ".conf": HASH,
    ".java": SLASH, ".go": SLASH, ".js": SLASH, ".jsx": SLASH, ".ts": SLASH, ".tsx": SLASH,
    ".c": SLASH, ".h": SLASH, ".cc": SLASH, ".cpp": SLASH, ".hpp": SLASH, ".cs": SLASH,
    ".rs": SLASH, ".kt": SLASH, ".scala": SLASH, ".swift": SLASH, ".php": SLASH,
    ".groovy": SLASH, ".dart": SLASH, ".proto": SLASH, ".m": SLASH, ".mm": SLASH,
    ".sql": ("--",), ".lua": ("--",), ".hs": ("--",),
}
# block-comment continuation: lines starting with * (inside /* */) or /* or */
BLOCK_COMMENT_RE = re.compile(r"^\s*(/\*|\*|\*/|<!--)")
TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
IMPORT_RES = {
    ".py": re.compile(r"^\s*(import\s+\w|from\s+[\w.]+\s+import\b)"),
    ".java": re.compile(r"^\s*import\s+[\w.]+"),
    ".go": re.compile(r'^\s*(import\s|\t?"[\w./-]+"$)'),  # import block lines handled loosely
    ".js": re.compile(r"^\s*(import\s|const\s+.*=\s*require\(|var\s+.*=\s*require\(|let\s+.*=\s*require\()"),
    ".jsx": re.compile(r"^\s*import\s"),
    ".ts": re.compile(r"^\s*import\s"),
    ".tsx": re.compile(r"^\s*import\s"),
    ".rb": re.compile(r"^\s*require(_relative)?\s"),
    ".rs": re.compile(r"^\s*use\s+[\w:]+"),
    ".kt": re.compile(r"^\s*import\s+[\w.]+"),
    ".scala": re.compile(r"^\s*import\s+[\w.]+"),
    ".php": re.compile(r"^\s*(use\s+[\w\\]+|require(_once)?\s|include(_once)?\s)"),
    ".cs": re.compile(r"^\s*using\s+[\w.]+;"),
}


def classify_file(path: str) -> str:
    """Order: generated > test > docs > config > source > other."""
    p = path.lower()
    base = p.rsplit("/", 1)[-1]
    ext = os.path.splitext(base)[1]
    if base in GENERATED_BASENAMES or any(r.search(p) for r in GENERATED_PATTERNS):
        return "generated"
    if any(r.search(p) for r in TEST_PATTERNS):
        return "test"
    if ext in DOCS_EXTS or any(r.search(p) for r in DOCS_PATTERNS):
        return "docs"
    if base in CONFIG_BASENAMES or ext in CONFIG_EXTS:
        return "config"
    if ext in SOURCE_EXTS:
        return "source"
    return "other"


def is_comment_line(content: str, ext: str) -> bool:
    s = content.strip()
    if not s:
        return False
    prefixes = COMMENT_SYNTAX.get(ext)
    if prefixes and s.startswith(prefixes):
        return True
    if ext in (".java", ".go", ".js", ".jsx", ".ts", ".tsx", ".c", ".h", ".cc", ".cpp",
               ".hpp", ".cs", ".rs", ".kt", ".scala", ".swift", ".php", ".groovy",
               ".dart", ".proto", ".css", ".m", ".mm"):
        if BLOCK_COMMENT_RE.match(s):
            return True
    if ext in (".html", ".xml", ".md", ".vue") and s.startswith("<!--"):
        return True
    if ext == ".py" and (s.startswith('"""') or s.startswith("'''")):
        return True
    return False


FILE_HEADER_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")

CATS = ["source", "test", "docs", "config", "generated", "other"]


def parse_diff(text: str) -> dict:
    f = {}
    n_files = n_new = n_deleted = n_renamed = n_binary = n_generated = 0
    exts = Counter()
    loc_added = {c: 0 for c in CATS}
    loc_removed = {c: 0 for c in CATS}
    comment_added = comment_removed = 0
    todo_added = todo_removed = 0
    import_added = import_removed = 0
    n_hunks = 0
    hunk_added_counts = []
    cur_hunk_added = 0
    in_hunk = False
    cat = "other"
    ext = ""
    cur_path = None
    saw_minus_header = False  # to avoid counting '--- a/x' as removal

    def close_hunk():
        nonlocal cur_hunk_added, in_hunk
        if in_hunk:
            hunk_added_counts.append(cur_hunk_added)
        cur_hunk_added = 0

    for line in text.splitlines():
        m = FILE_HEADER_RE.match(line)
        if m:
            close_hunk()
            in_hunk = False
            n_files += 1
            a_path, b_path = m.group(1), m.group(2)
            cur_path = b_path if b_path != "/dev/null" else a_path
            if a_path != b_path:
                n_renamed += 1
            cat = classify_file(cur_path)
            if cat == "generated":
                n_generated += 1
            ext = os.path.splitext(cur_path.lower())[1]
            exts[ext or "<none>"] += 1
            continue
        if line.startswith("new file mode"):
            n_new += 1
            continue
        if line.startswith("deleted file mode"):
            n_deleted += 1
            continue
        if line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            n_binary += 1
            continue
        if HUNK_RE.match(line):
            close_hunk()
            in_hunk = True
            n_hunks += 1
            continue
        if not in_hunk:
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            content = line[1:]
            loc_added[cat] += 1
            cur_hunk_added += 1
            if TODO_RE.search(content):
                todo_added += 1
            if is_comment_line(content, ext):
                comment_added += 1
            ir = IMPORT_RES.get(ext)
            if ir and ir.match(content):
                import_added += 1
        elif line.startswith("-"):
            content = line[1:]
            loc_removed[cat] += 1
            if TODO_RE.search(content):
                todo_removed += 1
            if is_comment_line(content, ext):
                comment_removed += 1
            ir = IMPORT_RES.get(ext)
            if ir and ir.match(content):
                import_removed += 1
    close_hunk()

    tot_added = sum(loc_added.values())
    tot_removed = sum(loc_removed.values())
    f["t2_n_files"] = n_files
    f["t2_n_new_files"] = n_new
    f["t2_n_deleted_files"] = n_deleted
    f["t2_n_renamed_files"] = n_renamed
    f["t2_n_binary_files"] = n_binary
    f["t2_new_file_ratio"] = n_new / n_files if n_files else 0.0
    f["t2_n_generated_files"] = n_generated
    f["t2_generated_file_ratio"] = n_generated / n_files if n_files else 0.0
    f["t2_n_distinct_exts"] = len(exts)
    f["t2_top_ext_frac"] = (exts.most_common(1)[0][1] / n_files) if n_files else 0.0
    for c in CATS:
        f[f"t2_loc_added_{c}"] = loc_added[c]
        f[f"t2_loc_removed_{c}"] = loc_removed[c]
    f["t2_loc_added"] = tot_added
    f["t2_loc_removed"] = tot_removed
    f["t2_loc_total"] = tot_added + tot_removed
    f["t2_add_del_ratio"] = tot_added / (tot_removed + 1)
    f["t2_n_hunks"] = n_hunks
    f["t2_hunks_per_file"] = n_hunks / n_files if n_files else 0.0
    f["t2_max_hunk_added"] = max(hunk_added_counts) if hunk_added_counts else 0
    f["t2_mean_hunk_added"] = (sum(hunk_added_counts) / len(hunk_added_counts)) if hunk_added_counts else 0.0
    f["t2_comment_lines_added"] = comment_added
    f["t2_comment_lines_removed"] = comment_removed
    f["t2_comment_delta"] = comment_added - comment_removed
    f["t2_comment_added_ratio"] = comment_added / (tot_added + 1)
    f["t2_todo_added"] = todo_added
    f["t2_todo_removed"] = todo_removed
    f["t2_todo_delta"] = todo_added - todo_removed
    f["t2_import_lines_added"] = import_added
    f["t2_import_lines_removed"] = import_removed
    f["t2_import_delta"] = import_added - import_removed
    f["t2_test_loc_frac"] = (loc_added["test"] + loc_removed["test"]) / (tot_added + tot_removed + 1)
    f["t2_docs_loc_frac"] = (loc_added["docs"] + loc_removed["docs"]) / (tot_added + tot_removed + 1)
    return f


def diff_path(owner, repo, pr_number):
    return DIFF_DIR / f"{owner}__{repo}__{pr_number}.diff"


def read_diff(p, cap_bytes=50_000_000):
    try:
        raw = open(p, "rb").read(cap_bytes)
        return raw.decode("utf-8", errors="replace")
    except OSError:
        return None


def cmd_validate():
    """Sample 30 diffs stratified by language; print features + independent cross-checks."""
    import pandas as pd
    pool = pd.read_parquet(OUT / "pr_stage0_pool.parquet")
    rng_sample = []
    for lang, k in [("Python", 8), ("Java", 7), ("Go", 6), ("JavaScript", 3),
                    ("TypeScript", 3), ("Other", 3)]:
        sub = pool[pool.file_language == lang]
        rng_sample.append(sub.sample(min(k, len(sub)), random_state=7))
    sample = pd.concat(rng_sample)
    print(f"validation sample: {len(sample)}")
    for _, r in sample.iterrows():
        p = diff_path(r.owner, r.repo, r.pr_number)
        text = read_diff(p)
        if text is None:
            print(f"!! missing {p}")
            continue
        feats = parse_diff(text)
        lines = text.splitlines()
        # independent cross-checks with dumb greps
        xc_files = sum(1 for l in lines if l.startswith("diff --git "))
        xc_hunks = sum(1 for l in lines if l.startswith("@@ -"))
        xc_plus = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        xc_minus = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
        xc_new = sum(1 for l in lines if l.startswith("new file mode"))
        ok_files = xc_files == feats["t2_n_files"]
        ok_hunks = xc_hunks >= feats["t2_n_hunks"]  # @@ can appear in content lines too
        ok_new = xc_new == feats["t2_n_new_files"]
        # +/- totals: parser only counts within hunks, cross-check counts everywhere; allow parser <= grep
        ok_loc = (feats["t2_loc_added"] <= xc_plus) and (feats["t2_loc_removed"] <= xc_minus) \
                 and (xc_plus - feats["t2_loc_added"] < 0.05 * max(xc_plus, 20)) \
                 and (xc_minus - feats["t2_loc_removed"] < 0.05 * max(xc_minus, 20))
        flag = "OK " if (ok_files and ok_hunks and ok_new and ok_loc) else "MISMATCH"
        print(f"[{flag}] {r.owner}/{r.repo}#{r.pr_number} lang={r.file_language} "
              f"files={feats['t2_n_files']}(grep {xc_files}) hunks={feats['t2_n_hunks']}(grep {xc_hunks}) "
              f"+{feats['t2_loc_added']}(grep {xc_plus}) -{feats['t2_loc_removed']}(grep {xc_minus}) "
              f"new={feats['t2_n_new_files']}(grep {xc_new}) "
              f"cmt+={feats['t2_comment_lines_added']} cmt-={feats['t2_comment_lines_removed']} "
              f"todoΔ={feats['t2_todo_delta']} impΔ={feats['t2_import_delta']} "
              f"gen={feats['t2_n_generated_files']} testfrac={feats['t2_test_loc_frac']:.2f}")


def _row_worker(args):
    owner, repo, pr_number = args
    p = diff_path(owner, repo, pr_number)
    text = read_diff(p)
    if text is None:
        return None
    feats = parse_diff(text)
    feats.update({"owner": owner, "repo": repo, "pr_number": pr_number})
    return feats


def cmd_run(pool_path, out_path, nproc=16):
    import pandas as pd
    from multiprocessing import Pool as MPPool
    pool = pd.read_parquet(pool_path)
    tasks = list(zip(pool.owner, pool.repo, pool.pr_number))
    print(f"extracting Tier2 for {len(tasks)} PRs with {nproc} procs")
    with MPPool(nproc) as mp:
        rows = [r for r in mp.imap_unordered(_row_worker, tasks, chunksize=64) if r is not None]
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"wrote {len(df)} rows -> {out_path}")


def cmd_full(nproc=24):
    import pandas as pd
    from multiprocessing import Pool as MPPool
    files = sorted(DIFF_DIR.glob("*.diff"))
    tasks = []
    for p in files:
        parts = p.stem.split("__")
        if len(parts) != 3:
            continue
        tasks.append((parts[0], parts[1], int(parts[2])))
    print(f"full corpus: {len(tasks)} diffs")
    with MPPool(nproc) as mp:
        rows = [r for r in mp.imap_unordered(_row_worker, tasks, chunksize=64) if r is not None]
    df = pd.DataFrame(rows)
    out = OUT / "pr_tier2_features_full.parquet"
    df.to_parquet(out)
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "run", "full"])
    ap.add_argument("--pool", default=str(OUT / "pr_stage0_pool.parquet"))
    ap.add_argument("--out", default=str(OUT / "pr_tier2_features_pool.parquet"))
    args = ap.parse_args()
    if args.cmd == "validate":
        cmd_validate()
    elif args.cmd == "run":
        cmd_run(args.pool, args.out)
    else:
        cmd_full()
