"""
Tier 3 static V-layer: environment-free lint DELTA per PR.

For each PR: blobless clone of repo, fetch refs/pull/N/head, base = merge-base(head,
origin/HEAD), checkout changed source files at base and head into temp dirs, run static
tools on both sides, delta = head - base. No deps installed, no build, CPU only.

Python tools: ruff (--isolated, families E,W,F,N,C90,S,D,B,SIM == pycodestyle, pyflakes,
naming, mccabe, bandit-rules, pydocstyle, bugbear, simplify) + radon cc/mi.
Go: gofmt -l (format violations); go vet attempted once per repo, skipped if it needs build.
JS/TS: eslint with fixed minimal flat config (no plugins), if available.

Usage:
  python pr_tier3_lint_delta.py pilot   # ~500 PRs / 10 Python repos
  python pr_tier3_lint_delta.py python  # all Python-dominant pool repos
  python pr_tier3_lint_delta.py gojs    # Go + JS/TS pool repos
  python pr_tier3_lint_delta.py full    # full corpus, all tooled-scope PRs (py/go/js/ts)
Resumable: skips (owner,repo,pr) already in the output shard dir.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT = ROOT / "outputs/v2_analysis"
SHARDS = OUT / "pr_tier3_shards"
CLONES = Path("/lfs/skampere3/0/alexspan/tmp_t3_clones")
PYBIN = Path("/lfs/skampere3/0/alexspan/miniconda3/bin")
RUFF = str(PYBIN / "ruff")
RADON = str(PYBIN / "radon")

PR_TIMEOUT = 240          # seconds per PR (all tool runs)
CLONE_TIMEOUT = 1200
MAX_FILES_PER_PR = 60
MAX_FILE_BYTES = 600_000
RUFF_FAMILIES = ["E", "W", "F", "N", "C90", "S", "D", "B", "SIM"]
RUFF_SELECT = ",".join(RUFF_FAMILIES)

sys.path.insert(0, str(ROOT / "scripts"))
from pr_tier2_diff_features import classify_file  # reuse generated/test detection


def run(cmd, cwd=None, timeout=120, env=None, check=False):
    e = dict(os.environ)
    e["GIT_TERMINAL_PROMPT"] = "0"
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, env=e, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------- tool runners: return dict of counts for a dir of files ----------------

def ruff_counts(d: Path) -> dict:
    """Family -> violation count; parse_fail = #files with E999/parse errors."""
    out = {f"ruff_{fam}": 0 for fam in RUFF_FAMILIES}
    out["ruff_total"] = 0
    out["py_parse_fail"] = 0
    files = list(d.rglob("*.py"))
    if not files:
        return out
    r = run([RUFF, "check", "--isolated", "--no-cache", "--exit-zero",
             "--select", RUFF_SELECT, "--output-format", "json", str(d)],
            timeout=PR_TIMEOUT // 2)
    try:
        viol = json.loads(r.stdout or "[]")
    except json.JSONDecodeError:
        return out
    parse_fail_files = set()
    for v in viol:
        code = v.get("code") or ""
        if code in ("E999",) or v.get("message", "").startswith("SyntaxError"):
            parse_fail_files.add(v.get("filename"))
            continue
        out["ruff_total"] += 1
        for fam in ("C90", "SIM"):  # multi-char families first
            if code.startswith(fam):
                out[f"ruff_{fam}"] += 1
                break
        else:
            fam = code[:1]
            if f"ruff_{fam}" in out:
                out[f"ruff_{fam}"] += 1
    out["py_parse_fail"] = len(parse_fail_files)
    return out


def radon_counts(d: Path) -> dict:
    out = {"radon_cc_sum": 0.0, "radon_cc_max": 0.0, "radon_n_blocks": 0,
           "radon_mi_mean": 0.0, "radon_mi_min": 0.0, "radon_n_mi_files": 0}
    if not list(d.rglob("*.py")):
        return out
    r = run([RADON, "cc", "-j", str(d)], timeout=PR_TIMEOUT // 2)
    try:
        cc = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        cc = {}
    ccs = []
    for blocks in cc.values():
        if isinstance(blocks, list):
            ccs.extend(b.get("complexity", 0) for b in blocks if isinstance(b, dict))
    if ccs:
        out["radon_cc_sum"] = float(sum(ccs))
        out["radon_cc_max"] = float(max(ccs))
        out["radon_n_blocks"] = len(ccs)
    r = run([RADON, "mi", "-j", str(d)], timeout=PR_TIMEOUT // 2)
    try:
        mi = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        mi = {}
    mis = [v["mi"] for v in mi.values() if isinstance(v, dict) and "mi" in v]
    if mis:
        out["radon_mi_mean"] = float(sum(mis) / len(mis))
        out["radon_mi_min"] = float(min(mis))
        out["radon_n_mi_files"] = len(mis)
    return out


def gofmt_counts(d: Path) -> dict:
    out = {"gofmt_bad_files": 0, "go_n_files": 0}
    files = list(d.rglob("*.go"))
    out["go_n_files"] = len(files)
    if not files:
        return out
    r = run(["gofmt", "-l", str(d)], timeout=PR_TIMEOUT // 2)
    out["gofmt_bad_files"] = len([l for l in r.stdout.splitlines() if l.strip()])
    return out


ESLINT = "/lfs/skampere3/0/alexspan/node_tools/node_modules/.bin/eslint"  # v8.57
ESLINT_CONFIG = json.dumps({
    "root": True,
    "parserOptions": {"ecmaVersion": "latest", "sourceType": "module",
                      "ecmaFeatures": {"jsx": True}},
    "env": {"es2022": True, "browser": True, "node": True},
    "rules": {
        "no-unused-vars": "warn", "eqeqeq": "warn", "no-var": "warn",
        "prefer-const": "warn", "no-eval": "warn", "no-console": "warn",
        "complexity": ["warn", 10], "max-depth": ["warn", 4], "no-shadow": "warn",
        "no-redeclare": "warn", "curly": "warn", "semi": "warn", "no-empty": "warn",
        "no-dupe-keys": "warn",
    },
})


def eslint_counts(d: Path, cfg_path: Path) -> dict:
    out = {"eslint_total": 0, "eslint_fatal": 0, "js_n_files": 0}
    files = [p for p in list(d.rglob("*.js")) + list(d.rglob("*.jsx"))]
    out["js_n_files"] = len(files)
    if not files:
        return out
    r = run([ESLINT, "--no-eslintrc", "--no-inline-config", "-c", str(cfg_path),
             "--format", "json", str(d)], timeout=PR_TIMEOUT // 2)
    stdout = r.stdout
    try:
        res = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        return out
    for fr in res:
        for m in fr.get("messages", []):
            if m.get("fatal"):
                out["eslint_fatal"] += 1
            else:
                out["eslint_total"] += 1
    return out


# ---------------- per-PR processing ----------------

LANG_EXTS = {"python": (".py",), "go": (".go",), "js": (".js", ".jsx")}


def extract_side(gitdir: Path, commit: str, paths: list, dest: Path, idx_file: Path):
    """Checkout given paths at commit into dest work-tree. Batched blob fetch."""
    dest.mkdir(parents=True, exist_ok=True)
    env = {"GIT_INDEX_FILE": str(idx_file)}
    # read-tree to populate the throwaway index, then checkout-index for our paths
    run(["git", "--git-dir", str(gitdir), "read-tree", commit], timeout=120, env=env, check=True)
    cmd = ["git", "--git-dir", str(gitdir), "--work-tree", str(dest),
           "checkout-index", "-f", "--"] + paths
    run(cmd, timeout=300, env=env, check=True)


DIFF_DIR = ROOT / "datasets/code-review/diffs"
FILE_HEADER_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$")


MODE_RE = re.compile(r"^(?:new file mode|deleted file mode|old mode|new mode) (\d+)")


def parse_changed_files(diff_text: str):
    """[(a_path, b_path, is_new, is_deleted, skip)] from a stored GitHub .diff.
    skip = binary, symlink (mode 120000) or submodule (mode 160000) entries."""
    out = []
    cur = None
    for line in diff_text.splitlines():
        m = FILE_HEADER_RE.match(line)
        if m:
            if cur:
                out.append(cur)
            cur = [m.group(1), m.group(2), False, False, False]
            continue
        if cur is None:
            continue
        mm = MODE_RE.match(line)
        if mm and mm.group(1) in ("120000", "160000"):
            cur[4] = True
        if line.startswith("new file mode"):
            cur[2] = True
        elif line.startswith("deleted file mode"):
            cur[3] = True
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            cur[4] = True
    if cur:
        out.append(cur)
    return out


def process_pr(gitdir: Path, owner: str, repo: str, pr_number: int, langs: list,
               eslint_cfg: Path) -> dict:
    """Head side = checkout of refs/pull/N/head changed files.
    Base side = head copy with the STORED diff reverse-applied (git apply -R).
    No merge-base needed; works for merged, squashed, and open PRs alike, and the
    delta corresponds exactly to the artifact the labels/diff corpus were built from."""
    row = {"owner": owner, "repo": repo, "pr_number": pr_number, "t3_status": "ok",
           "t3_n_src_changed": 0, "t3_n_src_skipped": 0}
    t0 = time.time()
    diff_file = DIFF_DIR / f"{owner}__{repo}__{pr_number}.diff"
    if not diff_file.exists():
        row["t3_status"] = "no_diff_file"
        return row
    try:
        diff_text = diff_file.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        row["t3_status"] = "diff_read_failed"
        return row
    exts = tuple(e for lang in langs for e in LANG_EXTS[lang])
    kept = []
    for a_path, b_path, is_new, is_del, is_bin in parse_changed_files(diff_text):
        path = a_path if is_del else b_path
        if is_bin or not path.lower().endswith(exts):
            continue
        if classify_file(path) == "generated":
            row["t3_n_src_skipped"] += 1
            continue
        kept.append((a_path, b_path, is_new, is_del))
    if len(kept) > MAX_FILES_PER_PR:
        row["t3_n_src_skipped"] += len(kept) - MAX_FILES_PER_PR
        kept = kept[:MAX_FILES_PER_PR]
    row["t3_n_src_changed"] = len(kept)
    if not kept:
        row["t3_status"] = "no_src_files"   # valid row: PR touches no analyzable source
        return row
    ref = f"refs/prx/{pr_number}"
    r = run(["git", "--git-dir", str(gitdir), "rev-parse", ref], timeout=60)
    if r.returncode != 0:
        row["t3_status"] = "no_pr_ref"
        return row
    head = r.stdout.strip()
    head_paths = sorted({b for _a, b, _n, is_del in kept if not is_del})
    with tempfile.TemporaryDirectory(dir=str(CLONES)) as td:
        td = Path(td)
        base_d, head_d = td / "base", td / "head"
        head_d.mkdir()
        try:
            if head_paths:
                # intersect with what actually exists at head (diff may name moved paths)
                r = run(["git", "--git-dir", str(gitdir), "ls-tree", "-r", "--name-only",
                         head], timeout=120)
                head_existing = sorted(set(r.stdout.splitlines()) & set(head_paths))
                if len(head_existing) < len(head_paths):
                    row["t3_n_head_missing"] = len(head_paths) - len(head_existing)
                if head_existing:
                    extract_side(gitdir, head, head_existing, head_d, td / "idx_head")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            row["t3_status"] = "checkout_failed"
            return row
        # base = head with the stored diff reverse-applied (restricted to kept paths)
        shutil.copytree(head_d, base_d, dirs_exist_ok=True)
        patch = td / "pr.diff"
        patch.write_text(diff_text)
        inc = []
        for a_path, b_path, _n, _d in kept:
            inc += ["--include=" + a_path, "--include=" + b_path]
        r = run(["git", "-C", str(base_d), "apply", "-R", "-p1", "--whitespace=nowarn"]
                + inc + [str(patch)], timeout=120)
        if r.returncode != 0:
            row["t3_status"] = "apply_failed"
            row["t3_apply_err"] = (r.stderr or "")[-200:]
            return row
        # drop oversized files from both sides
        for d in (base_d, head_d):
            for p in d.rglob("*"):
                if p.is_file() and p.stat().st_size > MAX_FILE_BYTES:
                    p.unlink()
        try:
            sides = {}
            for name, d in (("base", base_d), ("head", head_d)):
                c = {}
                if "python" in langs:
                    c.update(ruff_counts(d))
                    c.update(radon_counts(d))
                if "go" in langs:
                    c.update(gofmt_counts(d))
                if "js" in langs:
                    c.update(eslint_counts(d, eslint_cfg))
                sides[name] = c
        except subprocess.TimeoutExpired:
            row["t3_status"] = "tool_timeout"
            return row
        for k in sides["head"]:
            row[f"t3_{k}_head"] = sides["head"][k]
            row[f"t3_{k}_base"] = sides["base"].get(k, 0)
            row[f"t3_{k}_delta"] = sides["head"][k] - sides["base"].get(k, 0)
    row["t3_seconds"] = round(time.time() - t0, 1)
    return row


# ---------------- per-repo driver ----------------

def process_repo(repo_full: str, prs: list, langs: list, keep_clone=False) -> list:
    owner, repo = repo_full.split("/")
    shard = SHARDS / f"{owner}__{repo}.jsonl"
    done = set()
    if shard.exists():
        for line in open(shard):
            try:
                d = json.loads(line)
                done.add(d["pr_number"])
            except json.JSONDecodeError:
                pass
    todo = [p for p in prs if p not in done]
    if not todo:
        log(f"{repo_full}: all {len(prs)} done")
        return []
    CLONES.mkdir(parents=True, exist_ok=True)
    gitdir = CLONES / f"{owner}__{repo}.git"
    rows = []
    try:
        if not gitdir.exists():
            log(f"{repo_full}: blobless clone ...")
            r = run(["git", "clone", "--bare", "--filter=blob:none",
                     f"https://github.com/{repo_full}.git", str(gitdir)],
                    timeout=CLONE_TIMEOUT)
            if r.returncode != 0:
                # retry once (transient network failures shouldn't poison the shard)
                shutil.rmtree(gitdir, ignore_errors=True)
                time.sleep(10)
                r = run(["git", "clone", "--bare", "--filter=blob:none",
                         f"https://github.com/{repo_full}.git", str(gitdir)],
                        timeout=CLONE_TIMEOUT)
            if r.returncode != 0:
                log(f"{repo_full}: CLONE FAILED: {r.stderr[-300:]}")
                with open(shard, "a") as fh:
                    for p in todo:
                        fh.write(json.dumps({"owner": owner, "repo": repo,
                                             "pr_number": p, "t3_status": "clone_failed"}) + "\n")
                return []
            # ensure origin/HEAD exists
            run(["git", "--git-dir", str(gitdir), "remote", "set-head", "origin", "-a"],
                timeout=120)
        # ensure origin/HEAD resolvable
        r = run(["git", "--git-dir", str(gitdir), "rev-parse", "refs/remotes/origin/HEAD"],
                timeout=60)
        if r.returncode != 0:
            # bare clone: HEAD of the clone points at default branch
            r2 = run(["git", "--git-dir", str(gitdir), "symbolic-ref", "HEAD"], timeout=60)
            defbranch = r2.stdout.strip().rsplit("/", 1)[-1] or "master"
            run(["git", "--git-dir", str(gitdir), "update-ref",
                 "refs/remotes/origin/HEAD", f"refs/heads/{defbranch}"], timeout=60)
        # batch-fetch PR head refs, 40 per round trip
        for i in range(0, len(todo), 40):
            chunk = todo[i:i + 40]
            specs = [f"+pull/{n}/head:refs/prx/{n}" for n in chunk]
            run(["git", "--git-dir", str(gitdir), "fetch", "origin"] + specs,
                timeout=CLONE_TIMEOUT)
        eslint_cfg = CLONES / "eslintrc.json"
        if "js" in langs and not eslint_cfg.exists():
            eslint_cfg.write_text(ESLINT_CONFIG)
        with open(shard, "a") as fh:
            for n in todo:
                try:
                    row = process_pr(gitdir, owner, repo, n, langs, eslint_cfg)
                except subprocess.TimeoutExpired:
                    row = {"owner": owner, "repo": repo, "pr_number": n,
                           "t3_status": "pr_timeout"}
                except Exception as e:
                    row = {"owner": owner, "repo": repo, "pr_number": n,
                           "t3_status": f"error:{type(e).__name__}"}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
        ok = sum(1 for r in rows if r.get("t3_status") in ("ok", "no_src_files"))
        log(f"{repo_full}: {ok}/{len(rows)} ok")
    finally:
        if not keep_clone and gitdir.exists():
            shutil.rmtree(gitdir, ignore_errors=True)
    return rows


LANG_MAP = {"Python": ["python"], "Go": ["go"], "JavaScript": ["js"], "TypeScript": ["js"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "python", "gojs", "full"])
    ap.add_argument("--nproc", type=int, default=6, help="parallel repos")
    args = ap.parse_args()
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
    SHARDS.mkdir(parents=True, exist_ok=True)
    if args.cmd == "full":
        # full corpus: every PR (with a stored diff) whose comments touch a tooled language
        full = pd.read_csv(ROOT / "datasets/code-review/code_review_modeling_dataset.csv.gz",
                           usecols=["owner", "repo", "pr_number", "language"])
        full = full[full.language.isin(LANG_MAP)]
        full = full.drop_duplicates(["owner", "repo", "pr_number", "language"])
        t2 = pd.read_parquet(OUT / "pr_tier2_features_full.parquet",
                             columns=["owner", "repo", "pr_number"])
        full = full.merge(t2.drop_duplicates(), on=["owner", "repo", "pr_number"],
                          how="inner")
        full["repo_full"] = full.owner + "/" + full.repo
        sub = full.rename(columns={"language": "file_language"})
    else:
        pool = pd.read_parquet(OUT / "pr_stage0_pool.parquet")
        if args.cmd == "pilot":
            sub = pool[pool.file_language == "Python"]
            repos = sub.repo_full.value_counts().head(10).index.tolist()
            sub = sub[sub.repo_full.isin(repos)]
        elif args.cmd == "python":
            sub = pool[pool.file_language == "Python"]
        else:
            sub = pool[pool.file_language.isin(["Go", "JavaScript", "TypeScript"])]
    jobs = []
    for repo_full, grp in sub.groupby("repo_full"):
        langs = sorted({l for fl in grp.file_language.unique() for l in LANG_MAP.get(fl, [])})
        jobs.append((repo_full, sorted(set(grp.pr_number.astype(int).tolist())), langs))
    jobs.sort(key=lambda j: -len(j[1]))
    log(f"{args.cmd}: {len(jobs)} repos, {sum(len(j[1]) for j in jobs)} PRs")
    with ProcessPoolExecutor(max_workers=args.nproc) as ex:
        futs = {ex.submit(process_repo, rf, prs, langs): rf for rf, prs, langs in jobs}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                log(f"{futs[fut]}: REPO FAILED {type(e).__name__}: {e}")
    log("done")


if __name__ == "__main__":
    main()
