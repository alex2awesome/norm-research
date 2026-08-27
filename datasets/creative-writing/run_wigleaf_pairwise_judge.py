#!/usr/bin/env python3
"""Run the Wigleaf PAIRWISE probe through gpt-5.6-sol via `codex exec`.

Harness pattern (chunked prompts, JSON extraction, multi-pass retry, hard
coverage guard, anchors mixed in with real items) is taken from
methods/taste_decomposition/closure/run_bmerge_judges.py -- the repo's standing
frontier-judge wave harness (feedback_judge_checks_use_codex: judging waves run on
gpt-5.6-sol, never on Claude credits).

  python datasets/creative-writing/run_wigleaf_pairwise_judge.py \
      --packet datasets/creative-writing/wigleaf/pairwise/packet.json \
      --model gpt-5.6-sol --chunk 2 --workers 4 \
      --out datasets/creative-writing/wigleaf/pairwise/verdicts_sol.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LOCK = threading.Lock()


def preamble(criteria):
    lines = [
        "You are an expert literary editor performing a MEASUREMENT task.",
        "",
        "You are given PAIRS of short prose pieces (flash fiction). Both pieces in "
        "every pair were genuinely published in literary magazines, so both are "
        "competent work: your job is to make FINE comparative discriminations, not "
        "to find an obvious winner.",
        "",
        "For each pair and each criterion, decide which piece BETTER EXEMPLIFIES "
        "that criterion. Answer \"A\", \"B\", or \"TIE\". Use TIE sparingly - only "
        "when you genuinely cannot separate them on that specific dimension.",
        "",
        "Judge the writing on the text alone. Do NOT try to infer or predict which "
        "piece was selected for an anthology or 'best of' list, which magazine "
        "published it, magazine prestige, the author's reputation, or dataset "
        "membership. Brevity and open endings are normal in this form and are not "
        "in themselves faults.",
        "",
        "CRITERIA:",
    ]
    for m in criteria:
        lines.append(f"  {m['id']}: {m['name']} - {m['description']}")
    lines += [
        "",
        "Also answer 'overall': which piece is the stronger piece of writing overall.",
        "",
        "Output STRICT JSON ONLY, no prose, no code fences, in exactly this shape:",
        '{"verdicts": [{"pair_id": "...", "overall": "A|B|TIE", '
        '"criteria": {"a01": "A|B|TIE", "a02": "A|B|TIE", ...}}]}',
        "Include EVERY pair_id you were given and EVERY criterion id for each.",
        "",
    ]
    return "\n".join(lines)


def render(items):
    out = []
    for it in items:
        out += [f"=== PAIR {it['pair_id']} ===", "--- PIECE A ---", it["A"],
                "--- PIECE B ---", it["B"], ""]
    return "\n".join(out)


def extract_json(raw):
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M)
    for m in re.finditer(r'\{"verdicts"', raw):
        depth, i = 0, m.start()
        for j in range(m.start(), len(raw)):
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[i:j + 1])
                    except json.JSONDecodeError:
                        break
    # last resort: greedy outermost object
    a, b = raw.find("{"), raw.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(raw[a:b + 1])
        except json.JSONDecodeError:
            return None
    return None


def ask(model, prompt, wd, effort="high", timeout=1800):
    wd.mkdir(parents=True, exist_ok=True)
    cmd = ["codex", "exec", "--model", model, "-c", f"model_reasoning_effort={effort}",
           "-s", "read-only", "--skip-git-repo-check", "--cd", str(wd), "-"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           timeout=timeout)
        return p.stdout, time.time() - t0, p.returncode
    except subprocess.TimeoutExpired:
        return "", time.time() - t0, -9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--scratch", default="/lfs/skampere3/0/alexspan/tmp/wigleaf_pairwise")
    a = ap.parse_args()

    pk = json.loads(Path(a.packet).read_text())
    crit_ids = [m["id"] for m in pk["criteria"]]
    items = list(pk["items"]) + list(pk["anchors"])
    by_id = {it["pair_id"]: it for it in items}
    want = list(by_id)
    pre = preamble(pk["criteria"])
    wd = Path(a.scratch)
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    verdicts = {}
    if outp.exists():                      # resume
        verdicts = json.loads(outp.read_text()).get("verdicts", {})
        print(f"resuming with {len(verdicts)} verdicts already on disk", flush=True)

    def valid(v):
        return str(v).strip().upper() in ("A", "B", "TIE")

    for attempt in range(1, a.passes + 1):
        todo = [p for p in want if p not in verdicts]
        if not todo:
            break
        batches = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]
        print(f"pass {attempt}: {len(todo)} pairs in {len(batches)} prompts, "
              f"{a.workers} workers", flush=True)

        def run(b):
            raw, dt, rc = ask(a.model, pre + render([by_id[i] for i in b]), wd, a.effort)
            (wd / f"raw_p{attempt}_{b[0]}.txt").write_text(raw or "")
            obj = extract_json(raw)
            got = 0
            if obj:
                for v in obj.get("verdicts", []):
                    pid = v.get("pair_id")
                    if pid not in by_id:
                        continue
                    cr = {k: str(x).strip().upper()
                          for k, x in (v.get("criteria") or {}).items()
                          if k in crit_ids and valid(x)}
                    ov = str(v.get("overall", "")).strip().upper()
                    if len(cr) >= max(1, len(crit_ids) // 2):
                        with LOCK:
                            verdicts[pid] = {"pair_id": pid, "criteria": cr,
                                             "overall": ov if valid(ov) else None}
                        got += 1
            return b[0], dt, rc, got

        def checkpoint():
            with LOCK:
                outp.write_text(json.dumps(
                    {"judge": a.model, "judge_family": "openai/codex",
                     "effort": a.effort, "chunk": a.chunk, "packet": str(a.packet),
                     "n_want": len(want), "n_have": len(verdicts),
                     "criteria_ids": crit_ids, "verdicts": verdicts}, indent=1))

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run, b) for b in batches]
            for n, f in enumerate(as_completed(futs), 1):
                pid, dt, rc, got = f.result()
                if n % 5 == 0:
                    checkpoint()          # persist as we go; the wave is resumable
                if n % 10 == 0 or got == 0:
                    print(f"  [{n}/{len(batches)}] {pid} {dt:.0f}s rc={rc} got={got} "
                          f"| total {len(verdicts)}/{len(want)}", flush=True)
        with LOCK:
            outp.write_text(json.dumps(
                {"judge": a.model, "judge_family": "openai/codex",
                 "effort": a.effort, "chunk": a.chunk,
                 "packet": str(a.packet), "n_want": len(want),
                 "n_have": len(verdicts), "criteria_ids": crit_ids,
                 "verdicts": verdicts}, indent=1))

    missing = [p for p in want if p not in verdicts]
    print(f"COVERAGE {len(verdicts)}/{len(want)}; missing {len(missing)}", flush=True)
    if missing:
        print(f"  first missing: {missing[:8]}", flush=True)
    print("WIGLEAF_PAIRWISE_JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
