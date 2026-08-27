"""Step 2 — blind compile trials via the Codex companion (scripted, resumable).

Each trial t compiles every chunk with a FRESH Codex thread (independence across trials
and chunks). Output: roundtrip/output_rt_c<k>_codex_t<t>.py. A chunk is skipped iff its
output file exists AND contains the full function count (validation below), so the script
is safe to re-run after interruptions.

Usage: python3 run_compile_trials.py [n_trials]     (default 2)
"""
import json
import re
import subprocess
import sys
import time

from common import CODEX, INSTR, N_CHUNKS, WORK


def n_funcs(path):
    return len(re.findall(r"def score__", open(path).read())) if path.exists() else 0


def main(n_trials=2):
    for t in range(1, n_trials + 1):
        for k in range(1, N_CHUNKS + 1):
            inp = WORK / f"input_rt_c{k}.json"
            out = WORK / f"output_rt_c{k}_codex_t{t}.py"
            want = len(json.load(open(inp)))
            if n_funcs(out) >= want:
                print(f"SKIP c{k} t{t} (complete: {want})")
                continue
            prompt = (
                f"Read the file {INSTR} and follow it exactly for the rule list in {inp}. "
                f"ONE deviation from those instructions: write your output to {out} "
                f"(not output_rt_c{k}.py) — same format: one 'def score__<job_id>(text):' "
                f"function per rule, plus a JOB_IDS list at the end. Do not read any other "
                f"files in the repository; this is a blind compilation task. Reply with "
                f"just the count of functions written.")
            print(f"=== codex trial t{t} chunk c{k} {time.strftime('%H:%M:%S')} ===",
                  flush=True)
            # --write: the companion sandbox is READ-ONLY without it (verified 2026-08-12)
            r = subprocess.run(["node", CODEX, "task", "--write", prompt, "--fresh"],
                               timeout=900, capture_output=True, text=True)
            if not out.exists():        # fallback: reply-only response with a code fence
                m = re.findall(r"```(?:python)?\n(.*?)```", r.stdout or "", re.S)
                if m and "def score__" in m[-1]:
                    out.write_text(m[-1])
            got = n_funcs(out)
            print(f"c{k} t{t}: {got}/{want} functions", flush=True)
    print("COMPILE TRIALS DONE")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
