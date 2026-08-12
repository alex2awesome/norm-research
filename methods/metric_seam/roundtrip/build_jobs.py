"""Step 1 — build the frozen job packs (deterministic; seed in common.SEED).

Selects, per (task, aspect), the BEST code channel row from recon_results (max R), takes
its stored m_hat (the decoder's prose reconstruction of the code channel), adds the 6
planted calibration rules, shuffles with the frozen seed, and writes N_CHUNKS input packs.
Output: outputs/metric_seam_pilot/roundtrip/{jobs_full.json, input_rt_c<k>.json}
"""
import json
import random

from common import N_CHUNKS, PLANTED, RECON, SEED, WORK


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(RECON / "recon_results.jsonl")]
    best = {}
    for r in rows:
        if "R" not in r or not r["channel"].startswith("code_"):
            continue
        k = (r["task"], r["aspect"])
        if k not in best or r["R"] > best[k]["R"]:
            best[k] = r
    jobs = []
    for (t, a), r in sorted(best.items()):
        dp = RECON / "detail" / f"{t}__{a}__{r['channel']}.json"
        if not dp.exists():
            continue
        jobs.append({"job_id": f"{t}__{a}", "task": t, "aspect": a,
                     "channel": r["channel"], "R_mixed": r["R"], "m_hat": r["m_hat"]})
    for cid, rule in PLANTED:
        jobs.append({"job_id": f"CAL__{cid}", "task": "CALIBRATION", "aspect": cid,
                     "channel": "planted", "R_mixed": None, "m_hat": rule})
    random.Random(SEED).shuffle(jobs)
    per = (len(jobs) + N_CHUNKS - 1) // N_CHUNKS
    for i in range(N_CHUNKS):
        ch = jobs[i * per:(i + 1) * per]
        json.dump([{"job_id": j["job_id"], "rule": j["m_hat"]} for j in ch],
                  open(WORK / f"input_rt_c{i+1}.json", "w"), indent=1)
        print(f"c{i+1}: {len(ch)}")
    json.dump(jobs, open(WORK / "jobs_full.json", "w"), indent=1)
    print(f"total jobs: {len(jobs)} -> {WORK}")


if __name__ == "__main__":
    main()
