"""Build host-JSON bundles for the CUF pilot (run_unit_certificate.py).

Hosts per metric (plan): description (lineage seed rung), gepa (best-R lineage prompt),
checklist (head_selected criteria joined; needs the census pool npz with the criterion strings —
resolved on sk3; omitted with a warning if the pool file is absent).

Usage:
  python -m methods.metric_implementer.experiments.prep_cuf_hosts \
      --lineage data/prompt_optimality_20260703/gepa_lineage_cw_R3_g24_desc.json \
      --cert data/prompt_optimality_20260703/cert_8b_v2.json \
      [--pool-npz /lfs/.../cw_8b_census.npz] --out outputs/unit_cert/hosts_g24.json
"""
import argparse, json, os, re
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--lineage", required=True)
    p.add_argument("--cert", required=True)
    p.add_argument("--pool-npz", default=None,
                   help="census npz with 'prompts' or 'crits' array; indexes head_selected")
    p.add_argument("--task", default="creative-writing")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    j = json.load(open(a.lineage))
    lin = j["lineage"] if "lineage" in j else j
    name = j.get("name") or j.get("metric")
    desc = lin[0]["prompt"]
    best = max(lin, key=lambda x: x["R_bits"])["prompt"]

    hosts = {"description": desc}
    if best.strip() != desc.strip():
        hosts["gepa"] = best
    else:
        print("[prep] GEPA best == seed (no improvement); gepa host omitted (identical text)")

    def nn(s): return re.sub(r"\s+", " ", s.strip().lower())
    row = next(r for r in json.load(open(a.cert)) if nn(r["name"]) == nn(name))
    if a.pool_npz and os.path.exists(a.pool_npz):
        z = np.load(a.pool_npz, allow_pickle=True)
        key = "prompts" if "prompts" in z.files else ("crits" if "crits" in z.files else None)
        if key:
            pool = [str(x) for x in z[key]]
            sel = [pool[i] for i in row["head_selected"] if i < len(pool)]
            if sel:
                hosts["checklist"] = "Evaluate the text against each criterion:\n" + \
                    "\n".join(f"- {s}" for s in sel)
    if "checklist" not in hosts:
        print("[prep] checklist host NOT built (pool npz missing/unmatched) — pilot runs without it")

    out = {"metric": name, "task": a.task,
           "m_rubric": f"{name}: {desc}",
           "cert_row": {k: row[k] for k in ("opt_omega_bits", "H_M", "n_head", "verdict")},
           "hosts": hosts}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: metric='{name}', hosts={list(hosts)}")


if __name__ == "__main__":
    main()
