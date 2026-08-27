#!/usr/bin/env python3
"""One long-lived waiter for a round's sealed fleet.

Coordinator ruling 2026-08-09 (parseability, not existence) applied to the PROPOSER
leg: an out_*.txt is only "landed" when harness_maps.parse_output() actually yields
items, so a half-written or truncated model reply does not count as arrival.
Also exits if the non-Claude legs have all died AND no further file can arrive.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness_maps as H

tag = sys.argv[1]
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 14400
d = H.SCRATCH / tag
manifest = json.loads((d / "manifest.json").read_text())
want = [(m["track"], m["id"]) for m in manifest]
t0 = time.time()
while True:
    landed, pending = [], []
    for track, pid in want:
        f = d / f"out_{track}_{pid}.txt"
        try:
            n = len(H.parse_output(f.read_text(), track)) if f.exists() else 0
        except Exception:
            n = 0
        (landed if n else pending).append(f"{track}/{pid}")
    if not pending:
        print(f"FLEET_LANDED {tag} {len(landed)}/{len(want)} parseable", flush=True)
        break
    if time.time() - t0 > timeout:
        print(f"FLEET_TIMEOUT {tag} landed={len(landed)}/{len(want)} pending={pending}",
              flush=True)
        break
    time.sleep(45)
