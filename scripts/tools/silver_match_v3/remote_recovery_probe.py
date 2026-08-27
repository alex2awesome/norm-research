#!/usr/bin/env python3
"""Remote read-only probe used by remote_recovery_inspector.py."""

import base64
import hashlib
import json
import pathlib
import subprocess
import sys

payload = json.loads(base64.b64decode(sys.argv[1]))
output = {"roots": {}, "artifacts": {}, "gpu_probe": {}}
for pilot in payload.get("pilots", []):
    name = pilot["name"]
    output["roots"][name] = pathlib.Path(pilot["root"]).is_dir()
    rows = []
    for item in pilot["artifacts"]:
        path = pathlib.Path(item["path"])
        row = {
            "key": item["key"],
            "path": str(path),
            "mode": item["mode"],
            "exists": path.is_file(),
        }
        if path.is_file():
            data = path.read_bytes()
            row.update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
            if item["mode"] == "full":
                if len(data) <= payload["max"]:
                    row["content_b64"] = base64.b64encode(data).decode()
                else:
                    row["error"] = "artifact exceeds limit"
            elif item["mode"] == "tail":
                tail = data[-payload["tail"] :]
                row.update(
                    content_b64=base64.b64encode(tail).decode(),
                    content_sha256=hashlib.sha256(tail).hexdigest(),
                    tail_bytes=len(tail),
                )
        rows.append(row)
    output["artifacts"][name] = rows
try:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(query, text=True, capture_output=True, timeout=15)
    output["gpu_probe"] = {
        "available": result.returncode == 0,
        "stdout": result.stdout[-65536:],
        "stderr": result.stderr[-8192:],
    }
except Exception as exc:
    output["gpu_probe"] = {"available": False, "error": str(exc)}
print(json.dumps(output, sort_keys=True))
