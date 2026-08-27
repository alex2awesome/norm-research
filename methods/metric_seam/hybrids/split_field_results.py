"""Split batched field-extraction results into llm_fields/<aid>__<field>.json maps."""
import json, pathlib, re

OUT = pathlib.Path(__file__).resolve().parents[3] / "outputs/metric_seam_pilot/v1"
FD = OUT / "llm_fields"
FD.mkdir(exist_ok=True)

def clean(raw):
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    line = re.sub(r"^(answer|reply)\s*[:\-]\s*", "", line, flags=re.I).strip()
    return "" if line.upper().startswith("NONE") else line[:200]

def main():
    by = {}
    src = OUT / "field_results.jsonl"
    for line in open(src):
        r = json.loads(line)
        if r["channel"] != "field":
            continue
        by.setdefault(r["aspect_id"], {})[r["datapoint_id"]] = clean(r["raw"])
    for key, m in by.items():
        json.dump(m, open(FD / f"{key}.json", "w"))
        filled = sum(1 for v in m.values() if v)
        print(f"{key}: {len(m)} items, {filled} non-empty")

if __name__ == "__main__":
    main()
