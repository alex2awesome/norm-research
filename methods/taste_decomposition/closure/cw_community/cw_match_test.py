import gzip, json, sys

snips = []
with open("/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/closure/cw_community/cw_snips_test.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3:
            snips.append(parts)

print(f"testing {len(snips)} snippets against writingprompts_comments.jsonl.gz")
matches = {s[2]: [] for s in snips}
n = 0
with gzip.open("/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/writingprompts_comments.jsonl.gz", "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        n += 1
        for _, _, snip in snips:
            if snip in line:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                matches[snip].append((rec.get("created_utc"), rec.get("link_id"), rec.get("score")))
        if n % 500000 == 0:
            print(f"  scanned {n:,} lines...", flush=True)

print(f"total lines scanned: {n:,}")
for iid, pid, snip in snips:
    ms = matches[snip]
    print(f"ID={iid} PID={pid} SNIP={snip!r} N_MATCHES={len(ms)} -> {ms[:3]}")
