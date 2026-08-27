"""Summarize per-claim verdicts: by language, type, uncheckable reasons."""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True)
    args = ap.parse_args()
    rows = [json.loads(L) for L in open(args.verdicts)]
    print("n claims:", len(rows))

    def line(name, c):
        tot = sum(c.values()) or 1
        s, r, u = c["SUPPORTED"], c["REFUTED"], c["UNCHECKABLE"]
        print(f"{name:12s} n={tot:4d}  S={s:3d} ({100*s/tot:4.1f}%)  "
              f"R={r:3d} ({100*r/tot:4.1f}%)  U={u:3d} ({100*u/tot:4.1f}%)")

    print("\nby language:")
    by_lang = defaultdict(Counter)
    for r in rows:
        by_lang[r["language"]][r["verdict"]] += 1
    for lang, c in sorted(by_lang.items(), key=lambda kv: -sum(kv[1].values())):
        line(lang, c)

    print("\nby claim_type:")
    by_type = defaultdict(Counter)
    for r in rows:
        by_type[r["claim_type"]][r["verdict"]] += 1
    for t, c in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        line(t, c)

    print("\nUNCHECKABLE (checker, reason) top 25:")
    reasons = Counter()
    for r in rows:
        if r["verdict"] == "UNCHECKABLE":
            ev = r.get("evidence") or {}
            reasons[(r.get("checker"),
                     ev.get("reason") or ev.get("language")
                     or ev.get("claim_type") or "")] += 1
    for k, v in reasons.most_common(25):
        print(f"  {v:4d}  {k}")

    print("\nbinding_cue downgrades:",
          sum(1 for r in rows if r.get("binding_cue_downgrade")))

    print("\ncheckable-scope languages (python/unknown/js/java/c#):")
    scope = [r for r in rows
             if r["language"] in ("python", "unknown", "javascript", "java", "c#")]
    c = Counter(r["verdict"] for r in scope)
    line("in-scope", c)
    py = [r for r in rows if r["language"] in ("python", "unknown")]
    line("py+unk", Counter(r["verdict"] for r in py))

    print("\nSUPPORTED by checker:")
    for k, v in Counter(r["checker"] for r in rows
                        if r["verdict"] == "SUPPORTED").most_common():
        print(f"  {v:4d}  {k}")
    print("REFUTED by checker:")
    for k, v in Counter(r["checker"] for r in rows
                        if r["verdict"] == "REFUTED").most_common():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
