import pandas as pd, numpy as np, re
df = pd.read_csv("built/law_se_balanced.csv.gz")
def answer_part(t):
    p = re.split(r"\n\nAnswer:", t, maxsplit=1)
    return p[1].strip() if len(p) > 1 else ""
df["ans"] = df["text"].map(answer_part)
a = df["ans"].str.lower()
L = df["ans"].str.len().clip(lower=1)

def rate(name, pat):
    cnt = a.str.count(pat)
    perdoc = cnt / (L / 1000.0)
    pos = perdoc[df.judgement == 1]
    neg = perdoc[df.judgement == 0]
    has = (cnt > 0)
    print(f"{name:30s} per-1k  pos={pos.mean():.3f} neg={neg.mean():.3f} | "
          f"%docs pos={100*has[df.judgement==1].mean():.1f} neg={100*has[df.judgement==0].mean():.1f}")

print("=== Candidate confound / signal lexicons (per 1000 chars; % docs containing) ===")
rate("IANAL disclaimer", r"i am not a lawyer|i'?m not a lawyer|not legal advice|ianal|not your lawyer")
rate("hedging (maybe/think/guess)", r"\bmaybe\b|\bi think\b|\bnot sure\b|\bi guess\b|\bprobably\b|\bi believe\b|\bseems?\b")
rate("citation: case (v.)", r"\bv\.\s|\bvs\.\s")
rate("citation: USC/section/act", r"\bu\.?s\.?c\b|\bsection\b|§|\bstat\.|\bact\b")
rate("URL/link (http)", r"http")
rate("imperative legal (shall/must)", r"\bshall\b|\bmust\b")
rate("rhetorical question mark", r"\?")
