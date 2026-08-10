import numpy as np, pandas as pd, re
from collections import Counter
ML="/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
df=pd.read_parquet(ML+"/accept_reject_clean.parquet").reset_index(drop=True)
def strip_author(d):
    return "\n".join(l for l in str(d).split("\n") if not re.search(r"Copyright|Authors:|Released under|Apache 2\.0|described in the file LICENSE|SPDX-License|maintainer",l,re.I))
df["diff_noauth"]=df["diff"].astype(str).map(strip_author)
def area(d):
    ms=re.findall(r"(?:a|b)/Mathlib/([A-Za-z0-9_]+)/",str(d)); return Counter(ms).most_common(1)[0][0] if ms else "NONE"
df["area"]=df["diff"].astype(str).map(area)
TACTICS=["grind","aesop","simp","simpa","fun_prop","funprop","cat_disch","catdisch","decide","norm_num","ring","nlinarith","linarith","omega","intro","apply","have","unfold","rw","rewrite","cases","induction","exact","refine","rwa","trans","calc","change","ext","constructor","congr","simps","obtain"]
pat={t:re.compile(r"\b"+t+r"\b") for t in TACTICS}
for t in TACTICS:
    df["tac_"+t]=[len(pat[t].findall(str(d))) for d in df["diff"].astype(str)]
df.to_parquet(ML+"/accept_reject_clean_deconf.parquet",index=False)
print("saved accept_reject_clean_deconf.parquet  n=%d cols(+diff_noauth,area,tac_*)=%d" % (len(df),df.shape[1]))
print("author-token present after strip: %.3f" % df["diff_noauth"].str.contains("riou|avigad|carneiro",case=False).mean())
print("area distribution:",df.area.value_counts().head(6).to_dict())
