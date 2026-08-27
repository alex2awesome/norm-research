import json, numpy as np, pandas as pd
OUTD="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/v7_community/"
cb=json.load(open(OUTD+"v7_cat_by_year.json"))
print("=== citation_category coverage by CITING grant year ===")
rows=[]
for y,c in cb.items():
    if not y.isdigit(): continue
    yi=int(y)
    if yi<1995 or yi>2025: continue
    tot=sum(c.values())
    rows.append({"citing_year":yi,"n_edges":tot,
                 "examiner":c.get("cited by examiner",0)/tot,
                 "applicant":c.get("cited by applicant",0)/tot,
                 "other":c.get("cited by other",0)/tot,
                 "blank":c.get("",0)/tot})
d=pd.DataFrame(rows).sort_values("citing_year")
print(d.to_string(index=False,float_format=lambda v:f"{v:.3f}"))

agg=pd.read_parquet(OUTD+"v7_cite_aggregates.parquet")
print("\n=== tot5 distribution ===")
print(agg.tot5.describe().to_string())
print("zero share:",float((agg.tot5==0).mean()))
# cohort thresholds need CPC; approximate with grant_year only for the diagnostic
for key in ["grant_year"]:
    g=agg.groupby(key).tot5
    print(f"\ncohort={key}: median / q75 per cohort")
    print(pd.DataFrame({"median":g.median(),"q75":g.quantile(.75),"q90":g.quantile(.9),
                        "zero_share":g.apply(lambda s:(s==0).mean())}).to_string())
med=agg.groupby("grant_year").tot5.transform("median")
q75=agg.groupby("grant_year").tot5.transform("quantile",.75)
print("\nMEDIAN-SPLIT: tie share %.3f  pos-rate-among-untied %.3f"%(
    float((agg.tot5==med).mean()), float((agg.tot5>med)[agg.tot5!=med].mean())))
print("TOP-QUARTILE: pos rate %.3f"%float((agg.tot5>q75).mean()))
