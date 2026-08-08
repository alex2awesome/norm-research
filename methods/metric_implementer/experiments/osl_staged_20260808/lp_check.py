import numpy as np, itertools
O='/lfs/skampere3/0/alexspan/outputs/osl_multi'
EXECS=['llama1b','llama3b','llama8b','llama70b','qwen25-3b','qwen25-7b','qwen25-14b','qwen25-32b']
def load(tag,ex):
    f=f'{O}/mbar_zxa{tag}_humor_{ex}.npz'
    z=np.load(f,allow_pickle=True)
    names=[str(n) for n in z['names']]; kinds=[str(k) for k in z['kinds']]
    return names,kinds,z['m_bar'].astype(float)
def stats(m):
    fin=np.isfinite(m)
    const=np.array([fin[i].sum()>=30 and np.nanstd(m[i])==0 for i in range(m.shape[0])])
    return fin,const
print("=== A) const-row %% by exec x arm: v1(81ch) -> LP(1018ch) ===")
hdr=None
for ex in EXECS:
    try:
        n1,k1,m1=load('',ex); n2,k2,m2=load('LP',ex)
    except FileNotFoundError: continue
    arms=sorted(set(x.split('||')[1] for x in n1))
    if hdr is None: print('exec'.ljust(11), ' '.join(a[:12].ljust(14) for a in arms)); hdr=1
    _,c1=stats(m1); _,c2=stats(m2)
    row=[]
    for a in arms:
        i1=[i for i,x in enumerate(n1) if x.split('||')[1]==a]
        i2=[i for i,x in enumerate(n2) if x.split('||')[1]==a]
        row.append(f'{100*c1[i1].mean():3.0f}->{100*c2[i2].mean():3.0f}%'.ljust(14))
    print(ex.ljust(11), ' '.join(row))
print()
print("=== B) v1 unanimous-NO metrics (name arm, capable execs): do they develop variance on LP? ===")
CAP=['llama70b','qwen25-32b','qwen25-14b']
data={ex:load('',ex) for ex in CAP}; dataLP={ex:load('LP',ex) for ex in CAP}
n1,k1,_=data[CAP[0]]
bases=sorted(set(x.split('||')[0] for x in n1))
def yesrate(names,m,base,arm):
    ix=[i for i,x in enumerate(names) if x==f'{base}||{arm}']
    if not ix: return np.nan
    v=m[ix[0]]; return np.nanmean(v)
rows=[]
for b in bases:
    y1=[yesrate(*[data[e][0],data[e][2]][0:2],b,'name') for e in CAP]
    y1=[yesrate(data[e][0],data[e][2],b,'name') for e in CAP]
    y2=[yesrate(dataLP[e][0],dataLP[e][2],b,'name') for e in CAP]
    y1m=np.nanmean(y1); y2m=np.nanmean(y2)
    kind=k1[[i for i,x in enumerate(n1) if x.startswith(b+'||')][0]]
    if y1m<=0.07 or y1m>=0.93: rows.append((b,kind,y1m,y2m))
for b,kind,y1m,y2m in sorted(rows,key=lambda r:r[2]):
    move=' <-- DEVELOPS VARIANCE' if 0.10<y2m<0.90 else ''
    print(f'{b[:38]:40s} {kind[:14]:15s} v1 yes={y1m:.02f} -> LP yes={y2m:.02f}{move}')
print()
print("=== C) kappa (name arm, llama70b x qwen25-32b), by kind: v1 -> LP ===")
def kap(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<30: return np.nan
    x=(a[m]>0.5).astype(int); y=(b[m]>0.5).astype(int)
    po=(x==y).mean(); pe=(x.mean()*y.mean())+((1-x.mean())*(1-y.mean()))
    return (po-pe)/(1-pe) if pe<1 else np.nan
for kind in sorted(set(k1)):
    ks1=[];ks2=[]
    for b in bases:
        i=[i for i,x in enumerate(n1) if x==f'{b}||name']
        if not i or k1[i[0]]!=kind: continue
        e1,e2='llama70b','qwen25-32b'
        a1=data[e1][2][[j for j,x in enumerate(data[e1][0]) if x==f'{b}||name'][0]]
        b1=data[e2][2][[j for j,x in enumerate(data[e2][0]) if x==f'{b}||name'][0]]
        a2=dataLP[e1][2][[j for j,x in enumerate(dataLP[e1][0]) if x==f'{b}||name'][0]]
        b2=dataLP[e2][2][[j for j,x in enumerate(dataLP[e2][0]) if x==f'{b}||name'][0]]
        ks1.append(kap(a1,b1)); ks2.append(kap(a2,b2))
    ks1=[k for k in ks1 if np.isfinite(k)]; ks2=[k for k in ks2 if np.isfinite(k)]
    if ks1 or ks2:
        print(f'{kind:20s} v1 med-kappa={np.median(ks1) if ks1 else float("nan"):.02f} (n={len(ks1)})  ->  LP med-kappa={np.median(ks2) if ks2 else float("nan"):.02f} (n={len(ks2)})')
