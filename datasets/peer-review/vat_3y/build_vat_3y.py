import pandas as pd, os, re, hashlib, json
base='/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review'
out=os.path.join(base,'vat_3y'); os.makedirs(out, exist_ok=True)
def load(paths, cols): return pd.concat([pd.read_csv(os.path.join(base,p), usecols=cols) for p in paths], ignore_index=True)
def norm(s):
    s=str(s).lower(); s=re.sub(r'[^a-z0-9 ]',' ',s); return re.sub(r'\s+',' ',s).strip()
def get_title(t):
    m=re.match(r'\s*(?:title\s*[:\-]\s*)?(.+?)(?:\n|abstract\s*[:\-]|$)', str(t), re.I)
    return norm(m.group(1))[:200] if m else ''
def split_of(key):  # stable, salted, group-safe (group=key)
    h=int(hashlib.md5(('split::'+str(key)).encode()).hexdigest(),16)%10
    return 'train' if h<8 else ('eval' if h==8 else 'test')

# VERDICT: ICLR submissions accept/reject
V=load([f'splits/{s}.csv.gz' for s in ['train','eval','test']], ['id','text','venue','year','judgement'])
V=V[V.venue.astype(str).str.contains('ICLR',case=False)].copy()
V['ntitle']=V.text.map(get_title); V['rung']='verdict'; V['grp']=V.ntitle
V['split']=V.grp.map(split_of)

# CURATION: ICLR accepted oral/spotlight vs poster
C=load([f'oral_spotlight/{s}.csv.gz' for s in ['train','eval','test']], ['id','text','judgement','venue_key','year'])
C=C[C.venue_key.astype(str).str.contains('iclr',case=False)].copy()
C['ntitle']=C.text.map(get_title); C['venue']='ICLR'; C['rung']='curation'; C['grp']=C.ntitle
C['split']=C.grp.map(split_of)

# REVEALED: ICLR citation-percentile cell (raw v2)
R=pd.read_csv(os.path.join(base,'openalex_citations/openalex_citations_v2.csv.gz'),
              usecols=['id','text','title','judgement','percentile','venue','year'])
R=R[R.venue.astype(str).str.contains('ICLR',case=False)].copy()
R['ntitle']=R.title.map(norm); R['rung']='revealed'; R['grp']=R.ntitle
R['split']=R.grp.map(split_of)

# apples-to-apples shared set: normalized title in BOTH curation and revealed
shared=set(C.ntitle)&set(R.ntitle)-{''}
C['in_shared']=C.ntitle.isin(shared); R['in_shared']=R.ntitle.isin(shared)

keep=['id','ntitle','text','judgement','venue','year','rung','split','in_shared']
for name,df in [('verdict',V),('curation',C),('revealed',R)]:
    if 'in_shared' not in df: df['in_shared']=False
    d=df[[c for c in keep if c in df]].copy()
    p=os.path.join(out,f'{name}.jsonl'); d.to_json(p,orient='records',lines=True)
    pos=d.judgement.astype(float).mean()
    print(f'{name}: n={len(d)} pos={pos:.3f} shared={int(d.in_shared.sum())} splits={d.split.value_counts().to_dict()}')
print('shared curation<->revealed titles:', len(shared))
print('written to', out)
