import pandas as pd, os, json, random
random.seed(0)
d='/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/vat_3y'
V=pd.read_json(os.path.join(d,'verdict.jsonl'),lines=True)
C=pd.read_json(os.path.join(d,'curation.jsonl'),lines=True)
R=pd.read_json(os.path.join(d,'revealed.jsonl'),lines=True)
Cset=set(C.ntitle)
score={}
def add(row,src):
    k=row['ntitle']
    if not k: return
    score.setdefault(k,{'ntitle':k,'text':row['text'],'srcs':set()})['srcs'].add(src)
# revealed: all (smallest, balanced; ⊃ shared-4296)
for _,r in R.iterrows(): add(r,'revealed')
# curation: all pos + neg sample -> balanced ~ (also completes any shared not in R, none)
for _,r in C[C.judgement==1].iterrows(): add(r,'curation')          # 2016 pos
for _,r in C[C.judgement==0].sample(4000,random_state=0).iterrows(): add(r,'curation')  # neg
# verdict: balanced 3000 rej + 3000 acc(not already in curation)
for _,r in V[V.judgement==0].sample(3000,random_state=0).iterrows(): add(r,'verdict')
Vacc=V[(V.judgement==1)&(~V.ntitle.isin(Cset))]
for _,r in Vacc.sample(min(3000,len(Vacc)),random_state=0).iterrows(): add(r,'verdict')
out=os.path.join(d,'union_toscore.jsonl')
with open(out,'w') as fh:
    for k,v in score.items(): fh.write(json.dumps({'ntitle':k,'text':v['text'],'srcs':sorted(v['srcs'])})+'\n')
from collections import Counter
c=Counter()
for v in score.values():
    for s in v['srcs']: c[s]+=1
print('unique abstracts:', len(score), '| prompts ~', len(score)*154)
print('coverage by rung:', dict(c))
print('shared-4296 covered:', len(set(R[R.in_shared].ntitle)&set(score)), 'of', int(R.in_shared.sum()))
