import json, re
O='/lfs/skampere3/0/alexspan/outputs/osl_multi'
for author in ['llama70b','qwen25-72b']:
    d=json.load(open(f'{O}/zxa_authoring_fam/{author}.json'))
    rows=d if isinstance(d,list) else d.get('rows',d)
    if isinstance(rows,dict): rows=list(rows.values())
    print(f'--- {author}: {len(rows)} rows; keys={sorted(rows[0].keys())}')
    nval=sum(1 for r in rows if r.get('valid'))
    print(f'valid={nval}')
    fails={}
    for r in rows:
        if r.get('valid'): continue
        ex=r.get('explanation') or ''; ds=r.get('dossier') or ''
        ew=len(ex.split()); dw=len(ds.split())
        labs=[L for L in ['DEFINITION','WHAT COUNTS','CONTRAST EXEMPLARS','BOUNDARY CASES'] if L in ds]
        why=r.get('why_invalid') or r.get('invalid_reason') or ''
        key=(f'ew={"lo" if ew<130 else "hi" if ew>180 else "ok"}', f'dw={"lo" if dw<360 else "hi" if dw>450 else "ok"}', f'labs={len(labs)}', str(why)[:40])
        fails[key]=fails.get(key,0)+1
    for k,v in sorted(fails.items(),key=lambda x:-x[1])[:8]: print('  ',v,k)
    # sample one invalid row verbatim ends
    for r in rows:
        if not r.get('valid'):
            ex=r.get('explanation') or ''; ds=r.get('dossier') or ''
            print(f'  SAMPLE invalid base={str(r.get("base") or r.get("metric"))[:40]!r} ew={len(ex.split())} dw={len(ds.split())}')
            print('   expl-head:', repr(ex[:120]))
            print('   doss-head:', repr(ds[:160]))
            print('   doss-tail:', repr(ds[-120:]))
            break
