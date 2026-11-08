import re,sys
def conv(src, dst):
    txt=open(src).read()
    out=[]
    for name in ['nmos_rvt','pmos_rvt']:
        m=re.search(r'\.model\s+%s\s+(nmos|pmos)\s+level\s*=\s*72(.*?)(?=\n\.model|\Z)'%name, txt, re.S)
        body=m.group(2)
        lines=[]
        for ln in body.split('\n'):
            if ln.strip().startswith('*') or not ln.strip(): continue
            ln=re.sub(r'version\s*=\s*\S+','',ln)
            lines.append(ln)
        out.append('.model %s bsimcmg_va\n+type = %d\n'%(name, 1 if m.group(1)=='nmos' else 0)+'\n'.join(lines))
    open(dst,'w').write('\n'.join(out)+'\n')
for c in ['TT','SS','FF']:
    conv(f'asap7_pdk_r1p7/models/hspice/7nm_{c}.pm', f'asap7_{c}.mod')
