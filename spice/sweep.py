import subprocess, re, json, itertools, sys, os, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(tempfile.gettempdir(), 'nonsep2nm_fo4.sp')
tpl=open(os.path.join(HERE, 'inv.sp')).read()
def run(corner, V, T):
    s=tpl.replace('asap7_TT.mod',f'asap7_{corner}.mod').replace('.param vdd=0.7',f'.param vdd={V}').replace('.temp 25',f'.temp {T}')
    s=s.replace('val={vdd/2}',f'val={V/2}').replace('{vdd}',f'{V}')
    open(TMP,'w').write(s)
    o=subprocess.run(['ngspice','-b',TMP],capture_output=True,text=True,cwd=HERE).stdout
    tr=float(re.search(r'^tr\s*=\s*(\S+)',o,re.M).group(1)); tf=float(re.search(r'^tf\s*=\s*(\S+)',o,re.M).group(1))
    return (tr+tf)/2*1e12
res={}
Vs=[0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85]; Ts=[-40,0,25,50,85,125]
for c in ['TT','SS','FF']:
    for V,T in itertools.product(Vs,Ts):
        try: res[f'{c}|{V}|{T}']=run(c,V,T)
        except Exception as e: res[f'{c}|{V}|{T}']=None
    print(c, {T:round(res[f'{c}|0.7|{T}'],2) for T in Ts}, flush=True)
json.dump(res,open(os.path.join(HERE,'spice_fo4.json'),'w'),indent=1)
