import json, numpy as np
from scipy.optimize import least_squares
r=json.load(open('spice_fo4.json'))
Vs=[0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85]; Ts=[-40,0,25,50,85,125]
T0=298.15
def g(p,V,Tc,dv=0):
    Vth0,a,m,kT,c=p
    T=np.asarray(Tc)+273.15; vth=Vth0+dv-kT*(T-T0)
    return c*V/((T/T0)**(-m)*np.maximum(V-vth,0.02)**a)
pts=[(V,T,r[f'TT|{V}|{T}']) for V in Vs for T in Ts if V>=0.5]
V=np.array([p[0] for p in pts]); T=np.array([p[1] for p in pts]); D=np.array([p[2] for p in pts])
res=least_squares(lambda p: np.log(g(p,V,T))-np.log(D), [0.25,1.5,1.0,0.8e-3,5], bounds=([0.05,0.8,0,0,0.01],[0.5,3,3,3e-3,100]))
p=res.x; err=np.exp(np.abs(res.fun))-1
print('Vth0 %.3f alpha %.3f m %.3f kT %.2f mV/K  max err %.1f%% rms %.1f%%'%(p[0],p[1],p[2],p[3]*1e3,err.max()*100,np.sqrt(np.mean(res.fun**2))*100))
# ZTC
from scipy.optimize import brentq
f=lambda v: g(p,v,26)-g(p,v,24); print('ZTC %.3f V'%brentq(f,0.42,0.9))
# corner equivalent dVth (fit SS, FF over grid)
for c in ['SS','FF']:
    Dc=np.array([r[f'{c}|{v}|{t}'] for v,t in zip(V,T)])
    rr=least_squares(lambda q: np.log(g(p,V,T,q[0]))-np.log(Dc),[0.0])
    print(c,'dVth %.1f mV  rms %.1f%%'%(rr.x[0]*1e3,np.sqrt(np.mean(rr.fun**2))*100))
json.dump(dict(Vth0=p[0],alpha=p[1],m=p[2],kT=p[3],c=p[4]),open('fit.json','w'))
# also with 0.45 included check
