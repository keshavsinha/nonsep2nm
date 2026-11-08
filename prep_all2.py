import sys, time, pickle; sys.path.insert(0,'.')
import numpy as np
from nsim.liberty import read_liberty
from nsim.experiment import *
lib = read_liberty(LIBS)
designs = [('counter4','syn/counter4.v'),('simonpipe','syn/simonpipe.v'),('gcd','syn/gcd.v'),('riscv','syn/riscv.v'),('aes','syn/aes.v')]
for node in ['N2','N7']:
    for nm, f in designs:
        t=time.time()
        A = np.load(f'cache2/{nm}_N2_act.npy')
        P = prepare(nm, f, lib, node, act=A)
        P.d.lib = None; P.d.find = None
        pickle.dump(P, open(f'cache2/{nm}_{node}.pkl','wb'))
        print(node, nm, len(P.d.inst), 'Tnom %.1f ps'%P.tm.Tmin_nom, 'depth', len(P.tm.critical_path()), 'P %.3f mW'%(P.pdyn.sum()*1e3), 'clk %.1f'%P.clk_nom, '%.0fs'%(time.time()-t), flush=True)
    if node=='N2':
        # low-utilisation layouts for DTCO
        for nm, f in [('gcd','syn/gcd.v'),('aes','syn/aes.v'),('simonpipe','syn/simonpipe.v')]:
            P = prepare(nm, f, lib, 'N2', util=0.50, act=np.load(f'cache2/{nm}_N2_act.npy')); P.d.lib=None; P.d.find=None
            pickle.dump(P, open(f'cache2/{nm}_N2_u50.pkl','wb')); print('u50', nm, P.tm.Tmin_nom, flush=True)
