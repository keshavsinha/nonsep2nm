"""Step 0: switching activity for every benchmark.

Writes cache2/<design>_N2_act.npy, which prep_all2.py reads for both nodes
(activity comes from logic simulation and does not depend on the node).
Settings match nsim.experiment.prepare(): 500 cycles, 64 parallel random
input streams, seed 7, reset held inactive and enable held active.
"""
import os, sys
sys.path.insert(0, '.')
import numpy as np
from nsim.liberty import read_liberty
from nsim.design import Design
from nsim.experiment import LIBS

os.makedirs('cache2', exist_ok=True)
lib = read_liberty(LIBS)
designs = [('counter4', 'syn/counter4.v'), ('simonpipe', 'syn/simonpipe.v'), ('gcd', 'syn/gcd.v'),
           ('riscv', 'syn/riscv.v'), ('aes', 'syn/aes.v')]
for nm, f in designs:
    d = Design(f, lib, nm)
    held = {d.find(n): (1 if n in ('en', 'ld') else 0) for n in d.inputs
            if n in ('rst', 'reset', 'en', 'rst_n')}
    act = d.simulate_activity(cycles=500, width=64, held=held, seed=7)
    np.save(f'cache2/{nm}_N2_act.npy', act)
    print(nm, 'mean toggle rate %.3f' % act.mean(), flush=True)
