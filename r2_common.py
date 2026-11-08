import sys, json, pickle, time, os
sys.path.insert(0, '.')
import numpy as np
from nsim.experiment import *
os.makedirs('cache2', exist_ok=True)
QS = (0.5, 0.99, 0.99865, 0.999)
def L(nm, node='N2', tag=''):
    return pickle.load(open(f'cache2/{nm}_{node}{tag}.pkl', 'rb'))
def qs(D): return {str(p): q(D, p) for p in QS}
def boot_err(c, s, B=300, p=0.999, seed=0):
    rng = np.random.default_rng(seed); N = len(c); e = []; qc = []
    for _ in range(B):
        i = rng.integers(0, N, N); a = np.quantile(c[i], p); b = np.quantile(s[i], p)
        e.append((b - a) / a * 100); qc.append(a)
    return dict(err=float((np.quantile(s, p) - np.quantile(c, p)) / np.quantile(c, p) * 100),
                lo=float(np.percentile(e, 2.5)), hi=float(np.percentile(e, 97.5)),
                qlo=float(np.percentile(qc, 2.5)), qhi=float(np.percentile(qc, 97.5)))
class Store:
    def __init__(self, f):
        self.f = f; self.d = json.load(open(f)) if os.path.exists(f) else {}
    def put(self, k, v):
        self.d[k] = v; json.dump(self.d, open(self.f, 'w'), indent=1, default=float)
