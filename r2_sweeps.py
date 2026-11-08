from r2_common import *
S = Store('cache2/sweeps.json')
N = 20000
# rho sweep at budget and stress droop
for nm in ['simonpipe', 'gcd', 'aes']:
    P = L(nm)
    for scen, dt in [('budget', {}), ('stress', STRESS)]:
        if f'rho|{nm}|{scen}|1.0' in S.d: continue
        Nn = 10000 if nm == 'aes' else N
        s = run_mc(P, N=Nn, seed=31, dtco=dt, context=False, shared_w=False)
        for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
            c = run_mc(P, N=Nn, seed=31, dtco=dt, rho=rho)
            S.put(f'rho|{nm}|{scen}|{rho}', boot_err(c, s, B=200))
        print('rho', nm, scen, [round(S.d[f'rho|{nm}|{scen}|{r}']['err'], 2) for r in [0.0, 0.25, 0.5, 0.75, 1.0]], flush=True)
# supply sweep
for nm in ['simonpipe', 'gcd', 'aes']:
    P = L(nm)
    Vl = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80] if nm != 'aes' else [0.50, 0.60, 0.65, 0.70, 0.80]
    N = 10000 if nm == 'aes' else 20000
    for V0 in Vl:
        dt = dict(V0=V0)
        c = run_mc(P, N=N, seed=41, dtco=dt); s = run_mc(P, N=N, seed=41, dtco=dt, context=False, shared_w=False)
        r1 = run_mc(P, N=N, seed=41, dtco=dt, rho=1.0)
        cs = corner_stack(P, dtco=dt)
        S.put(f'V|{nm}|{V0}', dict(c=boot_err(c, s, B=200), r1=boot_err(r1, s, B=200), corner={k: float(v) for k, v in cs.items()}, cq=q(c, .999)))
        print('V', nm, V0, round(S.d[f'V|{nm}|{V0}']['c']['err'], 2), round(S.d[f'V|{nm}|{V0}']['r1']['err'], 2), round(cs['joint'] / cs['add'] - 1, 3), flush=True)
# IR sweep
for nm in ['simonpipe', 'gcd', 'aes']:
    P = L(nm)
    N = 10000 if nm == 'aes' else 20000
    for ir in ([0.02, 0.04, 0.06, 0.08, 0.12] if nm != 'aes' else [0.02, 0.04, 0.08, 0.12]):
        dt = dict(ir_target=ir)
        c = run_mc(P, N=N, seed=51, dtco=dt); s = run_mc(P, N=N, seed=51, dtco=dt, context=False, shared_w=False)
        r1 = run_mc(P, N=N, seed=51, dtco=dt, rho=1.0)
        S.put(f'IR|{nm}|{ir}', dict(c=boot_err(c, s, B=200), r1=boot_err(r1, s, B=200)))
        print('IR', nm, ir, round(S.d[f'IR|{nm}|{ir}']['c']['err'], 2), round(S.d[f'IR|{nm}|{ir}']['r1']['err'], 2), flush=True)
# tails x context x depth
def boot_inc(a, b, B=200, seed=0):
    rng = np.random.default_rng(seed); n = len(a); e = []
    for _ in range(B):
        i = rng.integers(0, n, n); e.append((np.quantile(a[i], .999) / np.quantile(b[i], .999) - 1) * 100)
    return dict(v=float((np.quantile(a, .999) / np.quantile(b, .999) - 1) * 100), lo=float(np.percentile(e, 2.5)), hi=float(np.percentile(e, 97.5)))
for nm in ['counter4', 'simonpipe', 'gcd', 'aes']:
    P = L(nm)
    N = 10000 if nm == 'aes' else 20000
    for reg, (p, med) in {'base': (0.008, 0.10), 'severe': (0.03, 0.25)}.items():
        nt = run_mc(P, N=N, seed=61, dtco=dict(p_tail=p, tail_med=med), tails=False)
        for kap in [0.0, 1.0, 2.0]:
            dt = dict(p_tail=p, tail_med=med, kappa=kap)
            c = run_mc(P, N=N, seed=61, dtco=dt)
            S.put(f'T|{nm}|{reg}|{kap}', dict(inc=boot_inc(c, nt), depth=len(P.tm.critical_path())))
        print('T', nm, reg, [round(S.d[f'T|{nm}|{reg}|{k}']['inc']['v'], 2) for k in [0.0, 1.0, 2.0]], flush=True)
