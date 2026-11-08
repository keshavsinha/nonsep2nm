from r2_common import *
S = Store('cache2/tornado.json')
N = 20000
# parameter: (low, high); device parameters perturb the calibrated fit
params = {'ir_target': (0.02, 0.08), 'chip_droop': (0.0075, 0.03), 'sig_w': (0.10, 0.30), 'rho': (0.25, 1.0),
          'sig_g': (0.006, 0.018), 'sig_sp': (0.0045, 0.0135), 'sig_loc': (0.009, 0.027), 'p_tail': (0.004, 0.016),
          'tail_med': (0.05, 0.20), 'kappa': (0.4, 1.6), 'T_rise': (20.0, 60.0), 'dT_loc': (2.25, 9.0),
          'Vth0': (0.30, 0.42), 'kT': (0.12e-3, 0.48e-3), 'alpha': (0.9, 1.3)}
for nm in ['simonpipe', 'gcd']:
    P = L(nm)
    c0 = run_mc(P, N=N, seed=81); s0 = run_mc(P, N=N, seed=81, context=False, shared_w=False)
    S.put(f'{nm}|base', dict(err=boot_err(c0, s0, B=100)['err'], c=q(c0, .999), sup=float(corner_stack(P)['joint'] / corner_stack(P)['add'] - 1)))
    for pn, (lo, hi) in params.items():
        out = {}
        for tag, v in (('lo', lo), ('hi', hi)):
            dt = {pn: v}
            c = run_mc(P, N=N, seed=81, dtco=dt); s = run_mc(P, N=N, seed=81, dtco=dt, context=False, shared_w=False)
            cs = corner_stack(P, dtco=dt)
            out[tag] = dict(err=boot_err(c, s, B=100)['err'], c=q(c, .999), sup=float(cs['joint'] / cs['add'] - 1), v=v)
        S.put(f'{nm}|{pn}', out); print(nm, pn, round(out['lo']['err'], 2), round(out['hi']['err'], 2), round(out['lo']['c'], 3), round(out['hi']['c'], 3), flush=True)
