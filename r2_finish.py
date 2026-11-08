from r2_common import *
import sys
part = sys.argv[1]
if part == 'sweeps':
    S = Store('cache2/sweeps.json')
    for nm in ['simonpipe', 'gcd']:
        P = L(nm)
        for ir in [0.02, 0.04, 0.06, 0.08, 0.12]:
            if f'IR|{nm}|{ir}' in S.d: continue
            dt = dict(ir_target=ir)
            c = run_mc(P, N=20000, seed=51, dtco=dt); s = run_mc(P, N=20000, seed=51, dtco=dt, context=False, shared_w=False)
            r1 = run_mc(P, N=20000, seed=51, dtco=dt, rho=1.0)
            S.put(f'IR|{nm}|{ir}', dict(c=boot_err(c, s, B=200), r1=boot_err(r1, s, B=200)))
        print('IR', nm, [round(S.d[f'IR|{nm}|{i}']['c']['err'], 2) for i in [0.02, 0.04, 0.06, 0.08, 0.12]], flush=True)
    def boot_inc(a, b, B=200):
        rng = np.random.default_rng(0); n = len(a); e = []
        for _ in range(B):
            i = rng.integers(0, n, n); e.append((np.quantile(a[i], .999) / np.quantile(b[i], .999) - 1) * 100)
        return dict(v=float((np.quantile(a, .999) / np.quantile(b, .999) - 1) * 100), lo=float(np.percentile(e, 2.5)), hi=float(np.percentile(e, 97.5)))
    for nm, N in [('counter4', 20000), ('simonpipe', 20000), ('gcd', 20000), ('aes', 5000)]:
        P = L(nm)
        for reg, (p, med) in {'base': (0.008, 0.10), 'severe': (0.03, 0.25)}.items():
            nt = run_mc(P, N=N, seed=61, dtco=dict(p_tail=p, tail_med=med), tails=False)
            for kap in [0.0, 1.0, 2.0]:
                c = run_mc(P, N=N, seed=61, dtco=dict(p_tail=p, tail_med=med, kappa=kap))
                S.put(f'T|{nm}|{reg}|{kap}', dict(inc=boot_inc(c, nt), depth=len(P.tm.critical_path()), N=N))
            print('T', nm, reg, [round(S.d[f'T|{nm}|{reg}|{k}']['inc']['v'], 2) for k in [0.0, 1.0, 2.0]], flush=True)
if part == 'dtco':
    S = Store('cache2/dtco.json')
    nm = 'aes'; base = L(nm); lo = L(nm, 'N2', '_u50')
    r = (lo.Pbin_nom.max() / (lo.W * lo.H)) / (base.Pbin_nom.max() / (base.W * base.H))
    th = dict(dT_loc=base.cfg['dT_loc'] * 2.0, T_rise=base.cfg['T_rise'] + 5.0)
    cfgs = {'baseline': (base, {}), 'D1_BSPDN': (base, dict(ir_scale=0.637)), 'D1t_BSPDN_thermal': (base, dict(ir_scale=0.637, **th)),
            'D2_lowutil': (lo, dict(ir_scale=r, dT_loc=base.cfg['dT_loc'] * r)),
            'D3_regular': (base, dict(kappa=0.4, p_tail=0.004)),
            'D4_all': (lo, dict(ir_scale=0.637 * r, dT_loc=base.cfg['dT_loc'] * r * 2.0, T_rise=45.0, kappa=0.4, p_tail=0.004))}
    for scen, extra in [('budget', {}), ('stress', STRESS)]:
        for k, (P, dt0) in cfgs.items():
            if f'{nm}|{scen}|{k}' in S.d: continue
            dt = dict(extra); dt.update(dt0)
            c = run_mc(P, N=6000, seed=71, dtco=dt); s = run_mc(P, N=6000, seed=71, dtco=dt, context=False, shared_w=False)
            cs = corner_stack(P, dtco=dt)
            S.put(f'{nm}|{scen}|{k}', dict(c50=q(c, .5), c99=q(c, .99), c999=q(c, .999), s999=q(s, .999), corner=float(cs['add']), corner_rss=float(cs['add_rss']),
                                           Tnom=P.tm.Tmin_nom, area=float(P.W * P.H * P.cfg['area_scale']), pdratio=float(r), N=6000))
            print(nm, scen, k, round(S.d[f'{nm}|{scen}|{k}']['c999'] * P.tm.Tmin_nom / base.tm.Tmin_nom, 4), flush=True)
if part == 'tornado':
    S = Store('cache2/tornado.json')
    params = {'ir_target': (0.02, 0.08), 'chip_droop': (0.0075, 0.03), 'sig_w': (0.10, 0.30), 'rho': (0.25, 1.0),
              'sig_g': (0.006, 0.018), 'sig_sp': (0.0045, 0.0135), 'sig_loc': (0.009, 0.027), 'p_tail': (0.004, 0.016),
              'tail_med': (0.05, 0.20), 'kappa': (0.4, 1.6), 'T_rise': (20.0, 60.0), 'dT_loc': (2.25, 9.0),
              'Vth0': (0.30, 0.42), 'kT': (0.12e-3, 0.48e-3), 'alpha': (0.9, 1.3)}
    for nm in ['gcd', 'simonpipe']:
        P = L(nm); N = 10000
        c0 = run_mc(P, N=N, seed=81); s0 = run_mc(P, N=N, seed=81, context=False, shared_w=False); cs0 = corner_stack(P)
        S.put(f'{nm}|base', dict(err=boot_err(c0, s0, B=50)['err'], c=q(c0, .999), sup=float(cs0['joint'] / cs0['add'] - 1)))
        for pn, (lo, hi) in params.items():
            out = {}
            for tag, v in (('lo', lo), ('hi', hi)):
                dt = {pn: v}
                c = run_mc(P, N=N, seed=81, dtco=dt); s = run_mc(P, N=N, seed=81, dtco=dt, context=False, shared_w=False)
                cs = corner_stack(P, dtco=dt)
                out[tag] = dict(err=boot_err(c, s, B=50)['err'], c=q(c, .999), sup=float(cs['joint'] / cs['add'] - 1), v=v)
            S.put(f'{nm}|{pn}', out)
        print('tornado', nm, 'done', flush=True)
