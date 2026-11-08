from r2_common import *
S = Store('cache2/dtco.json')
N = 20000
for nm in ['simonpipe', 'gcd', 'aes']:
    base = L(nm); lo = L(nm, 'N2', '_u50')
    pd_b = base.Pbin_nom.max() / (base.W * base.H / base.nb ** 2)
    pd_l = lo.Pbin_nom.max() / (lo.W * lo.H / lo.nb ** 2)
    r = pd_l / pd_b
    th = dict(dT_loc=base.cfg['dT_loc'] * 2.0, T_rise=base.cfg['T_rise'] + 5.0)
    cfgs = {'baseline': (base, {}),
            'D1_BSPDN': (base, dict(ir_scale=0.637)),
            'D1t_BSPDN_thermal': (base, dict(ir_scale=0.637, **th)),
            'D2_lowutil': (lo, dict(ir_scale=r, dT_loc=base.cfg['dT_loc'] * r)),
            'D3_regular': (base, dict(kappa=base.cfg['kappa'] * 0.5, p_tail=base.cfg['p_tail'] * 0.5)),
            'D4_all': (lo, dict(ir_scale=0.637 * r, dT_loc=base.cfg['dT_loc'] * r * 2.0, T_rise=base.cfg['T_rise'] + 5.0,
                                kappa=base.cfg['kappa'] * 0.5, p_tail=base.cfg['p_tail'] * 0.5))}
    for scen, extra in [('budget', {}), ('stress', STRESS)]:
        for k, (P, dt0) in cfgs.items():
            dt = dict(extra); dt.update(dt0)
            if 'ir_scale' not in dt0: pass
            c = run_mc(P, N=N, seed=71, dtco=dt); s = run_mc(P, N=N, seed=71, dtco=dt, context=False, shared_w=False)
            cs = corner_stack(P, dtco=dt)
            S.put(f'{nm}|{scen}|{k}', dict(c50=q(c, .5), c99=q(c, .99), c999=q(c, .999), s999=q(s, .999), corner=float(cs['add']),
                                           corner_rss=float(cs['add_rss']), Tnom=P.tm.Tmin_nom, area=float(P.W * P.H * P.cfg['area_scale']), pdratio=float(r)))
            print(nm, scen, k, round(S.d[f'{nm}|{scen}|{k}']['c999'] * P.tm.Tmin_nom / base.tm.Tmin_nom, 4), flush=True)
