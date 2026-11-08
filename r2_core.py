from r2_common import *
S = Store('cache2/core.json')
modes = {'coupled': dict(), 'sep': dict(context=False, shared_w=False), 'rho1': dict(rho=1.0),
         'no_ctx': dict(context=False), 'gauss_sep': dict(context=False, shared_w=False, tails=False),
         'noclk': dict(clock=False)}
for node in ['N2', 'N7']:
    for nm in ['simonpipe', 'gcd', 'aes', 'riscv']:
        P = L(nm, node); r = {}; arrs = {}
        for m, kw in modes.items():
            t = time.time(); D = run_mc(P, N=40000, seed=11, **kw); arrs[m] = D
            r[m] = qs(D); r[m]['mean'] = float(D.mean()); r[m]['std'] = float(D.std())
            np.save(f'cache2/D_{nm}_{node}_{m}.npy', D.astype(np.float32))
        r['boot_sep'] = boot_err(arrs['coupled'], arrs['sep'])
        r['boot_rho1'] = boot_err(arrs['rho1'], arrs['sep'])
        r['corner'] = {k: float(v) for k, v in corner_stack(P).items()}
        r['Tnom'] = P.tm.Tmin_nom; r['depth'] = len(P.tm.critical_path())
        S.put(f'{nm}_{node}', r)
        print(node, nm, 'c %.4f s %.4f rho1 %.4f err %.2f [%.2f,%.2f] corner add %.3f rss %.3f joint %.3f' % (
            r['coupled']['0.999'], r['sep']['0.999'], r['rho1']['0.999'], r['boot_sep']['err'], r['boot_sep']['lo'], r['boot_sep']['hi'],
            r['corner']['add'], r['corner']['add_rss'], r['corner']['joint']), flush=True)
# stress droop scenario (front-side PDN, revision-1 values)
for nm in ['simonpipe', 'gcd', 'aes', 'riscv']:
    P = L(nm, 'N2'); r = {}
    c = run_mc(P, N=40000, seed=12, dtco=STRESS); s = run_mc(P, N=40000, seed=12, dtco=STRESS, context=False, shared_w=False)
    r1 = run_mc(P, N=40000, seed=12, dtco=STRESS, rho=1.0)
    r = dict(coupled=qs(c), sep=qs(s), rho1=qs(r1), boot_sep=boot_err(c, s), boot_rho1=boot_err(r1, s))
    S.put(f'stress_{nm}', r); print('stress', nm, r['boot_sep'], r['boot_rho1'], flush=True)
