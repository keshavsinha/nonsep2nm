import sys, json, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figs import C, INK, MUTED, GRID, fig_device2
from nsim.experiment import corner_stack
from nsim.physics import Device
import os
OUT = 'figs/'
os.makedirs(OUT, exist_ok=True)
# use freshly generated results in cache2/ if present, otherwise the shipped results/
J = lambda f: json.load(open(('cache2/' if os.path.exists('cache2/' + f) else 'results/') + f))

def fig_maps():
    P = pickle.load(open('cache2/aes_N2.pkl', 'rb')); nb = P.nb
    pc = np.exp(P.cfg['kappa'] * P.z_bin); pc = pc / pc.mean() * P.cfg['p_tail']
    data = [(P.dens, 'Cell utilisation', 'Blues'), (P.rudy, 'RUDY demand / capacity', 'Blues'),
            (P.ir_map.reshape(nb, nb) * P.cfg['ir_target'] * 100, 'Nominal IR drop (% of V0)', 'Oranges'),
            (pc.reshape(nb, nb) * 100, 'Tail probability per cell (%)', 'Oranges')]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2))
    path = P.tm.critical_path(); px = P.gx[path] + 0.5; py = P.gy[path] + 0.5
    for a, (m, t, cm) in zip(ax, data):
        im = a.imshow(m, origin='lower', cmap=cm, extent=(0, nb, 0, nb))
        a.plot(px, py, color=INK, lw=1, marker='o', ms=2)
        a.set_title(t, fontsize=8); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(OUT + 'fig_maps.png'); plt.close(fig)

def fig_yield():
    P = pickle.load(open('cache2/aes_N2.pkl', 'rb'))
    lab = {'coupled': 'Coupled MC (ρ = 0.5)', 'sep': 'Separable MC', 'gauss_sep': 'Separable MC, Gaussian only', 'noclk': 'Coupled MC, ideal clock'}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for i, (m, l) in enumerate(lab.items()):
        D = np.sort(np.load(f'cache2/D_aes_N2_{m}.npy')); y = np.arange(1, len(D) + 1) / len(D)
        ls = ['-', '--', ':', '-.'][i]
        ax[0].plot(D, 1 - y + 1e-9, color=C[i], ls=ls, label=l)
    cs = corner_stack(P)
    for k, (lbl, col, ls) in {'add': ('additive', C[4], (0, (1, 1))), 'add_rss': ('RSS process', '#555555', '--'), 'rss_all': ('fully RSS', '#999999', ':')}.items():
        ax[0].axvline(cs[k], color=col, ls=ls, lw=1.3); ax[0].text(cs[k] + 0.005, 0.3 if k == 'add' else (0.05 if k == 'add_rss' else 0.006), lbl, fontsize=7, color=INK, rotation=90)
    ax[0].set_yscale('log'); ax[0].set_ylim(3e-5, 1); ax[0].set_xlim(1.0, 1.72)
    ax[0].set_xlabel('Required clock period / nominal'); ax[0].set_ylabel('Timing failure probability')
    ax[0].set_title('(a) Upper tail, AES at N2', fontsize=9, loc='left'); ax[0].legend(fontsize=6.5, loc='upper right', bbox_to_anchor=(1.0, 0.72))
    R = J('core.json'); names = ['simonpipe', 'gcd', 'aes', 'riscv']; lbls = ['Pipeline', 'GCD', 'AES', 'RISC-V']
    x = np.arange(4); w = 0.2
    for j, (k, l) in enumerate([('add', 'Additive'), ('add_rss', 'RSS process'), ('rss_all', 'Fully RSS'), ('joint', 'Simultaneous')]):
        v = [(R[f'{n}_N2']['corner'][k] / R[f'{n}_N2']['coupled']['0.99865'] - 1) * 100 for n in names]
        ax[1].bar(x + (j - 1.5) * w, v, w - 0.02, color=[C[0], C[1], C[2], C[4]][j], label=l)
    ax[1].axhline(0, color=MUTED, lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels(lbls, fontsize=8)
    ax[1].set_ylabel('Corner over coupled 99.865 % (%)'); ax[1].legend(fontsize=7, ncol=2)
    ax[1].set_title('(b) Corner pessimism at N2', fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(OUT + 'fig_yield.png'); plt.close(fig)

def fig_rho():
    S = J('sweeps.json'); rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for j, scen in enumerate(['budget', 'stress']):
        for i, (nm, l) in enumerate([('simonpipe', 'Pipeline'), ('gcd', 'GCD'), ('aes', 'AES')]):
            e = [S[f'rho|{nm}|{scen}|{r}'] for r in rhos]
            xs = np.array(rhos) + (i - 1) * 0.015
            ax[j].errorbar(xs, [d['err'] for d in e], yerr=[[d['err'] - d['lo'] for d in e], [d['hi'] - d['err'] for d in e]],
                           color=C[i], marker='o', ms=4, lw=1.5, capsize=2, ls=['-', '--', '-.'][i], label=l)
        ax[j].axhline(0, color=MUTED, lw=0.8)
        ax[j].set_xlabel('Workload correlation ρ (droop vs thermal)')
        ax[j].set_title('(%s) %s droop' % ('ab'[j], '10 % budget' if scen == 'budget' else 'Stress (front-side)'), fontsize=9, loc='left')
    ax[0].set_ylabel('Separable error ε_sep at 99.9 % (%)'); ax[0].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT + 'fig_rho.png'); plt.close(fig)

def fig_sweeps():
    S = J('sweeps.json')
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for i, (nm, l) in enumerate([('simonpipe', 'Pipeline'), ('gcd', 'GCD'), ('aes', 'AES')]):
        Vs = sorted(float(k.split('|')[2]) for k in S if k.startswith(f'V|{nm}|'))
        ax[0].plot(Vs, [S[f'V|{nm}|{v}']['r1']['err'] for v in Vs], color=C[i], marker='o', ms=3.5, ls=['-', '--', '-.'][i], label=l)
        Vb = [v for v in Vs if v >= 0.6]
        ax[1].plot(Vb, [(S[f'V|{nm}|{v}']['corner']['joint'] / S[f'V|{nm}|{v}']['corner']['add'] - 1) * 100 for v in Vb], color=C[i], marker='o', ms=3.5, ls=['-', '--', '-.'][i], label=l)
        irs = sorted(float(k.split('|')[2]) for k in S if k.startswith(f'IR|{nm}|'))
        if irs: ax[2].plot([r * 100 for r in irs], [S[f'IR|{nm}|{r}']['r1']['err'] for r in irs], color=C[i], marker='o', ms=3.5, ls=['-', '--', '-.'][i], label=l)
    z = Device(V0=0.65).ztc()
    ax[0].axvline(z, color=MUTED, lw=1, ls=':')
    ax[0].text(z + 0.005, ax[0].get_ylim()[1] * 0.85, 'ZTC', fontsize=8, color=MUTED)
    for a in (ax[0], ax[2]): a.axhline(0, color=MUTED, lw=0.8)
    ax[0].set_xlabel('Nominal supply V0 (V)'); ax[0].set_ylabel('ε_sep at 99.9 %, ρ = 1 (%)'); ax[0].legend(fontsize=7.5)
    ax[0].set_title('(a) Separable error', fontsize=9, loc='left')
    ax[1].set_xlabel('Nominal supply V0 (V)'); ax[1].set_ylabel('Simultaneous over additive (%)')
    ax[1].set_title('(b) Corner superadditivity', fontsize=9, loc='left')
    ax[2].set_xlabel('Worst-bin IR drop (% of V0)'); ax[2].set_ylabel('ε_sep at 99.9 %, ρ = 1 (%)')
    ax[2].set_title('(c) PDN strength', fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(OUT + 'fig_sweeps.png'); plt.close(fig)

def fig_tails():
    S = J('sweeps.json')
    names = [('counter4', 'Counter (3)'), ('simonpipe', 'Pipeline (4)'), ('aes', 'AES (16)'), ('gcd', 'GCD (16)')]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for j, reg in enumerate(['base', 'severe']):
        x = np.arange(len(names)); w = 0.26
        for i, kap in enumerate([0.0, 1.0, 2.0]):
            d = [S[f'T|{nm}|{reg}|{kap}']['inc'] for nm, _ in names]
            ax[j].bar(x + (i - 1) * w, [v['v'] for v in d], w - 0.02, color=C[i], label=f'κ = {kap:.0f}',
                      yerr=[[v['v'] - v['lo'] for v in d], [v['hi'] - v['v'] for v in d]], capsize=2, error_kw=dict(lw=0.8))
        ax[j].set_xticks(x); ax[j].set_xticklabels([l for _, l in names], fontsize=7.5); ax[j].axhline(0, color=MUTED, lw=0.8)
        ax[j].set_title('(%s) %s' % ('ab'[j], 'Baseline: p0 = 0.8 %, median 10 %' if reg == 'base' else 'Hypothetical severe: p0 = 3 %, median 25 %'), fontsize=8.5, loc='left')
    ax[0].set_ylabel('Increase of 99.9 % delay\nfrom EUV tails (%)'); ax[1].legend(fontsize=7.5, title='Context sensitivity', title_fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT + 'fig_tails.png'); plt.close(fig)

def fig_tornado():
    S = J('tornado.json')
    lab = {'ir_target': 'IR budget δ_IR', 'chip_droop': 'Chip droop δ_c', 'sig_w': 'Workload σ_w', 'rho': 'Correlation ρ', 'sig_g': 'Global σ(Vth)',
           'sig_sp': 'Spatial σ(Vth)', 'sig_loc': 'Local σ(Vth)', 'p_tail': 'Tail probability p0', 'tail_med': 'Tail median', 'kappa': 'Context κ',
           'T_rise': 'Workload ΔT', 'dT_loc': 'Local ΔT', 'Vth0': 'Device Vth0', 'kT': 'Device k_T', 'alpha': 'Device α'}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    nm = 'gcd'; base = S[f'{nm}|base']
    keys = sorted(lab, key=lambda k: abs(S[f'{nm}|{k}']['hi']['c'] - S[f'{nm}|{k}']['lo']['c']))
    y = np.arange(len(keys))
    for j, (met, ttl, sc) in enumerate([('c', '(a) Coupled 99.9 % period, change (%)', lambda v: (v / base['c'] - 1) * 100), ('err', '(b) ε_sep at 99.9 % (%)', lambda v: v)]):
        for i, k in enumerate(keys):
            lo = sc(S[f'{nm}|{k}']['lo'][met]); hi = sc(S[f'{nm}|{k}']['hi'][met])
            ax[j].barh(i - 0.18, lo - (0 if met == 'c' else 0), 0.34, color=C[0], label='low value' if i == 0 else None)
            ax[j].barh(i + 0.18, hi, 0.34, color=C[1], label='high value' if i == 0 else None)
        if met == 'err': ax[j].axvline(base['err'], color=MUTED, ls=':', lw=1)
        ax[j].axvline(0, color=MUTED, lw=0.8); ax[j].set_title(ttl, fontsize=8.5, loc='left')
    ax[0].set_yticks(y); ax[0].set_yticklabels([lab[k] for k in keys], fontsize=7.5); ax[0].legend(fontsize=7.5, loc='lower right')
    fig.tight_layout(); fig.savefig(OUT + 'fig_tornado.png'); plt.close(fig)

def fig_flow():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 42)
    def box(x, y, w, h, t, fc='#eef4fc'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=C[0], lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=7.0, color=INK)
    def arr(x0, y0, x1, y1): ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', color=MUTED, lw=1))
    box(1, 31, 15, 8, 'RTL benchmarks\n(counter, pipeline,\nGCD, RISC-V, AES)')
    box(21, 31, 16, 8, 'Yosys + ABC\n(ORFS speed script)\nto ASAP7 RVT')
    box(42, 31, 17, 8, 'Quadratic placement,\nrise/fall NLDM STA\n(checked vs OpenTimer)')
    box(64, 31, 16, 8, 'Simulated activity;\npower per cell/bin;\nclock quad-tree')
    box(84, 31, 15, 8, 'Maps: utilisation,\nRUDY, IR, thermal')
    for a, b in [(16.5, 20.5), (37.5, 41.5), (59.5, 63.5), (80.5, 83.5)]: arr(a, 35, b, 35)
    box(1, 16, 22, 9, 'BSIM-CMG SPICE (ngspice):\nFO4 over V, T and corners;\nalpha-power fit, ZTC', fc='#fdf0ea')
    box(27, 16, 22, 9, 'Variation: global, spatial,\nlocal Vth; EUV tails;\nfast and slow workload (ρ)', fc='#fdf0ea')
    box(53, 16, 22, 9, 'Vectorised Monte Carlo STA\nwith clock latency,\n40 000 chips per case', fc='#fdf0ea')
    box(79, 16, 20, 9, 'Couplings switched\non or off; sweeps\nand sensitivity', fc='#fdf0ea')
    arr(91, 30.5, 91, 25.5); arr(23.5, 20.5, 26.5, 20.5); arr(49.5, 20.5, 52.5, 20.5); arr(78.5, 20.5, 75.5, 20.5)
    box(10, 2, 24, 8, 'Corners: additive, RSS,\nsimultaneous', fc='#eaf7f1')
    box(38, 2, 24, 8, 'Separable statistical\n(mechanisms independent)', fc='#eaf7f1')
    box(66, 2, 24, 8, 'Coupled statistical\n(reference)', fc='#eaf7f1')
    arr(64, 15.5, 22, 10.5); arr(64, 15.5, 50, 10.5); arr(64, 15.5, 78, 10.5)
    fig.tight_layout(); fig.savefig(OUT + 'fig_flow.png'); plt.close(fig)

if __name__ == '__main__':
    for f in sys.argv[1:]: globals()[f]()
