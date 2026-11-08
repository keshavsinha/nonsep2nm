import sys, json, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from nsim.physics import Device
from nsim.experiment import NODES, corner_stack

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6']
INK, MUTED, GRID = '#222222', '#666666', '#e6e6e6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'savefig.dpi': 300, 'lines.linewidth': 2})
OUT = 'figs2/'


def fig_device():
    d = Device(V0=0.65)
    T = np.linspace(-40, 125, 100)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for i, V in enumerate([0.55, 0.60, 0.65, 0.70, 0.75]):
        ls = ['-', '--', '-', '-.', ':'][i]
        ax[0].plot(T, d.factor(V, T, 0), color=C[i], ls=ls, label=f'{V:.2f} V')
    ax[0].set_xlabel('Temperature (°C)'); ax[0].set_ylabel('Delay / delay(0.65 V, 25 °C)')
    ax[0].set_ylim(0.75, 1.72)
    ax[0].legend(ncol=3, fontsize=7.5, title='Supply', title_fontsize=8, loc='upper center')
    ax[0].set_title('(a) Temperature dependence of gate delay', fontsize=9, loc='left')
    V = np.linspace(0.5, 0.85, 120)
    s = (d.g(V, 26, 0) - d.g(V, 24, 0)) / 2 / d.g(V, 25, 0) * 100
    ax[1].plot(V, s, color=C[0])
    ax[1].axhline(0, color=MUTED, lw=0.8)
    z = d.ztc()
    ax[1].axvline(z, color=C[1], ls='--', lw=1.2)
    ax[1].text(z + 0.01, -0.17, f'ZTC ≈ {z:.2f} V', fontsize=8, color=INK)
    ax[1].text(0.51, 0.04, 'inverted\n(hotter is faster)', fontsize=8, color=MUTED)
    ax[1].text(0.75, -0.05, 'normal', fontsize=8, color=MUTED)
    ax[1].set_xlabel('Supply voltage (V)'); ax[1].set_ylabel('d ln(delay)/dT (% per K)')
    ax[1].set_title('(b) Temperature sensitivity and ZTC', fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(OUT + 'fig_device.png'); plt.close(fig)


def fig_maps(nm='aes', node='N2'):
    P = pickle.load(open(f'cache/{nm}_{node}.pkl', 'rb'))
    nb = P.nb
    kap = P.cfg['kappa']
    pc = np.exp(kap * P.z_bin); pc = pc / pc.mean() * P.cfg['p_tail']
    data = [(P.dens, 'Cell utilisation', 'Blues'), (P.rudy, 'RUDY demand / capacity', 'Blues'),
            (P.ir_map.reshape(nb, nb) * P.cfg['ir_target'] * 100, 'Nominal IR drop (% of V0)', 'Oranges'),
            (pc.reshape(nb, nb) * 100, 'Tail probability per cell (%)', 'Oranges')]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2))
    path = P.tm.critical_path()
    px = P.gx[path] + 0.5; py = P.gy[path] + 0.5
    for a, (m, t, cm) in zip(ax, data):
        im = a.imshow(m, origin='lower', cmap=cm, extent=(0, nb, 0, nb))
        a.plot(px, py, color=INK, lw=1, marker='o', ms=2)
        a.set_title(t, fontsize=8); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(OUT + f'fig_maps_{nm}.png'); plt.close(fig)


def fig_yield(nm='aes', node='N2'):
    P = pickle.load(open(f'cache/{nm}_{node}.pkl', 'rb'))
    lab = {'coupled': 'Coupled MC', 'sep': 'Separable MC', 'gauss_sep': 'Separable MC, Gaussian only'}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for i, (m, l) in enumerate(lab.items()):
        D = np.sort(np.load(f'cache/D_{nm}_{node}_{m}.npy'))
        y = np.arange(1, len(D) + 1) / len(D)
        ls = ['-', '--', ':'][i]
        ax[0].plot(D, y, color=C[i], ls=ls, label=l)
        ax[1].plot(D, 1 - y + 1e-9, color=C[i], ls=ls, label=l)
    add, parts = corner_stack(P)
    for a in ax:
        a.axvline(add, color=C[3], lw=1.4, ls='-.')
        a.axvline(parts['joint'], color=C[4], lw=1.4, ls=(0, (1, 1)))
    ax[0].text(add + 0.01, 0.55, 'additive\ncorner', fontsize=8, color=INK)
    ax[0].text(parts['joint'] - 0.27, 0.80, 'simultaneous\ncorner', fontsize=8, color=INK)
    ax[0].set_xlabel('Required clock period / nominal'); ax[0].set_ylabel('Timing yield')
    ax[0].set_title(f'(a) Timing yield, {nm.upper()} at N2', fontsize=9, loc='left')
    ax[0].legend(fontsize=7.5, loc='lower right')
    ax[1].set_yscale('log'); ax[1].set_ylim(5e-5, 1)
    ax[1].set_xlim(1.05, 1.45)
    ax[1].set_xlabel('Required clock period / nominal'); ax[1].set_ylabel('Timing failure probability')
    ax[1].set_title('(b) Upper tail (log scale)', fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(OUT + f'fig_yield_{nm}.png'); plt.close(fig)




def fig_sweeps():
    S = json.load(open('cache/sweeps.json'))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    Vs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for i, nm in enumerate(['gcd', 'aes']):
        e = [(S['B'][f'{nm}|{v}']['s'] - S['B'][f'{nm}|{v}']['c']) / S['B'][f'{nm}|{v}']['c'] * 100 for v in Vs]
        ax[0].plot(Vs, e, color=C[i], marker='o', ms=4, ls=['-', '--'][i], label=nm.upper())
        sup = [(S['B'][f'{nm}|{v}']['joint'] - S['B'][f'{nm}|{v}']['corner']) / S['B'][f'{nm}|{v}']['corner'] * 100 for v in Vs]
        ax[1].plot(Vs, sup, color=C[i], marker='o', ms=4, ls=['-', '--'][i], label=nm.upper())
        irs = [0.02, 0.05, 0.08, 0.12]
        e2 = [(S['C'][f'{nm}|{r}']['s'] - S['C'][f'{nm}|{r}']['c']) / S['C'][f'{nm}|{r}']['c'] * 100 for r in irs]
        ax[2].plot([r * 100 for r in irs], e2, color=C[i], marker='o', ms=4, ls=['-', '--'][i], label=nm.upper())
    z = Device(V0=0.65).ztc()
    ax[0].axvline(z, color=MUTED, lw=1, ls=':'); ax[0].text(z + 0.005, 0.5, 'ZTC', fontsize=8, color=MUTED)
    ax[0].set_xlabel('Nominal supply V0 (V)'); ax[0].set_ylabel('Separable error at 99.9 % (%)')
    ax[0].set_title('(a) Separable MC vs coupled', fontsize=9, loc='left'); ax[0].legend(fontsize=8)
    ax[1].set_xlabel('Nominal supply V0 (V)'); ax[1].set_ylabel('Simultaneous over additive (%)')
    ax[1].set_title('(b) Corner superadditivity', fontsize=9, loc='left')
    ax[2].set_xlabel('Worst-bin IR drop (% of V0)'); ax[2].set_ylabel('Separable error at 99.9 % (%)')
    ax[2].set_title('(c) Effect of PDN strength', fontsize=9, loc='left')
    for a in ax: a.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(OUT + 'fig_sweeps.png'); plt.close(fig)


def fig_tails():
    S = json.load(open('cache/sweeps.json'))['A']
    names = [('counter4', 'Counter (depth 3)'), ('aes', 'AES (depth 17)'), ('gcd', 'GCD (depth 20)')]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for j, reg in enumerate(['base', 'severe']):
        x = np.arange(3); w = 0.26
        for i, kap in enumerate([0.0, 1.0, 2.0]):
            vals = [(S[f'{nm}|{reg}|{kap}']['c'] - S[f'{nm}|{reg}|{kap}']['nt']) / S[f'{nm}|{reg}|{kap}']['nt'] * 100 for nm, _ in names]
            ax[j].bar(x + (i - 1) * w, vals, w - 0.02, color=C[i], label=f'κ = {kap:.0f}')
        ax[j].set_xticks(x); ax[j].set_xticklabels([l for _, l in names], fontsize=8)
        ax[j].axhline(0, color=MUTED, lw=0.8)
        t = 'p = 0.8 %, median 10 %' if reg == 'base' else 'p = 3 %, median 25 %'
        ax[j].set_title(f'({"ab"[j]}) Tail regime: {t}', fontsize=9, loc='left')
    ax[0].set_ylabel('Increase of 99.9 % delay\nfrom EUV tails (%)')
    ax[1].legend(fontsize=8, title='Context sensitivity', title_fontsize=8)
    fig.tight_layout(); fig.savefig(OUT + 'fig_tails.png'); plt.close(fig)


def fig_flow():
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 42)
    def box(x, y, w, h, t, fc='#eef4fc'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=C[0], lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=7.2, color=INK)
    def arr(x0, y0, x1, y1):
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', color=MUTED, lw=1))
    box(1, 31, 15, 8, 'RTL benchmarks\n(counter, GCD,\nRISC-V, AES)')
    box(21, 31, 16, 8, 'Yosys + ABC\nmapping to ASAP7\n(RVT, TT library)')
    box(42, 31, 17, 8, 'Quadratic placement,\ngate sizing,\nNLDM STA')
    box(64, 31, 16, 8, 'Activity by logic\nsimulation; power\nper cell and bin')
    box(84, 31, 15, 8, 'Maps: utilisation,\nRUDY, IR, thermal')
    arr(16.5, 35, 20.5, 35); arr(37.5, 35, 41.5, 35); arr(59.5, 35, 63.5, 35); arr(80.5, 35, 83.5, 35)
    box(1, 16, 22, 9, 'Device model: alpha-power\nwith T-dependent mobility\nand Vth, calibrated to\nASAP7 TT/SS/FF', fc='#fdf0ea')
    box(27, 16, 22, 9, 'Variation sources: global,\nspatial and local Vth;\ncontext-dependent EUV\ntails; workload; droop; T', fc='#fdf0ea')
    box(53, 16, 22, 9, 'Vectorised Monte Carlo STA\n(full graph, every sample)\n40 000 chips per case', fc='#fdf0ea')
    box(79, 16, 20, 9, 'Couplings switched\non or off: context,\nshared workload', fc='#fdf0ea')
    arr(91, 30.5, 91, 25.5); arr(23.5, 20.5, 26.5, 20.5); arr(49.5, 20.5, 52.5, 20.5); arr(78.5, 20.5, 75.5, 20.5)
    box(10, 2, 24, 8, 'Corner stacking\n(additive and simultaneous)', fc='#eaf7f1')
    box(38, 2, 24, 8, 'Separable statistical\n(mechanisms independent)', fc='#eaf7f1')
    box(66, 2, 24, 8, 'Coupled statistical\n(reference)', fc='#eaf7f1')
    arr(64, 15.5, 22, 10.5); arr(64, 15.5, 50, 10.5); arr(64, 15.5, 78, 10.5)
    fig.tight_layout(); fig.savefig(OUT + 'fig_flow.png'); plt.close(fig)




def fig_device2():
    import os
    r = json.load(open('cache2/spice_fo4.json' if os.path.exists('cache2/spice_fo4.json') else 'results/spice_fo4.json'))
    d = Device(V0=0.65, Vth0=0.363, alpha=1.067, m=0.544, kT=0.24e-3)
    ref = r['TT|0.65|25']
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    Ts = [-40, 0, 25, 50, 85, 125]; Tl = np.linspace(-40, 125, 100)
    for i, V in enumerate([0.5, 0.55, 0.6, 0.65, 0.7, 0.8]):
        ls = ['-', '--', '-', '-.', ':', (0, (5, 1, 1, 1))][i]
        col = (C + ['#555555'])[i]
        ax[0].plot(Tl, d.factor(V, Tl, 0), color=col, ls=ls, lw=1.6, label=f'{V:.2f} V')
        ax[0].plot(Ts, [r[f'TT|{V}|{T}'] / ref for T in Ts], 'o', color=col, ms=4)
    ax[0].set_xlabel('Temperature (°C)'); ax[0].set_ylabel('FO4 delay / delay(0.65 V, 25 °C)')
    ax[0].set_ylim(0.6, 2.2)
    ax[0].legend(ncol=3, fontsize=7.5, title='Supply (lines: model, dots: SPICE)', title_fontsize=7.5, loc='upper center')
    ax[0].set_title('(a) Model against BSIM-CMG SPICE', fontsize=9, loc='left')
    V = np.linspace(0.47, 0.85, 120)
    s = (d.g(V, 26, 0) - d.g(V, 24, 0)) / 2 / d.g(V, 25, 0) * 100
    ax[1].plot(V, s, color=C[0], label='model')
    Vs = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    sp = [np.log(r[f'TT|{v}|125'] / r[f'TT|{v}|-40']) / 165 * 100 for v in Vs]
    ax[1].plot(Vs, sp, 'o', color=C[1], ms=4, label='SPICE (−40 to 125 °C)')
    ax[1].axhline(0, color=MUTED, lw=0.8)
    z = d.ztc(); ax[1].axvline(z, color=C[1], ls='--', lw=1.2)
    ax[1].text(z + 0.01, -0.09, f'ZTC ≈ {z:.2f} V', fontsize=8, color=INK)
    ax[1].set_xlabel('Supply voltage (V)'); ax[1].set_ylabel('d ln(delay)/dT (% per K)')
    ax[1].legend(fontsize=7.5, loc='upper left')
    ax[1].set_title('(b) Temperature sensitivity and ZTC', fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(OUT + 'fig_device.png'); plt.close(fig)


if __name__ == '__main__':
    import os; os.makedirs(OUT, exist_ok=True)
    for f in sys.argv[1:]:
        globals()[f]()
