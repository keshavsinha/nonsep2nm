"""Scenario set-up and Monte Carlo experiments (revision 2).

Changes from revision 1: device model fitted to BSIM-CMG SPICE data; droop
budgets consistent with a 10 % sign-off margin at the 99.9th workload
percentile; partial correlation rho between the fast (droop) and slow
(thermal) workload; clock-tree latency under the same variation; RSS corners;
timing-driven synthesis (ORFS abc_speed script) without the greedy sizer.
"""
import numpy as np
from .design import Design
from .placement import place, net_geometry, maps
from .timing import Timer, BUF
from .liberty import interp2
from .physics import Device, pdn_impedance, thermal_impedance, gaussian_field

NODES = {
    'N7': dict(V0=0.70, area_scale=1.0, r_w=60.0, ir_target=0.025, chip_droop=0.010,
               sig_g=0.012, sig_sp=0.008, sig_loc=0.015, p_tail=0.002, kappa=0.8,
               tail_med=0.08, tail_sln=0.6, T_a=45.0, T_rise=40.0, dT_loc=2.0,
               sig_w=0.20, rho=0.5, util=0.65),
    'N2': dict(V0=0.65, area_scale=0.35, r_w=180.0, ir_target=0.040, chip_droop=0.015,
               sig_g=0.012, sig_sp=0.009, sig_loc=0.018, p_tail=0.008, kappa=0.8,
               tail_med=0.10, tail_sln=0.6, T_a=45.0, T_rise=40.0, dT_loc=4.5,
               sig_w=0.20, rho=0.5, util=0.65),
}
STRESS = dict(ir_target=0.08, chip_droop=0.04)   # front-side PDN stress case (revision 1 values)
DEVICE = dict(Vth0=0.363, alpha=1.067, m=0.544, kT=0.24e-3)

LIBS = ['lib/SIMPLE_TT.lib', 'lib/INVBUF_TT.lib', 'lib/SEQ_TT.lib']


class Prepared:
    pass


def prepare(name, vfile, lib, node, util=None, nb=16, seed=0, held=None, cycles=500, act=None):
    cfg = dict(NODES[node]) if isinstance(node, str) else dict(node)
    util = util or cfg['util']
    d = Design(vfile, lib, name)
    x, y, W, H, io = place(d, util=util, seed=seed)
    hp, bb = net_geometry(d, x, y, io)
    dens, rudy, gx, gy = maps(d, x, y, W, H, bb, nb=nb)
    s = np.sqrt(cfg['area_scale'])
    hp_s = hp * s
    cong_net = np.zeros(len(d.nets))
    for i in range(len(d.nets)):
        x0, y0, x1, y1 = bb[i]
        bx = min(int((x0 + x1) / 2 / W * nb), nb - 1); by = min(int((y0 + y1) / 2 / H * nb), nb - 1)
        cong_net[i] = rudy[by, bx]
    detour = 1.0 + 0.5 * np.clip(cong_net - 0.7, 0, None)
    tm = Timer(d, hp_s * detour, r_w=cfg['r_w'])
    if held is None:
        held = {d.find(n): (1 if n in ('en', 'ld') else 0) for n in d.inputs
                if n in ('rst', 'reset', 'en', 'rst_n')}
    if act is None:
        act = d.simulate_activity(cycles=cycles, held=held, seed=seed + 7)
    P = Prepared()
    P.__dict__.update(dict(name=name, d=d, x=x, y=y, W=W, H=H, hp=hp_s, bb=bb, dens=dens,
                           rudy=rudy, gx=gx, gy=gy, tm=tm, act=act, nb=nb, cfg=cfg, util=util))
    _power_maps(P)
    _clock_tree(P, lib)
    return P


def _power_maps(P):
    d, tm, nb, cfg = P.d, P.tm, P.nb, P.cfg
    V0 = cfg['V0']
    f = 1.0 / (tm.Tmin_nom * 1e-12)
    pdyn = np.zeros(len(d.inst))
    for net, (k, p) in d.driver.items():
        i = d.net_id[net]
        pdyn[k] += P.act[i] * tm.load[i] * 1e-15 * V0 ** 2 * f
    for k in np.where(d.is_seq)[0]:
        c = d.lib[d.inst[k][0]]
        pdyn[k] += c.inputs.get(c.clock_pin or 'CLK', 0.5) * 1e-15 * V0 ** 2 * f
    P.pdyn = pdyn
    P.cell_bin = P.gy * nb + P.gx
    Pb = np.zeros(nb * nb)
    np.add.at(Pb, P.cell_bin, pdyn)
    P.Pbin_nom = Pb
    P.Zpdn = pdn_impedance(nb)
    P.Zth = thermal_impedance(nb)
    ir = P.Zpdn @ (Pb / V0)
    P.ir_map = ir / ir.max()
    th = P.Zth @ Pb
    P.th_map = th / th.max()
    z = 0.5 * (P.dens - P.dens.mean()) / (P.dens.std() + 1e-9) + \
        0.5 * (P.rudy - P.rudy.mean()) / (P.rudy.std() + 1e-9)
    z = z.ravel()
    P.z_bin = (z - z.mean()) / (z.std() + 1e-9)
    amin = np.min(d.area[d.area > 0])
    P.loc_scale = np.sqrt(amin / np.maximum(d.area, amin))


def _clock_tree(P, lib):
    """Quad-tree (H-tree like) clock distribution: each level is a BUFx4 at the
    centre of a 2^l x 2^l partition of the die driving four children."""
    d, nb = P.d, P.nb
    seq = np.where(d.is_seq)[0]
    P.seq = seq
    if len(seq) == 0:
        P.clk_levels = 0
        return
    lev = max(1, int(np.ceil(np.log(max(len(seq), 2)) / np.log(4))))
    bc = lib[BUF]
    cin = bc.inputs['A']
    arc = bc.arcs[0]['t']
    side = P.W * np.sqrt(P.cfg['area_scale'])
    nodes = []          # per level: node id per flop, bin of node
    dl = []
    for l in range(lev + 1):
        n = 2 ** l
        ix = np.minimum((P.x[seq] / P.W * n).astype(int), n - 1)
        iy = np.minimum((P.y[seq] / P.H * n).astype(int), n - 1)
        nid = iy * n + ix
        cx = (ix + 0.5) / n; cy = (iy + 0.5) / n
        b = np.minimum((cy * nb).astype(int), nb - 1) * nb + np.minimum((cx * nb).astype(int), nb - 1)
        seg = side / n / 2
        load = 4 * cin + 0.18 * seg * 4 if l < lev else 4 * 1.0 + 0.18 * seg * 4
        dd = max(interp2(arc['cell_rise'], 20, load), interp2(arc['cell_fall'], 20, load))
        nodes.append((nid, b)); dl.append(dd)
    P.clk_levels = lev + 1
    P.clk_nodes = nodes
    P.clk_dl = np.array(dl)
    P.clk_nom = float(np.sum(dl))
    P.loc_buf = np.sqrt(np.min(d.area[d.area > 0]) / bc.area)


def _wl(sw, xi):
    return np.exp(sw * xi - sw ** 2 / 2)


def run_mc(P, N=20000, batch=1000, seed=1, context=True, shared_w=True, rho=None, tails=True,
           process=True, droop=True, thermal=True, clock=True, dtco=None, dev_kw=None):
    """Monte Carlo timing. Returns circuit delay per sample normalised to the
    nominal (TT, V0, 25 C) minimum clock period."""
    cfg = dict(P.cfg)
    if dtco:
        cfg.update(dtco)
    r = cfg['rho'] if rho is None else rho
    if not shared_w:
        r = 0.0
    ss = np.random.SeedSequence(seed)
    r_w, r_wT, r_chip, r_g, r_sp, r_loc, r_hit, r_mag, r_clk = [np.random.default_rng(c) for c in ss.spawn(9)]
    dk = dict(DEVICE); dk.update(dev_kw or {})
    dk.update({k: cfg[k] for k in ('Vth0', 'alpha', 'm', 'kT') if k in cfg})
    dev = Device(V0=cfg['V0'], **dk)
    nb = P.nb
    ncell = len(P.d.inst)
    cb = P.cell_bin
    corr_bins = nb * 0.5
    ir_scale = cfg.get('ir_scale', 1.0)
    p0 = cfg['p_tail'] if tails else 0.0
    if context and tails:
        pc = np.exp(cfg['kappa'] * P.z_bin[cb]); pc *= p0 / pc.mean()
        pc = np.minimum(pc, 0.5)
    else:
        pc = np.full(ncell, p0)
    out = []
    for b0 in range(0, N, batch):
        n = min(batch, N - b0)
        sw = cfg['sig_w']
        x1 = r_w.standard_normal(n); x2 = r_wT.standard_normal(n)
        wV = _wl(sw, x1)
        wT = _wl(sw, r * x1 + np.sqrt(max(0.0, 1 - r * r)) * x2)
        if process:
            dg = cfg['sig_g'] * r_g.standard_normal(n)
            dspb = cfg['sig_sp'] * gaussian_field(nb, corr_bins, n, r_sp)
            dl = (cfg['sig_loc'] * P.loc_scale)[:, None] * r_loc.standard_normal((ncell, n))
            dVth = (dg[None, :] + dspb[:, cb].T + dl).astype(np.float32)
        else:
            dg = np.zeros(n); dspb = np.zeros((n, nb * nb))
            dVth = np.zeros((ncell, n))
        if droop:
            chip = cfg['chip_droop'] * wV * (1 + 0.25 * r_chip.standard_normal(n))
            frac = cfg['ir_target'] * ir_scale * np.outer(wV, P.ir_map.ravel()) + chip[:, None]
            V = cfg['V0'] * (1 - np.clip(frac, 0, 0.4))
        else:
            V = np.full((n, nb * nb), cfg['V0'])
        if thermal:
            T = cfg['T_a'] + cfg['T_rise'] * wT[:, None] + cfg['dT_loc'] * np.outer(wT, P.th_map.ravel())
        else:
            T = np.full((n, nb * nb), dev.T_ref)
        f = dev.factor(V[:, cb].T, T[:, cb].T, dVth)
        if tails and p0 > 0:
            hit = r_hit.random((ncell, n)) < pc[:, None]
            mag = np.minimum(np.exp(np.log(cfg['tail_med']) + cfg['tail_sln'] * r_mag.standard_normal((ncell, n))), 0.5)
            f = f * (1 + hit * mag)
        clk = None; clk_po = None
        if clock and P.clk_levels > 0:
            lat = np.zeros((len(P.seq), n))
            for l, (nid, b) in enumerate(P.clk_nodes):
                un, inv = np.unique(nid, return_inverse=True)
                xi = r_clk.standard_normal((len(un), n))[inv]
                dv = dg[None, :] + dspb[:, b].T + (cfg['sig_loc'] * P.loc_buf * xi if process else 0)
                lat += P.clk_dl[l] * dev.factor(V[:, b].T, T[:, b].T, dv)
            clk = {k: lat[j].astype(np.float32) for j, k in enumerate(P.seq)}
            clk_po = np.float32(P.clk_nom)
        D = P.tm.mc(f.astype(np.float32), None, setup_scale=True, clk=clk, clk_po=clk_po)
        out.append(D / P.tm.Tmin_nom)
    return np.concatenate(out)


def corner_stack(P, dtco=None, v_margin=0.10, t_hot=125.0, t_cold=-40.0, dev_kw=None):
    """Corner sign-off. Returns dict with additive (linear process), RSS-process
    additive, fully RSS and simultaneous corners, normalised to nominal."""
    cfg = dict(P.cfg)
    if dtco:
        cfg.update(dtco)
    dk = dict(DEVICE); dk.update(dev_kw or {})
    dk.update({k: cfg[k] for k in ('Vth0', 'alpha', 'm', 'kT') if k in cfg})
    dev = Device(V0=cfg['V0'], **dk)
    depth = max(len(P.tm.critical_path()), 1)
    dv_lin = 3 * cfg['sig_g'] + 3 * cfg['sig_sp'] + 3 * cfg['sig_loc'] / np.sqrt(depth)
    dv_rss = np.sqrt((3 * cfg['sig_g']) ** 2 + (3 * cfg['sig_sp']) ** 2 + (3 * cfg['sig_loc'] / np.sqrt(depth)) ** 2)
    mP = dev.factor(cfg['V0'], dev.T_ref, dv_lin) - 1
    mPr = dev.factor(cfg['V0'], dev.T_ref, dv_rss) - 1
    mV = dev.factor(cfg['V0'] * (1 - v_margin), dev.T_ref, 0) - 1
    mT = max(dev.factor(cfg['V0'], t_hot, 0), dev.factor(cfg['V0'], t_cold, 0)) - 1
    joint = max(dev.factor(cfg['V0'] * (1 - v_margin), t, dv_lin) for t in (t_hot, t_cold))
    joint_rss = max(dev.factor(cfg['V0'] * (1 - v_margin), t, dv_rss) for t in (t_hot, t_cold))
    return dict(add=1 + mP + mV + mT, add_rss=1 + mPr + mV + mT,
                rss_all=1 + np.sqrt(mPr ** 2 + mV ** 2 + max(mT, 0) ** 2),
                joint=joint, joint_rss=joint_rss, mP=mP, mPr=mPr, mV=mV, mT=mT, depth=depth)


def q(D, p):
    return float(np.quantile(D, p))
