"""Analytical placement (SimPL-style quadratic placement with rough spreading),
cell-density and RUDY routing-congestion maps."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _spread(x, y, area, W, H, util, nb):
    """Area-proportional spreading: cells are sorted into vertical strips of
    equal area, then within each strip into rows of equal area (a
    look-ahead legalisation in the spirit of SimPL)."""
    n = len(x)
    ns = max(1, int(np.sqrt(n / 4)))
    order = np.argsort(x)
    cum = np.cumsum(area[order]); cum /= cum[-1]
    strip = np.minimum((cum * ns).astype(int), ns - 1)
    nx, ny = np.empty(n), np.empty(n)
    for s in range(ns):
        idx = order[strip == s]
        if len(idx) == 0:
            continue
        x0, x1 = W * s / ns, W * (s + 1) / ns
        # keep relative x inside strip
        rx = x[idx]
        rx = (rx - rx.min()) / (np.ptp(rx) + 1e-12)
        nx[idx] = x0 + (0.1 + 0.8 * rx) * (x1 - x0)
        oy = idx[np.argsort(y[idx])]
        cy = np.cumsum(area[oy]); cy = (cy - area[oy] / 2) / cy[-1]
        ny[oy] = cy * H
    return nx, ny


def place(design, util=0.65, iters=8, seed=0, density_cap=None):
    """Return x, y (um), die W, H."""
    rng = np.random.default_rng(seed)
    n = len(design.inst)
    A = design.area.sum()
    side = np.sqrt(A / util)
    W = H = side
    # IO pins evenly on the boundary
    ios = [design.find(p) for p in design.inputs + design.outputs]
    io_pos = {}
    for i, net in enumerate(ios):
        t = i / max(1, len(ios))
        per = 4 * side * t
        if per < side:
            io_pos[net] = (per, 0.0)
        elif per < 2 * side:
            io_pos[net] = (side, per - side)
        elif per < 3 * side:
            io_pos[net] = (3 * side - per, side)
        else:
            io_pos[net] = (0.0, 4 * side - per)
    # net -> pins
    nets = {}
    for k, (c, _, pins) in enumerate(design.inst):
        for p, net in pins.items():
            if net in design.clock_nets:
                continue
            nets.setdefault(net, []).append(k)
    rows, cols, vals = [], [], []
    bx = np.zeros(n); by = np.zeros(n); diag = np.zeros(n)
    for net, ks in nets.items():
        ks = list(set(ks))
        m = len(ks)
        if m > 50:   # very high fanout nets (reset/enable) are ignored
            continue
        fixed = io_pos.get(net)
        if m + (fixed is not None) < 2:
            continue
        w = 1.0 / (m - 1 + (fixed is not None)) if m > 1 else 1.0
        for a in range(m):
            for b in range(a + 1, m):
                i, j = ks[a], ks[b]
                rows += [i, j]; cols += [j, i]; vals += [-w, -w]
                diag[i] += w; diag[j] += w
            if fixed is not None:
                diag[ks[a]] += w
                bx[ks[a]] += w * fixed[0]; by[ks[a]] += w * fixed[1]
    Q0 = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    eps = 1e-3
    x = rng.uniform(0.4, 0.6, n) * W
    y = rng.uniform(0.4, 0.6, n) * H
    Q = Q0 + sp.diags(diag + eps)
    x = spla.spsolve(Q.tocsc(), bx + eps * W / 2)
    y = spla.spsolve(Q.tocsc(), by + eps * H / 2)
    for it in range(iters):
        ax, ay = _spread(x, y, design.area, W, H, util, None)
        wa = 0.02 * (it + 1) ** 1.5
        Q = Q0 + sp.diags(diag + wa)
        x = spla.spsolve(Q.tocsc(), bx + wa * ax)
        y = spla.spsolve(Q.tocsc(), by + wa * ay)
    x, y = _spread(x, y, design.area, W, H, util, None)
    return np.clip(x, 0, W), np.clip(y, 0, H), W, H, io_pos


def net_geometry(design, x, y, io_pos):
    """HPWL and bounding boxes per net (index aligned with design.nets)."""
    hp = np.zeros(len(design.nets))
    bb = np.zeros((len(design.nets), 4))
    pts = {}
    for k, (c, _, pins) in enumerate(design.inst):
        for p, net in pins.items():
            pts.setdefault(net, []).append((x[k], y[k]))
    for net, xy in io_pos.items():
        if net in pts:
            pts[net].append(xy)
    for i, net in enumerate(design.nets):
        P = np.array(pts.get(net, [(0, 0)]))
        x0, y0 = P.min(0); x1, y1 = P.max(0)
        bb[i] = (x0, y0, x1, y1)
        hp[i] = (x1 - x0) + (y1 - y0)
    return hp, bb


def maps(design, x, y, W, H, bb, nb=16, pitch=0.036, layers=4, clock=True):
    """Cell-area utilisation map and RUDY congestion map (demand/capacity)."""
    gx = np.minimum((x / W * nb).astype(int), nb - 1)
    gy = np.minimum((y / H * nb).astype(int), nb - 1)
    binA = (W / nb) * (H / nb)
    dens = np.zeros((nb, nb))
    np.add.at(dens, (gy, gx), design.area)
    dens /= binA
    rudy = np.zeros((nb, nb))
    xs = np.linspace(0, W, nb + 1); ys = np.linspace(0, H, nb + 1)
    for i, net in enumerate(design.nets):
        if net in design.clock_nets and not clock:
            continue
        x0, y0, x1, y1 = bb[i]
        w = max(x1 - x0, W / nb / 4); h = max(y1 - y0, H / nb / 4)
        wl = (x1 - x0) + (y1 - y0)
        if wl <= 0:
            continue
        d = wl / (w * h)            # wire length per unit area inside bbox
        ox = np.clip(np.minimum(xs[1:], x0 + w) - np.maximum(xs[:-1], x0), 0, None)
        oy = np.clip(np.minimum(ys[1:], y0 + h) - np.maximum(ys[:-1], y0), 0, None)
        rudy += d * np.outer(oy, ox)
    rudy /= binA
    cap = layers / pitch            # routable track length per unit area (um/um^2)
    return dens, rudy / cap, gx, gy
