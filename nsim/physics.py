"""Device delay model, PDN and thermal grids, and the coupled Monte Carlo
engine used for the separable / coupled comparison."""
import numpy as np
import scipy.sparse as sp

K_B = 8.617e-5
T0K = 298.15


# ----------------------------------------------------------------------------
# Device model: alpha-power law with temperature-dependent mobility and Vth.
# Parameters fitted to BSIM-CMG SPICE simulations of an ASAP7 RVT FO4 inverter
# (spice/fit.py; 0.5-0.85 V, -40..125 C, rms error 0.6 %).
# ----------------------------------------------------------------------------
class Device:
    def __init__(self, V0=0.70, Vth0=0.363, alpha=1.067, m=0.544, kT=0.24e-3, n_ss=1.3, T_ref=25.0):
        # defaults: least-squares fit to BSIM-CMG (ASAP7 RVT TT) FO4 delay, 0.5-0.85 V, -40..125 C
        self.V0, self.Vth0, self.alpha, self.m, self.kT, self.n = V0, Vth0, alpha, m, kT, n_ss
        self.T_ref = T_ref
        self.g0 = self.g(V0, T_ref, 0.0)

    def g(self, V, Tc, dVth):
        T = Tc + 273.15
        vth = self.Vth0 + dVth - self.kT * (T - T0K)
        ov = np.maximum(V - vth, 0.02)
        return V / ((T / T0K) ** (-self.m) * ov ** self.alpha)

    def factor(self, V, Tc, dVth):
        return self.g(V, Tc, dVth) / self.g0

    def leak_factor(self, Tc, dVth):
        T = Tc + 273.15
        vt = K_B * T
        vth = self.Vth0 + dVth - self.kT * (T - T0K)
        vth0 = self.Vth0
        return (T / T0K) ** 2 * np.exp(-(vth - vth0) / (self.n * vt)) * np.exp(0)  # relative to T_ref

    def ztc(self):
        from scipy.optimize import brentq
        f = lambda V: self.g(V, 26, 0) - self.g(V, 24, 0)
        try:
            return brentq(f, 0.42, 1.2)
        except ValueError:
            return np.nan


# ----------------------------------------------------------------------------
def grid_laplacian(nb, g_lat, g_gnd_mask):
    """Conductance matrix of an nb x nb resistive mesh with lateral conductance
    g_lat and per-node conductance to ground/supply g_gnd_mask (array)."""
    N = nb * nb
    rows, cols, vals = [], [], []
    diag = np.array(g_gnd_mask, float).ravel().copy()
    for r in range(nb):
        for c in range(nb):
            i = r * nb + c
            for (rr, cc) in ((r + 1, c), (r, c + 1)):
                if rr < nb and cc < nb:
                    j = rr * nb + cc
                    rows += [i, j]; cols += [j, i]; vals += [-g_lat, -g_lat]
                    diag[i] += g_lat; diag[j] += g_lat
    G = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr() + sp.diags(diag)
    return G


def pdn_impedance(nb, pad_pitch=4, g_pad=20.0):
    mask = np.zeros((nb, nb))
    off = pad_pitch // 2
    mask[off::pad_pitch, off::pad_pitch] = g_pad
    G = grid_laplacian(nb, 1.0, mask)
    return np.linalg.inv(G.toarray())


def thermal_impedance(nb, g_lat=4.0, g_v=1.0):
    G = grid_laplacian(nb, g_lat, np.full((nb, nb), g_v))
    return np.linalg.inv(G.toarray())


def gaussian_field(nb, corr_len_bins, n, rng):
    """Samples of a unit-variance Gaussian random field on an nb x nb grid with
    exponential correlation. Returns (n, nb*nb)."""
    yy, xx = np.mgrid[0:nb, 0:nb]
    P = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    C = np.exp(-d / corr_len_bins)
    Lc = np.linalg.cholesky(C + 1e-9 * np.eye(len(C)))
    return rng.standard_normal((n, len(C))) @ Lc.T
