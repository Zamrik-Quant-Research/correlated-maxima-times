#!/usr/bin/env python3
"""model.py: solve once for "The joint law of the times of the maxima of correlated Brownian motions".

W = (W1, W2) standard BMs with correlation rho on [0,1]; theta_i = argmax W_i.  With W1 = X, W2 = rho X + sqrt(1-rho^2) Y
the set {W1 < a1, W2 < a2} is a wedge of angle alpha = arccos(-rho).  The joint density of (theta1, theta2) on s < t is

    f(s,t) = (sin a / 4) int int D_s(r1) K_{t-s}(r1, r2) D_{1-t}(r2) dr1 dr2

D_s(r) = normal derivative of the wedge survival function at the point of edge 1 at distance r from the apex,
K_u = mixed normal derivative (edge 1 -> edge 2) of the Dirichlet heat kernel.  Scaled: D_s(r) = s^-1/2 delta(r/sqrt s),
K_u(r1, r2) = u^-2 k(r1/sqrt u, r2/sqrt u).  Everything below evaluates this two independent ways (eigen-series, images
at alpha = pi/n) and checks it against Monte Carlo with exact Brownian-bridge extremes.

Run:  nice -n 19 python3 model.py        (SMOKE=1 divides path counts by 100; plumbing only)
Writes results.npz, numbers.json, CHECKS.txt.  Parameters: PARAMS.md.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib
import json
import math
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate, special, stats
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"
CACHE.mkdir(exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"
SCALE = 100 if SMOKE else 1
WORKERS = 32
SQ2PI = math.sqrt(2.0 / math.pi)
ZETA3 = 1.2020569031595942

# ── tolerances (PARAMS.md) ────────────────────────────────────────────────────────────────────────────────────
TOL_EXACT_REL = 1e-8
TOL_KERNEL_ABS = 1e-10
TOL_MASS = 2e-6
TOL_DOUBLING = 2e-6
TOL_RS = 1e-7
TOL_C1 = 1e-4
Z_TOL = 4.0
CHI2_P = 1e-4
SLOPE_AN = 0.02
SLOPE_MC = 0.10
TOLERANCE_TEXT = f"""Tolerances (stated before any comparison; PARAMS.md)
  analytic vs exact closed form                   : {TOL_EXACT_REL} relative (kernels {TOL_KERNEL_ABS} absolute)
  mass, P(theta1<theta2), E theta, E theta^2      : {TOL_MASS} absolute
  quadrature doubling                             : {TOL_DOUBLING} absolute
  Rogers-Shepp c(rho) from the wedge survival      : {TOL_RS} absolute
  c1 exact vs Richardson slope                    : {TOL_C1} absolute
  Monte Carlo vs analytic scalar / per seed        : |z| <= {Z_TOL}
  histogram cell tests                            : chi-square p >= {CHI2_P}
  step halving                                    : |z| <= {Z_TOL}
  exponents: analytic local slope vs prediction   : {SLOPE_AN}; Monte Carlo slope vs analytic slope: {SLOPE_MC}
"""

RHO_MC = [-0.9, -math.sqrt(3) / 2, -1 / math.sqrt(2), -0.5, 0.0, 0.3, 0.5, 0.7, 0.9, 0.99]
RHO_CURVE = sorted(set([round(x, 10) for x in np.round(np.arange(-0.95, 0.951, 0.05), 10)] +
                       [-0.99, -0.98, -0.97, 0.97, 0.98, 0.99, 0.995, 0.999, -math.sqrt(3) / 2, -1 / math.sqrt(2),
                        0.3, 0.7, -0.01, 0.01, -0.02, 0.02, -0.1, 0.1]))
RHO_HEAT = [-1 / math.sqrt(2), -0.5, 0.0, 0.3, 0.7, 0.9]
RHO_BR = [-0.9, -math.sqrt(3) / 2, -1 / math.sqrt(2), -0.5, 0.0, 0.5, 0.9]
SEEDS = [11, 22, 33]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ══ analytic engine ═══════════════════════════════════════════════════════════════════════════════════════════
def _tail_N(logterm, nmax, tol):
    """Smallest N such that sum_{n > N} exp(logterm(n)) < tol (bound evaluated over n = 1..nmax, rest negligible)."""
    n = np.arange(1, nmax + 1)
    lt = logterm(n)
    tail = np.logaddexp.accumulate(lt[::-1])[::-1]          # tail[i] = log sum_{j >= i} term_j
    ok = np.nonzero(tail < math.log(tol))[0]
    if len(ok) == 0:
        raise RuntimeError("series bound never below tol")
    i = ok[0]                                               # terms with index >= i are dropped
    return int(n[i] - 1), float(np.exp(tail[i]))


def kmat(a, x, y, tol=1e-16):
    """k(x,y) = (2/(a x y)) e^{-(x^2+y^2)/2} sum_n (-1)^{n+1} nu^2 I_nu(xy), nu = n pi/a.  Returns (k, tail bound)."""
    p = math.pi / a
    z = np.outer(x, y)
    zmax = float(z.max())
    # bound per term (prefactor included): nu^2/(a Gamma(nu+1)) (z/2)^(nu-1)  >= |term|
    lt = lambda n: 2 * np.log(n * p) - math.log(a) - special.gammaln(n * p + 1) + (n * p - 1) * math.log(max(zmax, 1e-300) / 2)
    nmax = int((1.6 * zmax + 200) / p) + 50
    N, tail = _tail_N(lt, nmax, tol)
    N = max(N, 4)
    d2 = np.subtract.outer(x, y) ** 2
    band = d2 < 90.0                       # outside: |k| <= (2/pi) e^{-45} (|sum| <= z/p): set to 0
    zb = np.maximum(z[band], 1e-300)
    Sb = np.zeros_like(zb)
    for n in range(1, N + 1):
        nu = n * p
        Sb += (1 if n % 2 else -1) * nu * nu * special.ive(nu, zb)
    K = np.zeros_like(z)
    K[band] = (2.0 / a) * np.exp(-0.5 * d2[band]) * Sb / zb
    return K, tail, N


def delta_series(a, xs, tol=1e-16):
    """delta(x) = D_1(x) = (4/(a x)) sum_{m odd} G_m(x^2/4), G = sqrt(pi z/2) [ive((nu-1)/2, z) + ive((nu+1)/2, z)]."""
    p = math.pi / a
    z = xs ** 2 / 4.0
    zmax = float(z.max())
    # bound of the m-th term (m odd, nu = m p): (4/(a x)) sqrt(pi z/2) [(z/2)^((nu-1)/2)/Gamma((nu+1)/2) + (z/2)^((nu+1)/2)/Gamma((nu+3)/2)]
    xmin = float(xs.min())

    def lt(j):  # j = 1,2,... -> m = 2j-1
        nu = (2 * j - 1) * p
        l1 = 0.5 * (nu - 1) * math.log(zmax / 2) - special.gammaln(0.5 * (nu + 1))
        l2 = 0.5 * (nu + 1) * math.log(zmax / 2) - special.gammaln(0.5 * (nu + 3))
        return np.log(4 / (a * xmin) * math.sqrt(math.pi * zmax / 2)) + np.logaddexp(l1, l2)
    jmax = int((1.6 * zmax + 200) / p) + 50
    J, tail = _tail_N(lt, jmax, tol)
    J = max(J, 2)
    S = np.zeros_like(xs)
    for j in range(1, J + 1):
        nu = (2 * j - 1) * p
        S += np.sqrt(np.pi * z / 2) * (special.ive(0.5 * (nu - 1), z) + special.ive(0.5 * (nu + 1), z))
    return 4.0 / (a * xs) * S, tail, J


def tanh_sinh(N, tmax):
    t = np.linspace(-tmax, tmax, N)
    dt = t[1] - t[0]
    sh = 0.5 * math.pi * np.sinh(t)
    y = 0.5 * (1 + np.tanh(sh))
    ym = 0.5 * (1 - np.tanh(sh))                            # 1 - y, accurate
    ym = 1.0 / (1.0 + np.exp(2 * sh))
    y = 1.0 / (1.0 + np.exp(-2 * sh))
    w = 0.25 * math.pi * np.cosh(t) / np.cosh(sh) ** 2 * dt
    return y, ym, w


def gl01(N):
    v, w = leggauss(N)
    return 0.5 * (v + 1), 0.5 * w


class Wedge:
    def __init__(self, rho=None, alpha=None, Ecut=40.0, nfac=10.0, nmin=160, scale=1.0, images=False):
        a = math.acos(-rho) if alpha is None else alpha
        self.a, self.sa, self.p = a, math.sin(a), math.pi / a
        self.rho = -math.cos(a)
        self.X = math.sqrt(2 * Ecut) / (self.sa if a < math.pi / 2 else 1.0)   # |P1-P2|^2 >= x^2 sin^2 a (a<pi/2), x^2 (a>=pi/2)
        N = int(math.ceil(max(nmin, nfac * self.X) * scale))
        v, w = gl01(N)
        self.x, self.wx = self.X * v * v, w * 2 * self.X * v
        self.images = images
        if images:
            n = int(round(math.pi / a))
            assert abs(n * a - math.pi) < 1e-12
            self.n = n
            self.K = kimg(n, self.x, self.x)
            self.k_tail, self.k_N = 0.0, 0
        else:
            self.K, self.k_tail, self.k_N = kmat(a, self.x, self.x)
        self._delta_table()

    def _delta_table(self):
        self.xhp = 2 * self.X
        xs = np.geomspace(1e-7, self.xhp, 3000)
        if self.images:
            d = delta_img(self.n, xs)
            self.d_tail, self.d_J = 0.0, 0
        else:
            d, self.d_tail, self.d_J = delta_series(self.a, xs)
        self.d_end = float(d[-1] - SQ2PI)
        self.lx0, self.ld0 = math.log(xs[0]), math.log(d[0])
        self.spl = CubicSpline(np.log(xs), np.log(d))

    def delta(self, x):
        x = np.asarray(x, float)
        out = np.full(x.shape, SQ2PI)
        lo = x < 1e-7
        mid = (~lo) & (x < self.xhp)
        with np.errstate(divide="ignore"):
            lx = np.log(np.where(x > 0, x, 1e-300))
        out[mid] = np.exp(self.spl(lx[mid]))
        out[lo] = np.exp(self.ld0 + (self.p - 1) * (lx[lo] - self.lx0))
        return out

    def f_sul(self, s, u, L, chunk=4000):
        """joint density f(s, s+u) for s < s+u, L = 1 - s - u passed explicitly (accuracy near the ends)."""
        s, u, L = (np.atleast_1d(np.asarray(v, float)).ravel() for v in (s, u, L))
        out = np.empty(len(s))
        for i in range(0, len(s), chunk):
            sl = slice(i, i + chunk)
            A = self.wx[None, :] * self.delta(np.sqrt(u[sl] / s[sl])[:, None] * self.x[None, :])
            B = self.wx[None, :] * self.delta(np.sqrt(u[sl] / L[sl])[:, None] * self.x[None, :])
            q = np.einsum("ij,ij->i", A @ self.K, B)
            out[sl] = self.sa / 4.0 / u[sl] / np.sqrt(s[sl] * L[sl]) * q
        return out

    def f_st(self, s, t):
        s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
        lo, hi = np.minimum(s, t), np.maximum(s, t)
        return self.f_sul(lo, hi - lo, 1 - hi).reshape(s.shape)

    # ── integrals over the simplex s < t ──────────────────────────────────────────────────────────────────
    def _ugrid(self, Nz, top=1.0, uc=1e-280):
        """Gauss-Legendre in z, u = top z^(1/b), b = min(pi/a - 1, 1)/3, on u > uc; below uc an analytic tail is added."""
        b = min(self.p - 1.0, 1.0) / 3.0                  # u = z^(3/b): smooths u^(pi/a-2), log u, u^(pi/(2a)-1)
        zc = (uc / top) ** b
        z, wz = gl01(Nz)
        z, wz = zc + (1 - zc) * z, (1 - zc) * wz
        u = top * z ** (1 / b)
        wu = top * wz * (1 / b) * z ** (1 / b - 1)
        e = min(self.p - 1.0, self.p / 2.0)               # G(u) ~ u^(e-1) as u -> 0 (diagonal or corner)
        return np.append(u, uc), np.append(wu, uc / e)    # last node: tail int_0^uc G = G(uc) uc / e

    def moments(self, Nz=96, Ny=97, tmax=3.2):
        u, wu = self._ugrid(Nz)
        y, ym, wy = tanh_sinh(Ny, tmax)
        S = (1 - u)[:, None] * y[None, :]
        Lm = (1 - u)[:, None] * ym[None, :]
        U = np.broadcast_to(u[:, None], S.shape)
        Wt = (wu * (1 - u))[:, None] * wy[None, :]
        F = self.f_sul(S.ravel(), U.ravel(), Lm.ravel()).reshape(S.shape) * Wt
        T = S + U
        return dict(mass_half=F.sum(), E1=(F * (S + T)).sum(), E11=(F * (S * S + T * T)).sum(),
                    E12=2 * (F * S * T).sum(), Egap2=2 * (F * U * U).sum(), Egap=2 * (F * U).sum())

    def gap_density(self, us, Ny=121, tmax=3.4):
        """density of |theta2 - theta1| at u: 2 int_0^{1-u} f(s, s+u) ds."""
        y, ym, wy = tanh_sinh(Ny, tmax)
        out = []
        for u in us:
            s = (1 - u) * y
            L = (1 - u) * ym
            out.append(2 * (1 - u) * np.sum(wy * self.f_sul(s, np.full_like(s, u), L)))
        return np.array(out)

    def _ugrid_ts(self, Nz, top, tmax=3.4, uc=1e-280):
        """tanh-sinh in z on (zc, 1), u = top z^(1/b): handles u^(.) at 0 and sqrt(top-u) at top; returns u, top-u, w."""
        b = min(self.p - 1.0, 1.0) / 3.0
        zc = (uc / top) ** b
        y, ym, w = tanh_sinh(Nz, tmax)
        z = zc + (1 - zc) * y
        lz = np.where(z < 0.5, np.log(z), np.log1p(-(1 - zc) * ym))
        u = top * np.exp(lz / b)
        um = -top * np.expm1(lz / b)
        wu = top * (1 - zc) * w * (1 / b) * np.exp(lz * (1 / b - 1))
        e = min(self.p - 1.0, self.p / 2.0)
        return np.append(u, uc), np.append(um, top - uc), np.append(wu, uc / e)

    def diag(self, ts, Nz=97, Ny=97, tmax=3.4):
        """P(theta1 <= t, theta2 <= t) = 2 int int_{s < s+u <= t} f."""
        y, ym, wy = tanh_sinh(Ny, tmax)
        out = []
        for t in ts:
            u, um, wu = self._ugrid_ts(Nz, t)
            S = um[:, None] * y[None, :]
            U = np.broadcast_to(u[:, None], S.shape)
            Lr = (1 - t) + um[:, None] * ym[None, :]
            F = self.f_sul(S.ravel(), U.ravel(), Lr.ravel()).reshape(S.shape)
            out.append(2 * np.sum(F * (wu * um)[:, None] * wy[None, :]))
        return np.array(out)

    def joint_cdf(self, a, b, Nv=81, tmax=3.2):
        """P(theta1 <= a, theta2 <= b), a < b."""
        D = self.diag([a])[0]
        yv, yvm, wv = tanh_sinh(Nv, tmax)
        # s = a - v, v in (0,a): s = a*yvm when v = a*yv ; t' = a + w, w in (0, b-a)
        s = a * yvm
        v = a * yv
        w = (b - a) * yv
        L = 1 - a - w
        Sg, Vg = np.meshgrid(s, v, indexing="ij")[0], None
        S2 = np.repeat(s[:, None], Nv, 1)
        U2 = v[:, None] + w[None, :]
        L2 = np.repeat(L[None, :], Nv, 0)
        F = self.f_sul(S2.ravel(), U2.ravel(), L2.ravel()).reshape(S2.shape)
        return D + a * (b - a) * np.sum(F * wv[:, None] * wv[None, :])

    def marginal(self, s0, Nu=161, tmax=3.4):
        """int f(s0, t) dt over t != s0, tanh-sinh in z with u = U z^(1/b) (b as in _ugrid)."""
        b = min(self.p - 1.0, 1.0) / 3.0
        z, zm, w = tanh_sinh(Nu, tmax)
        lz = np.where(z < 0.5, np.log(np.maximum(z, 1e-320)), np.log1p(-zm))   # log z, accurate at both ends
        Z = np.exp(lz / b)
        Zm = -np.expm1(lz / b)                                                # 1 - Z
        dZ = (1 / b) * np.exp(lz * (1 / b - 1)) * w
        ok = (Z > 1e-290) & (Zm > 0)
        r = 0.0
        U1 = 1 - s0
        r += U1 * np.sum(dZ[ok] * self.f_sul(np.full(ok.sum(), s0), U1 * Z[ok], U1 * Zm[ok]))
        r += s0 * np.sum(dZ[ok] * self.f_sul(s0 * Zm[ok], s0 * Z[ok], np.full(ok.sum(), 1 - s0)))
        return r

    # ── bridge gap law: kappa(u) = (pi sin a / 2) int int K_u K_{1-u} ─────────────────────────────────────
    def kappa(self, us):
        out = []
        for u in us:
            v = 1 - u
            if u > 0.5:
                u, v = v, u
            lam = math.sqrt(u / v)
            if self.images:
                K2 = kimg(self.n, lam * self.x, lam * self.x)
            else:
                K2 = kmat(self.a, lam * self.x, lam * self.x)[0]
            W2 = self.wx[:, None] * self.wx[None, :]
            out.append(math.pi * self.sa / 2 / u / v ** 2 * np.sum(W2 * self.K * K2))
        return np.array(out)


# ── images at alpha = pi/n ────────────────────────────────────────────────────────────────────────────────────
def dihedral(n):
    a = math.pi / n
    els = []
    for k in range(n):
        c, s = math.cos(2 * k * a), math.sin(2 * k * a)
        els.append((np.array([[c, -s], [s, c]]), 1.0))      # rotation by 2k pi/n
        els.append((np.array([[c, s], [s, -c]]), -1.0))     # reflection in the line at angle k pi/n
    return els


def delta_img(n, xs):
    """delta(x) at alpha = pi/n: sqrt(2/pi) sum_{j<2n} (-1)^j cos(j pi/n) exp(-x^2 sin^2(j pi/n)/2) Phi(x cos(j pi/n)).
    The sum cancels to O(x^(n-1)) at small x: there it is evaluated with mpmath at 80 digits."""
    import mpmath as mpm
    xs = np.asarray(xs, float)
    out = np.zeros_like(xs)
    for j in range(2 * n):
        th = j * math.pi / n
        out += (-1) ** j * math.cos(th) * np.exp(-0.5 * (xs * math.sin(th)) ** 2) * special.ndtr(xs * math.cos(th))
    small = np.nonzero(xs < 0.2)[0]
    if len(small):
        with mpm.workdps(80):
            ths = [j * mpm.pi / n for j in range(2 * n)]
            for i in small:
                x = mpm.mpf(float(xs[i]))
                v = mpm.fsum((-1) ** j * mpm.cos(t) * mpm.exp(-(x * mpm.sin(t)) ** 2 / 2) * mpm.ncdf(x * mpm.cos(t)) for j, t in enumerate(ths))
                out[i] = float(v)
    return SQ2PI * out


def img_terms(n):
    """per group element g: (sgn, c_g = n2.g n1, G_g, S_g) with w = M_g r, r = (r1, r2)."""
    a = math.pi / n
    e0, ea = np.array([1.0, 0.0]), np.array([math.cos(a), math.sin(a)])
    n1, n2 = np.array([0.0, 1.0]), np.array([math.sin(a), -math.cos(a)])
    out = []
    for g, sg in dihedral(n):
        M = np.column_stack([-(g @ e0), ea])
        G = M.T @ M
        av, bv = M.T @ n2, M.T @ (g @ n1)
        S = 0.5 * (np.outer(av, bv) + np.outer(bv, av))
        out.append((sg, float(n2 @ (g @ n1)), G, S))
    return out


def kimg(n, x, y, u=1.0):
    """K_u(x, y) by images: sum_g sgn(g) [c_g/u - r^T S_g r / u^2] exp(-r^T G_g r/(2u)) / (2 pi u)."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    out = np.zeros_like(X)
    for sg, c, G, S in img_terms(n):
        q = G[0, 0] * X * X + 2 * G[0, 1] * X * Y + G[1, 1] * Y * Y
        qs = S[0, 0] * X * X + 2 * S[0, 1] * X * Y + S[1, 1] * Y * Y
        out += sg * (c / u - qs / u ** 2) * np.exp(-q / (2 * u)) / (2 * math.pi * u)
    return out


def J_int(b, k, A, B, C):
    """int_0^inf t^b / (C t^2 + 2 B t + A)^k dt, closed form (k <= 3, b <= 2k-2), Delta = AC - B^2 > 0."""
    D = A * C - B * B
    I = {1: math.atan2(math.sqrt(D), B) / math.sqrt(D)}
    for kk in range(1, 3):
        I[kk + 1] = (-B / A ** kk + (2 * kk - 1) * C * I[kk]) / (2 * kk * D)
    T = {}
    for kk in (1, 2, 3):
        T[(0, kk)] = I[kk]
    for kk in (2, 3):
        T[(1, kk)] = (A ** (1 - kk) / (kk - 1)) / (2 * C) - (B / C) * I[kk]
    for kk in (2, 3):
        T[(2, kk)] = (T[(0, kk - 1)] - 2 * B * T[(1, kk)] - A * T[(0, kk)]) / C
    T[(3, 3)] = (T[(1, 2)] - 2 * B * T[(2, 3)] - A * T[(1, 3)]) / C
    T[(4, 3)] = (T[(2, 2)] - 2 * B * T[(3, 3)] - A * T[(2, 3)]) / C
    return T[(b, k)]


def quad_mom(a_, b_, Q):
    """int int_{r1,r2>0} r1^a r2^b exp(-r^T Q r / 2) dr  (a+b even)."""
    k = (a_ + b_ + 2) // 2
    A, B, C = Q[0, 0], Q[0, 1], Q[1, 1]
    if A * C - B * B <= 1e-12 * A * C:                     # Q = c [[1,1],[1,1]] on the quadrant
        c = A
        return special.beta(a_ + 1, b_ + 1) * math.gamma(k) * 2 ** (k - 1) / c ** k
    return math.gamma(k) * 2 ** (k - 1) * J_int(b_, k, A, B, C)


def kappa_closed(n, u):
    """bridge gap density at alpha = pi/n in closed form (finite sum of elementary quadrant moments)."""
    a = math.pi / n
    v = 1 - u
    terms = img_terms(n)
    tot = 0.0
    for sg, cg, Gg, Sg in terms:
        for sh, ch, Gh, Sh in terms:
            Q = Gg / u + Gh / v
            m = {}
            for i in range(5):
                for j in range(5 - i):
                    if (i + j) % 2 == 0:
                        m[(i, j)] = quad_mom(i, j, Q)
            m0 = m[(0, 0)]
            mS = lambda S: S[0, 0] * m[(2, 0)] + 2 * S[0, 1] * m[(1, 1)] + S[1, 1] * m[(0, 2)]
            # (r^T Sg r)(r^T Sh r) as a quartic
            P4 = {(4, 0): Sg[0, 0] * Sh[0, 0],
                  (3, 1): 2 * Sg[0, 0] * Sh[0, 1] + 2 * Sg[0, 1] * Sh[0, 0],
                  (2, 2): Sg[0, 0] * Sh[1, 1] + Sg[1, 1] * Sh[0, 0] + 4 * Sg[0, 1] * Sh[0, 1],
                  (1, 3): 2 * Sg[1, 1] * Sh[0, 1] + 2 * Sg[0, 1] * Sh[1, 1],
                  (0, 4): Sg[1, 1] * Sh[1, 1]}
            m4 = sum(cf * m[e] for e, cf in P4.items())
            val = cg * ch / (u * v) * m0 - cg / (u * v * v) * mS(Sh) - ch / (u * u * v) * mS(Sg) + m4 / (u * u * v * v)
            tot += sg * sh * val / (4 * math.pi ** 2 * u * v)
    return math.pi * math.sin(a) / 2 * tot


# ── classical pieces ──────────────────────────────────────────────────────────────────────────────────────────
def rogers_shepp(rho):
    """c(rho) = E[M1 M2], Rogers & Shepp (2006) Eq. (1): rho = sin a, 2 gamma = a + pi/2."""
    a = math.asin(rho)
    g = 0.5 * (a + math.pi / 2)
    # cosh(nu a)/sinh(nu pi/2) = (e^{nu(a-pi/2)} + e^{-nu(a+pi/2)}) / (1 - e^{-nu pi}), overflow-free
    f = lambda nu: ((math.exp(nu * (a - math.pi / 2)) + math.exp(-nu * (a + math.pi / 2))) / (-math.expm1(-nu * math.pi))
                    * math.tanh(nu * g)) if nu > 0 else 2 * g / math.pi
    return math.cos(a) * integrate.quad(f, 0, np.inf, epsabs=1e-13, epsrel=1e-12, limit=400)[0]


def EM1M2_wedge(rho):
    """E[M1 M2] = int int P(M1 > a1, M2 > a2) da1 da2 from the wedge survival series (a1 = r sin phi, a2 = r sin(a-phi))."""
    a = math.acos(-rho)
    p = math.pi / a
    R = 8.0 * max(1 / math.sin(a / 2), 1 / math.sin(a) if a > math.pi / 2 else 1.0)   # {a1, a2 < 8} inside r < R
    Nr = int(max(200, 20 * R))
    Nphi = int(max(120, 2.5 * (8 * R + 60) * a / math.pi))
    r, wr = gl01(Nr)
    r, wr = R * r, R * wr
    ph, wph = gl01(Nphi)
    ph, wph = a * ph, a * wph
    Rg, Pg = np.meshgrid(r, ph, indexing="ij")
    z = Rg ** 2 / 4
    S = np.zeros_like(Rg)
    for m in range(1, 4000, 2):
        nu = m * p
        t = 4 / (m * math.pi) * np.sin(nu * Pg) * np.sqrt(np.pi * z / 2) * (special.ive(0.5 * (nu - 1), z) + special.ive(0.5 * (nu + 1), z))
        S += t
        if nu > 8 * R + 60:
            break
    a1, a2 = Rg * np.sin(Pg), Rg * np.sin(a - Pg)
    H = 1 - special.erf(a1 / math.sqrt(2)) - special.erf(a2 / math.sqrt(2)) + S
    return math.sin(a) * np.sum(H * Rg * wr[:, None] * wph[None, :])


def EthetaW_exact(u):
    """E[theta W(u)] = sqrt(2/pi) [ (2/3) u^1.5 + (2/3)(1 - (1-u)^1.5) - u ]   (h(u) = sqrt(2/pi)(sqrt u + sqrt(1-u) - 1))."""
    return SQ2PI * ((2 / 3) * u ** 1.5 + (2 / 3) * (1 - (1 - u) ** 1.5) - u)


C1_EXACT = 4 - 32 / (3 * math.pi)
CORR_MAXMIN = (6 - 7 * ZETA3) / 4            # MMS PRE 2020: cov = -(7 zeta3 - 6)/32, var = 1/8
COV_BB_MAXMIN = -(math.pi ** 2 - 9) / 36     # MMS PRE 2020, bridge


def log_antider(u, g):
    """antiderivative of g (sampled at increasing u > 0) via a cubic spline of g(e^l) e^l in l = log u."""
    cs = CubicSpline(np.log(u), g * u).antiderivative()
    return lambda x: cs(np.log(x))


def arcsine_cdf(t):
    return 2 / math.pi * np.arcsin(np.sqrt(t))


# ══ Monte Carlo ═══════════════════════════════════════════════════════════════════════════════════════════════
def argmax_location(rng, x, y, M, hh):
    """time of the max inside a Brownian bridge step from x to y of length hh, given its max M (exact)."""
    a = np.maximum(M - x, 1e-300)
    b = np.maximum(M - y, 1e-300)
    first = rng.random(len(a)) < b / (a + b)
    V1 = rng.wald(a / b, a * a / hh)
    V2 = 1.0 / rng.wald(b / a, b * b / hh)
    V = np.where(first, V1, V2)
    return hh * V / (1 + V)


def sample_argmax(rng, A, P, n, bridge, K=12, m=32, Wu_idx=None):
    d, N = A.shape[1], A.shape[0]
    h = 1.0 / n
    Z = np.zeros((P, n + 1, d))
    np.cumsum(rng.standard_normal((P, n, d)) * math.sqrt(h), axis=1, out=Z[:, 1:, :])
    if bridge:
        Z -= np.linspace(0, 1, n + 1)[None, :, None] * Z[:, -1:, :]
    W = Z @ A.T
    idx = np.sort(np.concatenate([np.argpartition(-np.maximum(W[:, :-1, j], W[:, 1:, j]), K, axis=1)[:, :K]
                                  for j in range(N)], axis=1), axis=1)
    dup = np.zeros(idx.shape, bool)
    dup[:, 1:] = idx[:, 1:] == idx[:, :-1]
    NK = idx.shape[1]
    Z0 = np.take_along_axis(Z, idx[:, :, None], axis=1)
    Z1 = np.take_along_axis(Z, idx[:, :, None] + 1, axis=1)
    hf = h / m
    F = np.cumsum(rng.standard_normal((P, NK, m, d)) * math.sqrt(hf), axis=2)
    frac = (np.arange(1, m + 1) / m)[None, None, :, None]
    Zf = np.empty((P, NK, m + 1, d))
    Zf[:, :, 0, :] = Z0
    Zf[:, :, 1:, :] = Z0[:, :, None, :] + F - frac * (F[:, :, -1:, :] - (Z1 - Z0)[:, :, None, :])
    Wf = Zf @ A.T
    maskb = np.zeros((P, n), bool)
    np.put_along_axis(maskb, idx, True, axis=1)
    r = np.arange(P)
    theta = np.empty((P, N))
    Mx = np.empty((P, N))
    for j in range(N):
        x, y = W[:, :-1, j], W[:, 1:, j]
        mb = 0.5 * (x + y + np.sqrt((y - x) ** 2 - 2 * h * np.log(rng.random((P, n)))))
        mb[maskb] = -np.inf
        kb = mb.argmax(1)
        Mb = mb[r, kb]
        xf, yf = Wf[:, :, :-1, j], Wf[:, :, 1:, j]
        mf = 0.5 * (xf + yf + np.sqrt((yf - xf) ** 2 - 2 * hf * np.log(rng.random((P, NK, m)))))
        mf[dup] = -np.inf
        flat = mf.reshape(P, -1)
        kf = flat.argmax(1)
        Mf = flat[r, kf]
        ist, isub = np.divmod(kf, m)
        use = Mf >= Mb
        x0 = np.where(use, xf[r, ist, isub], x[r, kb])
        y0 = np.where(use, yf[r, ist, isub], y[r, kb])
        t0 = np.where(use, idx[r, ist] * h + isub * hf, kb * h)
        hh = np.where(use, hf, h)
        M = np.where(use, Mf, Mb)
        theta[:, j] = t0 + argmax_location(rng, x0, y0, M, hh)
        Mx[:, j] = M
    Wu = W[:, Wu_idx, 0] if Wu_idx is not None else None
    return theta, Mx, Wu


def job_mc(args):
    key, A, bridge, n, Ptot, seed, tag, part, K, m = args
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag, part]))
    chunk = max(50, int(3_000_000 // (n * A.shape[1] + A.shape[0] * K * (m + 1) * A.shape[1] + 1)))
    Wu_idx = np.array([round(u * n) for u in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)])
    TH, MM, WU = [], [], []
    done = 0
    while done < Ptot:
        P = min(chunk, Ptot - done)
        th, Mx, Wu = sample_argmax(rng, A, P, n, bridge, K, m, Wu_idx)
        TH.append(th); MM.append(Mx.astype(np.float32)); WU.append(Wu.astype(np.float32))
        done += P
    return key, seed, np.concatenate(TH), np.concatenate(MM), np.concatenate(WU)


def m_refine(rho, h=1e-3):
    """sub-steps per refined step: sqrt(h/m) <= sqrt(1-rho^2) sqrt(1e-3)/5, at least 32 (PARAMS, revision 1)."""
    return int(max(32, math.ceil(25 * h / ((1 - rho * rho) * 1e-3))))


def A_pair(rho):
    return np.array([[1.0, 0.0], [rho, math.sqrt(max(0.0, 1 - rho * rho))]])


# ── discrete walks (F8) ───────────────────────────────────────────────────────────────────────────────────────
def walk_steps(law, rng, P, n, rho):
    if law == "gauss":
        X, Y = rng.standard_normal((2, P, n))
        return X, rho * X + math.sqrt(1 - rho * rho) * Y
    if law == "unif":
        X, Y = rng.uniform(-math.sqrt(3), math.sqrt(3), (2, P, n))
        return X, rho * X + math.sqrt(1 - rho * rho) * Y
    if law == "laplace":
        X, Y = rng.laplace(0, 1 / math.sqrt(2), (2, P, n))
        return X, rho * X + math.sqrt(1 - rho * rho) * Y
    if law == "copy":
        X, Y = rng.standard_normal((2, P, n))
        c = rng.random((P, n)) < abs(rho)
        return X, np.where(c, math.copysign(1, rho) * X, Y)
    raise ValueError(law)


def job_walk(args):
    law, rho, n, Ptot, seed, tag = args
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag, n]))
    chunk = max(1000, 4_000_000 // n)
    out = []
    done = 0
    while done < Ptot:
        P = min(chunk, Ptot - done)
        a, b = walk_steps(law, rng, P, n, rho)
        S1 = np.concatenate([np.zeros((P, 1)), np.cumsum(a, 1)], 1)
        S2 = np.concatenate([np.zeros((P, 1)), np.cumsum(b, 1)], 1)
        out.append(np.stack([S1.argmax(1), S2.argmax(1)], 1).astype(np.int32))
        done += P
    return law, rho, n, seed, np.concatenate(out)


# ══ checks bookkeeping ════════════════════════════════════════════════════════════════════════════════════════
class Checks:
    def __init__(self):
        self.lines, self.fails = [], 0

    def add(self, ok, name, detail):
        ok = bool(ok)
        self.lines.append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        self.fails += (not ok)
        return ok

    def close(self, name, val, tgt, tol, rel=False):
        err = abs(val - tgt) / (abs(tgt) if rel else 1.0)
        return self.add(err <= tol, name, f"{val:.12g} vs {tgt:.12g} ({'rel' if rel else 'abs'} err {err:.2e} <= {tol:g})")

    def z(self, name, est, se, tgt, tol=Z_TOL):
        z = (est - tgt) / se if se > 0 else 0.0
        return self.add(abs(z) <= tol, name, f"{est:.6f} +- {se:.6f} vs {tgt:.6f} (z = {z:+.2f})")


def write_checks(C, header):
    txt = header + TOLERANCE_TEXT + "\n" + "\n".join(C.lines) + f"\n\nSUMMARY: {len(C.lines) - C.fails} PASS, {C.fails} FAIL\n"
    (HERE / "CHECKS.txt").write_text(txt)


def cached(name, fn, mc=False):
    p = CACHE / (name + ("_smoke" if (SMOKE and mc) else "") + ".pkl")
    if p.exists():
        return pickle.loads(p.read_bytes())
    t0 = time.time()
    val = fn()
    p.write_bytes(pickle.dumps(val))
    print(f"  [{name}] {time.time() - t0:.1f}s", flush=True)
    return val


# ══ workers for the analytic stage ═══════════════════════════════════════════════════════════════════════════
def an_curve(rho):
    """corr(theta1, theta2) and moments at rho, two grids (doubling) for the error estimate."""
    W = Wedge(rho)
    m1 = W.moments()
    m2 = W.moments(Nz=192, Ny=193, tmax=3.6)
    W15 = Wedge(rho, scale=1.5)
    m3 = W15.moments()
    return dict(rho=rho, a=W.a, m1=m1, m2=m2, m3=m3, k_tail=W.k_tail, k_N=W.k_N, d_tail=W.d_tail, d_J=W.d_J,
                d_end=W.d_end, X=W.X, Nx=len(W.x))


GAP_UL = np.geomspace(1e-12, 0.5, 600)                  # u = |theta1 - theta2| near 0 (log spline in u)
GAP_VL = np.geomspace(1e-9, 0.5, 300)                   # v = 1 - u near 0 (log spline in v; g has sqrt(1-u) terms)
GAP_U = np.concatenate([GAP_UL, 1 - GAP_VL[::-1][1:]])
IMG_GAP_U = np.linspace(0.0025, 0.9975, 200)
DIAG_T = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
CDF_AB = [(0.1, 0.3), (0.1, 0.6), (0.2, 0.9), (0.3, 0.5), (0.4, 0.7), (0.5, 0.95), (0.7, 0.8)]
GAPCDF_U = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8]
EPS_CORNER = np.geomspace(1e-9, 1e-2, 15)
U_DIAG = np.geomspace(1e-9, 1e-2, 15)
KAP_U = np.concatenate([np.geomspace(1e-5, 0.02, 25), np.linspace(0.03, 0.5, 48)])


def an_profile(rho):
    """gap density, diagonal law, joint CDF points, local exponents, marginal check, kappa (bridge) at rho."""
    W = Wedge(rho)
    g = W.gap_density(GAP_U)
    D = W.diag(DIAG_T)
    Fab = [W.joint_cdf(a, b) for a, b in CDF_AB]
    corner = W.f_st(EPS_CORNER, 2 * EPS_CORNER)
    fdiag = W.f_sul(np.full_like(U_DIAG, 0.4), U_DIAG, 0.6 - U_DIAG)
    marg = [W.marginal(s0) for s0 in (0.05, 0.2, 0.5, 0.8)]
    kap = W.kappa(KAP_U)
    return dict(rho=rho, g=g, D=D, Fab=Fab, corner=corner, fdiag=fdiag, marg=marg, kap=kap)


def an_heat(rho, ng=160):
    W = Wedge(rho)
    c = (np.arange(ng) + 0.5) / ng
    Sg, Tg = np.meshgrid(c, c, indexing="ij")
    Tg2 = np.where(np.isclose(Sg, Tg), Tg + 0.25 / ng, Tg)
    return rho, W.f_st(Sg, Tg2)


def an_images(n):
    a = math.pi / n
    We = Wedge(alpha=a)
    Wi = Wedge(alpha=a, images=True)
    xs = np.geomspace(1e-3, 30, 200)
    d_e = delta_series(a, xs)[0]
    d_i = delta_img(n, xs)
    xq = np.linspace(0.05, 6, 40)
    k_e = kmat(a, xq, xq)[0]
    k_i = kimg(n, xq, xq)
    pts = [(0.1, 0.3), (0.2, 0.25), (0.05, 0.9), (0.45, 0.55), (0.6, 0.62), (0.01, 0.02), (0.3, 0.9)]
    f_e = np.array([We.f_st(s, t) for s, t in pts]).ravel()
    f_i = np.array([Wi.f_st(s, t) for s, t in pts]).ravel()
    mom_i = Wi.moments()
    g_i = Wi.gap_density(IMG_GAP_U)
    D_i = Wi.diag(DIAG_T)
    ku = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    kap_c = np.array([kappa_closed(n, u) for u in ku])
    kap_i = Wi.kappa(ku)
    kap_e = We.kappa(ku)
    kap_closed_curve = np.array([kappa_closed(n, u) for u in KAP_U])
    return dict(n=n, xs=xs, d_e=d_e, d_i=d_i, k_e=k_e, k_i=k_i, pts=pts, f_e=f_e, f_i=f_i, mom_i=mom_i, g_i=g_i,
                D_i=D_i, ku=ku, kap_c=kap_c, kap_i=kap_i, kap_e=kap_e, kap_closed_curve=kap_closed_curve)


def an_const(rho):
    """leading diagonal term f(s, s+u) ~ (sin a/4) c_a^2 I_a u^(pi/a-2) [s(1-s-u)]^(-pi/(2a)),
    c_a = (2/a) sqrt(pi/2) 8^(-(nu-1)/2) / Gamma((nu+1)/2) (delta(x) ~ c_a x^(nu-1)), I_a = int int x^b k(x,y) y^b, b = nu - 1."""
    W = Wedge(rho)
    nu = W.p
    ca = (2 / W.a) * math.sqrt(math.pi / 2) * 8 ** (-(nu - 1) / 2) / math.gamma((nu + 1) / 2)
    xb = W.wx * W.x ** (nu - 1)
    Ia = float(xb @ W.K @ xb)
    d_small = float(W.delta(np.array([1e-4]))[0] / (ca * 1e-4 ** (nu - 1)))
    u = np.array([1e-9, 1e-7, 1e-5])
    s0 = 0.4
    fnum = W.f_sul(np.full(3, s0), u, 1 - s0 - u)
    fas = W.sa / 4 * ca ** 2 * Ia * u ** (nu - 2) * (s0 * (1 - s0 - u)) ** (-nu / 2)
    return dict(rho=rho, ca=ca, Ia=Ia, d_small=d_small, ratio=(fnum / fas).tolist())


def an_rs(rho):
    return rho, rogers_shepp(rho), EM1M2_wedge(rho)


# ══ main ═════════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    pf = next((q for q in (HERE / "PARAMS.md", HERE / "PARAMS.txt") if q.exists()), None)   # the record of the parameters, if present
    header = (f"CHECKS.txt  correlated-maxima-times  ({time.strftime('%Y-%m-%d %H:%M')})\n"
              f"PARAMS sha256 {sha256(pf) if pf else 'absent (the parameters are the constants in this file)'}\n"
              f"model.py  sha256 {sha256(__file__)}\n"
              f"SMOKE={SMOKE}\n\n")
    (HERE / "CHECKS.txt").write_text(header + TOLERANCE_TEXT + "\n(running)\n")
    C = Checks()
    R, J = {}, {}
    pool = Pool(WORKERS)

    # ── analytic stage ────────────────────────────────────────────────────────────────────────────────────
    print("analytic: curve", flush=True)
    curve = cached("curve", lambda: pool.map(an_curve, RHO_CURVE, chunksize=1))
    print("analytic: profiles", flush=True)
    prof_rhos = sorted(set(RHO_MC + RHO_BR + [-0.97, -0.98, -0.99]))
    profiles = cached("profiles", lambda: pool.map(an_profile, prof_rhos, chunksize=1))
    print("analytic: heatmaps / images / RS", flush=True)
    heats = cached("heats", lambda: pool.map(an_heat, RHO_HEAT, chunksize=1))
    imgs = cached("images", lambda: pool.map(an_images, [2, 3, 4, 6], chunksize=1))
    rs = cached("rs", lambda: pool.map(an_rs, [r for r in RHO_MC if r <= 0.9], chunksize=1))
    RS_ALL = {r: rogers_shepp(r) for r in RHO_MC}
    consts = cached("consts", lambda: pool.map(an_const, [-0.5, 0.3, 0.5, 0.7, 0.9, 0.99], chunksize=1))
    PR = {round(p["rho"], 6): p for p in profiles}

    # ── Monte Carlo stage ─────────────────────────────────────────────────────────────────────────────────
    NP_MAIN, NP_HALF, NP_BR, NP_F7 = 400_000 // SCALE, 200_000 // SCALE, 200_000 // SCALE, 200_000 // SCALE
    PART = max(1000, 25_000 // SCALE)
    tasks = []
    for i, rho in enumerate(RHO_MC):
        for seed in SEEDS:
            for part in range(NP_MAIN // PART):
                tasks.append((("bm", i, 1000), A_pair(rho), False, 1000, PART, seed, 100 + i, part, 12, m_refine(rho)))
            for part in range(NP_HALF // PART):
                tasks.append((("bm", i, 2000), A_pair(rho), False, 2000, PART, seed, 200 + i, part, 12, m_refine(rho)))
    for i, rho in enumerate(RHO_BR):
        for seed in SEEDS:
            for part in range(NP_BR // PART):
                tasks.append((("bb", i, 1000), A_pair(rho), True, 1000, PART, seed, 300 + i, part, 12, m_refine(rho)))
    F7 = {"A3": np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]]) / math.sqrt(2),
          "A4": np.array([[-1, 1, 0, 0, 0], [0, -1, 1, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, 1]]) / math.sqrt(2),
          "EQ+0.5": np.linalg.cholesky(0.5 * np.eye(3) + 0.5),
          "EQ-0.4": np.linalg.cholesky(1.4 * np.eye(3) - 0.4)}
    for i, (nm, A) in enumerate(F7.items()):
        for seed in SEEDS:
            for part in range(NP_F7 // PART):
                mr = m_refine(float(np.max(np.abs(A @ A.T - np.eye(len(A))))))
                tasks.append((("f7", nm, 1000), A, False, 1000, PART, seed, 400 + i, part, 12, mr))
    print(f"MC: {len(tasks)} tasks", flush=True)

    def run_mc():
        res = {}
        for key, seed, th, Mx, Wu in pool.imap_unordered(job_mc, tasks, chunksize=1):
            res.setdefault((key, seed), []).append((th, Mx, Wu))
        return {k: tuple(np.concatenate([v[i] for v in L]) for i in range(3)) for k, L in res.items()}
    MC = cached("mc", run_mc, mc=True)

    wtasks = []
    for law in ("gauss", "unif", "laplace", "copy"):
        for rho in (-0.5, 0.5):
            for n in (10, 100, 1000):
                Pw = {10: 4_000_000, 100: 400_000, 1000: 100_000}[n] // SCALE
                for seed in SEEDS:
                    wtasks.append((law, rho, n, Pw, seed, 500 + ["gauss", "unif", "laplace", "copy"].index(law) * 10 + (rho > 0)))
    WK = cached("walks", lambda: {(l, r, n, s): a for l, r, n, s, a in pool.map(job_walk, wtasks, chunksize=1)}, mc=True)
    pool.close()

    # ══ checks ══════════════════════════════════════════════════════════════════════════════════════════════
    # 1. kernels and series
    for p in curve:
        rho = p["rho"]
        if rho in (-0.99, -0.5, 0.0, 0.5, 0.99):
            C.add(p["k_tail"] < 1e-15 and p["d_tail"] < 1e-15, f"series truncation rho={rho:+.3f}",
                  f"k: N={p['k_N']} terms, rigorous tail bound {p['k_tail']:.1e}; delta: J={p['d_J']} odd terms, tail {p['d_tail']:.1e}")
            C.add(abs(p["d_end"]) < 1e-12, f"delta(x_hp) = sqrt(2/pi) rho={rho:+.3f}", f"difference {p['d_end']:.2e} at x_hp = {2 * p['X']:.1f}")
    W0 = Wedge(0.0)
    xs = np.geomspace(1e-4, 20, 50)
    C.close("rho=0: delta = sqrt(2/pi) erf(x/sqrt2)", float(np.max(np.abs(W0.delta(xs) - SQ2PI * special.erf(xs / math.sqrt(2))))), 0.0, 1e-9)
    xq = np.linspace(0.05, 6, 30)
    k0 = kmat(math.pi / 2, xq, xq)[0]
    kex = (2 / math.pi) * np.outer(xq, xq) * np.exp(-0.5 * np.add.outer(xq ** 2, xq ** 2))
    C.close("rho=0: k(x,y) = (2/pi) x y exp(-(x^2+y^2)/2)", float(np.max(np.abs(k0 - kex))), 0.0, TOL_KERNEL_ABS)
    ss = np.array([0.03, 0.2, 0.5, 0.31, 0.77])
    tt = np.array([0.1, 0.7, 0.52, 0.9, 0.99])
    f0 = W0.f_st(ss, tt)
    fex = 1 / (math.pi ** 2 * np.sqrt(ss * (1 - ss) * tt * (1 - tt)))
    C.close("rho=0: f(s,t) = arcsine x arcsine (max rel err over 5 points)", float(np.max(np.abs(f0 / fex - 1))), 0.0, 1e-7)
    for im in imgs:
        n = im["n"]
        C.close(f"images n={n}: delta_img vs eigen-series delta", float(np.max(np.abs(im["d_e"] - im["d_i"]))), 0.0, TOL_KERNEL_ABS)
        C.close(f"images n={n}: k_img vs eigen-series k", float(np.max(np.abs(im["k_e"] - im["k_i"]))), 0.0, TOL_KERNEL_ABS)
        C.close(f"images n={n}: f_img vs f_eigen at 7 points (max rel)", float(np.max(np.abs(im["f_i"] / im["f_e"] - 1))), 0.0, 1e-7)
        C.close(f"images n={n}: kappa closed form vs image quadrature (max rel)", float(np.max(np.abs(im["kap_c"] / im["kap_i"] - 1))), 0.0, 1e-7)
        C.close(f"images n={n}: kappa closed form vs eigen-series quadrature (max rel)", float(np.max(np.abs(im["kap_c"] / im["kap_e"] - 1))), 0.0, 1e-6)
    # quadrant moments closed form vs quad
    Q = np.array([[1.7, 0.4], [0.4, 0.9]])
    errs = []
    for (i, j) in [(0, 0), (2, 0), (1, 1), (0, 2), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]:
        num = integrate.dblquad(lambda y, x: x ** i * y ** j * math.exp(-(Q[0, 0] * x * x + 2 * Q[0, 1] * x * y + Q[1, 1] * y * y) / 2),
                                0, 30, 0, 30, epsabs=1e-13, epsrel=1e-12)[0]
        errs.append(abs(quad_mom(i, j, Q) / num - 1))
    C.close("quadrant Gaussian moments: closed form vs dblquad (9 moments, max rel)", max(errs), 0.0, 1e-9)
    errs = []
    Qn = np.array([[1.3, -0.5], [-0.5, 2.0]])
    for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 4)]:
        num = integrate.dblquad(lambda y, x: x ** i * y ** j * math.exp(-(Qn[0, 0] * x * x + 2 * Qn[0, 1] * x * y + Qn[1, 1] * y * y) / 2),
                                0, 40, 0, 40, epsabs=1e-13, epsrel=1e-12)[0]
        errs.append(abs(quad_mom(i, j, Qn) / num - 1))
    C.close("quadrant moments, negative off-diagonal (4 moments, max rel)", max(errs), 0.0, 1e-9)

    # 2. mass, marginals, moments, doubling
    cur = {round(p["rho"], 6): p for p in curve}
    corr = {}
    for rho, p in sorted(cur.items()):
        m1, m2, m3 = p["m1"], p["m2"], p["m3"]
        corr[rho] = 8 * (m2["E12"] - 0.25)
        err = max(abs(m1["E12"] - m2["E12"]), abs(m3["E12"] - m2["E12"]))
        ok = (abs(m2["mass_half"] - 0.5) <= TOL_MASS and abs(m2["E1"] - 0.5) <= TOL_MASS and abs(m2["E11"] - 0.375) <= TOL_MASS
              and err <= TOL_DOUBLING)
        C.add(ok, f"quadrature rho={rho:+.4f}", f"P(theta1<theta2)={m2['mass_half']:.9f} E theta={m2['E1']:.9f} E theta^2={m2['E11']:.9f} "
              f"E[theta1 theta2]={m2['E12']:.9f} (doubling/1.5x-nodes diff {err:.1e}) corr={corr[rho]:.7f}")
    for rho in (-0.5, 0.0, 0.5, 0.9):
        pr = PR[round(rho, 6)]
        mex = [1 / (math.pi * math.sqrt(s0 * (1 - s0))) for s0 in (0.05, 0.2, 0.5, 0.8)]
        C.close(f"marginal of f is arcsine at s=0.05,0.2,0.5,0.8 rho={rho:+.2f} (max rel)",
                float(np.max(np.abs(np.array(pr["marg"]) / np.array(mex) - 1))), 0.0, 1e-5)
    C.close("rho=0 anchor: corr = 0", corr[0.0], 0.0, 1e-6)
    p0 = PR[0.0]
    C.close("rho=0: diagonal law = (2/pi arcsin sqrt t)^2 (max abs err, 15 t)", float(np.max(np.abs(p0["D"] - arcsine_cdf(DIAG_T) ** 2))), 0.0, 1e-8)
    C.close("rho=0: joint CDF = product of arcsine CDFs (max abs err, 7 points)",
            float(max(abs(F - arcsine_cdf(a_) * arcsine_cdf(b_)) for (a_, b_), F in zip(CDF_AB, p0["Fab"]))), 0.0, 1e-6)
    C.close("rho=0: bridge kappa = 1 (max abs err)", float(np.max(np.abs(p0["kap"] - 1))), 0.0, 1e-8)

    # 3. Rogers-Shepp
    for rho, c_rs, c_w in rs:
        C.close(f"Rogers-Shepp c(rho) from the wedge survival, rho={rho:+.4f}", c_w, c_rs, TOL_RS)
    C.close("Rogers-Shepp anchors c(0)=2/pi, c(1-)->1, c(-1+)->2log2-1", rogers_shepp(0.0), 2 / math.pi, 1e-10)

    # 4. c1 and chaos
    d1 = (corr[0.01] - corr[-0.01]) / 0.02
    d2 = (corr[0.02] - corr[-0.02]) / 0.04
    c1_num = (4 * d1 - d2) / 3
    C.close("c1 = 4 - 32/(3 pi): Richardson slope of corr at rho=0", c1_num, C1_EXACT, TOL_C1)
    c2_num = (corr[0.1] + corr[-0.1]) / (2 * 0.01)
    c2_a = (corr[0.05] + corr[-0.05]) / (2 * 0.0025)
    c2_num = (4 * c2_a - c2_num) / 3
    c3_num = ((corr[0.1] - corr[-0.1]) / 0.2 - c1_num) / 0.01
    rhos = np.array(sorted(corr))
    cv = np.array([corr[r] for r in rhos])
    C.add(bool(np.all(np.abs(cv) <= np.abs(rhos) + 1e-9)), "Gebelein |corr| <= |rho| on the grid",
          f"max(|corr|-|rho|) = {np.max(np.abs(cv) - np.abs(rhos)):.3e} over {len(rhos)} rho")
    pos = rhos >= 0
    dc = np.diff(cv[pos]) / np.diff(rhos[pos])
    C.add(bool(np.all(dc > 0) and np.all(np.diff(dc) > -1e-6)), "corr increasing and convex on [0,1) (c_n >= 0)",
          f"min slope {dc.min():.4f}, min slope increment {np.diff(dc).min():.2e}")
    # rho -> -1 : extrapolate to MMS
    rn = np.array([-0.99, -0.98, -0.97])
    cn = np.array([corr[r] for r in rn])
    pfit = np.polyfit(rn + 1, cn, 2)
    C.close("rho -> -1: quadratic extrapolation of corr from -0.99,-0.98,-0.97 -> (6 - 7 zeta3)/4 (MMS)", float(np.polyval(pfit, 0.0)), CORR_MAXMIN, 2e-3)
    # rho -> 1 : 1 - corr ~ kappa sqrt(1-rho)
    rp = np.array([0.97, 0.98, 0.99, 0.995, 0.999])
    om = np.array([1 - corr[r] for r in rp])
    Xd = np.column_stack([np.sqrt(1 - rp), 1 - rp])
    kap_fit, res_, *_ = np.linalg.lstsq(Xd, om, rcond=None)
    Xd2 = np.column_stack([np.ones_like(rp), np.sqrt(1 - rp), 1 - rp])
    fit2 = np.linalg.lstsq(Xd2, om, rcond=None)[0]
    C.add(abs(fit2[0]) < 2e-3, "rho -> 1: 1 - corr = k1 sqrt(1-rho) + k2 (1-rho), free intercept ~ 0 (Sum c_n = 1)",
          f"intercept {fit2[0]:+.2e}, k1 = {kap_fit[0]:.4f}, k2 = {kap_fit[1]:.4f}")
    J["chaos"] = dict(c1_exact=C1_EXACT, c1_num=c1_num, c2_num=c2_num, c3_num=c3_num, sum_odd=(7 * ZETA3 - 2) / 8,
                      sum_even=(10 - 7 * ZETA3) / 8, corr_maxmin=CORR_MAXMIN, kappa1=kap_fit[0], kappa2=kap_fit[1],
                      intercept=fit2[0], extrap_minus1=float(np.polyval(pfit, 0.0)))

    # 5. exponents (analytic)
    rows = []
    for rho in sorted(PR):
        pr = PR[rho]
        a = math.acos(-rho)
        p = math.pi / a
        sl_c = np.diff(np.log(pr["corner"])) / np.diff(np.log(EPS_CORNER))
        sl_d = np.diff(np.log(pr["fdiag"])) / np.diff(np.log(U_DIAG))
        gu = pr["g"]
        sel = GAP_U < 1e-10
        sl_g = np.diff(np.log(gu[sel])) / np.diff(np.log(GAP_U[sel]))
        pred_c, pred_d = p / 2 - 2, p - 2
        pred_g = p - 2 if rho > 1e-9 else (p / 2 - 1 if rho < -1e-9 else 0.0)
        rows.append(dict(rho=rho, alpha=a, corner=sl_c[0], corner_pred=pred_c, diag=sl_d[0], diag_pred=pred_d,
                         gap=sl_g[0], gap_pred=pred_g))
        if abs(rho) > 1e-9:
            C.add(abs(sl_c[0] - pred_c) < SLOPE_AN and abs(sl_d[0] - pred_d) < SLOPE_AN and abs(sl_g[0] - pred_g) < SLOPE_AN,
                  f"exponents rho={rho:+.4f} (alpha={a:.4f})",
                  f"corner {sl_c[0]:.4f} vs pi/(2a)-2={pred_c:.4f}; diagonal {sl_d[0]:.4f} vs pi/a-2={pred_d:.4f}; gap {sl_g[0]:.4f} vs {pred_g:.4f}")
        else:
            C.add(abs(sl_c[0] - pred_c) < SLOPE_AN and abs(sl_d[0] - pred_d) < SLOPE_AN, "exponents rho=0",
                  f"corner {sl_c[0]:.4f} vs -1; diagonal {sl_d[0]:.4f} vs 0; gap density ~ (2/pi^2) log(1/u): g(1e-6)/log(1e6) = {gu[0] / math.log(1e6):.4f} vs {2 / math.pi ** 2:.4f}")
    J["exponents"] = rows
    for cst in consts:
        C.close(f"diagonal constant rho={cst['rho']:+.2f}: f(0.4,0.4+u)/leading term at u=1e-9 (c_a={cst['ca']:.5f}, I_a={cst['Ia']:.6f})",
                cst["ratio"][0], 1.0, 1e-3)
        C.close(f"delta(x) ~ c_a x^(pi/a-1) at x=1e-4, rho={cst['rho']:+.2f}", cst["d_small"], 1.0, 1e-6)
    J["diag_constants"] = consts

    # 6. Monte Carlo vs analytic
    mcrows = []
    for i, rho in enumerate(RHO_MC):
        pr = PR[round(rho, 6)]
        a = math.acos(-rho)
        nL = len(GAP_UL)
        gL, gV = pr["g"][:nL], pr["g"][nL - 1:][::-1]          # gV on v = 1 - u, increasing v, ends at v = 0.5
        AL = log_antider(GAP_UL, gL)
        AV = log_antider(GAP_VL, gV)
        gap0 = gL[0] * GAP_UL[0] / min(math.pi / a - 1, math.pi / (2 * a)) if rho != 0 else gL[0] * GAP_UL[0] * (1 + 1 / math.log(1 / GAP_UL[0]))
        top0 = gV[0] * GAP_VL[0]
        upper = float(AV(0.5) - AV(GAP_VL[0])) + top0
        gtot = gap0 + float(AL(0.5) - AL(GAP_UL[0])) + upper
        C.close(f"gap density integrates to 1, rho={rho:+.4f}", float(gtot), 1.0, 1e-5)
        gapcdf = lambda u, AL=AL, AV=AV, upper=upper, top0=top0: (1.0 - (float(AL(0.5) - AL(u)) + upper) if u <= 0.5
                                                                  else 1.0 - (float(AV(1 - u) - AV(GAP_VL[0])) + top0))
        th_all = [MC[(("bm", i, 1000), s)][0] for s in SEEDS]
        M_all = [MC[(("bm", i, 1000), s)][1] for s in SEEDS]
        th = np.concatenate(th_all)
        Mx = np.concatenate(M_all)
        Nt = len(th)
        cor_seeds = [np.corrcoef(t[:, 0], t[:, 1])[0, 1] for t in th_all]
        cor_mc = np.corrcoef(th[:, 0], th[:, 1])[0, 1]
        # SE of corr via batch means (30 batches)
        bm = np.array([np.corrcoef(b[:, 0], b[:, 1])[0, 1] for b in np.array_split(th, 30)])
        se = bm.std(ddof=1) / math.sqrt(30)
        rho_r = round(rho, 6)
        cA = corr[rho_r] if rho_r in corr else None
        C.z(f"MC corr(theta1,theta2) rho={rho:+.4f} pooled", cor_mc, se, cA)
        for s, cs in zip(SEEDS, cor_seeds):
            C.z(f"MC corr seed {s} rho={rho:+.4f}", cs, se * math.sqrt(3), cA)
        # marginals arcsine
        for j in range(2):
            obs = np.histogram(th[:, j], bins=40, range=(0, 1))[0]
            e = Nt * np.diff(arcsine_cdf(np.linspace(0, 1, 41)))
            pch = stats.chisquare(obs, e).pvalue
            C.add(pch >= CHI2_P, f"MC theta{j + 1} arcsine rho={rho:+.4f}", f"chi-square 40 bins p = {pch:.3g}")
        pl = np.mean(th[:, 0] < th[:, 1])
        C.z(f"MC P(theta1<theta2)=1/2 rho={rho:+.4f}", pl, math.sqrt(0.25 / Nt), 0.5)
        # diagonal law
        mx = th.max(1)
        zs = []
        for t, Dv in zip(DIAG_T, pr["D"]):
            pe = np.mean(mx <= t)
            zs.append((pe - Dv) / math.sqrt(max(Dv * (1 - Dv), 1e-12) / Nt))
        C.add(np.max(np.abs(zs)) <= Z_TOL, f"MC diagonal law P(theta1,theta2<=t) rho={rho:+.4f}", f"15 values of t, max |z| = {np.max(np.abs(zs)):.2f}")
        # gap CDF
        gp = np.abs(th[:, 0] - th[:, 1])
        zs = []
        for u in GAPCDF_U:
            pa = float(gapcdf(u))
            pe = np.mean(gp <= u)
            zs.append((pe - pa) / math.sqrt(max(pa * (1 - pa), 1e-12) / Nt))
        C.add(np.max(np.abs(zs)) <= Z_TOL, f"MC gap law P(|theta1-theta2|<=u) rho={rho:+.4f}",
              f"u in {GAPCDF_U}, max |z| = {np.max(np.abs(zs)):.2f}")
        # joint CDF
        zs = []
        for (aa, bb), Fv in zip(CDF_AB, pr["Fab"]):
            pe = 0.5 * (np.mean((th[:, 0] <= aa) & (th[:, 1] <= bb)) + np.mean((th[:, 1] <= aa) & (th[:, 0] <= bb)))
            zs.append((pe - Fv) / math.sqrt(max(Fv * (1 - Fv), 1e-12) / Nt))
        C.add(np.max(np.abs(zs)) <= Z_TOL, f"MC joint CDF P(theta1<=a,theta2<=b) rho={rho:+.4f}", f"7 points (a,b), max |z| = {np.max(np.abs(zs)):.2f}")
        # Rogers-Shepp by MC
        c_rs = RS_ALL[rho]
        bmM = np.array([np.mean(b[:, 0] * b[:, 1]) for b in np.array_split(Mx, 30)])
        C.z(f"MC E[M1 M2] vs Rogers-Shepp rho={rho:+.4f}", np.mean(Mx[:, 0] * Mx[:, 1]), bmM.std(ddof=1) / math.sqrt(30), c_rs)
        # step halving
        th2 = np.concatenate([MC[(("bm", i, 2000), s)][0] for s in SEEDS])
        bm2 = np.array([np.corrcoef(b[:, 0], b[:, 1])[0, 1] for b in np.array_split(th2, 30)])
        c2_ = np.corrcoef(th2[:, 0], th2[:, 1])[0, 1]
        se2 = bm2.std(ddof=1) / math.sqrt(30)
        zh = (c2_ - cor_mc) / math.hypot(se, se2)
        gp2 = np.abs(th2[:, 0] - th2[:, 1])
        zg = []
        for u in GAPCDF_U[:4]:
            p1, p2 = np.mean(gp <= u), np.mean(gp2 <= u)
            pp = 0.5 * (p1 + p2)
            zg.append((p2 - p1) / math.sqrt(max(pp * (1 - pp), 1e-12) * (1 / Nt + 1 / len(gp2))))
        C.add(abs(zh) <= Z_TOL and np.max(np.abs(zg)) <= Z_TOL, f"step halving n=1000 -> 2000 rho={rho:+.4f}",
              f"corr {cor_mc:.5f} -> {c2_:.5f} (z={zh:+.2f}); gap CDF at u<=0.03 max |z| = {np.max(np.abs(zg)):.2f}")
        # MC exponents: corner mass and gap mass slopes on a window with >= 2000 events at its lower end, compared with
        # the formula's slope on the same window (the asymptotic exponent is reached only at much smaller scales)
        Wr = Wedge(rho)
        grid = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
        Dg = Wr.diag(grid)
        k1 = int(np.nonzero(Dg * Nt >= 2000)[0][0])
        e1, e2 = grid[k1], min(10 * grid[k1], 0.5)
        De = Wr.diag([e1, e2])
        pc = np.array([np.mean(mx <= e1), np.mean(mx <= e2)])
        mc_corner = math.log(pc[1] / pc[0]) / math.log(e2 / e1)
        an_corner = math.log(De[1] / De[0]) / math.log(e2 / e1)
        gg = np.array([float(gapcdf(u)) for u in grid])
        k2 = int(np.nonzero(gg * Nt >= 2000)[0][0])
        u1, u2 = grid[k2], min(10 * grid[k2], 0.5)
        pg = np.array([np.mean(gp <= u1), np.mean(gp <= u2)])
        mc_gap = math.log(pg[1] / pg[0]) / math.log(u2 / u1)
        an_gap = math.log(gapcdf(u2) / gapcdf(u1)) / math.log(u2 / u1)
        C.add(abs(mc_corner - an_corner) <= SLOPE_MC and abs(mc_gap - an_gap) <= SLOPE_MC,
              f"MC exponents rho={rho:+.4f}", f"corner mass slope on [{e1:g},{e2:g}]: MC {mc_corner:.3f} vs formula {an_corner:.3f} "
              f"(asymptote pi/(2a) = {math.pi / (2 * a):.3f}); gap mass slope on [{u1:g},{u2:g}]: MC {mc_gap:.3f} vs formula {an_gap:.3f}")
        mcrows.append(dict(rho=rho, N=Nt, corr_mc=cor_mc, corr_se=se, corr_an=cA, corr_seeds=cor_seeds, corr_half=c2_,
                           EM12_mc=float(np.mean(Mx[:, 0] * Mx[:, 1])), EM12_rs=c_rs,
                           corrM_mc=float(np.corrcoef(Mx[:, 0], Mx[:, 1])[0, 1]),
                           corrM_an=(c_rs - 2 / math.pi) / (1 - 2 / math.pi), mc_corner=mc_corner, an_corner=an_corner,
                           mc_gap=mc_gap, an_gap=an_gap))
        R[f"mc_hist2d_{i}"] = np.histogram2d(th[:, 0], th[:, 1], bins=80, range=[[0, 1], [0, 1]])[0]
        R[f"mc_gaphist_{i}"] = np.histogram(gp, bins=100, range=(0, 1))[0]
        R[f"mc_diag_{i}"] = np.array([np.mean(mx <= t) for t in DIAG_T])
        if i == 0:
            R["mc_pair_example"] = th[:5]
    J["mc"] = mcrows

    # E[theta W(u)] by MC (all rho pooled: theta1, W1 do not depend on rho)
    TW = np.concatenate([np.concatenate([MC[(("bm", i, 1000), s)][0][:, :1], MC[(("bm", i, 1000), s)][2]], 1)
                         for i in range(len(RHO_MC)) for s in SEEDS])
    us = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    zs = []
    for k_, u in enumerate(us):
        prod = TW[:, 0] * TW[:, 1 + k_]
        se_ = prod.std() / math.sqrt(len(prod))
        zs.append((prod.mean() - EthetaW_exact(u)) / se_)
    C.add(np.max(np.abs(zs)) <= Z_TOL, "first chaos: MC E[theta W(u)] vs sqrt(2/pi)[(2/3)u^1.5 + (2/3)(1-(1-u)^1.5) - u]",
          f"u = {list(us)}, z = {[round(z, 2) for z in zs]}")

    # 7. images at alpha = pi/n vs MC (n = 3, 4, 6 -> rho = -1/2, -1/sqrt2, -sqrt3/2)
    for im in imgs:
        n = im["n"]
        if n == 2:
            continue
        rho = -math.cos(math.pi / n)
        i = [k for k, r in enumerate(RHO_MC) if abs(r - rho) < 1e-9][0]
        th = np.concatenate([MC[(("bm", i, 1000), s)][0] for s in SEEDS])
        mx = th.max(1)
        zs = [(np.mean(mx <= t) - Dv) / math.sqrt(Dv * (1 - Dv) / len(th)) for t, Dv in zip(DIAG_T, im["D_i"])]
        C.add(np.max(np.abs(zs)) <= Z_TOL, f"images n={n}: diagonal law (image formula) vs MC", f"max |z| = {np.max(np.abs(zs)):.2f}")
        mi = im["mom_i"]
        ci = 8 * (mi["E12"] - 0.25)
        C.close(f"images n={n}: corr from image formula vs eigen-series", ci, corr[round(rho, 6)], 2e-6)
        J[f"images_{n}"] = dict(corr=ci, mass=mi["mass_half"], E12=mi["E12"])

    # 8. bridges
    brrows = []
    for i, rho in enumerate(RHO_BR):
        pr = PR[round(rho, 6)]
        th = np.concatenate([MC[(("bb", i, 1000), s)][0] for s in SEEDS])
        Nt = len(th)
        D = (th[:, 1] - th[:, 0]) % 1.0
        o1 = np.histogram(th[:, 0], bins=20, range=(0, 1))[0]
        p1 = stats.chisquare(o1).pvalue
        H = np.histogram2d(th[:, 0], D, bins=[10, 10], range=[[0, 1], [0, 1]])[0]
        pind = stats.chi2_contingency(H)[1]
        C.add(p1 >= CHI2_P and pind >= CHI2_P, f"bridge rho={rho:+.4f}: theta1 uniform, independent of D",
              f"uniform chi-square p = {p1:.3g}; independence (10x10) p = {pind:.3g}")
        # kappa normalisation and MC histogram (kappa symmetric about 1/2; spline in log u on (1e-5, 1/2])
        uu, kap = KAP_U, pr["kap"]
        a = math.acos(-rho)
        p = math.pi / a
        Ak = log_antider(uu, kap)
        Ak2 = log_antider(uu, kap * uu ** 2)
        Ak2m = log_antider(uu, kap * (1 - uu) ** 2)
        head = kap[0] * uu[0] / (p - 1)
        half = head + Ak(0.5) - Ak(uu[0])
        Ktot = 2 * half
        C.close(f"bridge rho={rho:+.4f}: int kappa = 1", float(Ktot), 1.0, 1e-5)
        cdf = lambda x: (head + Ak(x) - Ak(uu[0])) if x <= 0.5 else 1.0 - (head + Ak(1 - x) - Ak(uu[0]))
        edges = np.linspace(0, 1, 21)
        e = Nt * np.diff([0.0] + [float(cdf(x)) for x in edges[1:-1]] + [1.0])
        o = np.histogram(D, bins=edges)[0]
        pD = stats.chisquare(o, e * o.sum() / e.sum()).pvalue
        C.add(pD >= CHI2_P, f"bridge rho={rho:+.4f}: law of D vs kappa", f"chi-square 20 bins p = {pD:.3g}")
        ED2 = (Ak2(0.5) - Ak2(uu[0])) + (Ak2m(0.5) - Ak2m(uu[0])) + head * (1 + uu[0] ** 2)
        corr_an = 6 * ED2 - 2
        bmb = np.array([np.corrcoef(b[:, 0], b[:, 1])[0, 1] for b in np.array_split(th, 30)])
        cmc = np.corrcoef(th[:, 0], th[:, 1])[0, 1]
        C.z(f"bridge rho={rho:+.4f}: corr = 6 E[D^2] - 2", cmc, bmb.std(ddof=1) / math.sqrt(30), corr_an, tol=Z_TOL)
        brrows.append(dict(rho=rho, corr_an=corr_an, corr_mc=cmc, ED2=ED2))
        R[f"bb_Dhist_{i}"] = np.histogram(D, bins=50, range=(0, 1))[0]
    J["bridge"] = brrows
    for rho in RHO_BR:
        if abs(rho) < 1e-9:
            continue
        kap = PR[round(rho, 6)]["kap"]
        p = math.pi / math.acos(-rho)
        sl = math.log(kap[1] / kap[0]) / math.log(KAP_U[1] / KAP_U[0])
        C.add(abs(sl - (p - 2)) < 0.02, f"bridge exponent rho={rho:+.4f}: kappa(u) ~ u^(pi/a-2) at u -> 0",
              f"local slope at u = {KAP_U[0]:g}: {sl:.4f} vs pi/a - 2 = {p - 2:.4f}")
    # bridge rho -> -1 vs MMS
    kb = {}
    for rho in (-0.99, -0.98, -0.97):
        kap = PR[round(rho, 6)]["kap"]
        p = math.pi / math.acos(-rho)
        Ak2 = log_antider(KAP_U, kap * KAP_U ** 2)
        Ak2m = log_antider(KAP_U, kap * (1 - KAP_U) ** 2)
        head = kap[0] * KAP_U[0] / (p - 1)
        ED2 = (Ak2(0.5) - Ak2(KAP_U[0])) + (Ak2m(0.5) - Ak2m(KAP_U[0])) + head * (1 + KAP_U[0] ** 2)
        kb[rho] = ED2 / 2 - 1 / 6
    pf = np.polyfit(np.array(list(kb)) + 1, np.array(list(kb.values())), 2)
    C.close("bridge rho -> -1: cov extrapolated from -0.99,-0.98,-0.97 -> -(pi^2-9)/36 (MMS)", float(np.polyval(pf, 0)), COV_BB_MAXMIN, 2e-3)
    J["bridge_minus1"] = dict(extrap=float(np.polyval(pf, 0)), mms=COV_BB_MAXMIN)

    # 9. several motions (F7): numerical only
    f7 = {}
    for nm, A in F7.items():
        th = np.concatenate([MC[(("f7", nm, 1000), s)][0] for s in SEEDS])
        Nn = th.shape[1]
        cm = np.corrcoef(th.T)
        rho_ij = (A @ A.T)
        order = np.argsort(th, axis=1)
        perms, cnt = np.unique(order, axis=0, return_counts=True)
        f7[nm] = dict(corr=cm.tolist(), rho=rho_ij.tolist(), orders={"".join(map(str, p_ + 1)): int(c) for p_, c in zip(perms, cnt)},
                      N=len(th), E_prod=float(np.mean(np.prod(th, 1))))
        for i_ in range(Nn):
            for j_ in range(i_ + 1, Nn):
                rr = round(float(rho_ij[i_, j_]), 6)
                if rr in corr:
                    bmb = np.array([np.corrcoef(b[:, i_], b[:, j_])[0, 1] for b in np.array_split(th, 30)])
                    C.z(f"F7 {nm}: pair ({i_ + 1},{j_ + 1}) corr vs two-motion law at rho={rr:+.3f}", cm[i_, j_],
                        bmb.std(ddof=1) / math.sqrt(30), corr[rr])
    J["f7"] = f7

    # 10. discrete walks (F8)
    f8 = []
    for law in ("gauss", "unif", "laplace", "copy"):
        for rho in (-0.5, 0.5):
            for n in (10, 100, 1000):
                arr = np.concatenate([WK[(law, rho, n, s)] for s in SEEDS])
                Nw = len(arr)
                p0 = np.mean(arr[:, 0] == 0)
                sa = math.comb(2 * n, n) / 4 ** n
                eq = np.mean(arr[:, 0] == arr[:, 1])
                c00 = np.mean((arr[:, 0] == 0) & (arr[:, 1] == 0))
                cr = np.corrcoef(arr[:, 0] / n, arr[:, 1] / n)[0, 1]
                bmb = np.array([np.corrcoef(b[:, 0], b[:, 1])[0, 1] for b in np.array_split(arr, 30)])
                f8.append(dict(law=law, rho=rho, n=n, N=Nw, p0=p0, sparre=sa, p_eq=eq, p00=c00, corr=cr,
                               corr_se=bmb.std(ddof=1) / math.sqrt(30)))
                C.z(f"F8 {law} rho={rho:+.1f} n={n}: P(theta1 = 0) = C(2n,n)/4^n (Sparre Andersen)", p0, math.sqrt(sa * (1 - sa) / Nw), sa)
    J["f8"] = f8
    for rho in (-0.5, 0.5):
        zbest, info = 0.0, []
        for n in (10, 100):
            rows_ = [r for r in f8 if r["rho"] == rho and r["n"] == n]
            zmax = 0
            for r1 in rows_:
                for r2 in rows_:
                    se_ = math.sqrt(r1["p00"] * (1 - r1["p00"]) / r1["N"] + r2["p00"] * (1 - r2["p00"]) / r2["N"])
                    if se_ > 0:
                        zmax = max(zmax, abs(r1["p00"] - r2["p00"]) / se_)
            zbest = max(zbest, zmax)
            info.append(f"n={n}: max pairwise |z| = {zmax:.1f}, values {[round(float(r['p00']), 6) for r in rows_]}")
        C.add(zbest >= 10, f"F8 non-universality detected: P(theta1 = theta2 = 0) differs between step laws, rho={rho:+.1f}", "; ".join(info))
        for law in ("gauss", "unif", "laplace", "copy"):
            r3 = [r for r in f8 if r["rho"] == rho and r["law"] == law and r["n"] == 1000][0]
            C.z(f"F8 {law} rho={rho:+.1f} n=1000: corr(theta1,theta2) -> Brownian value", r3["corr"], math.hypot(r3["corr_se"], 2.5e-3), corr[rho])

    # ══ save ════════════════════════════════════════════════════════════════════════════════════════════════
    R["rho_curve"] = rhos
    R["corr_curve"] = cv
    R["gap_u"] = GAP_U
    R["diag_t"] = DIAG_T
    R["kap_u"] = KAP_U
    R["eps_corner"] = EPS_CORNER
    R["u_diag"] = U_DIAG
    for p in profiles:
        key = f"{p['rho']:+.6f}"
        R[f"gap_{key}"] = p["g"]
        R[f"D_{key}"] = p["D"]
        R[f"kap_{key}"] = p["kap"]
        R[f"corner_{key}"] = p["corner"]
        R[f"fdiag_{key}"] = p["fdiag"]
    for rho, H in heats:
        R[f"heat_{rho:+.6f}"] = H
    for im in imgs:
        R[f"img_kap_{im['n']}"] = im["kap_closed_curve"]
        R[f"img_gap_{im['n']}"] = im["g_i"]
        R[f"img_D_{im['n']}"] = im["D_i"]
    R["img_gap_u"] = IMG_GAP_U
    R["rs_all_rho"] = np.array(list(RS_ALL))
    R["rs_all_c"] = np.array(list(RS_ALL.values()))
    R["rs_rho"] = np.array([r for r, _, _ in rs])
    R["rs_c"] = np.array([c for _, c, _ in rs])
    rr = np.linspace(-0.999, 0.999, 201)
    R["rs_curve_rho"] = rr
    R["rs_curve_corrM"] = np.array([(rogers_shepp(x) - 2 / math.pi) / (1 - 2 / math.pi) for x in rr])
    # one illustrative pair of paths (rho = 0.5, 4000 steps, seed 2026) for the wedge figure
    rng = np.random.default_rng(2026)
    ne = 4000
    dZ = rng.standard_normal((ne, 2)) * math.sqrt(1 / ne)
    Ze = np.vstack([[0, 0], np.cumsum(dZ, 0)])
    We = Ze @ A_pair(0.5).T
    R["ex_t"], R["ex_W1"], R["ex_W2"] = np.linspace(0, 1, ne + 1), We[:, 0], We[:, 1]
    R["rho_mc"] = np.array(RHO_MC)
    R["rho_br"] = np.array(RHO_BR)
    np.savez_compressed(HERE / "results.npz", **R)
    J["corr_curve"] = {f"{r:+.4f}": float(c) for r, c in zip(rhos, cv)}
    J["curve_detail"] = [dict(rho=p["rho"], alpha=p["a"], E12=p["m2"]["E12"], Egap=p["m2"]["Egap"], Egap2=p["m2"]["Egap2"],
                              mass_half=p["m2"]["mass_half"], k_N=p["k_N"], k_tail=p["k_tail"], Nx=p["Nx"], X=p["X"]) for p in curve]
    J["rogers_shepp"] = [dict(rho=r, c_rs=c, c_wedge=w, corrM=(c - 2 / math.pi) / (1 - 2 / math.pi)) for r, c, w in rs]
    J["diag"] = {f"{p['rho']:+.6f}": list(map(float, p["D"])) for p in profiles}
    J["profiles_Fab"] = {f"{p['rho']:+.6f}": list(map(float, p["Fab"])) for p in profiles}
    J["runtime_s"] = time.time() - t_start
    (HERE / "numbers.json").write_text(json.dumps(J, indent=1, default=float))
    write_checks(C, header)
    print(f"done: {len(C.lines) - C.fails} PASS, {C.fails} FAIL, {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
