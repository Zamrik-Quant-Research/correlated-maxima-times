"""derive — every number the paper quotes, read from the saved research outputs (no new computation).

Reads numbers.json, results.npz and CHECKS.txt (all written once by model.py, which is NOT run here) and writes
derived.json: formatted scalars under "values" and the paper's Markdown tables under "tables". Nothing here simulates
or integrates a model: it selects, takes maxima of saved errors and z-scores, takes log-differences of saved arrays
(local slopes), interpolates the saved Rogers-Shepp curve with a cubic spline, and evaluates the closed forms the paper
states (c1 = 4 - 32/(3 pi), (6 - 7 zeta(3))/4, the explicit constants of Proposition 2 and Theorem 6 from the saved
c_alpha and I_alpha).
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import beta, gamma, zeta

HERE = Path(__file__).resolve().parent
J = json.loads((HERE / "numbers.json").read_text())
R = np.load(HERE / "results.npz")
CH = (HERE / "CHECKS.txt").read_text().splitlines()
V, T = {}, {}


def num(rx, cast=float):
    """All matches of one capture group over CHECKS.txt."""
    out = []
    for ln in CH:
        m = re.search(rx, ln)
        if m:
            out.append(cast(m.group(1)))
    return out


def sci(x, d=1):
    """1.2e-10 -> '1.2 \\times 10^{-10}' (for prose inside $...$)."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, d) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{d}f} \\times 10^{{{e}}}"


def up(x, d=1):
    """An upper bound written with d significant decimals, rounded UP (an error bound is never rounded down)."""
    e = int(math.floor(math.log10(abs(x))))
    m = math.ceil(x / 10 ** e * 10 ** d) / 10 ** d
    if m >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{d}f} \\times 10^{{{e}}}"


def rlab(rho):
    named = {-1 / math.sqrt(2): r"$-1/\sqrt{2}$", -math.sqrt(3) / 2: r"$-\sqrt{3}/2$"}
    for k, v in named.items():
        if abs(rho - k) < 1e-4:
            return v
    return "$0$" if abs(rho) < 1e-12 else f"${rho:+g}$"


def alpha(rho):
    return math.acos(-rho)


# ── 1. the decomposition: series, truncation, checks (Section 3) ─────────────────────────────────────────────────────
tails = num(r"series truncation .*rigorous tail bound ([0-9.e+-]+);") + num(r"series truncation .*delta: J=\d+ odd terms, tail ([0-9.e+-]+)")
V["trunc_tail_max"] = up(max(tails))
V["rho0_delta_err"] = up(num(r"rho=0: delta = sqrt\(2/pi\) erf.*abs err ([0-9.e+-]+)")[0])
V["rho0_k_err"] = up(num(r"rho=0: k\(x,y\).*abs err ([0-9.e+-]+)")[0])
V["rho0_f_err"] = up(num(r"rho=0: f\(s,t\) = arcsine x arcsine.*abs err ([0-9.e+-]+)")[0])
V["marg_err"] = up(max(num(r"marginal of f is arcsine.*abs err ([0-9.e+-]+)")))
V["mass_half_err"] = up(max(abs(d["mass_half"] - 0.5) for d in J["curve_detail"]))
V["n_curve"] = len(J["curve_detail"])
V["quad_doubling"] = up(max(num(r"doubling/1.5x-nodes diff ([0-9.e+-]+)\)")))
rs = J["rogers_shepp"]
V["rs_err"] = up(max(abs(r["c_rs"] - r["c_wedge"]) for r in rs))
V["rs_n"] = len(rs)
V["rs_rho_min"], V["rs_rho_max"] = f"{min(r['rho'] for r in rs):g}", f"{max(r['rho'] for r in rs):g}"
emz = num(r"MC E\[M1 M2\] vs Rogers-Shepp .*\(z = ([+-]?[0-9.]+)\)")
V["rs_mc_zmax"] = f"{max(abs(z) for z in emz):.2f}"
V["rs_mc_n"] = len(emz)
V["rs_half"] = f"{[r for r in rs if abs(r['rho'] + 0.5) < 1e-9][0]['c_rs']:.12f}"
c_half = [r for r in rs if abs(r["rho"] - 0.5) < 1e-9][0]["c_rs"]
V["rs_plus_half"] = f"{c_half:.10f}"
V["rs_plus_half_closed"] = f"{(3 * math.sqrt(3) - 2) / 4:.10f}"
V["rs_plus_half_diff"] = up(abs(c_half - (3 * math.sqrt(3) - 2) / 4))
V["rs_c0"] = f"{2 / math.pi:.10f}"

# ── 2. the density, the diagonal law, the gap (Section 4) ────────────────────────────────────────────────────────────
mcN = [m["N"] for m in J["mc"]]
V["mc_N_per_rho"] = f"{mcN[0]:,}"
V["mc_N_per_rho_millions"] = f"{mcN[0] / 1e6:g}"
V["mc_N_total_millions"] = f"{sum(mcN) / 1e6:g}"
V["mc_n_rho"] = len(mcN)
V["diag_rho0_err"] = up(num(r"rho=0: diagonal law = .*abs err ([0-9.e+-]+)")[0])
V["jointcdf_rho0_err"] = up(num(r"rho=0: joint CDF = product .*abs err ([0-9.e+-]+)")[0])
dz = num(r"MC diagonal law P\(theta1,theta2<=t\) .*max \|z\| = ([0-9.]+)")
V["diag_mc_zmax"] = f"{max(dz):.2f}"
gz = num(r"MC gap law P\(\|theta1-theta2\|<=u\) .*max \|z\| = ([0-9.]+)")
V["gap_mc_zmax"] = f"{max(gz):.2f}"
jz = num(r"MC joint CDF P\(theta1<=a,theta2<=b\) .*max \|z\| = ([0-9.]+)")
V["jcdf_mc_zmax"] = f"{max(jz):.2f}"
gint = num(r"gap density integrates to 1, .*abs err ([0-9.e+-]+)")
V["gap_int_err"] = up(max(gint))
pz = num(r"MC P\(theta1<theta2\)=1/2 .*\(z = ([+-]?[0-9.]+)\)")
V["p12_mc_zmax"] = f"{max(abs(z) for z in pz):.2f}"
az = num(r"MC theta[12] arcsine .*p = ([0-9.e+-]+)")
V["arcsine_chi2_pmin"] = f"{min(az):.3f}"
hz = num(r"step halving n=1000 -> 2000 .*\(z=([+-]?[0-9.]+)\)")
V["halving_zmax"] = f"{max(abs(z) for z in hz):.2f}"
hg = num(r"step halving .*gap CDF at u<=0.03 max \|z\| = ([0-9.]+)")
V["halving_gap_zmax"] = f"{max(hg):.2f}"
cz = num(r"MC corr\(theta1,theta2\) rho=.* pooled: .*\(z = ([+-]?[0-9.]+)\)")
V["corr_mc_zmax"] = f"{max(abs(z) for z in cz):.2f}"
sz = num(r"MC corr seed \d+ .*\(z = ([+-]?[0-9.]+)\)")
V["corr_seed_zmax"] = f"{max(abs(z) for z in sz):.2f}"
V["n_seed_tests"] = len(sz)

t = R["diag_t"]
cols = [0.1, 0.3, 0.5, 0.7, 0.9]
idx = [int(np.argmin(abs(t - c))) for c in cols]
rows = ["| $\\rho$ | " + " | ".join(f"$t = {c:g}$" for c in cols) + " | MC max $\\lvert z \\rvert$ (15 $t$) |", "|---|" + "---|" * (len(cols) + 1)]
rho_mc = list(R["rho_mc"])
zrho = dict(zip([round(r, 4) for r in rho_mc], dz))
for key in J["diag"]:
    rho = float(key)
    if key in ("-0.980000", "-0.970000", "-0.866025", "+0.300000", "+0.700000"):
        continue
    vals = J["diag"][key]
    i_mc = [k for k, r in enumerate(rho_mc) if abs(r - rho) < 1e-5]
    zc = f"{zrho[round(rho_mc[i_mc[0]], 4)]:.2f}" if i_mc else "--"
    rows.append(f"| {rlab(rho)} | " + " | ".join(f"{vals[i]:.5f}" for i in idx) + f" | {zc} |")
T["tab_diag"] = "\n".join(rows)
V["diag_half_m09"] = f"{J['diag']['-0.900000'][idx[2]]:.5f}"
V["diag_half_p09"] = f"{J['diag']['+0.900000'][idx[2]]:.5f}"
V["diag_half_0"] = f"{J['diag']['+0.000000'][idx[2]]:.5f}"

# ── 3. local behaviour (Section 5) ───────────────────────────────────────────────────────────────────────────────────
ku = R["kap_u"]
rows = ["| $\\rho$ | corner slope | $\\pi/(2\\alpha) - 2$ | diagonal slope | $\\pi/\\alpha - 2$ | gap slope | predicted | bridge slope |",
        "|---|---|---|---|---|---|---|---|"]
dev_corner, dev_diag, dev_gap, dev_br = [], [], [], []
for e in J["exponents"]:
    rho, a = e["rho"], e["alpha"]
    if abs(rho + 0.98) < 1e-9 or abs(rho + 0.97) < 1e-9:
        continue
    p = math.pi / a
    kap = R[f"kap_{rho:+.6f}"]
    brs = math.log(kap[1] / kap[0]) / math.log(ku[1] / ku[0]) if abs(rho) > 1e-9 else 0.0
    gp = "$0$ (log)" if abs(rho) < 1e-9 else f"{e['gap_pred']:.4f}"
    rows.append(f"| {rlab(rho)} | {e['corner']:.4f} | {e['corner_pred']:.4f} | {e['diag']:.4f} | "
                f"{e['diag_pred']:.4f} | {e['gap']:.4f} | {gp} | {brs:.4f} |")
    dev_corner.append(abs(e["corner"] - e["corner_pred"]))
    dev_diag.append(abs(e["diag"] - e["diag_pred"]))
    if abs(rho) > 1e-9:
        dev_gap.append(abs(e["gap"] - e["gap_pred"]))
        dev_br.append(abs(brs - (p - 2)))
T["tab_exp"] = "\n".join(rows).replace("| -0.0000 |", "| 0.0000 |")
V["exp_corner_dev"] = f"{max(dev_corner):.1e}"
V["exp_diag_dev"] = f"{max(dev_diag):.1e}"
V["exp_gap_dev"] = f"{max(dev_gap):.4f}"
V["exp_bridge_dev"] = f"{max(dev_br):.4f}"
V["kap_u0"] = f"{ku[0]:g}"
e0 = [e for e in J["exponents"] if abs(e["rho"]) < 1e-9][0]
V["gap_slope_rho0"] = f"{e0['gap']:.4f}"
# rho = 0: g(u) ~ (4/pi^2) log(1/u) -- coefficient from the saved gap array
u = R["gap_u"]
g0 = R["gap_+0.000000"]
i1, i2 = int(np.argmin(abs(u - 1e-12))), int(np.argmin(abs(u - 1e-6)))
A = (g0[i1] - g0[i2]) / math.log(u[i2] / u[i1])
V["gap_log_coef"] = f"{A:.6f}"
V["four_over_pi2"] = f"{4 / math.pi ** 2:.6f}"
V["gap_log_relerr"] = up(abs(A / (4 / math.pi ** 2) - 1))
V["gap_log_u1"], V["gap_log_u2"] = sci(u[i1], 0), sci(u[i2], 0)
# explicit constants: diagonal ratio, gap constant (rho > 0), bridge constant
ratios, gaprows, brrows = [], [], []
for d in J["diag_constants"]:
    rho = d["rho"]
    a = alpha(rho)
    p = math.pi / a
    ratios.append(abs(d["ratio"][0] - 1))
    if p < 2:
        Cg = math.sin(a) / 2 * d["ca"] ** 2 * d["Ia"] * beta(1 - p / 2, 1 - p / 2)
        g = R[f"gap_{rho:+.6f}"]
        iu = int(np.argmin(abs(u - 1e-12)))
        gaprows.append((rho, g[iu] / u[iu] ** (p - 2), Cg))
    Cb = p ** 3 * math.sin(a) * d["Ia"] / (2 ** p * gamma(p + 1))
    kap = R[f"kap_{rho:+.6f}"]
    brrows.append((rho, kap[0] / ku[0] ** (p - 2), Cb))
V["diag_ratio_dev"] = up(max(ratios))
V["diag_const_rhos"] = ", ".join(f"${d['rho']:g}$" for d in J["diag_constants"])
V["diag_const_table"] = "; ".join(f"$\\rho = {d['rho']:g}$: $c_\\alpha = {d['ca']:.6f}$, $I_\\alpha = {d['Ia']:.6f}$" for d in J["diag_constants"])
V["gap_const_cmp"] = "; ".join(f"$\\rho = {r:g}$: ${x:.5f}$ against ${c:.5f}$" for r, x, c in gaprows)
V["gap_const_relmax"] = f"{max(abs(x / c - 1) for _, x, c in gaprows):.4f}"
V["gap_const_relmax_excl03"] = up(max(abs(x / c - 1) for r, x, c in gaprows if r > 0.31))
p03 = math.pi / alpha(0.3)
V["gap_next_order_03"] = f"{1e-12 ** ((2 - p03) / 2):.3f}"
V["gap_const_p03"] = f"{p03:.4f}"
V["br_const_cmp"] = "; ".join(f"$\\rho = {r:g}$: ${x:.5f}$ against ${c:.5f}$" for r, x, c in brrows)
V["br_const_relmax"] = up(max(abs(x / c - 1) for _, x, c in brrows))
rows = ["| $\\rho$ | corner window | MC | formula | $\\pi/(2\\alpha)$ | gap window | MC | formula | asymptote |",
        "|---|---|---|---|---|---|---|---|---|"]
win = {}
for ln in CH:
    m = re.search(r"MC exponents rho=([+-][0-9.]+): corner mass slope on \[([0-9.]+),([0-9.]+)\].*gap mass slope on \[([0-9.]+),([0-9.]+)\]", ln)
    if m:
        win[round(float(m.group(1)), 4)] = (m.group(2), m.group(3), m.group(4), m.group(5))
dslope = []
for m in J["mc"]:
    rho = m["rho"]
    a = alpha(rho)
    p = math.pi / a
    w = win[round(rho, 4)]
    asym = "$1$ ($u \\log(1/u)$)" if abs(rho) < 1e-9 else (f"{p / 2:.3f}" if rho < 0 else f"{p - 1:.3f}")
    rows.append(f"| {rlab(rho)} | $[{w[0]}, {w[1]}]$ | {m['mc_corner']:.3f} | {m['an_corner']:.3f} | {p / 2:.3f} | "
                f"$[{w[2]}, {w[3]}]$ | {m['mc_gap']:.3f} | {m['an_gap']:.3f} | {asym} |")
    dslope += [abs(m["mc_corner"] - m["an_corner"]), abs(m["mc_gap"] - m["an_gap"])]
V["window_n"] = len(J["mc"])
V["window_slope_dev"] = f"{max(dslope):.3f}"

# ── 4. the correlation (Section 6) ───────────────────────────────────────────────────────────────────────────────────
ch = J["chaos"]
c1 = 4 - 32 / (3 * math.pi)
V["c1"] = f"{c1:.9f}"
V["c1_5"] = f"{c1:.5f}"
V["c1_num"] = f"{ch['c1_num']:.9f}"
V["c1_err"] = up(abs(ch["c1_num"] - c1))
V["corr_m1"] = f"{(6 - 7 * zeta(3)) / 4:.7f}"
V["cov_m1"] = f"{-(7 * zeta(3) - 6) / 32:.7f}"
V["S_odd"] = f"{(7 * zeta(3) - 2) / 8:.6f}"
V["S_even"] = f"{(10 - 7 * zeta(3)) / 8:.6f}"
V["S_odd4"] = f"{(7 * zeta(3) - 2) / 8:.4f}"
V["c2_num"] = f"{ch['c2_num']:.4f}"
V["c3_num"] = f"{ch['c3_num']:.4f}"
V["k1"] = f"{ch['kappa1']:.4f}"
V["k2"] = f"{ch['kappa2']:.4f}"
V["k_icpt"] = sci(ch["intercept"], 1)
V["k1_over"] = f"{ch['kappa1'] / (2 * math.sqrt(math.pi)):.4f}"
V["extrap_m1"] = f"{ch['extrap_minus1']:.7f}"
V["extrap_m1_err"] = up(abs(ch["extrap_minus1"] - (6 - 7 * zeta(3)) / 4))
geb = num(r"Gebelein \|corr\| <= \|rho\| on the grid: max\(\|corr\|-\|rho\|\) = ([0-9.e+-]+)")
V["gebelein_max"] = sci(geb[0], 1)
cv = re.search(r"min slope ([0-9.]+), min slope increment ([0-9.e+-]+)", "\n".join(CH))
V["convex_min_slope"], V["convex_min_incr"] = cv.group(1), sci(float(cv.group(2)), 2)
fcz = [float(x) for x in re.findall(r"np\.float64\(([0-9.]+)\)\]?", [l for l in CH if "first chaos" in l][0])[6:]]
V["first_chaos_zmax"] = f"{max(fcz):.2f}"
V["first_chaos_n"] = len(fcz)
cc = J["corr_curve"]
cs = CubicSpline(R["rs_curve_rho"], R["rs_curve_corrM"])
less = 0
nonzero = 0
for key, val in cc.items():
    rho = float(key)
    if abs(rho) < 1e-9:
        continue
    nonzero += 1
    if abs(val) < abs(float(cs(rho))):
        less += 1
V["obs_less"], V["obs_nonzero"] = less, nonzero
show = ["-0.9900", "-0.9000", "-0.7071", "-0.5000", "-0.1000", "+0.1000", "+0.5000", "+0.9000", "+0.9900", "+0.9990"]
det = {f"{d['rho']:+.4f}": d for d in J["curve_detail"]}
rows = ["| $\\rho$ | $\\operatorname{corr}(\\theta_1, \\theta_2)$ | lower bound | upper bound | $\\operatorname{corr}(M_1, M_2)$ |",
        "|---|---|---|---|---|"]
inside = True
S_odd, S_even = (7 * zeta(3) - 2) / 8, (10 - 7 * zeta(3)) / 8
for k in show:
    rho = float(k)
    d = det[k]
    a = alpha(rho)
    cval = cc[k] if abs(rho) > 1e-9 else 0.0
    if rho >= 0:
        lo, hi = c1 * rho, c1 * rho + (1 - c1) * rho ** 2
    else:
        r_ = abs(rho)
        lo, hi = -c1 * r_ - (S_odd - c1) * r_ ** 3, -c1 * r_ + S_even * r_ ** 2
    inside &= lo - 1e-12 <= cval <= hi + 1e-12
    cm = float(cs(rho)) if abs(rho) > 1e-9 else 0.0
    rows.append(f"| {rlab(rho)} | {cval:.7f} | {lo:.4f} | {hi:.4f} | {cm:.4f} |")
T["tab_corr"] = "\n".join(rows).replace("| -0.0000000 |", "| 0 |")
# the bounds hold on the whole 53-point curve
allin = True
for key, val in cc.items():
    rho = float(key)
    if rho >= 0:
        lo, hi = c1 * rho, c1 * rho + (1 - c1) * rho ** 2
    else:
        r_ = abs(rho)
        lo, hi = -c1 * r_ - (S_odd - c1) * r_ ** 3, -c1 * r_ + S_even * r_ ** 2
    allin &= lo - 1e-9 <= val <= hi + 1e-9
V["bounds_hold_all"] = "all" if allin else "NOT all"
rows = ["| $\\rho$ | corr, Monte Carlo | corr, formula | $z$ | $n = 2000$ | $E[M_1 M_2]$, Monte Carlo | Rogers and Shepp |",
        "|---|---|---|---|---|---|---|"]
for m in J["mc"]:
    z = (m["corr_mc"] - m["corr_an"]) / m["corr_se"]
    rows.append(f"| {rlab(m['rho'])} | ${m['corr_mc']:.5f} \\pm {m['corr_se']:.5f}$ | {m['corr_an']:.5f} | "
                f"{z:+.2f} | {m['corr_half']:.5f} | {m['EM12_mc']:.5f} | {m['EM12_rs']:.5f} |")
T["tab_mc"] = "\n".join(rows).replace("| -0.00000 |", "| 0 |")

# ── 5. images (Section 7) ────────────────────────────────────────────────────────────────────────────────────────────
V["img_delta_err"] = up(max(num(r"images n=\d: delta_img vs eigen-series delta: .*abs err ([0-9.e+-]+)")))
V["img_k_err"] = up(max(num(r"images n=\d: k_img vs eigen-series k: .*abs err ([0-9.e+-]+)")))
V["img_f_err"] = up(max(num(r"images n=\d: f_img vs f_eigen .*abs err ([0-9.e+-]+)")))
V["img_corr_err"] = up(max(num(r"images n=\d: corr from image formula vs eigen-series: .*abs err ([0-9.e+-]+)")))
V["img_diag_zmax"] = f"{max(num(r'images n=\d: diagonal law \(image formula\) vs MC: max \|z\| = ([0-9.]+)')):.2f}"
for n in (3, 4, 6):
    V[f"img_corr_{n}"] = f"{J[f'images_{n}']['corr']:.7f}"
    V[f"ser_corr_{n}"] = f"{J['corr_curve'][f'{-math.cos(math.pi / n):+.4f}']:.7f}"
V["kap_closed_err"] = up(max(num(r"images n=\d: kappa closed form vs .*abs err ([0-9.e+-]+)")))
V["quadrant_err"] = up(max(num(r"quadrant .*abs err ([0-9.e+-]+)")))

# ── 6. bridges (Section 8) ───────────────────────────────────────────────────────────────────────────────────────────
up_p = num(r"bridge rho=.*uniform chi-square p = ([0-9.e+-]+);")
ind_p = num(r"bridge rho=.*independence \(10x10\) p = ([0-9.e+-]+)")
law_p = num(r"bridge rho=.*law of D vs kappa: chi-square 20 bins p = ([0-9.e+-]+)")
V["br_unif_pmin"], V["br_ind_pmin"], V["br_law_pmin"] = f"{min(up_p):.3f}", f"{min(ind_p):.4f}", f"{min(law_p):.3f}"
V["br_int_err"] = up(max(num(r"bridge rho=.*int kappa = 1: .*abs err ([0-9.e+-]+)")))
V["br_rho0_err"] = up(num(r"rho=0: bridge kappa = 1 .*abs err ([0-9.e+-]+)")[0])
bz = num(r"bridge rho=.*corr = 6 E\[D\^2\] - 2: .*\(z = ([+-]?[0-9.]+)\)")
V["br_corr_zmax"] = f"{max(abs(z) for z in bz):.2f}"
V["br_n_rho"] = len(J["bridge"])
V["br_extrap"] = f"{J['bridge_minus1']['extrap']:.6f}"
V["br_mms_cov"] = f"{-(math.pi ** 2 - 9) / 36:.6f}"
V["br_mms_corr"] = f"{-(math.pi ** 2 - 9) / 3:.6f}"
V["br_extrap_err"] = up(abs(J["bridge_minus1"]["extrap"] - J["bridge_minus1"]["mms"]))
rows = ["| $\\rho$ | $E[D^2]$ | $6E[D^2] - 2$ (Theorem 6) | Monte Carlo |", "|---|---|---|---|"]
for b in J["bridge"]:
    rows.append(f"| {rlab(b['rho'])} | {b['ED2']:.6f} | {b['corr_an']:.5f} | {b['corr_mc']:.5f} |")
T["tab_bridge"] = "\n".join(rows).replace("| -0.00000 |", "| 0 |")

# ── 7. several motions (Section 9) ───────────────────────────────────────────────────────────────────────────────────
corr = J["corr_curve"]
rows = ["| configuration | pairs | $\\rho_{ij}$ | $\\operatorname{corr}(\\theta_i, \\theta_j)$, Monte Carlo | two-motion value | $E[\\prod_i \\theta_i]$ |",
        "|---|---|---|---|---|---|"]
NAMES = {"A3": "$A_3$ chain", "A4": "$A_4$ chain", "EQ+0.5": "equicorrelated", "EQ-0.4": "equicorrelated"}
for nm, f in J["f7"].items():
    C, Rh = np.array(f["corr"]), np.array(f["rho"])
    groups = {}
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            groups.setdefault(round(float(Rh[i, j]), 4), []).append(C[i, j])
    first = True
    for r, cs_ in sorted(groups.items()):
        two = 0.0 if abs(r) < 1e-9 else corr[f"{r:+.4f}"]
        rng = f"{min(cs_):+.4f} to {max(cs_):+.4f}" if len(cs_) > 1 else f"{cs_[0]:+.4f}"
        rows.append(f"| {NAMES[nm] if first else ''} | {len(cs_)} | {rlab(r)} | {rng} | {two:+.4f} | "
                    + (f"{f['E_prod']:.5f} |" if first else "|"))
        first = False
T["tab_several"] = "\n".join(rows).replace("| +0.0000 |", "| 0 |")
f7z = num(r"F7 .*\(z = ([+-]?[0-9.]+)\)")
V["f7_zmax"] = f"{max(abs(z) for z in f7z):.2f}"
V["f7_npairs"] = len(f7z)
V["f7_N"] = f"{J['f7']['A3']['N']:,}"
for nm in ("A3", "A4", "EQ+0.5", "EQ-0.4"):
    o = J["f7"][nm]["orders"]
    N = J["f7"][nm]["N"]
    top = sorted(o.items(), key=lambda kv: -kv[1])
    V[f"f7_{nm}_top"] = ", ".join(f"{k}: {c / N:.4f}" for k, c in top[:6])
    V[f"f7_{nm}_min"] = f"{min(o.values()) / N:.4f}"
    V[f"f7_{nm}_max"] = f"{max(o.values()) / N:.4f}"
o3 = J["f7"]["A3"]["orders"]
V["f7_A3_monotone"] = f"{(o3['123'] + o3['321']) / J['f7']['A3']['N']:.4f}"
o4 = J["f7"]["A4"]["orders"]
V["f7_A4_3142"] = f"{o4['3142'] / J['f7']['A4']['N']:.4f}"
V["f7_A4_2413"] = f"{o4['2413'] / J['f7']['A4']['N']:.4f}"
V["f7_A4_1234"] = f"{o4['1234'] / J['f7']['A4']['N']:.4f}"
V["f7_A4_4321"] = f"{o4.get('4321', 0) / J['f7']['A4']['N']:.4f}"
V["one_24"] = f"{1 / 24:.4f}"
V["one_6"] = f"{1 / 6:.4f}"
V["f7_binom_se"] = f"{math.sqrt((1 / 6) * (5 / 6) / J['f7']['A3']['N']):.5f}"

# ── 8. random walks (Section 10) ─────────────────────────────────────────────────────────────────────────────────────
LAW = {"gauss": "Gaussian", "unif": "uniform mix", "laplace": "Laplace mix", "copy": "copy-or-fresh"}
rows = ["| step law | $\\rho$ | $P(\\vartheta_1 = \\vartheta_2 = 0)$, $n = 10$ | corr, $n = 10$ | corr, $n = 1000$ | order of $P(\\vartheta_1 = \\vartheta_2)$ | predicted |",
        "|---|---|---|---|---|---|---|"]
zb = []
for law in LAW:
    for rho in (-0.5, 0.5):
        rr = sorted([r for r in J["f8"] if r["law"] == law and r["rho"] == rho], key=lambda r: r["n"])
        a = alpha(rho)
        pred = -(math.pi / a - 1) if rho > 0 else -math.pi / (2 * a)
        order = math.log(rr[2]["p_eq"] / rr[1]["p_eq"]) / math.log(10)
        rows.append(f"| {LAW[law]} | ${rho:+.1f}$ | {rr[0]['p00']:.6f} | ${rr[0]['corr']:+.4f} \\pm {rr[0]['corr_se']:.4f}$ | "
                    f"${rr[2]['corr']:+.4f} \\pm {rr[2]['corr_se']:.4f}$ | {order:.3f} | {pred:.3f} |")
        zb.append(abs(rr[2]["corr"] - corr[f"{rho:+.4f}"]) / rr[2]["corr_se"])
T["tab_walks"] = "\n".join(rows)
V["walk_brown_zmax"] = f"{max(zb):.2f}"
sp = num(r"F8 .*P\(theta1 = 0\) = C\(2n,n\)/4\^n .*\(z = ([+-]?[0-9.]+)\)")
V["sparre_zmax"] = f"{max(abs(z) for z in sp):.2f}"
V["sparre_n"] = len(sp)
nu = [l for l in CH if "non-universality detected" in l]
zz = {}
for l in nu:
    rho = re.search(r"rho=([+-][0-9.]+)", l).group(1)
    zz[rho] = re.findall(r"n=(\d+): max pairwise \|z\| = ([0-9.]+)", l)
V["nonuniv_m05"] = "; ".join(f"$n = {n}$: ${z}$" for n, z in zz["-0.5"])
V["nonuniv_p05"] = "; ".join(f"$n = {n}$: ${z}$" for n, z in zz["+0.5"])
V["walk_N10"] = f"{[r for r in J['f8'] if r['n'] == 10][0]['N']:,}"
V["walk_N100"] = f"{[r for r in J['f8'] if r['n'] == 100][0]['N']:,}"
V["walk_N1000"] = f"{[r for r in J['f8'] if r['n'] == 1000][0]['N']:,}"
rows = ["| step law | $\\rho$ | $n = 10$ | $n = 100$ | $n = 1000$ | observed order, $100 \\to 1000$ | predicted |", "|---|---|---|---|---|---|---|"]
od = []
for law in LAW:
    for rho in (-0.5, 0.5):
        rr = sorted([r for r in J["f8"] if r["law"] == law and r["rho"] == rho], key=lambda r: r["n"])
        pe = [r["p_eq"] for r in rr]
        a = alpha(rho)
        pred = -(math.pi / a - 1) if rho > 0 else -math.pi / (2 * a)
        order = math.log(pe[2] / pe[1]) / math.log(10)
        od.append(abs(order - pred))
        rows.append(f"| {LAW[law]} | ${rho:+.1f}$ | {pe[0]:.5f} | {pe[1]:.5f} | {pe[2]:.6f} | {order:.3f} | {pred:.3f} |")

V["coinc_dev"] = f"{max(od):.3f}"

# ── 9. Algorithm 1 parameters (PARAMS.md, revision 1) ────────────────────────────────────────────────────────────────
for rho in (0.7, 0.9, 0.99):
    V[f"m_{rho:g}"] = max(32, math.ceil(25 / (1 - rho * rho)))
V["m_min"] = 32

out = {"values": V, "tables": T}
(HERE / "derived.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"derived.json: {len(V)} values, {len(T)} tables")
