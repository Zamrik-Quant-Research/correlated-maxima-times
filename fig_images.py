"""fig_images: alpha = pi/n (rho = -1/2, -1/sqrt2, -sqrt3/2): image formulas against the eigen-series and Monte Carlo."""
import json, math
from pathlib import Path
import numpy as np
import mm_plot as mp

HERE = Path(__file__).parent
R = np.load(HERE / "results.npz")
t = R["diag_t"]
rho_mc = list(R["rho_mc"])
fig, (a, b) = mp.panels(2)
cols = {3: mp.NAVY, 4: mp.TEAL, 6: "#c9a227"}
nm = {3: r"$\pi/3$ ($\rho=-1/2$)", 4: r"$\pi/4$ ($\rho=-1/\sqrt{2}$)", 6: r"$\pi/6$ ($\rho=-\sqrt{3}/2$)"}
indep = (2 / math.pi * np.arcsin(np.sqrt(t))) ** 2
a.plot(t, indep, color=mp.GREY, lw=1.0, ls=":", label=r"independent ($\alpha=\pi/2$)")
for n in (3, 4, 6):
    rho = -math.cos(math.pi / n)
    a.plot(t, R[f"img_D_{n}"], color=cols[n], lw=2.0, label=r"images, $\alpha=$" + nm[n])
    a.plot(t, R[f"D_{rho:+.6f}"], color="white", lw=0.8, ls="--")
    i = [k for k, r in enumerate(rho_mc) if abs(r - rho) < 1e-9][0]
    a.plot(t, R[f"mc_diag_{i}"], "o", color=mp.RED, ms=4.5)
a.plot([], [], "o", color=mp.RED, label="Monte Carlo")
mp.label(a, r"$t$", r"$P(\theta_1 \leq t,\ \theta_2 \leq t)$")
mp.title(a, "diagonal law (white dashes: eigen-series)")
mp.legend(a, loc="upper left", fontsize=10)
u = R["img_gap_u"]
for n in (3, 4, 6):
    rho = -math.cos(math.pi / n)
    b.plot(u, R[f"img_gap_{n}"], color=cols[n], lw=2.0, label=r"$\alpha=$" + nm[n])
    i = [k for k, r in enumerate(rho_mc) if abs(r - rho) < 1e-9][0]
    h = R[f"mc_gaphist_{i}"]
    mids = (np.arange(100) + 0.5) / 100
    b.plot(mids[1::2], (h / h.sum() * 100)[1::2], "o", color=mp.RED, ms=3.2)
b.set_xlim(0, 1)
mp.label(b, r"$u = |\theta_1-\theta_2|$", "density")
mp.title(b, "gap density from the image formula")
mp.legend(b, loc="lower right", fontsize=10)
mp.save(fig, "fig_images.png")
