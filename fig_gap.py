"""fig_gap: density of |theta1 - theta2| (formula) with Monte Carlo histograms."""
import json, math
from pathlib import Path
import numpy as np
import mm_plot as mp

HERE = Path(__file__).parent
R = np.load(HERE / "results.npz")
J = json.loads((HERE / "numbers.json").read_text())
NMC = J["mc"][0]["N"]
u = R["gap_u"]
rho_mc = list(R["rho_mc"])
show = [-0.9, -0.5, 0.0, 0.5, 0.9]
cols = ["#1e3a5f", "#2a6f97", "#2a9d8f", "#7fb069", "#c9a227"]
fig, ax = mp.figure()
for rho, c in zip(show, cols):
    g = R[f"gap_{rho:+.6f}"]
    ax.plot(u, g, color=c, lw=1.8, label=rf"$\rho = {rho:g}$")
    i = [k for k, r in enumerate(rho_mc) if abs(r - rho) < 1e-9][0]
    h = R[f"mc_gaphist_{i}"]
    mids = (np.arange(100) + 0.5) / 100
    ax.plot(mids[1::2], (h / h.sum() * 100)[1::2], "o", color=mp.RED, ms=3.2)
ax.plot([], [], "o", color=mp.RED, ms=4, label=rf"Monte Carlo ({NMC / 1e6:g} million pairs per $\rho$)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 3.2)
mp.label(ax, r"$u = |\theta_1 - \theta_2|$", "density")
mp.title(ax, "the gap between the two argmax times")
mp.legend(ax, loc="upper right")
mp.save(fig, "fig_gap.png")
