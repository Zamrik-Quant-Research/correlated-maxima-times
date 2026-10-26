"""fig_bridge: law of the cyclic gap D for correlated Brownian bridges, formula and Monte Carlo; closed forms at pi/n."""
import json, math
from pathlib import Path
import numpy as np
import mm_plot as mp

HERE = Path(__file__).parent
R = np.load(HERE / "results.npz")
u = R["kap_u"]
uf = np.concatenate([u, 1 - u[::-1][1:]])
rho_br = list(R["rho_br"])
show = [-0.9, -0.5, 0.0, 0.5, 0.9]
cols = ["#1e3a5f", "#2a6f97", "#2a9d8f", "#7fb069", "#c9a227"]
fig, ax = mp.figure()
for rho, c in zip(show, cols):
    k = R[f"kap_{rho:+.6f}"]
    ax.plot(uf, np.concatenate([k, k[::-1][1:]]), color=c, lw=1.8, label=rf"$\rho = {rho:g}$")
    i = [j for j, r in enumerate(rho_br) if abs(r - rho) < 1e-9][0]
    h = R[f"bb_Dhist_{i}"]
    mids = (np.arange(50) + 0.5) / 50
    ax.plot(mids, h / h.sum() * 50, "o", color=mp.RED, ms=3.2)
kc = R["img_kap_3"]
ax.plot(uf, np.concatenate([kc, kc[::-1][1:]]), color="white", lw=0.9, ls="--")
hb, = ax.plot([], [], color="#2a6f97", lw=1.8)
hw, = ax.plot([], [], color="white", lw=0.9, ls="--")
ax.plot([], [], "o", color=mp.RED, ms=4, label="Monte Carlo (bridges)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 3.0)
mp.label(ax, r"$D = (\theta_2 - \theta_1)$ mod 1", "density")
mp.title(ax, r"correlated bridges: $\theta_1$ is uniform and independent of the cyclic gap $D$")
from matplotlib.legend_handler import HandlerTuple
h, l = ax.get_legend_handles_labels()
h.insert(len(show), (hb, hw)); l.insert(len(show), r"closed form at $\rho=-1/2$ (white dashes)")
mp.legend(ax, handles=h, labels=l, loc="upper center", ncol=2, handler_map={tuple: HandlerTuple(ndivide=1)})
mp.save(fig, "fig_bridge.png")
