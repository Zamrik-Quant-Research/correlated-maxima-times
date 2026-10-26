"""fig_heat: the joint density f_rho(s, t) of (theta1, theta2) for six correlations (log colour scale)."""
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mm_plot as mp

HERE = Path(__file__).parent
R = np.load(HERE / "results.npz")
rhos = [-1 / math.sqrt(2), -0.5, 0.0, 0.3, 0.7, 0.9]
names = [r"-1/\sqrt{2}", "-1/2", "0", "0.3", "0.7", "0.9"]
fig, axes = plt.subplots(2, 3, figsize=(mp.W_IN, mp.H_IN), facecolor="white", gridspec_kw={"wspace": 0.25, "hspace": 0.38})
for j, (ax, rho, nm) in enumerate(zip(axes.ravel(), rhos, names)):
    H = R[f"heat_{rho:+.6f}"]
    im = ax.imshow(np.log10(np.clip(H, 1e-3, None)).T, origin="lower", extent=[0, 1, 0, 1], cmap="jet", vmin=-1.5, vmax=1.5)
    ax.set_title(rf"$\rho = {nm}$", color=mp.BLACK, fontsize=12.5)
    ax.tick_params(colors=mp.AXIS, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color(mp.NAVY); sp.set_linewidth(2.0)
    ax.set_xlabel(r"$s$", color=mp.AXIS, fontsize=11)
    if j % 3 == 0:
        ax.set_ylabel(r"$t$", color=mp.AXIS, fontsize=11)
cb = fig.colorbar(im, cax=fig.add_axes([0.835, 0.12, 0.015, 0.78]))
cb.set_label(r"$\log_{10} f_\rho(s,t)$", color=mp.AXIS, fontsize=11.5)
cb.ax.tick_params(colors=mp.AXIS)
fig.subplots_adjust(left=0.12, right=0.80, top=0.93, bottom=0.09)
fig.savefig("fig_heat.png", dpi=mp.DPI, facecolor="white")
