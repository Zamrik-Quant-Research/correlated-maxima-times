"""fig_walks: discrete walks (four step laws) converge to the Brownian correlation of the argmax times."""
import json, math
from pathlib import Path
import numpy as np
import mm_plot as mp

HERE = Path(__file__).parent
J = json.loads((HERE / "numbers.json").read_text())
corr = J["corr_curve"]
fig, (a, b) = mp.panels(2)
cols = {"gauss": mp.NAVY, "unif": mp.TEAL, "laplace": "#c9a227", "copy": mp.GREY}
lab = {"gauss": "Gaussian", "unif": "uniform mix", "laplace": "Laplace mix", "copy": "copy-or-fresh"}
for ax, rho in ((a, -0.5), (b, 0.5)):
    for law in cols:
        rows = sorted([r for r in J["f8"] if r["law"] == law and r["rho"] == rho], key=lambda r: r["n"])
        ax.errorbar([r["n"] for r in rows], [r["corr"] for r in rows], yerr=[2 * r["corr_se"] for r in rows], color=cols[law],
                    marker="o", ms=5, lw=1.4, capsize=3, label=lab[law])
    ax.axhline(corr[f"{rho:+.4f}"], color=mp.RED, lw=1.4, ls="--", label="Brownian limit")
    ax.set_xscale("log")
    mp.label(ax, "number of steps $n$", r"corr$(\theta_1, \theta_2)$")
    mp.title(ax, rf"$\rho = {rho:g}$")
    mp.legend(ax, loc="best", fontsize=10)
mp.save(fig, "fig_walks.png")
