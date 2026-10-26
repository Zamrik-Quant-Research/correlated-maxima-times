"""fig_corr: corr(theta1, theta2) and corr(M1, M2) against rho, with Monte Carlo points."""
import json, math
from pathlib import Path
import numpy as np
import mm_plot as mp

HERE = Path(__file__).parent
R = np.load(HERE / "results.npz")
J = json.loads((HERE / "numbers.json").read_text())
r, c = R["rho_curve"], R["corr_curve"]
fig, ax = mp.figure()
rr = np.linspace(-1, 1, 3)
ax.plot(rr, rr, color=mp.GREY, lw=0.9, ls=":", label=r"$\pm\rho$ (Gebelein bound)")
ax.plot(rr, -rr, color=mp.GREY, lw=0.9, ls=":")
ax.plot(R["rs_curve_rho"], R["rs_curve_corrM"], color=mp.TEAL, lw=2.0, label=r"corr$(M_1, M_2)$, Rogers and Shepp")
ax.plot(np.concatenate([[-1], r, [1]]), np.concatenate([[J["chaos"]["corr_maxmin"]], c, [1]]), color=mp.NAVY, lw=2.2,
        label=r"corr$(\theta_1, \theta_2)$, this paper")
x = np.linspace(-0.45, 0.45, 2)
ax.plot(x, J["chaos"]["c1_exact"] * x, color=mp.AXIS, lw=1.0, ls="--", label=r"slope $c_1 = 4 - 32/(3\pi)$ at $\rho = 0$")
mc = J["mc"]
ax.plot([m["rho"] for m in mc], [m["corr_mc"] for m in mc], "o", color=mp.RED, ms=6, label="Monte Carlo")
ax.plot([m["rho"] for m in mc], [m["corrM_mc"] for m in mc], "o", color=mp.RED, ms=6)
ax.plot([-1], [J["chaos"]["corr_maxmin"]], "*", color=mp.RED, ms=12)
ax.annotate(r"$\rho=-1$: $(6-7\zeta(3))/4$ (Mori, Majumdar, Schehr)", (-1, J["chaos"]["corr_maxmin"]), xytext=(40, -70),
            textcoords="offset points", color=mp.AXIS, fontsize=11, arrowprops=dict(arrowstyle="-", color=mp.GREY, lw=0.8))
ax.set_xlim(-1.03, 1.03)
ax.set_ylim(-1.03, 1.03)
mp.label(ax, r"correlation $\rho$ of the motions", "correlation")
mp.title(ax, "times of the maxima against values of the maxima")
mp.legend(ax, loc="upper left")
mp.save(fig, "fig_corr.png")
