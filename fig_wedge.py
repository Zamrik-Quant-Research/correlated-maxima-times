"""fig_wedge: a pair of correlated paths with both maxima, and the same path in the plane inside the wedge."""
import math
from pathlib import Path
import numpy as np
import mm_plot as mp

HERE = Path(__file__).parent
R = np.load(HERE / "results.npz")
t, W1, W2 = R["ex_t"], R["ex_W1"], R["ex_W2"]
rho = 0.5
s = math.sqrt(1 - rho * rho)
i1, i2 = W1.argmax(), W2.argmax()
fig, (a, b) = mp.panels(2)
a.plot(t, W1, color=mp.NAVY, lw=1.1, label=r"$W_1$")
a.plot(t, W2, color=mp.TEAL, lw=1.1, label=r"$W_2$")
for i, W in ((i1, W1), (i2, W2)):
    a.axvline(t[i], color=mp.GREY, ls="--", lw=0.9)
    a.plot(t[i], W[i], "o", color=mp.RED, ms=7, zorder=5)
a.text(t[i1], a.get_ylim()[0], r" $\theta_1$", color=mp.AXIS, fontsize=12, va="bottom")
a.text(t[i2], a.get_ylim()[0], r" $\theta_2$", color=mp.AXIS, fontsize=12, va="bottom")
mp.label(a, "time", "value")
mp.title(a, r"two Brownian motions with $\rho = 0.5$ and their maxima")
mp.legend(a, loc="lower right")
X, Y = W1, (W2 - rho * W1) / s
M1, M2 = W1[i1], W2[i2]
ax_, ay_ = M1, (M2 - rho * M1) / s
b.plot(X, Y, color=mp.NAVY, lw=0.8)
L = 3.0
b.plot([ax_, ax_], [ay_ - L, ay_], color=mp.AXIS, lw=1.6)                       # edge 1: W1 = M1
d2 = np.array([-s, rho])                                                          # direction of edge 2 (W2 = M2) into the half-plane x < M1
b.plot([ax_, ax_ + L * d2[0]], [ay_, ay_ + L * d2[1]], color=mp.AXIS, lw=1.6)
b.plot(ax_, ay_, "s", color=mp.AXIS, ms=6)
b.plot(X[i1], Y[i1], "o", color=mp.RED, ms=7, zorder=5)
b.plot(X[i2], Y[i2], "o", color=mp.RED, ms=7, zorder=5)
b.plot(0, 0, "o", color=mp.TEAL, ms=6)
b.annotate(r"apex $(M_1, M_2)$", (ax_, ay_), xytext=(10, 6), textcoords="offset points", color=mp.AXIS, fontsize=11)
b.annotate(r"touches edge 1 at $\theta_1$", (X[i1], Y[i1]), xytext=(14, -40), textcoords="offset points", color=mp.AXIS, fontsize=11,
            arrowprops=dict(arrowstyle="-", color=mp.GREY, lw=0.8))
b.annotate(r"touches edge 2 at $\theta_2$", (X[i2], Y[i2]), xytext=(-60, 14), textcoords="offset points", color=mp.AXIS, fontsize=11)
b.set_aspect("equal")
b.set_xlim(min(X.min(), ax_ + L * d2[0]) - 0.15, ax_ + 1.2)
b.set_ylim(Y.min() - 0.15, ay_ + L * d2[1] + 0.1 if d2[1] > 0 else ay_ + 0.3)
mp.label(b, r"$X = W_1$", r"$Y = (W_2 - \rho W_1)/\sqrt{1-\rho^2}$")
mp.title(b, r"the planar path stays in a wedge of angle $\arccos(-\rho) = 2\pi/3$")
mp.save(fig, "fig_wedge.png")
