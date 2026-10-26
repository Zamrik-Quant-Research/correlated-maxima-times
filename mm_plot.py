"""mm_plot — copied from times-of-maximum-and-minimum.research/mm_plot.py (house defaults, 12.4 x 6.4 in at 170 dpi, NAVY/TEAL on white).
Change for this paper: titles and legend text are BLACK (paper pipeline rule)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W_IN, H_IN, DPI = 12.4, 6.4, 170
NAVY, TEAL, GREY = "#1e3a5f", "#2a9d8f", "#8a94a6"
AXIS, RED = "#1a1a1a", "#ff0000"          # axes near-black, meeting points red (Tim, 2026-10-11)
FS_LABEL, FS_LEGEND, FS_TICK = 12.5, 11.5, 11


def _dress(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelsize=FS_TICK)
    ax.grid(True, color=GREY, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def figure():
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), facecolor="white")
    return fig, _dress(ax)


def panels(n=2):
    fig, axes = plt.subplots(1, n, figsize=(W_IN, H_IN), facecolor="white")
    for ax in axes: _dress(ax)
    return fig, axes


def label(ax, x=None, y=None):
    if x: ax.set_xlabel(x, color=AXIS, fontsize=FS_LABEL)
    if y: ax.set_ylabel(y, color=AXIS, fontsize=FS_LABEL)


BLACK = "#000000"


def legend(ax, **kw):
    kw.setdefault("fontsize", FS_LEGEND)
    ax.legend(frameon=False, labelcolor=BLACK, **kw)


def title(ax, text, **kw):
    ax.set_title(text, color=BLACK, fontsize=FS_LABEL + 0.5, **kw)


def save(fig, name):
    fig.tight_layout(); fig.savefig(name, dpi=DPI, facecolor="white")
