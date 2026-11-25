"""SI figures: C-alpha RMSD against time and per-residue RMSF for all arms.

    python kras_ca_figs.py      (run from KRAS_unbound_basarim/)

Reads raw/kras_ca/<arm>_<rep>_{rmsd,rmsf}.dat (kras_ca.py) and writes
figS_ca_rmsd.{pdf,png} and figS_ca_rmsf.{pdf,png}.
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C = ["#1f6fb4", "#c8571f", "#3f8f5c", "#7d5ba6"]
ARMS = [("reference", "reference (2 fs)"), ("pme4", "PME order 4"),
        ("hmr", "HMR, 4 fs"), ("metad", "metadynamics (biased)")]
ARMC = {"reference": "#111111", "pme4": "#7d5ba6", "hmr": "#0177C1", "metad": "#E00000"}
REG = [(10, 17, "P-loop"), (30, 40, "Switch I"), (60, 76, "Switch II")]
plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def roll(x, w):
    if w < 2:
        return x
    k = np.ones(w) / w
    xp = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(xp, k, mode="valid")


def load(arm, kind):
    return [np.loadtxt(f) for f in sorted(glob.glob(f"raw/kras_ca/{arm}_rep*_{kind}.dat"))]


def fig_rmsd():
    cols = [(1, "all C$_\\alpha$ (fit on all)", 8), (2, "core C$_\\alpha$ (fit on core)", 8),
            (3, "Switch I C$_\\alpha$ (core fit)", 30)]
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 7.0), constrained_layout=True)
    for i, (arm, lab) in enumerate(ARMS):
        R = load(arm, "rmsd")
        unit, scale = ("µs", 1000.0)
        for j, (c, title, ymax) in enumerate(cols):
            ax = axes[i, j]
            for k, a in enumerate(R):
                t = a[:, 0] / scale
                w = max(1, len(a) // 150)
                ax.plot(t, a[:, c], color=C[k], lw=0.25, alpha=0.35)
                ax.plot(t, roll(a[:, c], w), color=C[k], lw=0.9, label=f"rep{k + 1}")
            ax.set_ylim(0, ymax)
            ax.set_xlim(0, max(a[-1, 0] for a in R) / scale)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{lab}\nRMSD (Å)", fontsize=8)
            if i == 3 or True:
                ax.set_xlabel(f"time ({unit})", fontsize=7.5, labelpad=1)
    h = [Line2D([], [], color=C[k], lw=1.2, label=f"rep{k + 1}") for k in range(4)]
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"figS_ca_rmsd.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("wrote figS_ca_rmsd")


def shade(ax, top):
    for lo, hi, lab in REG:
        ax.axvspan(lo - 0.5, hi + 0.5, color="0.92", lw=0, zorder=0)
        ax.text((lo + hi) / 2, top, lab, fontsize=6.5, ha="center", va="top", color="0.35")


def fig_rmsf():
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), constrained_layout=True, sharex=True)
    ref = load("reference", "rmsf")
    r = ref[0][:, 0]

    def band(ax, F, col, idx, lab, ls="-"):
        M = np.array([f[:, idx] for f in F])
        m, s = M.mean(0), M.std(0)
        ax.fill_between(r, m - s, m + s, color=col, alpha=0.15, lw=0)
        ax.plot(r, m, color=col, lw=1.0, ls=ls, label=lab)

    ax = axes[0]
    band(ax, ref, ARMC["reference"], 1, "reference, 4 × 1.64 µs")
    band(ax, load("pme4", "rmsf"), ARMC["pme4"], 1, "PME order 4, 3 × 100 ns")
    band(ax, load("hmr", "rmsf"), ARMC["hmr"], 2, "HMR, first 1.64 µs of each replicate")
    ax.set_ylim(0, 6); shade(ax, 5.9)
    ax.set_ylabel("C$_\\alpha$ RMSF (Å)")
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    ax.text(-0.06, 1.0, "a", transform=ax.transAxes, fontsize=10, weight="bold", va="top")

    ax = axes[1]
    band(ax, ref, ARMC["reference"], 1, "reference, 4 × 1.64 µs")
    band(ax, load("hmr", "rmsf"), ARMC["hmr"], 1, "HMR, 3 × 10 µs")
    band(ax, load("metad", "rmsf"), ARMC["metad"], 1, "metadynamics, 3 × 1.6 µs (biased)")
    ax.set_ylim(0, 12); shade(ax, 11.8)
    ax.set_ylabel("C$_\\alpha$ RMSF (Å)"); ax.set_xlabel("residue")
    ax.set_xlim(1, 166)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    ax.text(-0.06, 1.0, "b", transform=ax.transAxes, fontsize=10, weight="bold", va="top")
    for a in axes:
        a.tick_params(labelsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(f"figS_ca_rmsf.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("wrote figS_ca_rmsf")


if __name__ == "__main__":
    fig_rmsd()
    fig_rmsf()
