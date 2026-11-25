#!/usr/bin/env python3
"""SI: per-residue descriptors along the sequence and over time.

Reads raw/kras_biology/<arm>_<rep>_{residues.dat, stride/lddt/rsa/hbonds.dat.gz}
(written by kras_biology.py). Run from the data directory.
  figS_bio_profiles   mean +/- sigma along the sequence, one curve per arm
  figS_time_<desc>    residue x time maps for every replicate
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kras_biology as kb

RAW = os.path.join("raw", "kras_biology")
ARMS = ["baseline", "metaD", "HMR"]
LABEL = {"baseline": "reference", "metaD": "metadynamics", "HMR": "HMR"}
COLOR = {"baseline": "#111111", "metaD": "#E00000", "HMR": "#0177C1"}
DT = {"baseline": 0.1, "metaD": 0.1, "HMR": 1.0}       # ns per frame index
REGIONS = [("P-loop", 10, 17), ("Switch I", 30, 40), ("Switch II", 60, 76)]
HELIX, STRAND = set("HGI"), set("EB")

plt.rcParams.update({"font.size": 9, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.8})


def load():
    data = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*_residues.dat"))):
        stem = f[:-len("_residues.dat")]
        arm, rep = os.path.basename(stem).rsplit("_", 1)
        D = kb.load_raw(stem)
        ss = D["ss"]
        D["sscls"] = np.where(np.isin(ss, list(HELIX)), 0, np.where(np.isin(ss, list(STRAND)), 1, 2))
        D["coil"] = (D["sscls"] == 2).astype(float)
        D["t_us"] = D["frames"] * DT[arm] / 1000.0
        keep = (D["resids"] >= 1) & (D["resids"] <= 167)      # Gly0 (tag residue) excluded
        n = len(D["resids"])
        for k, v in list(D.items()):
            v = np.asarray(v) if not isinstance(v, np.ndarray) else v
            if v.ndim == 1 and len(v) == n and k != "frames" and k != "t_us":
                D[k] = v[keep]
            elif v.ndim == 2 and v.shape[1] == n:
                D[k] = v[:, keep]
        data.setdefault(arm, {})[rep] = D
    return data


def shade(ax, horizontal=False):
    for name, a, b in REGIONS:
        if horizontal:
            ax.axhspan(a - 0.5, b + 0.5, color="0.9", lw=0, zorder=0)
        else:
            ax.axvspan(a - 0.5, b + 0.5, color="0.9", lw=0, zorder=0)


def profiles(data):
    props = [("coil", "turn/coil fraction"), ("lddt", "lDDT-Cα"),
             ("hb_frames", "hydrogen bonds"), ("rsa", "RSA")]
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.0), sharex=True, constrained_layout=True)
    for ax, (key, lab) in zip(axes, props):
        shade(ax)
        for arm in ARMS:
            X = np.concatenate([d[key] for d in data[arm].values()], axis=0)
            res = next(iter(data[arm].values()))["resids"]
            m, s = X.mean(0), X.std(0)
            ax.plot(res, m, color=COLOR[arm], lw=1.0, label=LABEL[arm])
            ax.fill_between(res, m - s, m + s, color=COLOR[arm], alpha=0.12, lw=0)
        ax.set_ylabel(lab)
        if key in ("coil", "lddt"):
            ax.set_ylim(0, 1.05)
    for name, a, b in REGIONS:
        axes[0].text((a + b) / 2, 1.02, name, ha="center", va="bottom", fontsize=8,
                     transform=axes[0].get_xaxis_transform())
    axes[-1].set_xlabel("residue")
    axes[-1].set_xlim(res.min() - 0.5, res.max() + 0.5)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, fontsize=8.5, frameon=False,
               loc="outside upper center")
    for ext in ("pdf", "png"):
        fig.savefig(f"figS_bio_profiles.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figS_bio_profiles")


def timemap(data, key, title, out, cmap, vmin=None, vmax=None, cats=None):
    panels = [(arm, rep) for arm in ARMS for rep in sorted(data[arm])]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.0, 0.72 * len(panels) + 0.7),
                             constrained_layout=True)
    if cats:
        cm = ListedColormap(cats["colors"])
        norm = BoundaryNorm(np.arange(len(cats["colors"]) + 1) - 0.5, cm.N)
    for ax, (arm, rep) in zip(axes, panels):
        D = data[arm][rep]
        Z = D[key].T
        t = D["t_us"]
        res = D["resids"]
        ext = [t[0], t[-1], res.min() - 0.5, res.max() + 0.5]
        if cats:
            im = ax.imshow(Z, aspect="auto", origin="lower", extent=ext, cmap=cm, norm=norm,
                           interpolation="nearest")
        else:
            im = ax.imshow(Z, aspect="auto", origin="lower", extent=ext, cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
        for name, a, b in REGIONS:
            ax.axhline(a - 0.5, color="k", lw=0.6, ls=(0, (3, 2)))
            ax.axhline(b + 0.5, color="k", lw=0.6, ls=(0, (3, 2)))
        ax.set_yticks([0, 50, 100, 150])
        ax.tick_params(labelsize=7)
        ax.text(1.005, 0.5, f"{LABEL[arm]}\n{rep}", transform=ax.transAxes, fontsize=7.5,
                va="center", ha="left", color=COLOR[arm])
        ax.set_ylabel("residue", fontsize=7.5)
    axes[-1].set_xlabel("simulated time (µs)")
    if cats:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(cats["colors"], cats["labels"])],
                   loc="outside upper center", ncol=len(cats["labels"]), fontsize=8.5, frameon=False)
    else:
        cb = fig.colorbar(im, ax=axes, shrink=0.3, location="top", aspect=30)
        cb.set_label(title, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    data = load()
    profiles(data)
    timemap(data, "sscls", "secondary structure", "figS_time_ss", None,
            cats={"colors": ["#c8571f", "#1f6fb4", "#e8e8e8"],
                  "labels": ["helix (H, G, I)", "strand (E, B)", "turn/coil (T, C, b)"]})
    timemap(data, "lddt", "lDDT-Cα", "figS_time_lddt", "viridis", 0, 1)
    hbmax = max(np.percentile(d["hb_frames"], 99) for a in data.values() for d in a.values())
    timemap(data, "hb_frames", "hydrogen bonds per residue", "figS_time_hbonds", "magma_r",
            0, hbmax)
    timemap(data, "rsa", "RSA", "figS_time_rsa", "cividis", 0, 1)
