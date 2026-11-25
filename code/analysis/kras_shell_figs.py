"""
Active-site figures: Mg2+ first shell (a) and GTP environment (b), one figure
per arm, from the per-frame tables in raw/ (run from KRAS_unbound_basarim/):

    python kras_shell_figs.py

    fig_shell_hmr        main-text Fig. 4 (HMR; pooled reference dashed)
    figS_shell_reference SI Fig. S1
    figS_shell_pme       SI Fig. S2 (pooled reference dashed)
    figS_shell_metad     SI Fig. S3 (pooled reference dashed)
    tableS_gtp_shell.csv per-replicate means of the GTP descriptors

Mg2+ tables: raw/kras_baseline_all, kras_pme_mg, kras_hmr_switch1,
kras_metad_mg. GTP tables: raw/kras_gtp_shell/<arm>_rep*.dat written by
kras_gtp_shell.py (frames 1 ns apart; PME 0.1 ns).
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

W = 7.0
CUT = 2.6
C = ["#1f6fb4", "#c8571f", "#3f8f5c", "#7d5ba6"]
REF = "0.25"
FS_LAB, FS_TICK = 8, 7
plt.rcParams.update({
    "font.size": 9.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
LIG = ["SER17-OG", "THR35-OG1", "GTP-O2B", "GTP-O2G"]
LIG_LABEL = {"SER17-OG": "Ser17-OG", "THR35-OG1": "Thr35-OG1",
             "GTP-O2B": "GTP O2B", "GTP-O2G": "GTP O2G"}
AL = {"cis": ["cis", "cis_mean_deg"], "trans": ["trans", "trans_mean_deg"],
      "site": ["site", "mg_to_site_A"]}
for l in LIG:
    AL[l] = [f"dist:{l}", f"ligand_dist_A:{l}"]
MG_DIR = {"reference": "kras_baseline_all", "pme4": "kras_pme_mg",
          "hmr": "kras_hmr_switch1", "metad": "kras_metad_mg"}
# GTP panels: (column, x label, lo, hi, integer?)
GTP_PANELS = [
    ("d:K16", "Lys16 NZ–O$_{\\beta,\\gamma}$ (Å)", 2.3, 3.3, False),
    ("d:D119", "Asp119 O–GTP N1/N2 (Å)", 2.4, 3.4, False),
    ("d:G60", "Gly60 N–O$_\\gamma$ (Å)", 2.5, 12, False),
    ("d:T35", "Thr35 N–O$_\\gamma$ (Å)", 2.5, 25, False),
    ("n_prot", "protein N/O within 3.5 Å of GTP", 9, 24, True),
    ("n_wat_g", "waters within 3.5 Å of O$_\\gamma$", 1, 15, True),
    ("rmsd_pos", "GTP RMSD in the pocket (Å)", 0, 4.5, False),
    ("rmsd_int", "GTP internal RMSD (Å)", 0, 1.0, False),
]


def load_table(path):
    hdr = [l for l in open(path) if l.startswith("#")][-1][1:].split()
    return hdr, np.loadtxt(path, ndmin=2)


def load_mg(arm):
    reps = []
    for f in sorted(glob.glob(f"raw/{MG_DIR[arm]}/rep*.dat")):
        hdr, a = load_table(f)
        r = {k: a[:, next(hdr.index(n) for n in v if n in hdr)] for k, v in AL.items()}
        r["label"] = os.path.basename(f)[:-4]
        reps.append(r)
    return reps


def load_gtp(arm):
    reps = []
    for f in sorted(glob.glob(f"raw/kras_gtp_shell/{arm}_rep*.dat")):
        hdr, a = load_table(f)
        r = {k: a[1:, i] for i, k in enumerate(hdr)}   # frame 0 is the RMSD reference
        r["label"] = os.path.basename(f)[:-4].split("_")[1]
        reps.append(r)
    return reps


def pool(reps):
    return {k: np.concatenate([r[k] for r in reps]) for k in reps[0] if k != "label"}


def dens(ax, x, color, lo, hi, ref=False, integer=False):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return
    if integer:
        bins = np.arange(lo - 0.5, hi + 1.5)
        y, e = np.histogram(x, bins=bins, density=True)
        ax.stairs(y, e, color=color, lw=1.4 if ref else 1.0,
                  ls="--" if ref else "-", fill=False)
        if not ref:
            ax.stairs(y, e, color=color, alpha=0.12, fill=True)
        return
    g = np.linspace(lo, hi, 400)
    y = gaussian_kde(x)(g) if np.ptp(x) > 0 else np.zeros_like(g)
    ax.plot(g, y, color=color, lw=1.4 if ref else 1.0, ls="--" if ref else "-")
    if not ref:
        ax.fill_between(g, y, color=color, alpha=0.12, lw=0)


def style(ax, xl):
    ax.set_xlabel(xl, fontsize=FS_LAB, labelpad=1.5)
    ax.tick_params(labelsize=FS_TICK, pad=1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(bottom=0)


def figure(arm, name, ref_arm=None, arm_label=""):
    mg, gt = load_mg(arm), load_gtp(arm)
    rmg = pool(load_mg(ref_arm)) if ref_arm else None
    rgt = pool(load_gtp(ref_arm)) if ref_arm else None

    fig = plt.figure(figsize=(W, 5.6), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)
    A = outer[0].subgridspec(4, 2, height_ratios=[1, 1, 1, 0.001])
    B = outer[1].subgridspec(4, 2)

    # (a) Mg2+ shell
    axes = []
    for j, lig in enumerate(LIG):
        ax = fig.add_subplot(A[j // 2, j % 2]); axes.append(ax)
        for i, r in enumerate(mg):
            x = r[lig]; dens(ax, x[x < CUT], C[i], 1.6, 2.6)
        if rmg is not None:
            x = rmg[lig]; dens(ax, x[x < CUT], REF, 1.6, 2.6, ref=True)
        ax.set_xlim(1.6, 2.6); style(ax, f"Mg–O, {LIG_LABEL[lig]} (Å)")
    ax = fig.add_subplot(A[2, 0]); axes.append(ax)
    for i, r in enumerate(mg):
        dens(ax, r["site"], C[i], 1.65, 2.0)
    if rmg is not None:
        dens(ax, rmg["site"], REF, 1.65, 2.0, ref=True)
    ax.set_xlim(1.65, 2.0); style(ax, "Mg–nearest protein/GTP O (Å)")
    sub = A[2, 1].subgridspec(1, 2, width_ratios=[1, 1.3], wspace=0.08)
    axc, axt = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    for i, r in enumerate(mg):
        dens(axc, r["cis"], C[i], 88, 92); dens(axt, r["trans"], C[i], 164, 180)
    if rmg is not None:
        dens(axc, rmg["cis"], REF, 88, 92, ref=True); dens(axt, rmg["trans"], REF, 164, 180, ref=True)
    axc.set_xlim(88.5, 91.5); axt.set_xlim(164, 180)
    axc.set_xticks([89, 91]); axt.set_xticks([166, 172, 178])
    style(axc, "cis (°)"); style(axt, "trans (°)")
    axes[0].text(-0.08, 1.06, "a", transform=axes[0].transAxes, fontsize=10, weight="bold")
    axes[0].text(1.05, 1.06, "Mg$^{2+}$ first shell", transform=axes[0].transAxes,
                 fontsize=FS_LAB + 0.5, ha="center")

    # (b) GTP environment
    for j, (col, xl, lo, hi, integ) in enumerate(GTP_PANELS):
        ax = fig.add_subplot(B[j // 2, j % 2])
        for i, r in enumerate(gt):
            dens(ax, r[col], C[i], lo, hi, integer=integ)
        if rgt is not None:
            dens(ax, rgt[col], REF, lo, hi, ref=True, integer=integ)
        ax.set_xlim(lo - (0.5 if integ else 0), hi + (0.5 if integ else 0))
        style(ax, xl)
        if col == "d:T35":
            ax.set_xticks([5, 10, 15, 20, 25])
        if j == 0:
            ax.text(-0.08, 1.06, "b", transform=ax.transAxes, fontsize=10, weight="bold")
            ax.text(1.05, 1.06, "GTP environment", transform=ax.transAxes,
                    fontsize=FS_LAB + 0.5, ha="center")

    h = [Line2D([], [], color=C[i], lw=1.6, label=f"{arm_label} {r['label']}".strip())
         for i, r in enumerate(mg)]
    if ref_arm:
        h.append(Line2D([], [], color=REF, lw=1.6, ls="--", label="reference (pooled)"))
    fig.legend(handles=h, loc="outside lower center", ncol=len(h), fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    fig.savefig(f"{name}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", name)


COLS = ["n_prot", "n_ploop", "n_wat", "n_wat_g", "rmsd_pos", "rmsd_int", "chi", "p_ang",
        "d:K16", "d:G60", "d:T35", "d:Y32", "d:F28", "d:K117", "d:D119", "d:A146"]


def table():
    rows = []
    thr = {}
    for f in sorted(glob.glob("raw/kras_hmr_switch1/rep*.dat")):
        hdr, a = load_table(f)
        thr[os.path.basename(f)[:-4]] = a[:, hdr.index("ligand_dist_A:THR35-OG1")] < CUT
    for arm, lab in [("reference", "Ref."), ("pme4", "PME-4"), ("hmr", "HMR"), ("metad", "MetaD")]:
        for r in load_gtp(arm):
            rows.append([lab, r["label"][-1], "all"] + [np.mean(r[c]) for c in COLS])
            if arm == "hmr":
                c = thr[r["label"]][1:len(r["t"]) + 1]
                for s, m in (("Thr35 in", c), ("Thr35 out", ~c)):
                    rows.append([lab, r["label"][-1], s] + [np.mean(r[k][m]) for k in COLS])
    with open("tableS_gtp_shell.csv", "w") as fh:
        fh.write("arm,rep,frames," + ",".join(COLS) + "\n")
        for r in rows:
            fh.write(",".join(r[:3]) + "," + ",".join(f"{v:.3f}" for v in r[3:]) + "\n")
    print("wrote tableS_gtp_shell.csv")


if __name__ == "__main__":
    figure("hmr", "fig_shell_hmr", ref_arm="reference", arm_label="HMR")
    figure("reference", "figS_shell_reference", arm_label="reference")
    figure("pme4", "figS_shell_pme", ref_arm="reference", arm_label="PME-4")
    figure("metad", "figS_shell_metad", ref_arm="reference", arm_label="metaD")
    table()
