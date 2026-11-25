"""
Combined main-text figures (run from KRAS_unbound_basarim/):

    python kras_main_figs.py

    fig_throughput_combined   (a) baseline throughput per replicate,
                              (b) core-count sweep, (c) accelerated-protocol
                              throughput against time, (d) its densities
    fig_switch1_combined      (a) Switch I in the unbiased HMR replicates,
                              (b) Switch I under well-tempered metadynamics

The data readers are those of kras_baseline_all.py, kras_cpu_sweep.py,
kras_hmr_benchmark.py, kras_hmr_switch1.py and kras_metad_summary.py, so the
values are identical to the single figures they replace.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import kras_baseline_all as kb
import kras_cpu_sweep as kc
import kras_hmr_benchmark as kh
import kras_hmr_switch1 as ks
import kras_metad_summary as km
import rawio

W = 7.0                      # double-column width, inches
C = ["#1f6fb4", "#c8571f", "#3f8f5c", "#7d5ba6"]
FS_LAB, FS_TICK, FS_LEG = 8.5, 7.5, 7.5
plt.rcParams.update({
    "font.size": 9.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def letter(ax, s, dx=-0.13, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight="bold",
            ha="left", va="bottom")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    fig.savefig(f"{name}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------
def fig_throughput():
    base = {lab: kb.read_logs(d) for d, lab in kb.DIRS.items()}
    base = {l: g for l, g in base.items() if g is not None}
    hmr = {}
    for d, lab in kh.DIRS.items():
        r = kh.read_logs(d)
        # step5_production_HMR.log is a separate benchmark, not part of the
        # 10 us production chain
        keep = np.array(["_HMR" not in os.path.basename(r["files"][p]) for p in r["piece"]])
        hmr[lab] = {k: r[k][keep] for k in ("steps", "rate", "piece")}
    sweep = []
    for d, lab in kc.SWEEP_DIRS.items():
        runs = [r for r in (kc.parse(f) for f in sorted(glob.glob(os.path.join(d, kc.LOG_GLOB)))) if r]
        runs = [r for r in runs if r["resident"] and r["ncpu"] and "nogpu" not in r["name"]]
        if lab == "baseline":
            # 20-core baseline point: the GPU-resident production run of the
            # same configuration (as in the paper)
            prod = kc.parse(os.path.join(d, "step5_production.log"))
            prod["ncpu"] = 20
            runs.append(prod)
        runs = sorted(runs, key=lambda r: r["ncpu"])
        sweep.append(("baseline" if lab == "baseline" else "default pairlist", runs))

    fig = plt.figure(figsize=(W, 4.9), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.06)
    left = outer[0].subgridspec(2, 1, hspace=0.12)
    right = outer[1].subgridspec(4, 1, height_ratios=[1, 1, 1, 1.45], hspace=0.05)

    # (a) baseline throughput densities
    ax = fig.add_subplot(left[0])
    pooled = np.concatenate([g["rate"] for g in base.values()])
    lo, hi = 144, 152
    for i, (lab, g) in enumerate(base.items()):
        kb.density(ax, g["rate"], C[i], lo, hi, fill=True, label=lab)
    kb.density(ax, pooled, "#333333", lo, hi, lw=1.5, ls="--",
               label=f"pooled ({np.median(pooled):.0f})")
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)
    ax.set_xlabel("simulation speed, baseline (ns/day)", fontsize=FS_LAB)
    ax.set_ylabel("density", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG, loc="upper right", handlelength=1.4)
    letter(ax, "a")

    # (b) core-count sweep
    ax = fig.add_subplot(left[1])
    cores = sorted({r["ncpu"] for _, runs in sweep for r in runs})
    pos = {n: i for i, n in enumerate(cores)}
    for k, (lab, runs) in enumerate(sweep):
        x = np.array([pos[r["ncpu"]] for r in runs], float)
        y = np.array([r["rate"].mean() for r in runs])
        e = np.array([r["rate"].std() for r in runs])
        ls, mk = kc.STYLES[k]
        ax.errorbar(x, y, yerr=e, color="k", ls=ls, marker=mk, markersize=4,
                    lw=1.1, elinewidth=0.9, capsize=2.5, label=lab)
    ax.set_xticks(range(len(cores)), [str(n) for n in cores])
    ax.set_xlim(-0.4, len(cores) - 0.6); ax.set_ylim(140, 210)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("CPU cores per GPU", fontsize=FS_LAB)
    ax.set_ylabel("simulation speed (ns/day)", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG, loc="center right")
    letter(ax, "b")

    # (c) accelerated protocol against time, (d) densities
    labels = list(hmr)
    hp = np.concatenate([hmr[l]["rate"] for l in labels])
    lo, hi = max(0.0, np.percentile(hp, 0.5) - 20), np.percentile(hp, 99.5) + 20
    lo_t = 405.0        # time panels: show the 415 ns/day dip in replicate 3
    ax0 = None
    for i, lab in enumerate(labels):
        d = hmr[lab]
        starts, run = {}, 0
        for p in np.unique(d["piece"]):
            starts[p] = run
            run += d["steps"][d["piece"] == p].max()
        x = (np.array([starts[p] for p in d["piece"]]) + d["steps"]) * kh.TIMESTEP_FS / 1e9
        ax = fig.add_subplot(right[i], sharex=ax0); ax0 = ax0 or ax
        ax.plot(x, d["rate"], color=C[i], lw=0.45)
        med = np.median(d["rate"])
        ax.axhline(med, color="0.35", lw=0.7, ls="--")
        ax.text(0.985, 0.06, f"median {med:.0f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=FS_LEG, color="0.35")
        ax.text(0.015, 0.92, lab, transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_LEG + 0.5, color=C[i])
        ax.set_ylim(lo_t, hi); ax.set_ylabel("ns/day", fontsize=FS_LAB)
        ax.tick_params(labelsize=FS_TICK)
        if i < len(labels) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("simulated time (µs)", fontsize=FS_LAB)
        if i == 0:
            letter(ax, "c", dx=-0.13, dy=1.0)
    ax = fig.add_subplot(right[3])
    for i, lab in enumerate(labels):
        kh.density(ax, hmr[lab]["rate"], C[i], lo, hi, label=lab)
    kh.density(ax, hp, "#333333", lo, hi, lw=1.5, ls="--",
               label=f"pooled ({np.median(hp):.0f})")
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)
    ax.set_xlabel("simulation speed, HMR 4 fs (ns/day)", fontsize=FS_LAB)
    ax.set_ylabel("density", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEG, loc="upper left", handlelength=1.4)
    letter(ax, "d")
    save(fig, "fig_throughput_combined")


# ----------------------------------------------------------------------------
def fig_switch1():
    reps = [ks.record(rawio.get("kras_hmr_switch1", lab, lambda d=d: ks.analyse(d)), lab)
            for d, lab in ks.DIRS.items()]
    met = {r: km.load_rep(d) for r, d in km.REPS.items() if os.path.isdir(d)}
    for D in met.values():
        D["trap"] = km.trap_start(D["t"], D["cv"])
        D["use"] = D["t"] < D["trap"] if D["trap"] is not None else np.ones(len(D["t"]), bool)

    fig = plt.figure(figsize=(W, 5.0), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, wspace=0.08)
    L = outer[0].subgridspec(4, 1, height_ratios=[1, 1, 1, 1.25], hspace=0.06)
    R = outer[1].subgridspec(4, 1, height_ratios=[1, 1, 1, 1.25], hspace=0.06)
    CL, OP = ks.CLOSED_MAX, ks.OPEN_MIN

    # (a) unbiased HMR
    tmax = max(r["t"][-1] for r in reps) / 1000.0
    ax0 = None
    for i, r in enumerate(reps):
        c = C[i]; t = r["t"] / 1000.0
        w = max(3, (len(r["t"]) // 300) | 1)
        t35 = r["dist"]["THR35-OG1"]
        ax = fig.add_subplot(L[i], sharex=ax0); ax0 = ax0 or ax
        ax.axhspan(0, CL, color="#eef1f4", lw=0); ax.axhspan(OP, 180, color="#e2e6ea", lw=0)
        ax.fill_between(t, 0, 180, where=(t35 >= ks.MG_SHELL_CUT), color="#d8b26a",
                        alpha=0.26, lw=0, step="mid")
        ax.plot(t, r["cv"], color=c, lw=0.2, alpha=0.28)
        ax.plot(t, ks.roll(r["cv"], w), color=c, lw=1.0)
        late = True
        if r["exit"] is not None:
            xe = t[r["exit"]]; late = xe > 0.5 * tmax
            ax.axvline(xe, color="k", lw=0.8, ls="--")
            ax.text(xe, 172, f"{xe * 1000:.0f} ns " if late else f" {xe * 1000:.0f} ns",
                    fontsize=FS_LEG, va="top", ha="right" if late else "left")
        ax.set_ylim(0, 180); ax.set_xlim(0, tmax); ax.set_yticks([0, 90, 180])
        ax.set_ylabel("CV (°)", fontsize=FS_LAB); ax.tick_params(labelsize=FS_TICK)
        ax.text(0.015 if late else 0.985, 0.92, r["label"], transform=ax.transAxes,
                ha="left" if late else "right", va="top", fontsize=FS_LEG + 0.5, color=c)
        if i < len(reps) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("simulated time (µs)", fontsize=FS_LAB)
        axr = ax.twinx()
        axr.plot(t, ks.roll(t35, w), color="0.30", lw=0.7)
        axr.axhline(ks.MG_SHELL_CUT, color="0.25", lw=0.5, ls=":")
        axr.set_ylim(0, 30); axr.set_yticks([0, 15, 30])
        axr.set_ylabel("Thr35–Mg (Å)", color="0.30", fontsize=FS_LAB)
        axr.tick_params(axis="y", colors="0.30", labelsize=FS_TICK, direction="in")
        axr.spines["right"].set_visible(True); axr.spines["right"].set_color("0.6")
        if i == 0:
            letter(ax, "a", dx=-0.17, dy=1.0)
            ax.set_title("unbiased, HMR 4 fs", fontsize=FS_LAB, loc="center")
    ax = fig.add_subplot(L[3])
    ax.axvspan(0, CL, color="#eef1f4", lw=0, zorder=0); ax.axvspan(OP, 180, color="#e2e6ea", lw=0, zorder=0)
    for i, r in enumerate(reps):
        ks.density(ax, r["cv"], C[i], lo=0, hi=180, label=r["label"])
    ax.set_xlim(0, 180); ax.set_ylim(bottom=0); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Switch I CV (°)", fontsize=FS_LAB); ax.set_ylabel("density", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)

    # (b) metadynamics
    ax0 = None
    names = list(met)
    for i, rr in enumerate(names):
        D = met[rr]
        ax = fig.add_subplot(R[i], sharex=ax0); ax0 = ax0 or ax
        km.shade(ax, vertical=False)
        t, cv, u = D["t"] / 1000, D["cv"], D["use"]
        ax.plot(t[u], cv[u], color=C[i], lw=0.4)
        if D["trap"] is not None:
            ax.plot(t[~u], cv[~u], color="0.6", lw=0.4)
        ax.set_ylim(0, 180); ax.set_yticks([0, 90, 180]); ax.set_xlim(0, max(d["t"][-1] for d in met.values()) / 1000)
        ax.set_ylabel("CV (°)", fontsize=FS_LAB); ax.tick_params(labelsize=FS_TICK)
        ax.text(0.985, 0.92, rr, transform=ax.transAxes, ha="right", va="top",
                fontsize=FS_LEG + 0.5, color=C[i])
        if i < len(names) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("simulated time (µs)", fontsize=FS_LAB)
        if i == 0:
            letter(ax, "b", dx=-0.15, dy=1.0)
            ax.set_title("well-tempered metadynamics, 2 fs", fontsize=FS_LAB, loc="center")
    ax = fig.add_subplot(R[3])
    km.shade(ax)
    bins = np.arange(0, 182.5, 2.5)
    for i, rr in enumerate(names):
        D = met[rr]
        x, y = km.smooth_density(D["cv"][D["use"]], bins)
        ax.fill_between(x, y, color=C[i], alpha=0.2, lw=0)
        ax.plot(x, y, color=C[i], lw=1.0, label=rr)
    ax.set_xlim(0, 180); ax.set_ylim(bottom=0); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Switch I CV (°)", fontsize=FS_LAB); ax.set_ylabel("density", fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK)
    save(fig, "fig_switch1_combined")


if __name__ == "__main__":
    fig_throughput()
    fig_switch1()
