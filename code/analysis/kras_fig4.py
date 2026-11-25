"""
Main-text Fig. 4: the active site under hydrogen mass repartitioning.

    python kras_fig4.py          (run from KRAS_unbound_basarim/)

(a) schematic of the measured quantities; (b) Mg2+ first shell;
(c) GTP anchors; (d) switch contacts; (e) packing and hydration;
(f) GTP placement. HMR frames are pooled over the three replicates and split
by whether Thr35-OG1 coordinates the ion; the pooled reference arm is dashed.
Writes fig4_active_site.{pdf,png,tiff}.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon

import kras_shell_figs as ksf

W, H = 7.0, 7.6
IN, OUT, REF = "#0868AC", "#D55E00", "#222222"
MGC, ANC, SWC, WAC = "#2E8B57", "#6B6B6B", "#D55E00", "#3A8FD0"
FS_LAB, FS_TICK = 7.5, 6.5
CUT = 2.6


def split_hmr():
    mg, gt = ksf.load_mg("hmr"), ksf.load_gtp("hmr")
    out = {"in": {}, "out": {}}
    for r_mg, r_gt in zip(mg, gt):
        bound = r_mg["THR35-OG1"] < CUT
        for st, m in (("in", bound), ("out", ~bound)):
            for k, v in r_mg.items():
                if k != "label":
                    out[st].setdefault("mg:" + k, []).append(v[m])
            mg_ = m[1:len(r_gt["t"]) + 1]           # GTP tables start at frame 1
            for k, v in r_gt.items():
                if k != "label":
                    out[st].setdefault(k, []).append(v[mg_])
    for st in out:
        out[st] = {k: np.concatenate(v) for k, v in out[st].items()}
    frac = len(out["in"]["t"]) / (len(out["in"]["t"]) + len(out["out"]["t"]))
    return out, frac


def reference():
    mg, gt = ksf.pool(ksf.load_mg("reference")), ksf.pool(ksf.load_gtp("reference"))
    r = {"mg:" + k: v for k, v in mg.items()}
    r.update(gt)
    return r


def dens(ax, x, color, lo, hi, kind, integer=False):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 20:
        return
    if integer:
        e = np.arange(lo - 0.5, hi + 1.5)
        y, e = np.histogram(x, bins=e, density=True)
        if kind == "ref":
            ax.stairs(y, e, color=REF, lw=1.1, ls=(0, (3, 1.6)), zorder=4)
        else:
            ax.stairs(y, e, color=color, fill=True, alpha=0.22, zorder=2)
            ax.stairs(y, e, color=color, lw=1.0, zorder=3)
        return
    from scipy.stats import gaussian_kde
    g = np.linspace(lo, hi, 500)
    y = gaussian_kde(x)(g)
    if kind == "ref":
        ax.plot(g, y, color=REF, lw=1.1, ls=(0, (3, 1.6)), zorder=4)
    else:
        ax.fill_between(g, y, color=color, alpha=0.22, lw=0, zorder=2)
        ax.plot(g, y, color=color, lw=1.0, zorder=3)


def panel(ax, key, xl, lo, hi, S, R, integer=False, states=("in", "out"), ticks=None,
          title=None, tcol="k"):
    for st in states:
        dens(ax, S[st][key], IN if st == "in" else OUT, lo, hi, "hmr", integer)
    dens(ax, R[key], REF, lo, hi, "ref", integer)
    pad = 0.5 if integer else 0
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(bottom=0)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel(xl, fontsize=FS_LAB, labelpad=1.5)
    ax.tick_params(labelsize=FS_TICK, pad=1.5, length=2)
    if ticks is not None:
        ax.set_xticks(ticks)
    if title:
        ax.set_title(title, fontsize=FS_LAB, color=tcol, pad=2, loc="left")


def group(fig, axes, letter, title, sub, color):
    """Header above a group of panels: bold letter, title, one-line message."""
    x0 = min(a.get_position().x0 for a in axes)
    x1 = max(a.get_position().x1 for a in axes)
    y1 = max(a.get_position().y1 for a in axes)
    fig.text(x0 - 0.012, y1 + 0.052, letter, fontsize=10, weight="bold", va="bottom")
    fig.text(x0 + 0.012, y1 + 0.052, title, fontsize=8.2, weight="bold", va="bottom", color=color)
    fig.text(x0 + 0.012, y1 + 0.033, sub, fontsize=6.8, va="bottom", color="0.35", style="italic")
    fig.add_artist(Line2D([x0, x1], [y1 + 0.028, y1 + 0.028], color=color, lw=0.8,
                          transform=fig.transFigure))


def schematic(ax):
    ax.set_xlim(-0.3, 9.0); ax.set_ylim(-3.6, 9.4); ax.set_aspect("equal"); ax.axis("off")
    mg = np.array([3.6, 6.6])
    lig = {"Ser17": (2.3, 8.0), "Thr35": (4.9, 8.0), "W1": (1.9, 6.6), "W2": (5.3, 6.6),
           "O2B": (2.9, 5.1), "O2G": (4.3, 5.1)}
    for p in lig.values():
        ax.plot([mg[0], p[0]], [mg[1], p[1]], color=MGC, lw=1.5, zorder=1)
    Pg, Pb, Pa = np.array([4.9, 4.1]), np.array([2.5, 4.1]), np.array([0.9, 3.3])
    ax.plot([Pg[0], Pb[0], Pa[0]], [Pg[1], Pb[1], Pa[1]], color="0.45", lw=1.6, zorder=0)
    for o, p in (("O2B", Pb), ("O2G", Pg)):
        ax.plot([lig[o][0], p[0]], [lig[o][1], p[1]], color="0.45", lw=1.2, zorder=0)
    ax.plot([Pa[0], 0.5], [Pa[1], 2.2], color="0.45", lw=1.2, zorder=0)
    ax.add_patch(Polygon([[0.5, 2.2], [1.1, 1.6], [0.9, 0.9], [0.2, 0.9], [0.0, 1.6]],
                         closed=True, fc="#EDE3C8", ec="0.45", lw=0.8, zorder=1))
    ax.add_patch(Polygon([[1.1, 1.6], [1.9, 1.2], [2.7, 1.5], [2.8, 0.6], [2.0, 0.2], [1.3, 0.6]],
                         closed=True, fc="#E6E6F0", ec="0.45", lw=0.8, zorder=1))
    ax.text(1.95, 0.9, "guanine", fontsize=5.2, ha="center", va="center", color="0.3")
    ax.text(0.55, 1.55, "ribose", fontsize=4.6, ha="center", va="center", color="0.3", rotation=0)
    for p, t in ((Pg, "P$_\\gamma$"), (Pb, "P$_\\beta$"), (Pa, "P$_\\alpha$")):
        ax.add_patch(Circle(p, 0.38, fc="#F2A541", ec="0.35", lw=0.6, zorder=3))
        ax.text(p[0], p[1] - 0.03, t, fontsize=6, ha="center", va="center", zorder=4)
    for k, p in lig.items():
        ax.add_patch(Circle(p, 0.3, fc="white" if k.startswith("W") else "#F7D7D7",
                            ec=WAC if k.startswith("W") else "0.35", lw=0.7, zorder=3))
    ax.add_patch(Circle(mg, 0.45, fc=MGC, ec="none", zorder=4))
    ax.text(*mg, "Mg", color="white", fontsize=6.5, weight="bold", ha="center", va="center", zorder=5)
    lab = dict(fontsize=6, ha="center", va="center")
    ax.text(2.3, 8.6, "Ser17 OG", color=MGC, **lab)
    ax.text(4.9, 8.6, "Thr35 OG1", color=MGC, **lab)
    ax.text(1.9, 7.2, "W", color=WAC, **lab); ax.text(5.3, 7.2, "W", color=WAC, **lab)
    ax.text(2.0, 5.2, "O2B", fontsize=5.5, ha="center", va="center", color="0.3")
    ax.text(5.2, 5.2, "O2G", fontsize=5.5, ha="center", va="center", color="0.3")

    def contact(a, b, col, label, lpos, ha="center"):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=1.1, ls=(0, (2, 1.4)), zorder=2)
        ax.add_patch(Circle(a, 0.18, fc=col, ec="none", zorder=3))
        ax.text(*lpos, label, fontsize=6, color=col, ha=ha, va="center")
    contact((7.4, 5.2), Pg, SWC, "Thr35 N", (7.4, 5.65))
    contact((7.4, 3.1), Pg, SWC, "Gly60 N", (7.4, 2.65))
    contact((3.1, 2.6), Pb, ANC, "Lys16 NZ", (3.1, 2.15))
    contact((4.4, 0.7), (2.8, 0.85), ANC, "Asp119", (5.25, 0.7))
    for q in ((5.2, 3.05), (5.85, 3.5), (4.55, 3.0)):
        ax.add_patch(Circle(q, 0.2, fc="white", ec=WAC, lw=0.8, zorder=2))

    # key
    y = -0.6
    items = [("line", MGC, "-", "Mg²⁺ ligands", "b"),
             ("line", ANC, (0, (2, 1.4)), "GTP anchors", "c"),
             ("line", SWC, (0, (2, 1.4)), "switch contacts", "d"),
             ("circ", WAC, None, "waters at P$_\\gamma$; protein N/O near GTP", "e"),
             ("none", "0.2", None, "GTP position and shape", "f")]
    for kind, col, ls, txt, let in items:
        if kind == "line":
            ax.plot([0.0, 0.8], [y, y], color=col, lw=1.4, ls=ls)
        elif kind == "circ":
            ax.add_patch(Circle((0.4, y), 0.2, fc="white", ec=col, lw=0.8))
        else:
            ax.add_patch(Circle((0.4, y), 0.2, fc="#F2A541", ec="0.35", lw=0.6))
        ax.text(1.1, y, f"({let}) {txt}", fontsize=6, va="center", color="0.15")
        y -= 0.62


def main():
    S, frac = split_hmr()
    R = reference()
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1], hspace=0.42,
                          left=0.045, right=0.985, top=0.93, bottom=0.075)
    top = gs[0].subgridspec(1, 2, width_ratios=[0.9, 2.1], wspace=0.06)
    axs = fig.add_subplot(top[0]); schematic(axs)

    mg = top[1].subgridspec(2, 3, hspace=0.62, wspace=0.12)
    axes = []
    spec = [("mg:SER17-OG", "Mg–Ser17 OG (Å)"), ("mg:THR35-OG1", "Mg–Thr35 OG1 (Å)"),
            ("mg:GTP-O2B", "Mg–GTP O2B (Å)"), ("mg:GTP-O2G", "Mg–GTP O2G (Å)")]
    for j, (k, xl) in enumerate(spec):
        ax = fig.add_subplot(mg[j // 3, j % 3]); axes.append(ax)
        Sx = {st: {k: np.where(S[st][k] < CUT, S[st][k], np.nan)} for st in S}
        panel(ax, k, xl, 1.62, 2.45, Sx, {k: R[k][R[k] < CUT]},
              states=("in",) if "THR35" in k else ("in", "out"), ticks=[1.7, 1.9, 2.1, 2.3])
    ax = fig.add_subplot(mg[1, 1]); axes.append(ax)
    panel(ax, "mg:cis", "mean cis O–Mg–O (°)", 89.4, 90.6, S, R, ticks=[89.5, 90, 90.5])
    ax = fig.add_subplot(mg[1, 2]); axes.append(ax)
    panel(ax, "mg:trans", "mean trans O–Mg–O (°)", 165, 179, S, R, ticks=[166, 170, 174, 178])
    fig.canvas.draw()
    y1 = max(a.get_position().y1 for a in axes)
    p = axs.get_position()
    fig.text(0.012, y1 + 0.052, "a", fontsize=10, weight="bold", va="bottom")
    fig.text(0.036, y1 + 0.052, "What is measured", fontsize=8.2, weight="bold", va="bottom")
    fig.text(0.036, y1 + 0.033, "active-site schematic; letters give the panels", fontsize=6.8, va="bottom",
             color="0.35", style="italic")
    fig.add_artist(Line2D([0.036, p.x1 - 0.01], [y1 + 0.028] * 2, color="0.2", lw=0.8,
                          transform=fig.transFigure))
    group(fig, axes, "b", "Mg²⁺ first shell",
          "six-coordinate in both states; water replaces Thr35 OG1 when it leaves",
          MGC)

    bot = gs[1].subgridspec(2, 4, hspace=0.75, wspace=0.14)
    G = [  # (row, col, key, label, lo, hi, integer, ticks)
        (0, 0, "d:K16", "Lys16 NZ–O$_{\\beta,\\gamma}$ (Å)", 2.35, 3.0, False, [2.4, 2.6, 2.8, 3.0]),
        (1, 0, "d:D119", "Asp119 O–guanine N1/N2 (Å)", 2.45, 3.2, False, [2.5, 2.75, 3.0]),
        (0, 1, "d:T35", "Thr35 N–O$_\\gamma$ (Å)", 3, 25, False, [5, 10, 15, 20, 25]),
        (1, 1, "d:G60", "Gly60 N–O$_\\gamma$ (Å)", 2.5, 12, False, [3, 6, 9, 12]),
        (0, 2, "n_prot", "protein N/O within 3.5 Å of GTP", 10, 23, True, [10, 15, 20]),
        (1, 2, "n_wat_g", "waters within 3.5 Å of O$_\\gamma$", 2, 14, True, [2, 6, 10, 14]),
        (0, 3, "rmsd_pos", "GTP RMSD in the pocket (Å)", 0, 4.0, False, [0, 1, 2, 3, 4]),
        (1, 3, "rmsd_int", "GTP internal RMSD (Å)", 0, 0.9, False, [0, 0.3, 0.6, 0.9]),
    ]
    cols = {}
    for r, c, k, xl, lo, hi, integ, t in G:
        ax = fig.add_subplot(bot[r, c]); cols.setdefault(c, []).append(ax)
        panel(ax, k, xl, lo, hi, S, R, integer=integ, ticks=t)
    fig.canvas.draw()
    group(fig, cols[0], "c", "GTP anchors", "unchanged", ANC)
    group(fig, cols[1], "d", "Switch contacts", "Thr35 lost; Gly60 lost in 2/3", SWC)
    group(fig, cols[2], "e", "Packing and hydration", "fewer contacts, more water", WAC)
    group(fig, cols[3], "f", "GTP placement", "shifts, but keeps its shape", "0.2")

    h = [Patch(fc=IN, alpha=0.35, ec=IN, label=f"HMR, Thr35 bound ({100 * frac:.0f}% of frames)"),
         Patch(fc=OUT, alpha=0.35, ec=OUT, label=f"HMR, Thr35 out ({100 * (1 - frac):.0f}%)"),
         Line2D([], [], color=REF, lw=1.2, ls=(0, (3, 1.6)), label="reference, 4 × 1.64 µs")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7.2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"fig4_active_site.{ext}", dpi=600 if ext == "png" else None)
    fig.savefig("fig4_active_site.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    print("wrote fig4_active_site", f"bound fraction {frac:.3f}")


if __name__ == "__main__":
    main()
