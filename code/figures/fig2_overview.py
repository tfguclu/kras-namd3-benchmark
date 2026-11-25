"""Fig. 2: overview of the compared protocols (schematic + throughput strip)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon

plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Liberation Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

RED, RED_L = "#C81E1E", "#FBEAEA"
BLUE, BLUE_L = "#0868AC", "#E8F1F8"
INK, GREY, GREY_L = "#1A1A1A", "#6B6B6B", "#D9D9D9"

W, H = 7.0, 6.0
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect("equal"); ax.axis("off")


def card(x0, y0, w, h, col, fill):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill, ec=col, lw=0.9, zorder=0))


def header(x0, ytop, n, title, sub, col):
    ax.add_patch(Circle((x0 + 0.17, ytop - 0.2), 0.105, fc=col, ec="none", zorder=3))
    ax.text(x0 + 0.17, ytop - 0.203, str(n), color="white", fontsize=8, weight="bold",
            ha="center", va="center", zorder=4)
    ax.text(x0 + 0.34, ytop - 0.2, title, color=col, fontsize=9.5, weight="bold", va="center")
    ax.text(x0 + 0.34, ytop - 0.39, sub, color=GREY, fontsize=7, va="center")


# ---------------- row A: the two reference protocols ----------------
def well(x0, y0, w, h):
    u = np.linspace(0.07, 0.93, 300)
    v = (u - 0.22) ** 2 * (u - 0.78) ** 2 + 0.0012 * u
    s = (u - 0.07) / 0.86
    v = (v - v.min()) / (v.max() - v.min())
    return s, x0 + s * w, y0 + v * h


yA, hA = 4.3, 1.55
for i, (x0, col, fill, n, title, sub, note) in enumerate([
        (0.05, RED, RED_L, 1, "Unbiased MD", "CHARMM-GUI baseline, 2 fs",
         "Switch I stays closed\nin 4 × 1.64 µs"),
        (3.55, RED, RED_L, 2, "Well-tempered metadynamics", "baseline + Colvars bias on the Switch I angle",
         "angle leaves closed range,\nwith a bias, at lower speed")]):
    card(x0, yA, 3.4, hA, col, fill)
    header(x0, yA + hA, n, title, sub, col)
    ax.text(x0 + 0.17, yA + 0.45, note, fontsize=7.5, color=INK, va="center", linespacing=1.3)
    gx, gy, gw, gh = x0 + 1.9, yA + 0.28, 1.3, 0.78
    s, X, Y = well(gx, gy, gw, gh)
    iC = np.argmin(np.where(s < 0.5, Y, 9)); iO = np.argmin(np.where(s > 0.5, Y, 9))
    iB = iC + np.argmax(Y[iC:iO])
    if n == 2:  # deposited Gaussians fill the closed well up to the barrier
        level = Y[iB] - 0.04
        m = (s < s[iB]) & (Y < level)
        ax.fill_between(X[m], Y[m], level, color="#F2B8B8", lw=0, zorder=1)
        for k in range(1, 5):
            lv = Y[m].min() + k * (level - Y[m].min()) / 4
            mm = (s < s[iB]) & (Y < lv)
            ax.plot(X[mm], np.full(mm.sum(), lv), color=RED, lw=0.5, alpha=0.7, zorder=2)
        ax.text(X[iC] - 0.02, level + 0.06, "bias", fontsize=6.5, color=RED, ha="center")
        ball = (X[iO] - 0.02, Y[iO] + 0.06)
        ax.add_patch(FancyArrowPatch((X[iC] + 0.2, level + 0.12), (ball[0] - 0.06, ball[1] + 0.09),
                                     connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                     mutation_scale=7, color=INK, lw=0.7, zorder=4))
    else:
        ball = (X[iC], Y[iC] + 0.06)
        ax.text(X[iB], Y[iB] + 0.05, "barrier", fontsize=6.5, color=GREY, ha="center")
    ax.plot(X, Y, color=INK, lw=1.1, zorder=3)
    ax.add_patch(Circle(ball, 0.055, fc=RED, ec="white", lw=0.6, zorder=5))
    ax.text(X[iC], gy - 0.13, "closed", fontsize=6.5, color=GREY, ha="center")
    ax.text(X[iO], gy - 0.13, "open", fontsize=6.5, color=GREY, ha="center")

# ---------------- row B: cumulative changes ----------------
ax.text(W / 2, 4.13, "cumulative changes to the baseline (1)", color=BLUE, fontsize=8,
        weight="bold", ha="center", va="center")
yB, hB, wB = 1.75, 2.22, 2.22
xs = [0.05, 2.39, 4.73]
titles = [("Default pairlist", "explicit pairlist parameters removed",
           "forces unchanged"),
          ("PME order 6 → 4", "B-spline interpolation order",
           "Mg²⁺ shell unchanged"),
          ("HMR, 4 fs", "solute H 1 → 3 amu; Δt 2 → 4 fs",
           "Mg²⁺ shell within tolerance")]
for j, (x0, (t, sub, note)) in enumerate(zip(xs, titles)):
    card(x0, yB, wB, hB, BLUE, BLUE_L)
    header(x0, yB + hB, j + 3, t, sub, BLUE)
    ax.text(x0 + wB / 2, yB + 0.17, note, fontsize=7.5, color=INK, ha="center", va="center")
    if j < 2:
        ax.annotate("", xy=(xs[j + 1] + 0.005, yB + hB / 2), xytext=(x0 + wB - 0.005, yB + hB / 2),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0, mutation_scale=9))

# (3) pairlist radius
cx, cy = xs[0] + 0.78, yB + 0.98
ax.add_patch(Circle((cx, cy), 0.6, fc="#F3F3F3", ec=GREY, ls=(0, (3, 2)), lw=0.8, zorder=1))
ax.add_patch(Circle((cx, cy), 0.45, fc="white", ec=BLUE, lw=1.1, zorder=2))
rng = np.random.default_rng(3)
for r0, r1, kind, nn in [(0.12, 0.38, "in", 7), (0.49, 0.57, "mid", 6), (0.66, 0.74, "out", 5)]:
    th = np.sort(rng.uniform(0, 2 * np.pi, nn)) + rng.uniform(0, 1)
    rr = rng.uniform(r0, r1, nn)
    for a, r in zip(th, rr):
        p = (cx + r * np.cos(a), cy + r * np.sin(a))
        if kind == "in":
            ax.add_patch(Circle(p, 0.035, fc=INK, ec="none", zorder=3))
        elif kind == "mid":
            ax.add_patch(Circle(p, 0.035, fc="white", ec=INK, lw=0.6, zorder=3))
        else:
            ax.add_patch(Circle(p, 0.035, fc=GREY_L, ec="none", zorder=3))
ax.add_patch(Circle((cx, cy), 0.045, fc=RED, ec="none", zorder=4))
lx = xs[0] + 1.48
for k, (lab, kw) in enumerate([("< 12 Å: in cutoff", dict(fc=INK, ec="none")),
                               ("12–16 Å: old list", dict(fc="white", ec=INK, lw=0.6)),
                               ("> 16 Å: ignored", dict(fc=GREY_L, ec="none"))]):
    yy = cy + 0.36 - k * 0.2
    ax.add_patch(Circle((lx, yy), 0.035, zorder=3, **kw))
    ax.text(lx + 0.07, yy, lab, fontsize=6, va="center", color=INK)
ax.text(lx - 0.04, cy - 0.37, "new list: 12 Å\n(= cutoff)", fontsize=6,
        color=BLUE, va="center", linespacing=1.2)
ax.text(cx, cy - 0.45, "12 Å", fontsize=6, color=BLUE, ha="center", va="center",
        bbox=dict(fc="white", ec="none", pad=0.4), zorder=5)
ax.text(cx, cy + 0.6, "16 Å", fontsize=6, color=GREY, ha="center", va="center",
        bbox=dict(fc="#F3F3F3", ec="none", pad=0.4), zorder=5)

# (4) PME stencils
for k, (order, col, ox) in enumerate([(6, GREY, xs[1] + 0.58), (4, BLUE, xs[1] + 1.64)]):
    cy = yB + 1.0; d = 0.105
    for gxv in np.arange(-4, 5) * d:
        for gyv in np.arange(-4, 5) * d:
            ax.add_patch(Circle((ox + gxv, cy + gyv), 0.011, fc=GREY_L, ec="none", zorder=1))
    half = (order - 1) / 2 * d
    ax.add_patch(FancyBboxPatch((ox - half - 0.05, cy - half - 0.05), 2 * half + 0.1, 2 * half + 0.1,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec=col, lw=0.9, zorder=2))
    for i in range(order):
        for jj in range(order):
            ax.add_patch(Circle((ox - half + i * d, cy - half + jj * d), 0.022,
                                fc=INK if order == 6 else BLUE, ec="none", zorder=3))
    ax.add_patch(Circle((ox + 0.03, cy + 0.02), 0.035, fc=RED, ec="white", lw=0.4, zorder=4))
    ax.text(ox, cy - 0.58, f"order {order}: {order}³ = {order**3}", fontsize=6.5,
            color=INK if order == 6 else BLUE, ha="center", weight="bold")
ax.annotate("", xy=(xs[1] + 1.14, yB + 1.0), xytext=(xs[1] + 1.02, yB + 1.0),
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8, mutation_scale=7))

# (5) HMR dumbbells
def dumb(xc, yc, mh, mx, hcol):
    rX, rH = 0.075 * mh ** (1 / 3) * 0.95, 0.075 * mx ** (1 / 3) * 0.95
    ax.plot([xc, xc + 0.42], [yc, yc], color=INK, lw=1.6, zorder=1)
    ax.add_patch(Circle((xc, yc), rX, fc="#4D4D4D", ec="none", zorder=2))
    ax.add_patch(Circle((xc + 0.42, yc), rH, fc="white", ec=hcol, lw=1.0, zorder=2))
    ax.text(xc, yc, f"{mh:.1f}", color="white", fontsize=6.5, weight="bold", ha="center", va="center", zorder=3)
    ax.text(xc + 0.42, yc, f"{mx:.1f}", color=hcol, fontsize=6, weight="bold", ha="center", va="center", zorder=3)

yc = yB + 1.45
dumb(xs[2] + 0.4, yc, 12.0, 1.0, INK)
ax.text(xs[2] + 0.61, yc - 0.33, "standard", fontsize=6.5, color=GREY, ha="center")
dumb(xs[2] + 0.4, yc - 0.82, 10.0, 3.0, BLUE)
ax.text(xs[2] + 0.61, yc - 1.12, "repartitioned", fontsize=6.5, color=BLUE, ha="center")
ax.add_patch(FancyArrowPatch((xs[2] + 1.33, yc - 0.12), (xs[2] + 1.33, yc - 0.7),
                             arrowstyle="-|>", mutation_scale=9, color=BLUE, lw=1.1))
ax.text(xs[2] + 1.42, yc - 0.41, "2 amu\nheavy → H", fontsize=6.5, color=BLUE, va="center",
        linespacing=1.2)

# ---------------- row C: throughput strip ----------------
axb = fig.add_axes([0.215, 0.07, 0.66, 0.2])
lab = ["(1) baseline", "(2) + metadynamics", "(3) + default pairlist", "(4) + PME order 4", "(5) + HMR, 4 fs"]
val = [147, 129, 206, 254, 499]
days = [112, 128, 81, 65, 32]
cols = [RED, "#E58A8A", "#9EC5E2", "#4A97CC", BLUE]
yy = np.arange(5)[::-1]
axb.barh(yy, val, color=cols, height=0.62, zorder=2)
for y_, v, d in zip(yy, val, days):
    axb.text(v + 6, y_, f"{v} ns/day", va="center", fontsize=7, color=INK, weight="bold")
    axb.text(640, y_, f"{d} d", va="center", ha="right", fontsize=7, color=INK)
axb.text(640, 4.4, "time for 10 µs", ha="right", va="bottom", fontsize=6.5, color=GREY)
axb.set_yticks(yy, lab, fontsize=7)
axb.set_xlim(0, 640); axb.set_ylim(-0.55, 4.55)
axb.set_xticks([0, 100, 200, 300, 400, 500])
axb.tick_params(axis="x", labelsize=6.5, length=2, color=GREY)
axb.tick_params(axis="y", length=0)
axb.set_xlabel("simulation speed on one H100 GPU (ns/day)", fontsize=7, labelpad=2)
for sp in ["top", "right", "left"]:
    axb.spines[sp].set_visible(False)
axb.spines["bottom"].set_color(GREY); axb.spines["bottom"].set_linewidth(0.6)
axb.grid(axis="x", color="#EEEEEE", lw=0.6, zorder=0)
axb.axvline(147, color=RED, lw=0.6, ls=(0, (2, 2)), zorder=1)

for ext in ["png", "pdf", "tiff"]:
    kw = dict(dpi=600) if ext in ("png", "tiff") else {}
    if ext == "tiff":
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(f"fig2_overview.{ext}", **kw)
fig.savefig("fig2_overview.eps")
print("ok")
