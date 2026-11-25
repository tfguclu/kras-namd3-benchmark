#!/usr/bin/env python3
"""GTP heavy-atom RMSD in the pocket against the equilibrated starting structure.

Protein C-alpha atoms (residues 1-167) of every frame are superposed on
step4_equilibration.coor of the arm (reference, PME and HMR share one
equilibrated structure; metadynamics has its own), and the RMSD of the GTP heavy
atoms is computed. Frames as in kras_gtp_shell.py.

usage: python3 kras_gtp_rmsd_eq.py <project_dir> <out_dir> <arm> <rep>
       python3 kras_gtp_rmsd_eq.py --merge <raw_dir>
           writes <out_dir>/<arm>_<rep>_rmsd_eq.dat; --merge copies these values
           into the rmsd_pos column of raw/kras_gtp_shell/<arm>_<rep>.dat
           (the column used in Fig. 4f and Table S3)
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kras_gtp_shell import ARMS, read_psf, DCD, mic, order, kabsch_rmsd


def merge(raw):
    for f in sorted(glob.glob(os.path.join(raw, "kras_gtp_shell", "*_rep?.dat"))):
        name = os.path.basename(f)[:-4]
        e = os.path.join(raw, "kras_gtp_eq", f"{name}_rmsd_eq.dat")
        hdr = open(f).readline()[2:].strip()
        a = np.loadtxt(f); b = np.loadtxt(e)
        a[:, hdr.split().index("rmsd_pos")] = b[:, 1]
        np.savetxt(f, a, fmt="%.5g", header=hdr, comments="# ")
        print("merged", name)


def main(proj, out, arm, rep):
    d, psf, pat, stride, dt = ARMS[arm][rep]
    d = os.path.join(proj, d)
    at = read_psf(os.path.join(d, psf))
    ca = np.array([i for i, a in enumerate(at) if a[0].startswith("PRO") and a[3] == "CA" and 1 <= a[1] <= 167])
    gh = np.array([i for i, a in enumerate(at) if a[2] == "GTP" and not a[3].startswith("H")])
    pb = [i for i, a in enumerate(at) if a[2] == "GTP" and a[3] == "PB"][0]
    raw = open(os.path.join(d, "step4_equilibration.coor"), "rb").read()
    n = np.frombuffer(raw[:4], "<i4")[0]
    X0 = np.frombuffer(raw[4:4 + 24 * n], "<f8").reshape(n, 3)
    xsc = [l for l in open(os.path.join(d, "step4_equilibration.xsc")) if not l.startswith("#")][0].split()
    box0 = np.array([float(xsc[1]), float(xsc[5]), float(xsc[9])])
    L0 = lambda idx: X0[pb] + mic(X0[idx] - X0[pb], box0)
    rca, rg = L0(ca), L0(gh)
    rows = []
    for p in sorted(glob.glob(os.path.join(d, pat)), key=order):
        dc = DCD(p)
        for k in range(0, dc.nframes, stride):
            x, box = dc.frame(k)
            c0 = x[pb]
            L = lambda idx: c0 + mic(x[idx] - c0, box)
            rows.append(kabsch_rmsd(L(ca), rca, L(gh), rg))
    r = np.array(rows)
    t = np.arange(len(r)) * stride * dt
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, f"{arm}_{rep}_rmsd_eq.dat"), np.c_[t, r], fmt="%.5g",
               header="t rmsd_pos_eq", comments="# ")
    print(arm, rep, len(r), round(r.mean(), 3))


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge(sys.argv[2])
    else:
        main(*sys.argv[1:5])
