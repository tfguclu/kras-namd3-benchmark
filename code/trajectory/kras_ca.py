#!/usr/bin/env python3
"""C-alpha RMSD and RMSF per replicate (numpy only).

Frames as in kras_gtp_shell.py (1 ns apart; PME control 0.1 ns).
C-alpha atoms of residues 1-167; the chain is made whole by unwrapping each
C-alpha onto the previous one under the minimum image.

  <arm>_<rep>_rmsd.dat   t, rmsd_all (fit and RMSD on all CA), rmsd_core (fit
                         and RMSD on CA outside Switch I 30-40 and Switch II
                         60-76), rmsd_sw1, rmsd_sw2 (Switch I / II CA after
                         the core fit); reference = first frame
  <arm>_<rep>_rmsf.dat   resid, rmsf (CA after iterative fit to the average
                         structure), rmsf_1640 (first 1.64 us only; HMR)

usage: python3 kras_ca.py <project_dir> <out_dir> <arm> <rep>
"""
import os, sys, glob
import numpy as np
from kras_gtp_shell import ARMS, read_psf, DCD, mic, order

SW1, SW2 = (30, 40), (60, 76)


def fit(X, Y, w=None):
    """rotate+translate each frame of X (n,k,3) onto Y (k,3) using mask w."""
    w = np.ones(X.shape[1], bool) if w is None else w
    xc = X[:, w].mean(1, keepdims=True); yc = Y[w].mean(0)
    A = X - xc
    H = np.einsum("nki,kj->nij", A[:, w], Y[w] - yc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("nij,njk->nik", U, Vt)))
    D = np.ones((len(X), 3)); D[:, 2] = d
    R = np.einsum("nij,nj,njk->nik", U, D, Vt)
    return np.einsum("nki,nij->nkj", A, R) + yc


def rmsd(A, B, m):
    return np.sqrt(((A[:, m] - B[m]) ** 2).sum(-1).mean(1))


def rmsf(X):
    ref = X[0]
    for _ in range(3):
        Xf = fit(X, ref)
        ref = Xf.mean(0)
    return np.sqrt(((Xf - ref) ** 2).sum(-1).mean(0))


if __name__ == "__main__":
    proj, out, arm, rep = sys.argv[1:5]
    d, psf, pat, stride, dt = ARMS[arm][rep]
    d = os.path.join(proj, d)
    at = read_psf(os.path.join(d, psf))
    ca = [(i, a[1]) for i, a in enumerate(at) if a[0].startswith("PRO") and a[3] == "CA" and 1 <= a[1] <= 167]
    idx = np.array([i for i, _ in ca]); res = np.array([r for _, r in ca])
    frames = []
    for p in sorted(glob.glob(os.path.join(d, pat)), key=order):
        dcd = DCD(p)
        for k in range(0, dcd.nframes, stride):
            x, box = dcd.frame(k)
            c = x[idx]
            w = np.empty_like(c); w[0] = c[0]
            for j in range(1, len(c)):
                w[j] = w[j - 1] + mic(c[j] - c[j - 1], box)
            frames.append(w)
    X = np.array(frames)
    t = np.arange(len(X)) * stride * dt
    core = ~(((res >= SW1[0]) & (res <= SW1[1])) | ((res >= SW2[0]) & (res <= SW2[1])))
    s1 = (res >= SW1[0]) & (res <= SW1[1]); s2 = (res >= SW2[0]) & (res <= SW2[1])
    ref = X[0]
    Xa = fit(X, ref)
    Xc = fit(X, ref, core)
    allm = np.ones(len(res), bool)
    R = np.c_[t, rmsd(Xa, ref, allm), rmsd(Xc, ref, core), rmsd(Xc, ref, s1), rmsd(Xc, ref, s2)]
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, f"{arm}_{rep}_rmsd.dat"), R, fmt="%.5g",
               header="t rmsd_all rmsd_core rmsd_sw1 rmsd_sw2", comments="# ")
    f_all = rmsf(X)
    f_win = rmsf(X[t < 1640]) if t[-1] > 1700 else f_all
    np.savetxt(os.path.join(out, f"{arm}_{rep}_rmsf.dat"), np.c_[res, f_all, f_win], fmt="%.5g",
               header="resid rmsf rmsf_1640", comments="# ")
    print(f"{arm} {rep}: {len(X)} frames, mean RMSD {R[:,1].mean():.2f} A", flush=True)
