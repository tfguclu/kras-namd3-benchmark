#!/usr/bin/env python3
"""Mg2+ first-shell descriptors for the metadynamics replicates (numpy only).

Same definitions as kras_baseline_all.analyse(): oxygen pool = all atoms named
O* except water oxygens, plus water oxygens; shell cutoff 2.6 A; cis/trans
split at 135 deg; distances to Ser17 OG, Thr35 OG1, GTP O2B and O2G; Mg to
nearest of these four; Switch I CV = angle Ser17 CA - Gln25 CA - Thr35 CA.
Reads the biased segments (step6..step12 *_metadynamics_1d.dcd) with STRIDE.

usage: python3 kras_metad_mg.py <project_dir> <out_dir> [stride]
writes <out_dir>/rep{1,2,3}.dat in the rawio column format.
"""
import os, re, sys, glob, struct
import numpy as np

CUT, TRANS_MIN = 2.6, 135.0
REPS = {"rep1": "KRAS_unbound_mg_gtp_metaD/namd",
        "rep2": "KRAS_unbound_mg_gtp_metaD_2/namd",
        "rep3": "KRAS_unbound_mg_gtp_metaD_4/namd"}
WATER = {"TIP3", "HOH", "WAT", "SOL"}


def read_psf(path):
    lines = open(path, errors="ignore").read().splitlines()
    i = next(k for k, l in enumerate(lines) if "!NATOM" in l)
    n = int(lines[i].split()[0])
    atoms = []
    for l in lines[i + 1:i + 1 + n]:
        f = l.split()
        atoms.append((int(f[2]) if f[2].lstrip("-").isdigit() else f[2], f[3], f[4]))
    return atoms  # (resid, resname, name)


class DCD:
    def __init__(self, path):
        self.f = open(path, "rb")
        h = self.f.read(4)
        self.e = "<" if struct.unpack("<i", h)[0] == 84 else ">"
        self.f.seek(0)
        icntrl = struct.unpack(self.e + "4s20i", self.f.read(92)[4:88])[1:]
        self.cell = icntrl[10] != 0
        n = struct.unpack(self.e + "i", self.f.read(4))[0]
        self.f.seek(n + 4, 1)
        self.f.read(4)
        self.n = struct.unpack(self.e + "i", self.f.read(4))[0]
        self.f.read(4)
        self.hdr = self.f.tell()
        self.fs = 3 * (4 * self.n + 8) + (56 if self.cell else 0)
        self.nframes = (os.path.getsize(path) - self.hdr) // self.fs

    def frame(self, k):
        self.f.seek(self.hdr + k * self.fs)
        box = None
        if self.cell:
            c = np.frombuffer(self.f.read(56)[4:52], dtype=self.e + "f8")
            box = np.array([c[0], c[2], c[5]])   # A, B, C (orthorhombic)
        xyz = []
        for _ in range(3):
            b = self.f.read(4 * self.n + 8)
            xyz.append(np.frombuffer(b[4:-4], dtype=self.e + "f4"))
        return np.stack(xyz, axis=1).astype(np.float64), box


def mic(v, box):
    return v - box * np.round(v / box)


def angle(a, b, c, box):
    u, w = mic(a - b, box), mic(c - b, box)
    cosv = (u * w).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(w, axis=-1)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))


def order(p):
    m = re.search(r"step(\d+)_", os.path.basename(p))
    return int(m.group(1)) if m else 0


def analyse(d, stride):
    atoms = read_psf(os.path.join(d, "step3_input.psf"))
    def idx(resid, name, resname=None):
        for i, (r, rn, n) in enumerate(atoms):
            if n == name and (r == resid if resname is None else rn == resname):
                return i
        raise SystemExit(f"atom not found: {resid} {resname} {name}")
    mg = next(i for i, a in enumerate(atoms) if a[1] == "MG" and a[2] == "MG")
    s17, q25, t35 = idx(17, "CA"), idx(25, "CA"), idx(35, "CA")
    named = {"SER17-OG": idx(17, "OG"), "THR35-OG1": idx(35, "OG1"),
             "GTP-O2B": idx(None, "O2B", "GTP"), "GTP-O2G": idx(None, "O2G", "GTP")}
    water = np.array([i for i, a in enumerate(atoms) if a[1] in WATER and a[2] in ("OH2", "OW", "O")])
    other = np.array([i for i, a in enumerate(atoms) if a[2].startswith("O") and a[1] not in WATER])
    pool = np.concatenate([other, water])
    is_wat = np.r_[np.zeros(len(other), bool), np.ones(len(water), bool)]
    keys = list(named); nid = np.array([named[k] for k in keys])

    dcds = sorted(glob.glob(os.path.join(d, "step*_metadynamics_1d.dcd")), key=order)
    rows = []
    for p in dcds:
        dcd = DCD(p)
        for k in range(0, dcd.nframes, stride):
            x, box = dcd.frame(k)
            cv = angle(x[s17], x[q25], x[t35], box)
            v = mic(x[pool] - x[mg], box)
            r = np.linalg.norm(v, axis=1)
            sel = np.where(r < CUT)[0]
            cn, nw = len(sel), int(is_wat[sel].sum())
            if cn >= 2:
                vv = v[sel] / r[sel, None]
                ai, bi = np.triu_indices(cn, k=1)
                a = np.degrees(np.arccos(np.clip((vv[ai] * vv[bi]).sum(1), -1, 1)))
                cis = a[a < TRANS_MIN].mean() if (a < TRANS_MIN).any() else np.nan
                tra = a[a >= TRANS_MIN].mean() if (a >= TRANS_MIN).any() else np.nan
            else:
                cis = tra = np.nan
            dn = np.linalg.norm(mic(x[nid] - x[mg], box), axis=1)
            rows.append([len(rows), cv, cn, nw, cis, tra, dn.min(), *dn])
        print(f"  {os.path.basename(p)}: {dcd.nframes} frames", flush=True)
    return keys, np.array(rows)


if __name__ == "__main__":
    proj, out = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    os.makedirs(out, exist_ok=True)
    for rep, sub in REPS.items():
        print(rep, flush=True)
        keys, a = analyse(os.path.join(proj, sub), stride)
        a[:, 0] *= stride * 0.1           # time, ns (frames saved every 0.1 ns)
        hdr = "# t cv cn nwat cis trans site " + " ".join(f"dist:{k}" for k in keys)
        np.savetxt(os.path.join(out, f"{rep}.dat"), a, fmt="%.5g", header=hdr[2:], comments="# ")
        print(f"wrote {rep}.dat ({len(a)} frames)", flush=True)
