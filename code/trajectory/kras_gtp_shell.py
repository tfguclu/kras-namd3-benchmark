#!/usr/bin/env python3
"""GTP environment ("GTP shell") per frame, numpy only.

For each arm/replicate, writes raw_gtp/<arm>_<rep>.dat with columns

  t          time (ns)
  cv         Switch I angle Ser17 CA - Gln25 CA - Thr35 CA (deg)
  rmsd_pos   GTP heavy-atom RMSD after superposition of protein CA 1-167 on
             the first frame of the replicate (A): displacement in the pocket
  rmsd_int   GTP heavy-atom RMSD after superposition of GTP on itself (A):
             internal conformational change
  chi        glycosidic torsion O4'-C1'-N9-C4 (deg)
  p_ang      PA-PB-PG angle (deg)
  n_prot     protein N/O atoms within 3.5 A of any GTP N/O
  n_ploop    backbone N of residues 13-18 within 3.5 A of any phosphate O
  n_wat      water O within 3.5 A of any GTP N/O
  n_wat_g    water O within 3.5 A of any gamma-phosphate O
  d:K16      Lys16 NZ  - nearest beta/gamma O
  d:G60      Gly60 N   - nearest gamma O        (Switch II, DxxG)
  d:T35      Thr35 N   - nearest gamma O        (Switch I, state 2)
  d:Y32      Tyr32 OH  - nearest gamma O        (Switch I)
  d:F28      Phe28 ring centroid - purine ring centroid
  d:K117     Lys117 NZ - O4'                    (NKxD)
  d:D119     Asp119 OD1/OD2 - nearest of N1, N2  (NKxD, guanine specificity)
  d:A146     Ala146 N  - O6                      (SAK)

All distances use the minimum image in the orthorhombic cell of each frame.

usage: python3 kras_gtp_shell.py <project_dir> <out_dir> <arm> <rep>
"""
import os, re, sys, glob, struct
import numpy as np

CUT = 3.5
P = "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/"
ARMS = {  # arm: {rep: (dir, psf, dcd glob, stride, ns per saved frame)}
    "reference": {f"rep{i}": (f"KRAS_unbound_mg_gtp_{i}/namd", "step3_input.psf",
                              "step5_production*.dcd", 10, 0.1) for i in (1, 2, 3, 4)},
    "pme4": {"rep1": (P + "namd_2", "step3_input.psf", "step5_production.dcd", 1, 0.1),
             "rep2": (P + "namd_2_2", "step3_input.psf", "step5_production.dcd", 1, 0.1),
             "rep3": (P + "namd_2_3", "step3_input.psf", "step5_production.dcd", 1, 0.1)},
    "hmr": {"rep1": (P + "namd_3", "step3_input_hmr.psf", "step5_production*.dcd", 1, 1.0),
            "rep2": (P + "namd_3_2", "step3_input_hmr.psf", "step5_production*.dcd", 1, 1.0),
            "rep3": (P + "namd_3_3", "step3_input_hmr.psf", "step5_production*.dcd", 1, 1.0)},
    "metad": {"rep1": ("KRAS_unbound_mg_gtp_metaD/namd", "step3_input.psf",
                       "step*_metadynamics_1d.dcd", 10, 0.1),
              "rep2": ("KRAS_unbound_mg_gtp_metaD_2/namd", "step3_input.psf",
                       "step*_metadynamics_1d.dcd", 10, 0.1),
              "rep3": ("KRAS_unbound_mg_gtp_metaD_4/namd", "step3_input.psf",
                       "step*_metadynamics_1d.dcd", 10, 0.1)},
}
WATER = {"TIP3", "HOH", "WAT", "SOL"}
GAMMA = ["O1G", "O2G", "O3G"]
BETA_GAMMA = ["O1B", "O2B", "O3B"] + GAMMA
PHOS_O = ["O1A", "O2A", "O3A", "O5'"] + BETA_GAMMA
PURINE = ["N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"]
DKEYS = ["K16", "G60", "T35", "Y32", "F28", "K117", "D119", "A146"]


def read_psf(path):
    lines = open(path, errors="ignore").read().splitlines()
    i = next(k for k, l in enumerate(lines) if "!NATOM" in l)
    n = int(lines[i].split()[0])
    out = []
    for l in lines[i + 1:i + 1 + n]:
        f = l.split()
        out.append((f[1], int(f[2]), f[3], f[4]))   # seg, resid, resname, name
    return out


class DCD:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.e = "<" if struct.unpack("<i", self.f.read(4))[0] == 84 else ">"
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
            box = np.array([c[0], c[2], c[5]])
        xyz = [np.frombuffer(self.f.read(4 * self.n + 8)[4:-4], dtype=self.e + "f4")
               for _ in range(3)]
        return np.stack(xyz, axis=1).astype(np.float64), box


def mic(v, box):
    return v - box * np.round(v / box)


def kabsch_rmsd(P_, Q, Pm=None, Qm=None):
    """RMSD of Pm onto Qm after the superposition that best fits P onto Q."""
    pc, qc = P_.mean(0), Q.mean(0)
    H = (P_ - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    A = P_ if Pm is None else Pm
    B = Q if Qm is None else Qm
    X = (A - pc) @ R.T + qc
    return np.sqrt(((X - B) ** 2).sum(1).mean())


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))


def order(p):
    b = os.path.basename(p)
    m = re.search(r"step(\d+)_\d+ns_metadynamics", b)
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)\.dcd$", b)
    return int(m.group(1)) if m else -1


def analyse(d, psf, pat, stride):
    at = read_psf(os.path.join(d, psf))
    prot = np.array([a[0].startswith("PRO") for a in at])
    gtp = np.array([a[2] == "GTP" for a in at])
    def one(resid, name, resname=None):
        for i, (s, r, rn, n) in enumerate(at):
            if n == name and ((resname is None and prot[i] and r == resid) or rn == resname):
                return i
        raise SystemExit(f"not found {resid} {resname} {name}")
    def g(names):
        return np.array([one(None, n, "GTP") for n in names])
    ca = np.array([i for i, a in enumerate(at) if prot[i] and a[3] == "CA" and 1 <= a[1] <= 167])
    gh = np.array([i for i, a in enumerate(at) if gtp[i] and not a[3].startswith("H")])
    gpol = np.array([i for i, a in enumerate(at) if gtp[i] and a[3][0] in "NO"])
    ppol = np.array([i for i, a in enumerate(at) if prot[i] and a[3][0] in "NO"])
    ploopN = np.array([i for i, a in enumerate(at) if prot[i] and a[3] == "N" and 13 <= a[1] <= 18])
    wat = np.array([i for i, a in enumerate(at) if a[2] in WATER and a[3] in ("OH2", "OW", "O")])
    gam, bg, pho = g(GAMMA), g(BETA_GAMMA), g(PHOS_O)
    pur = g(PURINE)
    pg, pb, pa = g(["PG"])[0], g(["PB"])[0], g(["PA"])[0]
    o4, c1, n9, c4 = g(["O4'", "C1'", "N9", "C4"])
    f28 = np.array([one(28, n) for n in ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]])
    tgt = {"K16": ([one(16, "NZ")], bg), "G60": ([one(60, "N")], gam),
           "T35": ([one(35, "N")], gam), "Y32": ([one(32, "OH")], gam),
           "K117": ([one(117, "NZ")], g(["O4'"])),
           "D119": ([one(119, "OD1"), one(119, "OD2")], g(["N1", "N2"])),
           "A146": ([one(146, "N")], g(["O6"]))}
    s17, q25, t35 = one(17, "CA"), one(25, "CA"), one(35, "CA")

    dcds = sorted(glob.glob(os.path.join(d, pat)), key=order)
    rows, ref = [], None
    for p in dcds:
        dcd = DCD(p)
        for k in range(0, dcd.nframes, stride):
            x, box = dcd.frame(k)
            c0 = x[pb]
            # local, whole copy of everything around GTP
            L = lambda idx: c0 + mic(x[idx] - c0, box)
            xca, xg = L(ca), L(gh)
            if ref is None:
                ref = (xca.copy(), xg.copy())
            rpos = kabsch_rmsd(xca, ref[0], xg, ref[1])
            rint = kabsch_rmsd(xg, ref[1])
            u, w = x[pa] - x[pb], x[pg] - x[pb]
            u, w = mic(u, box), mic(w, box)
            pang = np.degrees(np.arccos(u @ w / np.linalg.norm(u) / np.linalg.norm(w)))
            q = L(np.array([o4, c1, n9, c4]))
            chi = dihedral(*q)
            xgp, xpp = L(gpol), L(ppol)
            dpp = np.linalg.norm(xpp[:, None] - xgp[None], axis=2)
            n_prot = int((dpp.min(1) < CUT).sum())
            dpl = np.linalg.norm(L(ploopN)[:, None] - L(pho)[None], axis=2)
            n_ploop = int((dpl.min(1) < CUT).sum())
            xw = L(wat)
            near = np.abs(xw - c0).max(1) < 25
            xw = xw[near]
            dw = np.linalg.norm(xw[:, None] - xgp[None], axis=2)
            n_wat = int((dw.min(1) < CUT).sum())
            dwg = np.linalg.norm(xw[:, None] - L(gam)[None], axis=2)
            n_wat_g = int((dwg.min(1) < CUT).sum())
            ds = []
            for key in DKEYS:
                if key == "F28":
                    ds.append(np.linalg.norm(L(f28).mean(0) - L(pur).mean(0)))
                else:
                    a, b = tgt[key]
                    ds.append(np.linalg.norm(L(np.array(a))[:, None] - L(b)[None], axis=2).min())
            cv_u, cv_w = mic(x[s17] - x[q25], box), mic(x[t35] - x[q25], box)
            cv = np.degrees(np.arccos(cv_u @ cv_w / np.linalg.norm(cv_u) / np.linalg.norm(cv_w)))
            rows.append([len(rows), cv, rpos, rint, chi, pang, n_prot, n_ploop, n_wat, n_wat_g, *ds])
        print(f"  {os.path.basename(p)}: {dcd.nframes} frames", flush=True)
    return np.array(rows)


if __name__ == "__main__":
    proj, out, arm, rep = sys.argv[1:5]
    nmax = int(sys.argv[5]) if len(sys.argv) > 5 else None
    d, psf, pat, stride, dt = ARMS[arm][rep]
    a = analyse(os.path.join(proj, d), psf, pat, stride)
    a[:, 0] *= stride * dt
    os.makedirs(out, exist_ok=True)
    hdr = ("t cv rmsd_pos rmsd_int chi p_ang n_prot n_ploop n_wat n_wat_g "
           + " ".join(f"d:{k}" for k in DKEYS))
    np.savetxt(os.path.join(out, f"{arm}_{rep}.dat"), a, fmt="%.5g", header=hdr, comments="# ")
    print(f"wrote {arm}_{rep}.dat ({len(a)} frames)", flush=True)
