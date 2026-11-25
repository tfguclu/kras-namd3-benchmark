#!/usr/bin/env python3
"""Concatenate NAMD DCD segments of one replicate, keep solute + Mg2+ only, write at 1 ns.

usage: strip_concat.py <psf> <out_prefix> <stride> <dcd> [<dcd> ...]

Kept atoms: segments PROA (KRAS), HETA (GTP), HETB (Mg2+); water (WATA, SOLV) and
ions (IONS) are removed. Segments are ordered by chain position in the file name (step number, then
restart suffix) and checked for continuity (the step count continues or restarts at 0). Every <stride>-th frame is written (stride 10 for 0.1 ns output,
1 for 1 ns output). Unit-cell records are kept. Writes <out_prefix>.dcd, and with
--topology also <out_prefix>.psf / .pdb (stripped topology).
NumPy only.
"""
import struct, sys, os, re
import numpy as np

KEEP = ("PROA", "HETA", "HETB")


def psf_sections(path):
    with open(path) as f:
        lines = f.read().split("\n")
    return lines


def n_keep_from_psf(lines):
    i = next(k for k, l in enumerate(lines) if "!NATOM" in l)
    nat = int(lines[i].split()[0])
    seg = [lines[i + 1 + j].split()[1] for j in range(nat)]
    n = 0
    while n < nat and seg[n] in KEEP:
        n += 1
    assert all(s not in KEEP for s in seg[n:]), "kept segments not contiguous at start"
    return n, nat


def read_header(f):
    def rec():
        (n,) = struct.unpack("<i", f.read(4))
        b = f.read(n)
        (m,) = struct.unpack("<i", f.read(4))
        assert n == m
        return b
    h = rec()
    assert h[:4] == b"CORD"
    icntrl = struct.unpack("<20i", h[4:84])
    nset, istart, nsavc = icntrl[0], icntrl[1], icntrl[2]
    delta = struct.unpack("<f", h[40:44])[0]
    has_cell = icntrl[10] == 1
    four_d = icntrl[11] == 1
    assert not four_d
    title = rec()
    natom = struct.unpack("<i", rec())[0]
    return dict(head=h, title=title, nset=nset, istart=istart, nsavc=nsavc,
                delta=delta, cell=has_cell, natom=natom, off=f.tell())


def fortran(b):
    return struct.pack("<i", len(b)) + b + struct.pack("<i", len(b))


def main():
    args = sys.argv[1:]
    topo = "--topology" in args
    args = [a for a in args if a != "--topology"]
    psf, out, stride, dcds = args[0], args[1], int(args[2]), args[3:]
    lines = psf_sections(psf)
    nk, nat = n_keep_from_psf(lines)

    hs = []
    for d in dcds:
        with open(d, "rb") as f:
            h = read_header(f)
        assert h["natom"] == nat, (d, h["natom"], nat)
        fsize = os.path.getsize(d)
        frame = (56 if h["cell"] else 0) + 3 * (4 * nat + 8)
        nfr = (fsize - h["off"]) // frame
        if nfr != h["nset"]:
            print(f"note: {os.path.basename(d)} header nset={h['nset']} file has {nfr} frames; using {nfr}")
        h.update(path=d, frame=frame, nfr=nfr)
        hs.append(h)
    # chain order from file names: step5_production, _1, _2 ... / step6_..., step7_ ...
    def order(h):
        b = os.path.basename(h["path"])[:-4]
        stepn = int(re.match(r"step(\d+)", b).group(1))
        m = re.search(r"_(\d+)$", b)
        return (stepn, int(m.group(1)) if m else -1)
    hs.sort(key=order)
    for a, b in zip(hs, hs[1:]):
        exp = a["istart"] + a["nfr"] * a["nsavc"]
        # a segment either continues the step count or restarts it at 0 (firsttimestep 0)
        assert b["istart"] in (exp, b["nsavc"]), f"gap: {a['path']} ends {exp}, {b['path']} starts {b['istart']}"
        assert b["nsavc"] == a["nsavc"]

    nsavc = hs[0]["nsavc"]
    total = sum(h["nfr"] for h in hs)
    nout = total // stride
    # header for output
    h0 = hs[0]
    icntrl = list(struct.unpack("<20i", h0["head"][4:84]))
    icntrl[0] = nout
    icntrl[1] = h0["istart"] + (stride - 1) * nsavc
    icntrl[2] = nsavc * stride
    icntrl[3] = nout * nsavc * stride
    if icntrl[3] > 2**31 - 1:  # last-step field overflows int32 for 10 us at 4 fs
        icntrl[3] = 0
    head = b"CORD" + struct.pack("<9i", *icntrl[:9]) + h0["head"][40:44] + struct.pack("<10i", *icntrl[10:])
    title_txt = [b"REMARKS KRAS-GTP-Mg; solute+Mg only (PROA HETA HETB), water/ions removed".ljust(80),
                 (f"REMARKS concatenated from {len(hs)} NAMD segments, every {stride} frame(s)").encode().ljust(80)]
    title = struct.pack("<i", len(title_txt)) + b"".join(title_txt)

    with open(out + ".dcd.part", "wb") as fo:
        fo.write(fortran(head)); fo.write(fortran(title)); fo.write(fortran(struct.pack("<i", nk)))
        g = 0
        written = 0
        for h in hs:
            with open(h["path"], "rb") as f:
                for k in range(h["nfr"]):
                    g += 1
                    if g % stride:
                        continue
                    if written >= nout:
                        break
                    base = h["off"] + k * h["frame"]
                    f.seek(base)
                    if h["cell"]:
                        fo.write(f.read(56))
                    for c in range(3):
                        f.seek(base + (56 if h["cell"] else 0) + c * (4 * nat + 8) + 4)
                        xyz = f.read(4 * nk)
                        fo.write(fortran(xyz))
                    written += 1
    assert written == nout
    os.replace(out + ".dcd.part", out + ".dcd")
    dt = nsavc * stride * h0["delta"] * 48.88821 / 1e6
    print(f"{out}.dcd: {nk} atoms, {nout} frames x {dt:.3f} ns = {nout*dt:.1f} ns, "
          f"from {len(hs)} segments ({total} input frames: " + ", ".join(f"{os.path.basename(h['path'])}:{h['nfr']}" for h in hs) + ")")
    if topo:
        write_topology(lines, nk, out, psf)


def write_topology(lines, nk, out, psf):
    out_l = []
    i = 0
    L = len(lines)
    while i < L:
        l = lines[i]
        if "!N" in l and "!NTITLE" not in l and l.split()[0].lstrip("-").isdigit():
            n = int(l.split()[0]); key = l.split("!")[1].split(":")[0].strip()
            per = {"NBOND": 2, "NTHETA": 3, "NPHI": 4, "NIMPHI": 4, "NCRTERM": 8}.get(key)
            if key == "NATOM":
                out_l.append(f"{nk:10d} !NATOM")
                out_l += lines[i + 1:i + 1 + nk]
                out_l.append("")
                i += 1 + n
                while i < L and lines[i].strip() == "":
                    i += 1
                continue
            j = i + 1
            vals = []
            while j < L and lines[j].strip() != "":
                vals += lines[j].split(); j += 1
            if per:
                arr = np.array(vals, int).reshape(-1, per)
                arr = arr[(arr <= nk).all(1)]
                cols = {2: 4, 3: 3, 4: 2, 8: 1}[per]
                out_l.append(f"{len(arr):10d} !{l.split('!')[1]}")
                flat = arr.ravel()
                step = per * cols
                for s in range(0, len(flat), step):
                    out_l.append("".join(f"{v:10d}" for v in flat[s:s + step]))
            else:
                # drop donors/acceptors/NNB/groups/cross-terms we do not filter: write empty
                if key == "NGRP NST2":
                    out_l.append(f"{0:10d}{0:10d} !NGRP NST2")
                elif key == "NUMLP NUMLPH":
                    out_l.append(f"{0:10d}{0:10d} !NUMLP NUMLPH")
                else:
                    out_l.append(f"{0:10d} !{l.split('!')[1]}")
            out_l.append("")
            i = j + 1
            if key == "NNB":  # IBLO list, one entry per atom
                while i < L and lines[i].strip() == "":
                    i += 1
                while i < L and lines[i].strip() != "":
                    i += 1
                z = [0] * nk
                for s in range(0, nk, 8):
                    out_l.append("".join(f"{v:10d}" for v in z[s:s + 8]))
                out_l.append("")
                i += 1
            continue
        out_l.append(l)
        i += 1
    with open(out + ".psf", "w") as f:
        f.write("\n".join(out_l).rstrip("\n") + "\n")
    pdb = psf.replace(".psf", ".pdb")
    with open(pdb) as f, open(out + ".pdb", "w") as g:
        n = 0
        for l in f:
            if l.startswith(("ATOM", "HETATM")):
                if n < nk:
                    g.write(l)
                n += 1
            elif l.startswith("CRYST1"):
                g.write(l)
        g.write("END\n")


if __name__ == "__main__":
    main()
