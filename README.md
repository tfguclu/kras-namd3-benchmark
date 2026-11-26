# GPU-resident NAMD 3 benchmark of KRAS-GTP-Mg²⁺ (MareNostrum 5)

NAMD inputs, analysis code and per-frame data for

> T. F. Guclu, *Physics-validated optimization of GPU-resident molecular dynamics
> for sampling Switch I opening in unbound KRAS*

A 46,326-atom effector-free KRAS-GTP-Mg²⁺ system is simulated with NAMD 3.0.1
(GPU-resident) on one H100 GPU of MareNostrum 5. Three cumulative changes to the
CHARMM-GUI configuration (default pairlist settings, PME interpolation order
6 → 4, hydrogen mass repartitioning with a 4 fs timestep) raise the simulation
speed from 147 to 499 ns/day, with the Mg²⁺ first coordination shell as the
acceptance test.

Trajectories (one DCD per replicate, KRAS + GTP + Mg²⁺, 1 ns per frame) are on
Zenodo: https://zenodo.org/records/24536911.

## Contents

```
run_all.sh            rebuilds every figure, table and printed summary of the paper
namd/                 NAMD inputs (one per protocol) and the system
  step3_input.{psf,pdb,str}, step3_input_hmr.psf, restraints/, toppar/
  step4_equilibration.inp              minimization + restrained NVT
  step4_equilibration.{coor,vel,xsc}   equilibrated start of the reference, PME and HMR arms
  reference_step5_production[_restart].inp   reference arm, CHARMM-GUI settings, 2 fs
  pairlist_step5_production.inp              default pairlist settings (core-count sweep)
  pme4_step5_production.inp                  + PME interpolation order 4
  hmr_step5_production[_restart].inp         + HMR, 4 fs
  metad_step5_production.inp, metad_step6_metadynamics.inp,
  metad_step7_metadynamics_restart.inp, colvars_kras_wt_gtp_1d.in   well-tempered metadynamics
code/
  analysis/    figure, table and summary scripts
  figures/     schematic figures (Fig. 1b, Fig. 2, TOC graphic)
  trajectory/  trajectory readers that produced data/raw; strip_concat.py made the Zenodo files
  setup/       make_hmr.py (hydrogen mass repartitioning of the PSF)
  slurm/       MareNostrum 5 job script
data/
  raw.tar.xz        per-frame measurements from the trajectories (unpacked to data/raw/)
  namd_logs.tar.xz  NAMD log timing lines and Colvars traces of all runs
```

`run_all.sh` unpacks both archives on its first run and writes figures to
`outputs/figures/`, tables to `outputs/tables/` and the printed numbers quoted in the
paper to `outputs/summaries/`.

Restart inputs differ from the first input of an arm only in the input/output
names and `numsteps`; longer trajectories are chains of such jobs. The
metadynamics arm was equilibrated separately with `step4_equilibration.inp` and
run 100 ns unbiased (`metad_step5_production.inp`) before the biased segments
(steps 6–12). Only the three parameter files that contain terms for this system
are included; the remaining files of the CHARMM-GUI `toppar` set, which the
original runs also read, add no parameter for any atom type, bonded term or NBFIX
present here.

## Paper items and scripts (`code/analysis/` unless stated)

| Paper | Script |
|---|---|
| Fig. 1b, Fig. 2 | `code/figures/fig1b_shell.py`, `code/figures/fig2_overview.py` |
| Fig. 3, Tables 1–3 | `kras_main_figs.py`, `kras_baseline_all.py`, `kras_cpu_sweep.py`, `kras_hmr_benchmark.py`, `kras_hmr_walltime.py` |
| Fig. 4 | `kras_fig4.py` |
| Table 5, Table S1 | `kras_baseline_all.py`, `kras_pme_mg.py`, `kras_hmr_switch1.py` |
| Fig. 5, Table 6 | `kras_main_figs.py`, `kras_hmr_switch1.py`, `kras_metad_summary.py` |
| Fig. 6 | `kras_biology.py --plot` |
| Figs. S1–S3, Tables S2–S3 | `kras_shell_figs.py` |
| Figs. S4, S5 | `kras_ca_figs.py` |
| Figs. S6–S10 | `kras_si_residue.py` |
| TOC graphic | `code/figures/toc_graphic.py` |

## Per-frame data (`data/raw/`)

| Directory | Content (column names in each file header) |
|---|---|
| `kras_baseline_all/`, `kras_hmr_switch1/`, `kras_pme_mg/`, `kras_metad_mg/` | Switch I angle, Mg²⁺ coordination number, Mg–O distances (Ser17, Thr35, GTP O2B, O2G), shell waters, cis/trans O–Mg–O angles, Switch I RMSD, Thr35–core distance |
| `kras_gtp_shell/` | GTP environment: pocket and internal RMSD, glycosidic torsion, Pα–Pβ–Pγ angle, protein/water contacts, key residue distances |
| `kras_ca/` | Cα RMSD (all, core, Switch I, Switch II) and RMSF |
| `kras_biology/` | per-residue STRIDE code, lDDT-Cα, RSA, hydrogen bonds |

Residues 1–167 are analysed (Gly0 of the construct excluded). Mg²⁺ data are at
0.1 ns (1 ns for HMR); GTP, Cα and per-residue data at 1 ns.

## Requirements

Python 3.10+ with NumPy, SciPy and Matplotlib for `run_all.sh`; MDAnalysis,
mdtraj, FreeSASA and STRIDE only to recompute `data/raw/` from full trajectories.
NAMD 3.0.1 (SMP CUDA build) for the simulations:

```bash
namd3 +p20 +setcpuaffinity +devices 0 --GPUresident on hmr_step5_production.inp > hmr_step5_production.log
```

Equilibration was run with GPU-resident mode off. Set a distinct thermostat seed
per replicate.

## Zenodo trajectories

13 DCD files (`reference_rep1-4`, `pme4_rep1-3`, `hmr_rep1-3`, `metad_rep1-3`),
2,700 atoms (water and ions removed), frame *k* at *k* ns, with
`kras_solute_mg.psf/.pdb`. Cα, Switch I and GTP-pocket quantities can be
recomputed from them; water-dependent quantities (Mg²⁺ shell waters) are only in
`data/raw/`.

## Citation and license

Please cite the article and the Zenodo record. Code: MIT (`LICENSE`). Data: CC BY 4.0.
