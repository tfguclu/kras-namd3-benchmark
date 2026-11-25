#!/usr/bin/env bash
# Rebuild every figure, table and printed summary of the paper from data/,
# without trajectories. Trajectory-derived quantities are read from the per-frame
# tables in data/raw.tar.xz; simulation speed and metadynamics are read from the NAMD
# logs and Colvars traces in data/namd_logs.tar.xz (unpacked on the first run).
#
#   bash run_all.sh
#
# The per-frame tables can be recomputed from the trajectories (Zenodo record)
# with the scripts in code/trajectory/ (see README).
set -uo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
cd "$REPO/data"
# NAMD logs (timing lines only) and Colvars traces, in the original run-directory layout
[ -d KRAS_unbound_mg_gtp_1 ] || tar -xJf namd_logs.tar.xz
# per-frame measurements from the trajectories
[ -d raw ] || tar -xJf raw.tar.xz
mkdir -p "$REPO/outputs/figures" "$REPO/outputs/summaries" "$REPO/outputs/tables"
export PYTHONPATH="$REPO/code/analysis:$REPO/code/trajectory:${PYTHONPATH:-}"

run () {  # run <dir> <script> [args]
  echo "== $2"
  python "$REPO/code/$1/$2.py" "${@:3}" > "$REPO/outputs/summaries/$2.txt" 2>&1 \
    || echo "   failed, see outputs/summaries/$2.txt"
}

# printed summaries (numbers quoted in the text and tables)
run analysis kras_baseline_all      # reference arm: Table 5, Table 6, Sec. 3.1-3.3
run analysis kras_cpu_sweep         # core-count sweep (Fig. 3b)
run analysis kras_hmr_benchmark     # accelerated-arm speed (Table 1, Fig. 3c,d)
run analysis kras_hmr_walltime      # wall time per chained job (Table 2)
run analysis kras_pme_mg            # PME order 4 Mg shell (Table 5)
run analysis kras_hmr_switch1       # HMR Mg shell and Switch I (Tables 5, 6)
run analysis kras_metad_summary     # metadynamics speed and Switch I angle (Sec. 3.4)
run analysis kras_biology --plot    # per-residue comparison (Fig. 6, outputs/tables/bio_*.csv)

# figures of the paper
run figures  fig2_overview          # Fig. 2
run analysis kras_main_figs         # Fig. 3 (simulation speed) and Fig. 5 (Switch I)
run analysis kras_fig4              # Fig. 4 (active site under HMR)
run analysis kras_shell_figs        # Figs. S1-S3, tableS_gtp_shell.csv (Tables S2, S3)
run analysis kras_ca_figs           # Figs. S4, S5 (C-alpha RMSD and RMSF)
run analysis kras_si_residue        # Figs. S6-S10 (per-residue profiles and time maps)

mv -f ./*.png ./*.pdf ./*.tiff ./*.eps "$REPO/outputs/figures/" 2>/dev/null
mv -f ./bio_*.csv ./tableS_*.csv "$REPO/outputs/tables/" 2>/dev/null
echo "figures -> outputs/figures, tables -> outputs/tables, printed summaries -> outputs/summaries"
