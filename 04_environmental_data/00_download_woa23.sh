#!/usr/bin/env bash
# ==============================================================================
# 00_download_woa23.sh — Download WOA23 + MLD climatology NetCDFs
#
# Downloads surface-layer annual climatology for:
#   - Nitrate (NO₃)      — WOA23
#   - Phosphate (PO₄)    — WOA23
#   - Silicate (SiO₄)    — WOA23
#   - Dissolved oxygen    — WOA23
#   - Mixed layer depth   — de Boyer Montégut climatology
#
# All files are 1°×1° resolution, annual mean, surface (0 m) depth.
#
# Usage:
#   bash scripts/nutrients/00_download_woa23.sh [--hpc]
#
# With --hpc flag, downloads to /scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/woa23_nutrients/
# Without flag, downloads to local data/ directory for development.
#
# Author: TARA-OMEN analysis pipeline
# Date: 2026-03-20
# ==============================================================================

set -euo pipefail

# --- Destination directory ---
if [[ "${1:-}" == "--hpc" ]]; then
    OUTDIR="/scratch/drn2/PROJECTS/TARA-LA4SR/03_analyses/woa23_nutrients"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    OUTDIR="${SCRIPT_DIR}/../../data/woa23_nutrients"
fi

mkdir -p "${OUTDIR}"
echo "Download directory: ${OUTDIR}"

# ==============================================================================
# WOA23 — NCEI THREDDS server
# URL pattern for 1° objectively analyzed annual climatology (decav = decadal average):
#   https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA/{var}/netcdf/{decade}/{res}/
#
# Variable codes:
#   n = nitrate, p = phosphate, i = silicate, o = dissolved oxygen
# File naming:
#   woa23_{decade}_{var}{period}_{res}.nc
#   decade = decav91C0 (1991-2020 climatology, latest)
#   period = 00 (annual)
#   res    = 01 (1° grid)
# ==============================================================================

WOA23_BASE="https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA"
# WOA23 uses "all" for the full climatology period; period 00 = annual mean
PERIOD="all"

declare -A WOA_VARS
WOA_VARS[nitrate]="n"
WOA_VARS[phosphate]="p"
WOA_VARS[silicate]="i"
WOA_VARS[oxygen]="o"

for varname in nitrate phosphate silicate oxygen; do
    code="${WOA_VARS[$varname]}"
    filename="woa23_${PERIOD}_${code}00_01.nc"
    url="${WOA23_BASE}/${varname}/netcdf/${PERIOD}/1.00/${filename}"
    outpath="${OUTDIR}/${filename}"

    if [[ -f "${outpath}" ]]; then
        echo "[SKIP] ${varname}: ${outpath} already exists"
    else
        echo "[DOWNLOAD] ${varname}: ${url}"
        wget -q --show-progress -O "${outpath}" "${url}" || {
            echo "[FAIL] Could not download ${varname}"
            rm -f "${outpath}"
        }
    fi
done

# ==============================================================================
# Mixed Layer Depth — de Boyer Montégut climatology (SEANOE)
#
# Ref: de Boyer Montégut et al. (2004), updated dataset hosted at SEANOE
# DOI: https://doi.org/10.17882/91774
# Criterion: density threshold (0.03 kg/m³ from 10m reference)
# ==============================================================================

# MLD data is distributed as tar archives on SEANOE
MLD_TAR_URL="https://www.seanoe.org/data/00806/91774/data/103667.tar"
MLD_TAR="${OUTDIR}/mld_archive.tar"
MLD_DIR="${OUTDIR}/mld_extracted"

# Check if we already have any MLD NetCDF
MLD_NC=$(ls "${OUTDIR}"/mld*.nc "${MLD_DIR}"/*.nc 2>/dev/null | head -1)

if [[ -n "${MLD_NC}" ]]; then
    echo "[SKIP] MLD: ${MLD_NC} already exists"
else
    echo "[DOWNLOAD] MLD climatology archive: ${MLD_TAR_URL}"
    wget -q --show-progress -O "${MLD_TAR}" "${MLD_TAR_URL}" || {
        # Try alternate file ID
        alt_url="https://www.seanoe.org/data/00806/91774/data/97836.tar"
        echo "[RETRY] Trying alternate: ${alt_url}"
        wget -q --show-progress -O "${MLD_TAR}" "${alt_url}" || {
            echo "[WARN] MLD download failed. Manual download required."
            echo "       Visit: https://doi.org/10.17882/91774"
            echo "       Download the tar archive and extract to: ${OUTDIR}/"
            rm -f "${MLD_TAR}"
        }
    }

    if [[ -f "${MLD_TAR}" ]]; then
        echo "[EXTRACT] Extracting MLD archive..."
        mkdir -p "${MLD_DIR}"
        tar xf "${MLD_TAR}" -C "${MLD_DIR}"
        # Move any .nc files to main directory
        find "${MLD_DIR}" -name "*.nc" -exec cp {} "${OUTDIR}/" \;
        echo "  Extracted files:"
        ls -lh "${OUTDIR}"/mld*.nc "${MLD_DIR}"/*.nc 2>/dev/null
        rm -f "${MLD_TAR}"
    fi
fi

# ==============================================================================
# Verify downloads
# ==============================================================================

echo ""
echo "=== Download Summary ==="
for f in "${OUTDIR}"/*.nc; do
    if [[ -f "$f" ]]; then
        size=$(du -h "$f" | cut -f1)
        echo "  ${size}  $(basename "$f")"
    fi
done

n_files=$(ls -1 "${OUTDIR}"/*.nc 2>/dev/null | wc -l)
echo ""
echo "Total files: ${n_files} / 5 expected"

if [[ ${n_files} -lt 5 ]]; then
    echo "[WARN] Some downloads may have failed. Check above for errors."
    echo "       WOA23 files can be manually downloaded from:"
    echo "       https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/"
fi
