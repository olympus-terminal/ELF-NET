#!/usr/bin/env python3
"""
Batch PyMOL headless rendering of 138 well-folded AF3 dark proteome domains.
Colors by per-residue pLDDT (B-factor in CIF). White background, ray-traced.

Input:  figures/af3_renders/cif/job_XXXXX.cif
Output: figures/af3_renders/render_job_XXXXX.png
"""

import subprocess
import tempfile
from pathlib import Path

PYMOL = '/home/drn2/miniconda3/envs/pymol-env/bin/pymol'
CIF_DIR = Path('/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures/af3_renders/cif')
OUT_DIR = Path('/media/drn2/External/TARA-Oceans/MANUSCRIPT/figures/af3_renders')

OUT_DIR.mkdir(parents=True, exist_ok=True)

cif_files = sorted(CIF_DIR.glob('job_*.cif'))
print(f"Found {len(cif_files)} CIF files to render")

for i, cif in enumerate(cif_files):
    job_id = cif.stem
    out_png = OUT_DIR / f'render_{job_id}.png'

    if out_png.exists():
        print(f"  [{i+1}/{len(cif_files)}] {job_id} — already rendered, skipping")
        continue

    pml_script = f"""load {cif}, protein
remove solvent
hide everything
show cartoon, protein
spectrum b, red_white_blue, protein, 0, 100
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
set cartoon_loop_radius, 0.3
bg_color white
set ray_opaque_background, 1
set antialias, 2
set ray_shadows, 0
orient protein
zoom protein, 3
ray 800, 600
png {out_png}, dpi=150
quit
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pml', delete=False) as f:
        f.write(pml_script)
        pml_path = f.name

    result = subprocess.run(
        [PYMOL, '-c', '-Q', pml_path],
        capture_output=True,
        text=True,
        timeout=120
    )

    Path(pml_path).unlink(missing_ok=True)

    if out_png.exists():
        print(f"  [{i+1}/{len(cif_files)}] {job_id} — rendered ({out_png.stat().st_size // 1024} KB)")
    else:
        print(f"  [{i+1}/{len(cif_files)}] {job_id} — FAILED")
        if result.stderr:
            print(f"    stderr: {result.stderr[:300]}")
        if result.stdout:
            print(f"    stdout: {result.stdout[:300]}")

rendered = list(OUT_DIR.glob('render_job_*.png'))
print(f"\nDone: {len(rendered)}/{len(cif_files)} renders completed")
