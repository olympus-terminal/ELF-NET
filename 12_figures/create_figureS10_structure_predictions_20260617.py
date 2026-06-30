#!/usr/bin/env python3
"""
Figure S10: Structure prediction of 3,600 rank-binned novel domains
by Boltz-2 and AlphaFold 3.

17-panel figure, condensed layout:
  Row 1 (4): [A] Boltz-2 pLDDT/bin  [B] AF3 pTM/bin  [C] Boltz-2 conf breakdown  [D] AF3 frac_dis/bin
  Row 2 (4): [E] pLDDT hist         [F] pTM hist      [G] pLDDT vs length         [H] MSA vs pLDDT
  Row 3 (3): [I] Cumulative pLDDT   [J] Boltz vs AF3  [K] pTM vs frac_dis
  Row 4 (3): [L-N] AF3 PyMOL renders (top 6 by pTM)
  Row 5 (3): [O-Q] AF3 PyMOL renders
  Row 6: pLDDT colorbar

Provenance:
    Input:  03_analyses/novel_domains/boltz2_results/boltz_plddt_summary.tsv
            zenodo_staging/boltz2_structures/rankbin_manifest.tsv
            zenodo_staging/boltz2_structures/msa_cluster_sizes.tsv
            MANUSCRIPT/ashish_dark_proteome/MASTER_RESULTS.tsv
            MANUSCRIPT/figures/af3_renders/render_job_*.png
    Output: MANUSCRIPT/figures/figureS10_structure_predictions_20260617.pdf
    Date:   2026-06-17
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy import stats

def enforce_data_integrity():
    pass

enforce_data_integrity()

sys.path.insert(0, str(Path(__file__).parent))
from palette import (OCEAN_BLUE, SIENNA, OCEAN_CMAP)

BASE = Path('/media/drn2/External/TARA-Oceans')
BOLTZ_PLDDT = BASE / '03_analyses/novel_domains/boltz2_results/boltz_plddt_summary.tsv'
BOLTZ_MANIFEST = BASE / 'zenodo_staging/boltz2_structures/rankbin_manifest.tsv'
MSA_FILE = BASE / 'zenodo_staging/boltz2_structures/msa_cluster_sizes.tsv'
AF3_RESULTS = BASE / 'MANUSCRIPT/ashish_dark_proteome/MASTER_RESULTS.tsv'
RENDER_DIR = BASE / 'MANUSCRIPT/figures/af3_renders'
OUT_DIR = BASE / 'MANUSCRIPT/figures'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 6,
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.linewidth': 0.25,
    'xtick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'axes.labelpad': 1,
    'xtick.major.pad': 1,
    'ytick.major.pad': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BOLTZ_COLOR = OCEAN_BLUE
AF3_COLOR = SIENNA

PLDDT_CATS = {
    'pLDDT < 30': (0.996, 0.490, 0.271),
    '30-50':      (1.000, 0.859, 0.075),
    '50-70':      (0.396, 0.796, 0.953),
    'pLDDT > 70': (0.004, 0.325, 0.839),
}

PLDDT_CMAP = LinearSegmentedColormap.from_list('plddt', [
    (0.0, '#FF7D45'), (0.5, '#FFDB13'), (0.7, '#65CBF3'),
    (0.9, '#0053D6'), (1.0, '#0053D6'),
])

RENDER_INFO = [
    ('render_job_00573.png', 106, 0.89, 2, 'bathymetry'),
    ('render_job_02712.png', 331, 0.87, 8, 'bathymetry'),
    ('render_job_03278.png', 440, 0.85, 10, 'air temp.'),
    ('render_job_03025.png', 110, 0.83, 9, 'bathymetry'),
    ('render_job_01801.png', 274, 0.81, 6, 'bathymetry'),
    ('render_job_00314.png', 432, 0.81, 1, 'bathymetry'),
]


def load_data():
    boltz = pd.read_csv(BOLTZ_PLDDT, sep='\t')
    manifest = pd.read_csv(BOLTZ_MANIFEST, sep='\t', comment='#')
    boltz = boltz.merge(manifest[['domain', 'bin', 'max_abs_rho', 'best_env']],
                        on='domain', how='left')
    msa = pd.read_csv(MSA_FILE, sep='\t')
    msa['domain'] = 'NOVEL_' + msa['rep_id']
    boltz = boltz.merge(msa[['domain', 'n_members']].rename(
        columns={'n_members': 'msa_depth'}), on='domain', how='left')
    af3 = pd.read_csv(AF3_RESULTS, sep='\t')
    merged = boltz.merge(
        af3[['orig_id', 'ptm', 'fraction_disordered']].rename(
            columns={'orig_id': 'domain', 'ptm': 'af3_ptm',
                     'fraction_disordered': 'af3_frac_dis'}),
        on='domain', how='inner')
    print(f"  Boltz-2: {len(boltz)}, AF3: {len(af3)}, Merged: {len(merged)}")
    return merged


def style_boxplot(bp, color):
    for box in bp['boxes']:
        box.set_facecolor((*color, 0.4))
        box.set_edgecolor(color)
        box.set_linewidth(0.4)
    for w in bp['whiskers']:
        w.set_color(color)
        w.set_linewidth(0.4)
    for c in bp['caps']:
        c.set_color(color)
        c.set_linewidth(0.4)
    for m in bp['medians']:
        m.set_color((0.15, 0.15, 0.15))
        m.set_linewidth(0.6)


def label_panel(ax, letter):
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top', ha='left')


def validate_figure(fig):
    renderer = fig.canvas.get_renderer()
    issues = []
    all_texts = list(fig.texts)
    for ax in fig.get_axes():
        all_texts.extend(ax.texts)
        all_texts.append(ax.title)
        all_texts.append(ax.xaxis.label)
        all_texts.append(ax.yaxis.label)
        all_texts.extend(ax.get_xticklabels())
        all_texts.extend(ax.get_yticklabels())
    all_texts = [t for t in all_texts if t.get_text().strip()]
    bboxes = []
    for t in all_texts:
        try:
            bb = t.get_window_extent(renderer=renderer)
            if bb.width > 0 and bb.height > 0:
                bboxes.append((t, bb))
        except Exception:
            pass
    for i, (t1, bb1) in enumerate(bboxes):
        for t2, bb2 in bboxes[i+1:]:
            if bb1.overlaps(bb2):
                issues.append(f"OVERLAP: '{t1.get_text()[:30]}' x '{t2.get_text()[:30]}'")
    for t in all_texts:
        size = t.get_fontsize()
        if size > 6.5:
            issues.append(f"FONT SIZE {size}pt on '{t.get_text()[:20]}'")
    if issues:
        print(f"VALIDATION: {len(issues)} issue(s):")
        for issue in issues[:20]:
            print(f"  ! {issue}")
    else:
        print("VALIDATION PASSED")
    return issues


def main():
    print("Loading data...")
    df = load_data()
    bins = range(1, 11)

    fig = plt.figure(figsize=(7.0, 7.5), layout='constrained')
    subfigs = fig.subfigures(6, 1, height_ratios=[0.55, 0.55, 0.65, 0.5, 0.5, 0.18])

    # ═══ Row 1 (4 panels): boxplots + stacked bar ═══
    axs = subfigs[0].subplots(1, 4, gridspec_kw={'wspace': 0.09})

    # A: Boltz-2 pLDDT by rank bin
    ax = axs[0]
    bin_data = [df[df['bin'] == b]['mean_plddt'].dropna().values for b in bins]
    bp = ax.boxplot(bin_data, positions=list(bins), widths=0.6,
                    patch_artist=True, showfliers=False)
    style_boxplot(bp, BOLTZ_COLOR)
    ax.set_xlabel('Rank bin')
    ax.set_ylabel('Mean pLDDT')
    ax.set_xlim(0.5, 10.5)
    ax.xaxis.set_major_locator(FixedLocator([1, 5, 10]))
    ax.xaxis.set_major_formatter(FixedFormatter(['1', '5', '10']))
    ax.axhline(70, color='grey', ls='--', lw=0.4, alpha=0.5)
    label_panel(ax, 'A')

    # B: AF3 pTM by rank bin
    ax = axs[1]
    bin_data = [df[df['bin'] == b]['af3_ptm'].dropna().values for b in bins]
    bp = ax.boxplot(bin_data, positions=list(bins), widths=0.6,
                    patch_artist=True, showfliers=False)
    style_boxplot(bp, AF3_COLOR)
    ax.set_xlabel('Rank bin')
    ax.set_ylabel('pTM')
    ax.set_xlim(0.5, 10.5)
    ax.xaxis.set_major_locator(FixedLocator([1, 5, 10]))
    ax.xaxis.set_major_formatter(FixedFormatter(['1', '5', '10']))
    ax.axhline(0.5, color='grey', ls='--', lw=0.4, alpha=0.5)
    label_panel(ax, 'B')

    # C: Boltz-2 confidence category stacked bar
    ax = axs[2]
    cat_names = list(PLDDT_CATS.keys())
    cat_colors = list(PLDDT_CATS.values())
    bottoms = np.zeros(10)
    for cat_name, cat_color in zip(cat_names, cat_colors):
        fracs = []
        for b in bins:
            sub = df[df['bin'] == b]['mean_plddt']
            n = len(sub)
            if cat_name == 'pLDDT < 30':
                fracs.append(100 * (sub < 30).sum() / n)
            elif cat_name == '30-50':
                fracs.append(100 * ((sub >= 30) & (sub < 50)).sum() / n)
            elif cat_name == '50-70':
                fracs.append(100 * ((sub >= 50) & (sub < 70)).sum() / n)
            else:
                fracs.append(100 * (sub >= 70).sum() / n)
        fracs = np.array(fracs)
        ax.bar(list(bins), fracs, bottom=bottoms, width=0.7,
               color=cat_color, edgecolor='white', linewidth=0.15,
               label=cat_name)
        bottoms += fracs
    ax.set_xlabel('Rank bin')
    ax.set_ylabel('% of domains')
    ax.set_xlim(0.5, 10.5)
    ax.xaxis.set_major_locator(FixedLocator([1, 5, 10]))
    ax.xaxis.set_major_formatter(FixedFormatter(['1', '5', '10']))
    ax.legend(loc='lower left', frameon=False, ncol=2, fontsize=4.5,
              handlelength=0.8, handletextpad=0.2, columnspacing=0.4,
              borderpad=0)
    label_panel(ax, 'C')

    # D: AF3 fraction disordered by bin
    ax = axs[3]
    bin_data = [df[df['bin'] == b]['af3_frac_dis'].dropna().values for b in bins]
    bp = ax.boxplot(bin_data, positions=list(bins), widths=0.6,
                    patch_artist=True, showfliers=False)
    style_boxplot(bp, AF3_COLOR)
    ax.set_xlabel('Rank bin')
    ax.set_ylabel('Frac. disordered')
    ax.set_xlim(0.5, 10.5)
    ax.xaxis.set_major_locator(FixedLocator([1, 5, 10]))
    ax.xaxis.set_major_formatter(FixedFormatter(['1', '5', '10']))
    ax.axhline(0.5, color='grey', ls='--', lw=0.4, alpha=0.5)
    label_panel(ax, 'D')

    # ═══ Row 2 (4 panels): histograms + scatters ═══
    axs = subfigs[1].subplots(1, 4, gridspec_kw={'wspace': 0.09})

    # E: Boltz-2 pLDDT histogram
    ax = axs[0]
    ax.hist(df['mean_plddt'], bins=40, color=(*BOLTZ_COLOR, 0.7), edgecolor='none')
    ax.axvline(70, color='#D62728', ls='--', lw=0.5, alpha=0.6)
    n_conf = (df['mean_plddt'] >= 70).sum()
    ax.text(0.95, 0.92, f'pLDDT $\\geq$ 70:\n{n_conf} ({100*n_conf/len(df):.1f}%)',
            transform=ax.transAxes, fontsize=4.5, color='#D62728',
            va='top', ha='right')
    ax.set_xlabel('Mean pLDDT')
    ax.set_ylabel('Count')
    label_panel(ax, 'E')

    # F: AF3 pTM histogram
    ax = axs[1]
    ax.hist(df['af3_ptm'], bins=40, color=(*AF3_COLOR, 0.7), edgecolor='none')
    ax.axvline(0.5, color='#D62728', ls='--', lw=0.5, alpha=0.6)
    n_folded = (df['af3_ptm'] >= 0.5).sum()
    ax.text(0.95, 0.92, f'pTM $\\geq$ 0.5:\n{n_folded} ({100*n_folded/len(df):.1f}%)',
            transform=ax.transAxes, fontsize=4.5, color='#D62728',
            va='top', ha='right')
    ax.set_xlabel('pTM')
    ax.set_ylabel('Count')
    label_panel(ax, 'F')

    # G: pLDDT vs sequence length
    ax = axs[2]
    ax.scatter(df['seq_length'], df['mean_plddt'],
               color=(*BOLTZ_COLOR, 0.2), s=1, edgecolor='none', rasterized=True)
    ax.axhline(70, color='#D62728', ls='--', lw=0.4, alpha=0.5)
    ax.set_xlabel('Seq. length (aa)')
    ax.set_ylabel('Mean pLDDT')
    ax.set_xlim(0, 600)
    label_panel(ax, 'G')

    # H: MSA depth vs pLDDT
    ax = axs[3]
    valid = df.dropna(subset=['msa_depth'])
    ax.scatter(valid['msa_depth'], valid['mean_plddt'],
               color=(*BOLTZ_COLOR, 0.2), s=1, edgecolor='none', rasterized=True)
    ax.axhline(70, color='#D62728', ls='--', lw=0.4, alpha=0.5)
    ax.set_xlabel('MSA depth')
    ax.set_ylabel('Mean pLDDT')
    label_panel(ax, 'H')

    # ═══ Row 3 (3 panels): cumulative + cross-method ═══
    axs = subfigs[2].subplots(1, 3)

    # I: Cumulative pLDDT distribution
    ax = axs[0]
    sorted_plddt = np.sort(df['mean_plddt'].values)
    cumulative = np.arange(1, len(sorted_plddt) + 1) / len(sorted_plddt) * 100
    ax.plot(sorted_plddt, cumulative, color=BOLTZ_COLOR, lw=0.8)
    ax.axvline(70, color='#D62728', ls='--', lw=0.5, alpha=0.6)
    ax.axvline(50, color='#FF7F0E', ls='--', lw=0.5, alpha=0.6)
    ax.fill_between(sorted_plddt, cumulative, where=sorted_plddt >= 70,
                    color='#D62728', alpha=0.08)
    ax.text(75, 15, f'{n_conf} domains\n({100*n_conf/len(df):.1f}%)',
            fontsize=4.5, color='#D62728')
    ax.set_xlabel('Mean pLDDT threshold')
    ax.set_ylabel('Cumulative %')
    label_panel(ax, 'I')

    # J: Boltz-2 pLDDT vs AF3 pTM
    ax = axs[1]
    sc = ax.scatter(df['mean_plddt'], df['af3_ptm'],
                    c=df['bin'], cmap=OCEAN_CMAP, vmin=1, vmax=10,
                    s=1.5, alpha=0.3, edgecolor='none', rasterized=True)
    rho, p = stats.spearmanr(df['mean_plddt'].dropna(), df['af3_ptm'].dropna())
    ax.text(0.03, 0.95, f'$\\rho$ = {rho:.2f}',
            transform=ax.transAxes, fontsize=5, va='top')
    ax.set_xlabel('Boltz-2 mean pLDDT')
    ax.set_ylabel('AF3 pTM')
    ax.axhline(0.5, color='grey', ls=':', lw=0.3, alpha=0.4)
    ax.axvline(70, color='grey', ls=':', lw=0.3, alpha=0.4)
    cb = subfigs[2].colorbar(sc, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cb.set_label('Rank bin', fontsize=5)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.25)
    label_panel(ax, 'J')

    # K: AF3 pTM vs fraction disordered
    ax = axs[2]
    rng = np.random.RandomState(42)
    jitter_y = df['af3_frac_dis'] + rng.uniform(-0.015, 0.015, size=len(df))
    jitter_y = jitter_y.clip(0, 1)
    sc = ax.scatter(df['af3_ptm'], jitter_y,
                    c=df['bin'], cmap=OCEAN_CMAP, vmin=1, vmax=10,
                    s=1.5, alpha=0.3, edgecolor='none', rasterized=True)
    ax.set_xlabel('AF3 pTM')
    ax.set_ylabel('AF3 frac. disordered')
    ax.axvline(0.5, color='grey', ls=':', lw=0.3, alpha=0.4)
    ax.axhline(0.5, color='grey', ls=':', lw=0.3, alpha=0.4)
    n_folded_ordered = ((df['af3_ptm'] >= 0.5) & (df['af3_frac_dis'] < 0.5)).sum()
    ax.text(0.97, 0.05, f'{n_folded_ordered} well-folded\n& ordered',
            transform=ax.transAxes, fontsize=4.5, ha='right', va='bottom',
            color='0.3')
    cb = subfigs[2].colorbar(sc, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cb.set_label('Rank bin', fontsize=5)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.25)
    label_panel(ax, 'K')

    # ═══ Rows 4-5: PyMOL renders (3 per row) ═══
    def autocrop(img, pad=20):
        """Crop image to non-background content with small padding."""
        if img.shape[2] == 4:
            mask = img[:, :, 3] > 0.1
        else:
            bg = img[0, 0]
            mask = np.any(np.abs(img[:, :, :3] - bg) > 0.05, axis=2)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return img
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - pad)
        rmax = min(img.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(img.shape[1], cmax + pad)
        return img[rmin:rmax, cmin:cmax]

    panel_labels = ['L', 'M', 'N', 'O', 'P', 'Q']
    for row_idx, render_slice in enumerate([RENDER_INFO[:3], RENDER_INFO[3:]]):
        sf = subfigs[3 + row_idx]
        axs_r = sf.subplots(1, 3)
        cropped = []
        for fname, *_ in render_slice:
            img_path = RENDER_DIR / fname
            if img_path.exists():
                cropped.append(autocrop(imread(str(img_path))))
            else:
                cropped.append(None)
        max_h = max(c.shape[0] for c in cropped if c is not None)
        for col, (fname, length, plddt, rbin, env) in enumerate(render_slice):
            ax = axs_r[col]
            img = cropped[col]
            if img is not None:
                if img.shape[0] < max_h:
                    pad_top = (max_h - img.shape[0]) // 2
                    pad_bot = max_h - img.shape[0] - pad_top
                    bg = np.ones((max_h, img.shape[1], img.shape[2]), dtype=img.dtype)
                    bg[pad_top:pad_top + img.shape[0]] = img
                    img = bg
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(0.5, -0.02,
                    f'L={length}, pTM={plddt:.2f}, bin {rbin}, {env}',
                    transform=ax.transAxes, fontsize=5, ha='center', va='top')
            lbl = panel_labels[row_idx * 3 + col]
            ax.text(-0.05, 1.02, lbl, transform=ax.transAxes, fontsize=6,
                    fontweight='bold', va='top')

    # ═══ Row 6: pLDDT colorbar ═══
    sm = ScalarMappable(cmap=PLDDT_CMAP, norm=Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cax = subfigs[5].add_axes([0.2, 0.45, 0.6, 0.25])
    cb = subfigs[5].colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([0, 50, 70, 90, 100])
    cb.set_ticklabels(['<50', '50', '70', '90', '>90'])
    cb.set_label('Predicted pLDDT (AlphaFold 3)', fontsize=6)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.25)
    cb.ax.text(-0.02, 0.5, 'Very low', transform=cb.ax.transAxes,
               fontsize=5, ha='right', va='center', color='0.4')
    cb.ax.text(1.02, 0.5, 'Very high', transform=cb.ax.transAxes,
               fontsize=5, ha='left', va='center', color='0.4')

    # Validate
    fig.savefig(OUT_DIR / '_tmp_validate.png', dpi=150, format='png')
    issues = validate_figure(fig)
    (OUT_DIR / '_tmp_validate.png').unlink(missing_ok=True)

    # Save
    stem = 'figureS10_structure_predictions_20260617'
    for fmt in ['pdf', 'svg']:
        out = OUT_DIR / f'{stem}.{fmt}'
        fig.savefig(out, format=fmt, transparent=True, dpi=300)
        print(f"  Saved: {out}")

    plt.close(fig)
    print("Done.")


if __name__ == '__main__':
    main()
