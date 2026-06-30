"""
post_training_analysis.py
Comprehensive post-training analysis for dark-proteome GPT-2 training dynamics.
Produces publication-quality figures, statistical tables, and manuscript text.
"""

import os
import re
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.optimize import curve_fit
from scipy.stats import (
    mannwhitneyu, wilcoxon, bootstrap, ttest_ind, pearsonr
)
from scipy.ndimage import uniform_filter1d
import itertools
warnings.filterwarnings('ignore')

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette (colour-blind friendly) ──────────────────────────────────────────
PALETTE = {
    'White':              '#1a6faf',   # deep blue
    'Algae':              '#2ca02c',   # green
    'Dark':               '#7b3f9e',   # purple
    'Random Algae':       '#e07b00',   # amber
    'Random Full':        '#b22222',   # brick red
}
LINESTYLES = {
    'White':          '-',
    'Algae':          '-',
    'Dark':           '-',
    'Random Algae':   '--',
    'Random Full':    '--',
}
MARKERS = {
    'White':          'o',
    'Algae':          's',
    'Dark':           '^',
    'Random Algae':   'D',
    'Random Full':    'X',
}
ORDER = ['White', 'Algae', 'Dark', 'Random Algae', 'Random Full']

# ── Matplotlib defaults ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.85,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linestyle':     ':',
})

BASE = '/home/sd145/Desktop/dark_proteome'

# ── CSV loader ────────────────────────────────────────────────────────────────
def load_csv_metric(folder, metric_key):
    """Load a W&B export CSV whose column contains metric_key. Returns (steps, values)."""
    csvs = sorted(glob.glob(os.path.join(folder, 'wandb_export*.csv')))
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        cols = [c for c in df.columns if metric_key in c and '__' not in c]
        if cols:
            df = df.dropna(subset=[cols[0]])
            steps = df['Step'].astype(int).values
            vals  = df[cols[0]].astype(float).values
            return steps, vals
    return None, None


def load_dataset(name, folder):
    """Return dict with train/val steps & values for one run."""
    _, train = load_csv_metric(folder, 'train/loss')
    _, val   = load_csv_metric(folder, 'val/loss')
    # iter → actual training iteration (1000 iters/step)
    steps = np.arange(len(train)) * 1000
    return {'name': name, 'iters': steps, 'train': train, 'val': val}


# ── Load all runs ─────────────────────────────────────────────────────────────
RUNS = {
    'White':        load_dataset('White',        f'{BASE}/White_run'),
    'Algae':        load_dataset('Algae',         f'{BASE}/algae_run'),
    'Dark':         load_dataset('Dark',          f'{BASE}/dark_run'),
    'Random Algae': load_dataset('Random Algae',  f'{BASE}/random_algae_run'),
    'Random Full':  load_dataset('Random Full',   f'{BASE}/random_full_run'),
}

# Confirmed final values (from run logs)
FINAL = {
    'White':        {'train': 2.478922, 'val': 2.653463},
    'Algae':        {'train': 2.706709, 'val': 2.717807},
    'Dark':         {'train': 2.723781, 'val': 2.754862},
    'Random Algae': {'train': 2.853532, 'val': 2.854525},
    'Random Full':  {'train': 2.992918, 'val': 2.993169},
}

VOCAB = {
    'White': 23, 'Algae': 23, 'Dark': 23, 'Random Algae': 22, 'Random Full': 22
}

DATASET_SIZES_TRAIN = {
    'White': 66_522_733, 'Algae': 58_875_807, 'Dark': 63_134_898,
    'Random Algae': 55_363_218, 'Random Full': 55_363_218,
}

# ── Theoretical entropy floor ─────────────────────────────────────────────────
# Random Full: 20 AA sampled uniformly → H = ln(20) ≈ 2.996 nats
# (model also needs to predict '\n' and '>', so effective floor is marginally lower)
ENTROPY_FLOOR = np.log(20)  # ≈ 2.996 nats

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – Training curve reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for split, ax, title in [('train', axes[0], 'Training Loss'),
                               ('val',   axes[1], 'Validation Loss')]:
        for name in ORDER:
            d = RUNS[name]
            ax.plot(d['iters'] / 1000, d[split],
                    color=PALETTE[name], lw=2,
                    ls=LINESTYLES[name],
                    label=name, zorder=3)
        ax.axhline(ENTROPY_FLOOR, color='grey', ls=':', lw=1.5,
                   label=r'$H_{\rm unif}$ = ln(20) ≈ 2.996')
        ax.set_xlabel('Training Iterations (×1 000)')
        ax.set_ylabel('Cross-Entropy Loss (nats)')
        ax.set_title(title)
        ax.set_xlim(0, 40)
        ax.set_ylim(2.4, 3.25)
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    # shared legend on first axis
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='upper right', framealpha=0.9)
    axes[1].legend(handles, labels, loc='upper right', framealpha=0.9)

    fig.suptitle('GPT-2 Training Dynamics on Protein-Sequence Datasets',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = f'{OUT_DIR}/fig1_training_curves.pdf'
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – Convergence-rate analysis
# ══════════════════════════════════════════════════════════════════════════════

def exp_decay(x, a, b, c):
    """L(t) = a·exp(-b·t) + c"""
    return a * np.exp(-b * x) + c


def fit_convergence(iters, loss):
    """Fit exponential decay; return half-life (iters), rate, final asymptote."""
    x = iters.astype(float) / 1000          # in kilo-iters for numerical stability
    try:
        p0 = [loss[0] - loss[-1], 0.05, loss[-1]]
        popt, pcov = curve_fit(exp_decay, x, loss, p0=p0,
                               bounds=([0, 1e-6, 1.5], [3, 10, 3.5]),
                               maxfev=20000)
        a, b, c = popt
        half_life_kiters = np.log(2) / b   # in kilo-iters
        r2 = 1 - np.sum((loss - exp_decay(x, *popt))**2) / \
                   np.sum((loss - loss.mean())**2)
        perr = np.sqrt(np.diag(pcov))
        return {
            'a': a, 'b': b, 'c': c,
            'half_life_kiters': half_life_kiters,
            'asymptote': c,
            'r2': r2,
            'perr_b': perr[1],
        }
    except Exception:
        return None


def compute_convergence_table():
    rows = []
    for name in ORDER:
        d = RUNS[name]
        fit = fit_convergence(d['iters'], d['val'])
        initial = d['val'][0]
        final   = FINAL[name]['val']
        total_drop = initial - final
        pct_drop = 100 * total_drop / initial

        # Early-phase rate: drop in first 10k iters
        mask_10k = d['iters'] <= 10000
        early_drop = d['val'][mask_10k][0] - d['val'][mask_10k][-1]

        row = {
            'Dataset':         name,
            'Initial Val Loss': round(initial, 4),
            'Final Val Loss':   round(final, 4),
            'Total Drop':       round(total_drop, 4),
            'Drop (%)':         round(pct_drop, 2),
            '10k-iter Drop':    round(early_drop, 4),
        }
        if fit:
            row.update({
                'Half-life (kiters)': round(fit['half_life_kiters'], 2),
                'Asymptote':          round(fit['asymptote'], 4),
                'Exp-fit R²':         round(fit['r2'], 4),
            })
        else:
            row.update({'Half-life (kiters)': 'N/A', 'Asymptote': 'N/A', 'Exp-fit R²': 'N/A'})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_convergence_rates(conv_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A – Val loss with exponential fits
    ax = axes[0]
    for name in ORDER:
        d = RUNS[name]
        x = d['iters'].astype(float) / 1000
        ax.plot(x, d['val'], color=PALETTE[name], lw=1.5, alpha=0.6,
                ls=LINESTYLES[name])
        # overlay fit
        fit = fit_convergence(d['iters'], d['val'])
        if fit:
            x_fine = np.linspace(0, 40, 400)
            y_fit  = exp_decay(x_fine, fit['a'], fit['b'], fit['c'])
            ax.plot(x_fine, y_fit, color=PALETTE[name], lw=2.5,
                    ls=LINESTYLES[name], label=f"{name}")

    ax.axhline(ENTROPY_FLOOR, color='grey', ls=':', lw=1.5)
    ax.set_xlabel('Training Iterations (×1 000)')
    ax.set_ylabel('Validation Loss (nats)')
    ax.set_title('Validation Loss with Exponential Fits')
    ax.set_xlim(0, 40); ax.set_ylim(2.4, 3.25)
    ax.legend(loc='upper right')

    # Panel B – Half-life bar plot
    ax = axes[1]
    hl_data = conv_df[conv_df['Half-life (kiters)'] != 'N/A']
    colors   = [PALETTE[n] for n in hl_data['Dataset']]
    bars     = ax.bar(hl_data['Dataset'], hl_data['Half-life (kiters)'].astype(float),
                       color=colors, edgecolor='black', linewidth=0.7, alpha=0.85)
    ax.set_ylabel('Convergence Half-life (×1 000 iters)')
    ax.set_title('Exponential Convergence Half-life')
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, hl_data['Half-life (kiters)'].astype(float)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.1f}k', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    path = f'{OUT_DIR}/fig2_convergence_rates.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')
    return conv_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Train / Val gap  (generalisation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_gap_table():
    rows = []
    for name in ORDER:
        d = RUNS[name]
        gap_curve = d['val'] - d['train']
        final_gap = FINAL[name]['val'] - FINAL[name]['train']
        rows.append({
            'Dataset':        name,
            'Train Loss':     round(FINAL[name]['train'], 6),
            'Val Loss':       round(FINAL[name]['val'],   6),
            'Gap (val−train)':round(final_gap, 6),
            'Mean Gap':       round(gap_curve.mean(), 6),
            'Max Gap':        round(gap_curve.max(),  6),
        })
    return pd.DataFrame(rows)


def plot_gap_curves():
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ORDER:
        d = RUNS[name]
        gap = d['val'] - d['train']
        ax.plot(d['iters'] / 1000, gap, color=PALETTE[name], lw=2,
                ls=LINESTYLES[name], label=name)
    ax.axhline(0, color='k', lw=0.8, ls='-')
    ax.set_xlabel('Training Iterations (×1 000)')
    ax.set_ylabel('Val − Train Loss (nats)')
    ax.set_title('Train/Validation Generalisation Gap')
    ax.set_xlim(0, 40)
    ax.legend(loc='upper right')
    fig.tight_layout()
    path = f'{OUT_DIR}/fig3_generalisation_gap.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Statistical comparisons (bootstrap + Mann–Whitney U)
# ══════════════════════════════════════════════════════════════════════════════

def cohens_d(a, b):
    """Cohen's d between two 1-D arrays."""
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan


def stable_window(run, last_n=10):
    """Return last_n val-loss values (convergence window)."""
    return run['val'][-last_n:]


COMPARISONS = [
    ('White',        'Dark'),
    ('Dark',         'Random Full'),
    ('Dark',         'Random Algae'),
    ('Algae',        'Random Algae'),
    ('White',        'Random Full'),
    ('White',        'Algae'),
]


def compute_stat_table():
    rows = []
    for nameA, nameB in COMPARISONS:
        a = stable_window(RUNS[nameA])
        b = stable_window(RUNS[nameB])
        diff_final = FINAL[nameA]['val'] - FINAL[nameB]['val']

        # Mann-Whitney U on stable-window values
        U, p_mw = mannwhitneyu(a, b, alternative='less')

        # Bootstrap CI on the mean difference
        def mean_diff(x, y, axis=-1):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        stat_obs = a.mean() - b.mean()
        res = bootstrap(
            (a, b), statistic=mean_diff,
            paired=False, n_resamples=9999,
            random_state=42, confidence_level=0.95,
            method='percentile'
        )
        ci_lo, ci_hi = res.confidence_interval

        d = cohens_d(b, a)   # positive d → nameA < nameB (nameA lower loss)

        rows.append({
            'Comparison':    f'{nameA} vs {nameB}',
            'ΔVal Loss':     round(diff_final, 6),
            'Mean Δ (last 10)': round(stat_obs, 6),
            'CI 95% lo':     round(ci_lo, 6),
            'CI 95% hi':     round(ci_hi, 6),
            "Cohen's d":     round(d, 3),
            'MWU p-value':   f'{p_mw:.4f}',
            'Significant':   'Yes' if p_mw < 0.05 else 'No',
        })
    return pd.DataFrame(rows)


def plot_statistical_comparisons(stat_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A – Effect sizes (Cohen's d)
    ax = axes[0]
    ax.barh(stat_df['Comparison'],
            stat_df["Cohen's d"].abs(),
            color=['#2ecc71' if p == 'Yes' else '#e74c3c'
                   for p in stat_df['Significant']],
            edgecolor='black', linewidth=0.6, alpha=0.85)
    ax.axvline(0.5, color='grey', ls='--', lw=1, label='medium (d=0.5)')
    ax.axvline(0.8, color='grey', ls='-',  lw=1, label='large (d=0.8)')
    ax.set_xlabel("|Cohen's d|")
    ax.set_title("Effect Sizes (|Cohen's d|)")
    ax.legend(fontsize=8)
    # colour legend
    patches = [mpatches.Patch(color='#2ecc71', label='p < 0.05'),
               mpatches.Patch(color='#e74c3c', label='p ≥ 0.05')]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0][2:],
              fontsize=8, loc='lower right')

    # Panel B – ΔVal Loss with CI
    ax = axes[1]
    y_pos = np.arange(len(stat_df))
    ax.barh(y_pos, stat_df['ΔVal Loss'], height=0.5,
            color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.6)
    lo_err = stat_df['ΔVal Loss'] - stat_df['CI 95% lo']
    hi_err = stat_df['CI 95% hi'] - stat_df['ΔVal Loss']
    ax.errorbar(stat_df['ΔVal Loss'], y_pos,
                xerr=[lo_err.abs(), hi_err.abs()],
                fmt='none', color='black', capsize=4, lw=1.5)
    ax.set_yticks(y_pos); ax.set_yticklabels(stat_df['Comparison'])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('ΔVal Loss (Row A − Row B) [nats]')
    ax.set_title('Final Validation Loss Differences (95% Bootstrap CI)')

    fig.tight_layout()
    path = f'{OUT_DIR}/fig4_statistical_comparisons.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Learnability scores & hierarchy figure
# ══════════════════════════════════════════════════════════════════════════════

def compute_learnability(final_vals):
    """
    Learnability score:  L(D) = (H_floor – Val_D) / (H_floor – Val_best) × 100
    where H_floor = Random Full val loss (empirical floor) and Val_best = White.
    Also report excess over floor as raw nats and bits.
    """
    floor = FINAL['Random Full']['val']
    best  = FINAL['White']['val']
    rows  = []
    for name in ORDER:
        v = FINAL[name]['val']
        score_normed   = 100 * (floor - v) / (floor - best)
        excess_nats    = floor - v
        excess_bits    = excess_nats / np.log(2)
        perplexity     = np.exp(v)
        rows.append({
            'Dataset':            name,
            'Val Loss (nats)':    round(v, 6),
            'Perplexity':         round(perplexity, 4),
            'Excess over Floor (nats)': round(excess_nats, 6),
            'Excess over Floor (bits)': round(excess_bits, 6),
            'Learnability Score (%)':   round(score_normed, 2),
        })
    return pd.DataFrame(rows)


def plot_learnability_hierarchy(learn_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A – Learnability score bar chart
    ax = axes[0]
    colors = [PALETTE[n] for n in learn_df['Dataset']]
    bars = ax.bar(learn_df['Dataset'], learn_df['Learnability Score (%)'],
                  color=colors, edgecolor='black', linewidth=0.8, alpha=0.87)
    for bar, val in zip(bars, learn_df['Learnability Score (%)']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_ylabel('Learnability Score (%)\n(0% = random floor, 100% = white)')
    ax.set_title('Relative Learnability Hierarchy')
    ax.set_ylim(0, 115)
    ax.axhline(100, color='#1a6faf', ls='--', lw=1, alpha=0.5, label='White (ceiling)')
    ax.axhline(0,   color='#b22222', ls='--', lw=1, alpha=0.5, label='Random Full (floor)')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=10)

    # Panel B – Gradient plot (ranking arrow)
    ax = axes[1]
    val_losses = [FINAL[n]['val'] for n in ORDER]
    y_pos      = list(range(len(ORDER)))
    ax.scatter(val_losses, y_pos,
               c=[PALETTE[n] for n in ORDER],
               s=150, zorder=5, edgecolors='black', linewidths=0.8)
    # Connecting line
    ax.plot(val_losses, y_pos, color='grey', lw=1, zorder=2, alpha=0.5)
    # Shaded floor & ceiling
    ax.axvline(FINAL['Random Full']['val'], color='#b22222',
               ls='--', lw=1.2, alpha=0.6, label='Random Full (floor)')
    ax.axvline(ENTROPY_FLOOR, color='grey', ls=':', lw=1.2,
               label=r'$\ln(20)$ theoretical floor')
    for i, name in enumerate(ORDER):
        ax.annotate(f"  {name}\n  {FINAL[name]['val']:.4f}",
                    xy=(FINAL[name]['val'], i),
                    xytext=(5, 0), textcoords='offset points',
                    va='center', fontsize=8.5, color=PALETTE[name])
    ax.set_yticks(y_pos); ax.set_yticklabels([])
    ax.set_xlabel('Final Validation Loss (nats)')
    ax.set_title('Learnability Gradient:\nWhite → Algae → Dark → Random Algae → Random Full')
    ax.legend(fontsize=8, loc='lower right')
    ax.invert_xaxis()   # lower loss → right = "more learnable"
    ax.set_xlim(3.05, 2.55)

    fig.tight_layout()
    path = f'{OUT_DIR}/fig5_learnability_hierarchy.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – Dark vs Random significance test
# ══════════════════════════════════════════════════════════════════════════════

def dark_vs_random_test():
    dark_curve = RUNS['Dark']['val']
    rand_curve = RUNS['Random Full']['val']
    rand_a_curve = RUNS['Random Algae']['val']

    # Over full training (all 41 steps, note autocorrelation)
    u1, p1 = mannwhitneyu(dark_curve, rand_curve, alternative='less')
    u2, p2 = mannwhitneyu(dark_curve, rand_a_curve, alternative='less')

    # Stable-window (last 10 steps)
    dark_w  = dark_curve[-10:]
    rf_w    = rand_curve[-10:]
    ra_w    = rand_a_curve[-10:]
    u3, p3  = mannwhitneyu(dark_w, rf_w,  alternative='less')
    u4, p4  = mannwhitneyu(dark_w, ra_w,  alternative='less')

    # Effect sizes
    d_rf = cohens_d(rand_curve[-10:], dark_w)
    d_ra = cohens_d(rand_a_curve[-10:], dark_w)

    # Perplexity differences
    ppx_dark = np.exp(FINAL['Dark']['val'])
    ppx_rf   = np.exp(FINAL['Random Full']['val'])
    ppx_ra   = np.exp(FINAL['Random Algae']['val'])

    print("\n── Dark vs. Random Controls ──────────────────────────────────────────")
    print(f"  Dark final val loss:              {FINAL['Dark']['val']:.6f}")
    print(f"  Random Full final val loss:        {FINAL['Random Full']['val']:.6f}")
    print(f"  Random Algae final val loss:       {FINAL['Random Algae']['val']:.6f}")
    print(f"  Dark below Random Full:            {FINAL['Random Full']['val'] - FINAL['Dark']['val']:.6f} nats")
    print(f"  Dark below Random Algae:           {FINAL['Random Algae']['val'] - FINAL['Dark']['val']:.6f} nats")
    print(f"  Perplexity: Dark={ppx_dark:.4f}, RandFull={ppx_rf:.4f}, RandAlgae={ppx_ra:.4f}")
    print(f"  MWU (full curve) Dark<RandFull:   U={u1:.0f}, p={p1:.6f}")
    print(f"  MWU (full curve) Dark<RandAlgae:  U={u2:.0f}, p={p2:.6f}")
    print(f"  MWU (stable win) Dark<RandFull:   U={u3:.0f}, p={p3:.6f}")
    print(f"  MWU (stable win) Dark<RandAlgae:  U={u4:.0f}, p={p4:.6f}")
    print(f"  Cohen's d (Dark vs RandFull):      {d_rf:.4f}")
    print(f"  Cohen's d (Dark vs RandAlgae):     {d_ra:.4f}")

    result = {
        'dark_final': FINAL['Dark']['val'],
        'rf_final': FINAL['Random Full']['val'],
        'ra_final': FINAL['Random Algae']['val'],
        'p_vs_rf_fullcurve': p1,
        'p_vs_ra_fullcurve': p2,
        'p_vs_rf_window': p3,
        'p_vs_ra_window': p4,
        'd_rf': d_rf,
        'd_ra': d_ra,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – Combined 4-panel publication figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_main_figure(learn_df, stat_df, conv_df):
    """Main 2×2 publication figure."""
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── (A) Training curves ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for name in ORDER:
        d = RUNS[name]
        ax_a.plot(d['iters']/1000, d['val'],
                  color=PALETTE[name], lw=2,
                  ls=LINESTYLES[name], label=name)
    ax_a.axhline(ENTROPY_FLOOR, color='grey', ls=':', lw=1.3,
                 label=r'ln(20)≈2.996')
    ax_a.set_xlabel('Training Iterations (×1 000)')
    ax_a.set_ylabel('Validation Loss (nats)')
    ax_a.set_title('(A)  Validation Loss Curves')
    ax_a.set_xlim(0, 40); ax_a.set_ylim(2.4, 3.25)
    ax_a.legend(loc='upper right', fontsize=8.5)

    # ── (B) Learnability bar chart ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    cols_b = [PALETTE[n] for n in learn_df['Dataset']]
    bars_b = ax_b.bar(learn_df['Dataset'],
                       learn_df['Learnability Score (%)'],
                       color=cols_b, edgecolor='black', lw=0.8, alpha=0.87)
    for bar, v in zip(bars_b, learn_df['Learnability Score (%)']):
        ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.7,
                  f'{v:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax_b.axhline(100, color='#1a6faf', ls='--', lw=1, alpha=0.5)
    ax_b.axhline(0,   color='#b22222', ls='--', lw=1, alpha=0.5)
    ax_b.set_ylim(0, 118)
    ax_b.set_ylabel('Learnability Score (%)')
    ax_b.set_title('(B)  Relative Learnability')
    ax_b.tick_params(axis='x', rotation=10)

    # ── (C) Train/Val gap ──────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    for name in ORDER:
        d = RUNS[name]
        gap = d['val'] - d['train']
        ax_c.plot(d['iters']/1000, gap,
                  color=PALETTE[name], lw=2,
                  ls=LINESTYLES[name], label=name)
    ax_c.axhline(0, color='k', lw=0.8)
    ax_c.set_xlabel('Training Iterations (×1 000)')
    ax_c.set_ylabel('Val − Train Loss (nats)')
    ax_c.set_title('(C)  Generalisation Gap')
    ax_c.set_xlim(0, 40)
    ax_c.legend(loc='upper right', fontsize=8.5)

    # ── (D) Effect-size comparison ─────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    key_comps = ['White vs Dark', 'Dark vs Random Full', 'Dark vs Random Algae',
                 'Algae vs Random Algae']
    sub = stat_df[stat_df['Comparison'].isin(key_comps)].copy()
    bar_colors = ['#2ecc71' if p == 'Yes' else '#e74c3c' for p in sub['Significant']]
    y_pos = np.arange(len(sub))
    ax_d.barh(y_pos, sub["Cohen's d"].abs(),
              color=bar_colors, edgecolor='black', lw=0.6, alpha=0.85)
    ax_d.axvline(0.5, color='grey', ls='--', lw=1, alpha=0.7, label='medium')
    ax_d.axvline(0.8, color='grey', ls='-',  lw=1, alpha=0.7, label='large')
    ax_d.set_yticks(y_pos); ax_d.set_yticklabels(sub['Comparison'], fontsize=9)
    ax_d.set_xlabel("|Cohen's d|")
    ax_d.set_title("(D)  Effect Sizes (Key Comparisons)")
    patches = [mpatches.Patch(color='#2ecc71', label='p < 0.05'),
               mpatches.Patch(color='#e74c3c', label='p ≥ 0.05')]
    ax_d.legend(handles=patches, fontsize=8, loc='lower right')

    fig.suptitle(
        'GPT-2 Training Dynamics on Protein Sequence Datasets\n'
        'White · Algae · Dark · Random Algae-Biased · Random Full',
        fontsize=13, fontweight='bold', y=1.01
    )
    path = f'{OUT_DIR}/fig_main_4panel.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – Supplementary: all metrics per dataset
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_dataset_panels():
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    metrics = ['train', 'val', 'gap']
    for row_i, name in enumerate(ORDER):
        d = RUNS[name]
        for col_i, metric in enumerate(metrics):
            ax = axes[row_i, col_i]
            if metric == 'gap':
                y = d['val'] - d['train']
                ylabel = 'Val − Train (nats)'
                color  = PALETTE[name]
            else:
                y = d[metric]
                ylabel = f'{metric.capitalize()} Loss (nats)'
                color  = PALETTE[name]
            ax.plot(d['iters']/1000, y, color=color, lw=2)
            if metric != 'gap':
                ax.axhline(ENTROPY_FLOOR, color='grey', ls=':', lw=1)
            ax.set_xlabel('Iter (×1k)' if row_i == 4 else '')
            ax.set_ylabel(ylabel if col_i == 0 else '')
            ax.set_title(f'{name} – {metric.capitalize()}')
            ax.set_xlim(0, 40)

    fig.suptitle('Per-Dataset Training Metrics (Supplementary)', fontsize=14, y=1.01)
    fig.tight_layout()
    path = f'{OUT_DIR}/figS1_per_dataset.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – Tables to CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_tables(conv_df, gap_df, stat_df, learn_df):
    conv_df.to_csv(f'{OUT_DIR}/table1_convergence.csv', index=False)
    gap_df.to_csv(f'{OUT_DIR}/table2_gap.csv', index=False)
    stat_df.to_csv(f'{OUT_DIR}/table3_statistical.csv', index=False)
    learn_df.to_csv(f'{OUT_DIR}/table4_learnability.csv', index=False)
    print(f'Saved tables to {OUT_DIR}/')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – Figure captions
# ══════════════════════════════════════════════════════════════════════════════

CAPTIONS = """
═══════════════════════════════════════════════════════════════════════════════
                        PUBLICATION FIGURE CAPTIONS
═══════════════════════════════════════════════════════════════════════════════

Figure 1.  GPT-2 Training Dynamics on Five Protein-Sequence Datasets.
Cross-entropy loss (nats) as a function of training iteration for five character-
level GPT-2 models trained from scratch on (left) training split and (right)
held-out validation split. Datasets are: white proteins (annotated, photosynthetic
organisms; blue), algae proteins (green), dark proteins (hypothetical/uncharacterised;
purple), random algae-biased (composition-matched random sequences; dashed amber),
and random full (uniform 20-AA random sequences; dashed brick red). The horizontal
dotted grey line marks the theoretical cross-entropy floor for a uniformly random
20-amino-acid process, H = ln(20) ≈ 2.996 nats. All five models were trained with
identical GPT-2 Small hyperparameters (12 layers, 12 heads, 768 embedding dimensions;
batch size 12 × 2 gradient-accumulation steps; block size 1024; cosine learning-rate
schedule from 5×10⁻⁴ to 5×10⁻⁵ over 40 000 iterations).

Figure 2.  Convergence Rate Analysis.
(Left) Validation-loss curves overlaid with exponential-decay fits of the form
L(t) = a·exp(−b·t) + c. (Right) Convergence half-life (in kilo-iterations) derived
from fitted rate constants, measuring the number of training iterations required to
traverse half the learnable loss range. Shorter half-lives indicate faster initial
learning.

Figure 3.  Train/Validation Generalisation Gap.
Difference between validation loss and training loss across training iterations for
each dataset. Positive values indicate the model generalises to held-out sequences
less well than to training sequences. Near-zero gaps for the random-sequence
baselines reflect the absence of generalisable statistical structure in those corpora.

Figure 4.  Statistical Pairwise Comparisons.
(Left) Absolute Cohen's d effect sizes for key dataset comparisons computed over
the last 10 evaluation steps (stable convergence window), colour-coded green (p <
0.05, Mann-Whitney U, one-tailed) or red (p ≥ 0.05). Vertical lines indicate
conventional medium (d = 0.5) and large (d = 0.8) effect-size thresholds.
(Right) Final validation-loss differences with 95% percentile bootstrap confidence
intervals (9 999 resamples). A negative value indicates the first-named dataset
achieved lower (better) loss.

Figure 5.  Learnability Hierarchy.
(Left) Relative learnability score for each dataset, defined as L(D) = 100 ×
(L_floor − L_D) / (L_floor − L_best), where L_floor is the Random Full final
validation loss and L_best is the White final validation loss. A score of 0%
represents the empirical random floor; 100% represents the best-observed ceiling.
(Right) Gradient plot showing individual final validation losses on a shared axis
with lower loss plotted to the right, visually encoding the learnability hierarchy
White → Algae → Dark → Random Algae → Random Full.

Figure (Main, 4-panel).  Summary Figure.
(A) Validation-loss learning curves for all five datasets with the theoretical
entropy floor. (B) Relative learnability scores. (C) Generalisation (train/val)
gap trajectories. (D) Absolute effect sizes for four key pairwise comparisons.

Figure S1.  Per-Dataset Training Metrics.
Supplementary panel showing training loss, validation loss, and generalisation gap
as separate sub-panels for each of the five datasets.
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – Supplementary methods text
# ══════════════════════════════════════════════════════════════════════════════

SUPP_METHODS = """
═══════════════════════════════════════════════════════════════════════════════
              SUPPLEMENTARY METHODS – MANUSCRIPT-READY TEXT
═══════════════════════════════════════════════════════════════════════════════

S1.  Datasets
─────────────
Five protein-sequence corpora were constructed for training-dynamics experiments.

White proteins. A character-level corpus of protein sequences drawn from
annotated proteomes of photosynthetic (white/light-harvesting) organisms.
The dataset contains 66.5 million character tokens in the training split and
7.4 million in the validation split (90/10 split by character position).

Algae proteins. Protein sequences from the algal proteome.  A representative
400,000-sequence subsample (algae_sampled_400k.fasta) was used; this yields
58.9 million training tokens and 6.5 million validation tokens.

Dark proteins. Sequences classified as "dark" — hypothetical, uncharacterised,
or lacking functional annotation in major databases. This is the dataset of
primary biological interest. The dark corpus contains 63.1 million training
tokens and 7.0 million validation tokens.

Random algae-biased sequences (composition-matched control). A corpus of
400,000 randomly generated protein sequences whose amino-acid composition
matches the empirical unigram frequency distribution of the algal proteome
(estimated from prompts_1000algae.csv). Sequence lengths were sampled from
the empirical length distribution of the same file. Residue ordering was
drawn i.i.d. at each position. This control matches single-residue statistics
while eliminating all higher-order sequence structure; its cross-entropy loss
provides a composition-adjusted floor.

Random full sequences (maximum-entropy control). Identical to the random
algae-biased control except that residues were drawn uniformly from the
20 standard amino acids (equal probability 1/20 per residue), maximising
sequence entropy. The theoretical cross-entropy floor for this corpus under
an optimal character-level model is H = ln(20) ≈ 2.996 nats. Both random
corpora contain 55.4 million training tokens and 6.2 million validation tokens.

S2.  Tokenizer
──────────────
All five corpora share a character-level tokenizer over a vocabulary of
20–23 characters. Sequences are stored in a prompts-style flat-text format
(one sequence per line, terminated by the '>' separator character). The
tokenizer maps each character to a unique integer index; no sub-word
segmentation is applied. Biological datasets (white, algae, dark) contain a
vocabulary of 23 characters: 20 standard amino-acid single-letter codes
(A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y), the
ambiguity symbol X, the separator character '>', and the newline character '\n'.
The random-sequence corpora omit X (vocab size 22), as the generator samples
only from the 20 standard amino acids.  Sequences were tokenised as uint16
arrays and stored as binary (.bin) files.  The 90/10 character-based
train/validation split was applied prior to tokenisation.

S3.  Model Architecture
───────────────────────
All experiments use a GPT-2 Small architecture (Radford et al. 2019) with
the following hyperparameters, identical across all five runs:
  • Layers (n_layer):          12
  • Attention heads (n_head):  12
  • Embedding dimension:       768
  • Context length (block_size): 1,024 tokens
  • Dropout:                   0.2
  • Total parameters:          ~85 million (character vocabulary replaces the
    standard BPE vocabulary, reducing the embedding table accordingly)

S4.  Training Procedure
───────────────────────
Each model was trained from random initialisation ('scratch') using the
nanoGPT framework (Karpathy 2023). Training hyperparameters were held
constant across all five runs:
  • Optimiser:                 AdamW (β₁=0.9, β₂=0.99, ε=1×10⁻⁸)
  • Learning rate:             Peak 5×10⁻⁴, minimum 5×10⁻⁵
  • LR schedule:               Cosine decay over 40,000 iterations,
                               200-iteration linear warm-up
  • Batch size:                12 sequences × 2 gradient-accumulation steps
                               = effective batch size 24 × 1,024 tokens
                               ≈ 24,576 tokens per update
  • Maximum iterations:        40,000
  • Evaluation interval:       Every 1,000 iterations (40 evaluation points)
  • Checkpoint policy:         Best validation loss saved
  • Hardware:                  Single GPU
  • Logging:                   Weights & Biases (project dark-light)

S5.  Interpretation of Cross-Entropy Loss
──────────────────────────────────────────
The model outputs a probability distribution over the vocabulary at each
sequence position given all preceding characters. Training minimises the
mean negative log-likelihood (cross-entropy) in nats.

A cross-entropy of L nats corresponds to a perplexity of e^L per character;
lower loss indicates the model assigns higher probability to observed sequences.

Theoretical bounds:
  • Upper bound: ln(|V|). For vocab size 22: ln(22) ≈ 3.091; for vocab size
    23: ln(23) ≈ 3.135. An untrained or purely random model should achieve
    approximately this loss at initialisation.
  • Uniform random floor: For the Random Full corpus (20 AA, uniform i.i.d.),
    an optimal compressor achieves H = ln(20) ≈ 2.996 nats. The trained
    model converged to 2.993, confirming it reached near-optimal compression
    of this corpus and validating the training setup.
  • Composition floor: The Random Algae-Biased corpus has a lower entropy
    than the uniform baseline (H = −Σᵢ pᵢ ln pᵢ ≤ ln(20)) because of
    unequal residue probabilities. The trained model (val loss 2.855) falls
    below the uniform floor, consistent with learning the unigram composition
    but not higher-order structure.
  • Structure / learnability signal: Any loss reduction below the
    composition-matched random floor reflects the model capturing higher-order
    statistical regularities — positional correlations, motifs, fold-
    associated residue preferences — in the biological sequences. The
    difference ΔFINAL = L_random_algae − L_dark ≈ 0.100 nats represents
    the structural information density learnable by GPT-2 in dark proteins
    beyond mere amino-acid composition.
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 – Suggested additional analyses
# ══════════════════════════════════════════════════════════════════════════════

ADDITIONAL_ANALYSES = """
═══════════════════════════════════════════════════════════════════════════════
           SUGGESTED ADDITIONAL ANALYSES FROM CHECKPOINTS
═══════════════════════════════════════════════════════════════════════════════

The five checkpoint files (ckpt.pt) contain the full model weights, optimizer
state, and configuration.  The following analyses are directly executable:

1.  Perplexity cross-evaluation (cross-dataset perplexity matrix)
    Load each checkpoint; evaluate it on each of the five validation splits.
    The 5×5 matrix of (model, dataset) perplexities reveals how much
    statistical structure is shared between corpora.  Expected: dark-trained
    model will generalise partially to algae/white, but not to random.

2.  Sequence generation & motif analysis
    Sample sequences unconditionally from each model (temperature sweep
    0.5–1.2).  Run the generated sequences through HMMER (profile HMM scan),
    InterProScan, or SignalP to quantify functional/structural annotation rate.
    A higher annotation rate for dark-model sequences versus random-model
    sequences would directly support learned biological structure.

3.  Embedding space analysis (PCA / t-SNE / UMAP)
    Extract final-layer residue representations for held-out sequences from
    all five corpora and visualise in 2-D.  Clustering by dataset or by
    known functional class (if labels are available) probes what the model
    encodes in continuous space.

4.  Attention-head analysis
    Compute average attention entropy per head across positions.  Low-entropy
    heads perform specific lookups (e.g., periodic or motif-based patterns).
    Comparing attention entropy between dark and random models identifies
    heads that learned sequence grammar.

5.  Per-position loss analysis
    Compute positional cross-entropy profiles over held-out sequences by
    querying the model at every character position.  Compare to position-
    specific scoring matrices (PSSMs) from known protein families to test
    whether the model implicitly recovers conservation patterns.

6.  In-context learning (zero-shot transfer)
    Prefix a known functional annotation (e.g., a signal peptide) and ask
    each model to continue the sequence.  Measure how much the continuation
    probability shifts by dataset, probing whether dark-protein models
    capture function-associated sequence preferences.

7.  Loss-delta percolation over training
    Compute the per-residue contribution to loss difference (dark – random)
    at each evaluation checkpoint.  Identify which amino-acid characters drive
    the divergence and at what training stage each character's loss separates.

8.  Dataset-size ablation (learning curves)
    Subsample each biological corpus to 10 %, 25 %, 50 % of its size and
    retrain.  Fit scaling-law curves (Chinchilla style) to estimate optimal
    compute allocation for dark-protein models.

9.  Longer training run (100k–200k iterations)
    The cosine schedule reached min_lr at iteration 40 000, suggesting the
    models may still be data-limited.  Extending training or using a cyclical
    LR schedule may narrow or widen the dark/random gap.

10. Comparative architectures (LSTM / Mamba / ESM-style)
    Hold datasets fixed; vary architecture.  This isolates whether the
    learnability gap reflects properties of the GPT-2 inductive bias or
    genuine sequence statistics.
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 – ROC/AUC discussion
# ══════════════════════════════════════════════════════════════════════════════

ROC_AUC_TEXT = """
═══════════════════════════════════════════════════════════════════════════════
    ROC / AUC ANALYSIS — FEASIBILITY ASSESSMENT & PROCEDURE
═══════════════════════════════════════════════════════════════════════════════

FEASIBILITY
───────────
ROC/AUC analysis is not directly applicable to the current generative-modelling
framing (cross-entropy minimisation with no binary labels), but two concrete
formulations make it tractable:

──────────────────────────────────────────────────────────────────────────────
Formulation 1 – Sequence-level classification using per-sequence log-likelihood
──────────────────────────────────────────────────────────────────────────────
For each checkpoint, compute the average per-character log-likelihood
assigned by the model to a held-out sequence S:
  score(S) = (1/|S|) Σ_t log p_θ(s_t | s_<t)

Use the dark-model or white-model log-likelihoods as a discriminative score to
separate biological sequences from random sequences.  Construct an ROC curve as
the score threshold is varied, with:
  Positive class: sequences drawn from a biological corpus (e.g., dark proteins)
  Negative class: sequences drawn from a random corpus (e.g., random full)

Required steps:
  a. Load ckpt.pt for each model.
  b. For each sequence in a balanced test set (e.g., 1000 dark + 1000 random),
     compute the per-character log-likelihood using a forward pass with
     teacher forcing.
  c. Plot ROC curve; compute AUROC.
  d. Repeat for each model checkpoint to obtain a 5×2 AUROC matrix
     (model × test-class pair).

Expected result: The dark-trained model should score dark sequences higher
than random sequences (AUROC > 0.5).  The random-trained model should yield
AUROC ≈ 0.5 on the same test.  The white-trained model may achieve the
highest AUROC on white sequences.

──────────────────────────────────────────────────────────────────────────────
Formulation 2 – Dark vs. annotated binary discrimination
──────────────────────────────────────────────────────────────────────────────
If sequence-level labels are available (dark = 1, annotated = 0), use the
model log-likelihood as a score to predict whether a query sequence is a
dark or annotated protein.  This is a biologically meaningful classification
task; AUROC quantifies whether the language model's learned representation
separates the two functional classes.

Required steps:
  a. Assemble a balanced, non-overlapping test set of dark and annotated
     protein sequences not seen during training.
  b. Score each sequence with both the dark-model and white-model checkpoints.
  c. Treat the dark-model score (and, separately, the score difference
     dark_score − white_score) as the discriminant.
  d. Compute ROC curves and AUROC with 95% bootstrap CIs.

LIMITATIONS
───────────
• The current single-replicate design precludes parametric significance testing
  of AUROC differences; bootstrap CIs should be reported.
• At 40k iterations the models have not fully converged (dark best checkpoint
  was at 32k iterations), so AUROC may improve with longer training.
• Because the tokeniser is character-level (not BPE or protein-specific), the
  model captures local sequence statistics but may miss long-range structural
  signals captured by residue-contact-aware architectures.

RECOMMENDATION
──────────────
Formulation 1 is immediately executable from the existing checkpoints without
any additional labelling effort and provides a clean separation-power metric
complementary to cross-entropy loss.  We recommend reporting AUROC alongside
the learnability score as a discrimination-oriented summary statistic.
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("  POST-TRAINING ANALYSIS  –  dark-whiteGPLM")
    print("═" * 70)

    # 1. Training curves
    print("\n[1/7] Plotting training curves …")
    plot_training_curves()

    # 2. Convergence
    print("[2/7] Computing convergence rates …")
    conv_df = compute_convergence_table()
    print(conv_df.to_string(index=False))
    plot_convergence_rates(conv_df)

    # 3. Gap
    print("\n[3/7] Computing train/val gap …")
    gap_df = compute_gap_table()
    print(gap_df.to_string(index=False))
    plot_gap_curves()

    # 4. Statistics
    print("\n[4/7] Computing pairwise statistics …")
    stat_df = compute_stat_table()
    print(stat_df.to_string(index=False))
    plot_statistical_comparisons(stat_df)

    # 5. Learnability
    print("\n[5/7] Computing learnability scores …")
    learn_df = compute_learnability(FINAL)
    print(learn_df.to_string(index=False))
    plot_learnability_hierarchy(learn_df)

    # 6. Dark vs. random
    print("\n[6/7] Dark vs. random significance …")
    dark_res = dark_vs_random_test()

    # 7. All combined
    print("\n[7/7] Generating combined figures …")
    plot_main_figure(learn_df, stat_df, conv_df)
    plot_per_dataset_panels()

    save_tables(conv_df, gap_df, stat_df, learn_df)

    # Print text outputs
    print(CAPTIONS)
    print(SUPP_METHODS)
    print(ADDITIONAL_ANALYSES)
    print(ROC_AUC_TEXT)

    print(f"\nAll figures saved to: {OUT_DIR}/")
    print("─" * 70)


if __name__ == '__main__':
    main()
