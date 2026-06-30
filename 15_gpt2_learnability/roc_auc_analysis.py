"""
roc_auc_analysis.py
ROC/AUC analysis: score held-out sequences from all five datasets with each
of the five trained GPT-2 checkpoints, then compute AUROC for key binary
classification tasks.

Scoring: mean per-token negative log-likelihood over the sequence
  score(S) = (1/|S|-1) Σ_{t=1}^{|S|-1}  -log p_θ(s_t | s_0..s_{t-1})

ROC classification tasks  (positive class listed first):
  1.  Dark   vs  Random Full       (main biological vs. max-entropy control)
  2.  Dark   vs  Random Algae      (biological vs. composition-matched control)
  3.  White  vs  Random Full
  4.  Algae  vs  Random Algae
  5.  Dark   vs  White             (inter-biological discrimination)
  6.  Bio (White+Algae+Dark) vs Random (RA+RF)   (aggregate)

For each task the scoring model is varied (all 5 checkpoints), giving a
rich picture of cross-model generalisation.
"""

import os, sys, pickle, glob, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import bootstrap as scipy_bootstrap
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/sd145/dark-whiteGPLM')
from model import GPTConfig, GPT

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = '/home/sd145/dark-whiteGPLM'
OUT_DIR = f'{ROOT}/analysis/figures'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

DATASETS = {
    'White':        f'{ROOT}/data/char_white',
    'Algae':        f'{ROOT}/data/char_algae',
    'Dark':         f'{ROOT}/data/char_dark',
    'Random Algae': f'{ROOT}/data/char_random_algae_biased',
    'Random Full':  f'{ROOT}/data/char_random_full',
}
CHECKPOINTS = {
    'White':        f'{ROOT}/out_white_random_control/ckpt.pt',
    'Algae':        f'{ROOT}/out_algae/ckpt.pt',
    'Dark':         f'{ROOT}/out_dark_random_control/ckpt.pt',
    'Random Algae': f'{ROOT}/out_random_algae_biased/ckpt.pt',
    'Random Full':  f'{ROOT}/out_random_full/ckpt.pt',
}
# stoi for each model's training vocabulary (loaded once, stored here)
MODEL_STOI = {}   # populated in load_model()
PALETTE = {
    'White':        '#1a6faf',
    'Algae':        '#2ca02c',
    'Dark':         '#7b3f9e',
    'Random Algae': '#e07b00',
    'Random Full':  '#b22222',
}

N_SEQS   = 1000   # sequences sampled per dataset
MAX_LEN  = 512    # max tokens scored per sequence (well under block_size=1024)
MIN_LEN  = 30     # discard very short fragments
BATCH_SZ = 32
SEED     = 42

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'legend.framealpha': 0.85,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': ':',
})


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Sequence extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_sequences(dataset_path, n_seqs, max_len, min_len, seed):
    """
    Parse val.bin into individual protein sequences.
    Returns list of np.uint16 arrays, each the token ids for one sequence
    (including the terminal '>').
    """
    meta  = pickle.load(open(f'{dataset_path}/meta.pkl', 'rb'))
    stoi  = meta['stoi']
    data  = np.fromfile(f'{dataset_path}/val.bin', dtype=np.uint16)
    sep   = stoi['>']
    sep_pos = np.where(data == sep)[0]

    seqs = []
    prev = 0
    for pos in sep_pos:
        # tokens from prev to pos inclusive (pos is the '>')
        chunk = data[prev : pos + 1]
        # strip any leading/trailing newlines
        nl = stoi.get('\n', None)
        if nl is not None:
            chunk = chunk[chunk != nl]  # keep only AA + '>'
        if min_len <= len(chunk) <= max_len:
            seqs.append(chunk)
        prev = pos + 1   # skip '\n' after '>'

    rng = random.Random(seed)
    if len(seqs) > n_seqs:
        seqs = rng.sample(seqs, n_seqs)
    else:
        print(f'  WARNING: only {len(seqs)} valid sequences (requested {n_seqs})')
    return seqs, meta


print('Extracting validation sequences …')
SEQ_STORE  = {}   # name → list of token arrays
META_STORE = {}   # name → meta dict
for name, path in DATASETS.items():
    seqs, meta = extract_sequences(path, N_SEQS, MAX_LEN, MIN_LEN, SEED)
    SEQ_STORE[name]  = seqs
    META_STORE[name] = meta
    print(f'  {name}: {len(seqs)} sequences, '
          f'lengths {min(len(s) for s in seqs)}–{max(len(s) for s in seqs)}, '
          f'median {np.median([len(s) for s in seqs]):.0f}')


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Model loader
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, device, model_name):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt['model_args']
    cfg  = GPTConfig(**args)
    model = GPT(cfg)
    state = {k.replace('_orig_mod.', '').replace('module.', ''): v
             for k, v in ckpt['model'].items()}
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    # Cache the model's own stoi (from its training dataset)
    meta = pickle.load(open(f'{DATASETS[model_name]}/meta.pkl', 'rb'))
    MODEL_STOI[model_name] = meta['stoi']
    return model, args['vocab_size']


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Per-sequence NLL scorer
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def score_sequences(model, model_name, model_vocab_size, sequences_tok,
                    dataset_meta, device, batch_size=BATCH_SZ):
    """
    Compute mean per-token NLL for a list of token-id arrays.
    Tokens are remapped src_itos → model_stoi so cross-dataset scoring is
    correct even when vocabularies differ in size (e.g., 'Y' has different
    integer IDs in biological vs. random tokenisers).
    Sequences containing characters absent from the model vocab are skipped.
    Returns np.array of shape (len(sequences_tok),).
    """
    src_itos = dataset_meta['itos']          # src dataset id → char
    model_stoi = MODEL_STOI[model_name]      # model's char → id

    model.eval()
    scores = []
    for tok_arr in sequences_tok:
        # Remap: src ids → characters → model ids
        remapped = []
        skip = False
        for tid in tok_arr:
            char = src_itos[int(tid)]
            mid  = model_stoi.get(char, None)
            if mid is None:              # character absent from model vocab
                skip = True; break
            remapped.append(mid)
        if skip or len(remapped) < 2:
            scores.append(np.nan)
            continue

        t   = torch.tensor(remapped, dtype=torch.long, device=device).unsqueeze(0)
        inp = t[:, :-1]
        tgt = t[:, 1:]
        _, nll_tensor = model(inp, tgt)
        scores.append(nll_tensor.item())
    return np.array(scores, dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Build full score matrix  [model × dataset]
# ══════════════════════════════════════════════════════════════════════════════

SCORE_MATRIX = {}   # (model_name, dataset_name) → array of per-seq scores

ORDER = ['White', 'Algae', 'Dark', 'Random Algae', 'Random Full']

for model_name in ORDER:
    print(f'\nLoading model: {model_name} …')
    model, vocab_sz = load_model(CHECKPOINTS[model_name], DEVICE, model_name)
    SCORE_MATRIX[model_name] = {}
    for ds_name in ORDER:
        seqs = SEQ_STORE[ds_name]
        ds_meta = META_STORE[ds_name]
        sc = score_sequences(model, model_name, vocab_sz, seqs, ds_meta, DEVICE)
        n_valid = np.sum(~np.isnan(sc))
        SCORE_MATRIX[model_name][ds_name] = sc
        print(f'  → scored {ds_name}: {n_valid}/{len(sc)} valid, '
              f'mean NLL={np.nanmean(sc):.4f}')
    del model
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ROC / AUC computation helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_roc(scores_pos, scores_neg):
    """
    Positive class = lower NLL (model prefers these).
    We flip sign so higher score → model prefers the sequence.
    Returns fpr, tpr, auroc.
    """
    y_true  = np.array([1]*len(scores_pos) + [0]*len(scores_neg))
    y_score = np.concatenate([-scores_pos, -scores_neg])   # negate: lower loss = better
    mask    = ~np.isnan(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(np.unique(y_true)) < 2:
        return None, None, np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = auc(fpr, tpr)
    return fpr, tpr, auroc


def auroc_bootstrap_ci(scores_pos, scores_neg, n_resamples=5000, confidence=0.95, seed=42):
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.default_rng(seed)
    aurocs = []
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    for _ in range(n_resamples):
        pos_sample = rng.choice(scores_pos[~np.isnan(scores_pos)], size=n_pos, replace=True)
        neg_sample = rng.choice(scores_neg[~np.isnan(scores_neg)], size=n_neg, replace=True)
        y_true  = np.array([1]*n_pos + [0]*n_neg)
        y_score = np.concatenate([-pos_sample, -neg_sample])
        try:
            aurocs.append(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    aurocs = np.array(aurocs)
    alpha = 1 - confidence
    return np.percentile(aurocs, 100*alpha/2), np.percentile(aurocs, 100*(1-alpha/2))


# Define the six classification tasks
TASKS = [
    # (task_label, positive_class, negative_class, scoring_model)
    ('Dark vs Random Full\n(Dark model)',    'Dark',  'Random Full',  'Dark'),
    ('Dark vs Random Algae\n(Dark model)',   'Dark',  'Random Algae', 'Dark'),
    ('White vs Random Full\n(White model)',  'White', 'Random Full',  'White'),
    ('Algae vs Random Algae\n(Algae model)', 'Algae', 'Random Algae', 'Algae'),
    ('Dark vs White\n(Dark model)',          'Dark',  'White',        'Dark'),
    ('Dark vs White\n(White model)',         'Dark',  'White',        'White'),
]


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Figure A – ROC curves for primary tasks
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    roc_results = []
    task_colors = ['#7b3f9e', '#7b3f9e', '#1a6faf', '#2ca02c', '#7b3f9e', '#1a6faf']

    for idx, (label, pos_cls, neg_cls, mdl) in enumerate(TASKS):
        ax = axes[idx]
        s_pos = SCORE_MATRIX[mdl][pos_cls]
        s_neg = SCORE_MATRIX[mdl][neg_cls]
        fpr, tpr, auroc = compute_roc(s_pos, s_neg)
        ci_lo, ci_hi = auroc_bootstrap_ci(
            s_pos[~np.isnan(s_pos)], s_neg[~np.isnan(s_neg)])

        # ROC curve
        if fpr is not None:
            ax.plot(fpr, tpr, color=task_colors[idx], lw=2.5,
                    label=f'AUC = {auroc:.3f}')
            ax.fill_between(fpr, tpr, alpha=0.12, color=task_colors[idx])
        # diagonal
        ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random (0.5)')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(label, fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.text(0.60, 0.12, f'95% CI [{ci_lo:.3f}, {ci_hi:.3f}]',
                transform=ax.transAxes, fontsize=8, color='grey')

        roc_results.append({
            'Task': label.replace('\n', ' '),
            'Scoring Model': mdl,
            'Positive Class': pos_cls,
            'Negative Class': neg_cls,
            'AUROC': round(auroc, 4) if not np.isnan(auroc) else 'N/A',
            'CI 95% lo': round(ci_lo, 4),
            'CI 95% hi': round(ci_hi, 4),
            'N pos': int(np.sum(~np.isnan(s_pos))),
            'N neg': int(np.sum(~np.isnan(s_neg))),
        })

    fig.suptitle('ROC Curves: Per-Sequence Log-Likelihood Discrimination\n'
                 '(positive class = higher model log-likelihood)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = f'{OUT_DIR}/fig6_roc_curves.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'\nSaved {path}')
    return pd.DataFrame(roc_results)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Figure B – Cross-model AUROC heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_auroc_heatmap():
    """
    For every (scoring_model, positive_class, negative_class) triplet with
    the same binary task, compute AUROC.  Show as heatmap.
    Key tasks: Dark vs Random Full across all 5 scoring models.
    """
    bio_classes = ['White', 'Algae', 'Dark']
    rand_classes = ['Random Algae', 'Random Full']

    # Cross-model AUROC: Dark vs Random Full with every scoring model
    scoring_models = ORDER
    tasks_cm = [
        ('Dark vs\nRandom Full',  'Dark',  'Random Full'),
        ('Dark vs\nRandom Algae', 'Dark',  'Random Algae'),
        ('Algae vs\nRandom Algae','Algae', 'Random Algae'),
        ('White vs\nRandom Full', 'White', 'Random Full'),
        ('Dark vs\nWhite',        'Dark',  'White'),
    ]
    mat = np.zeros((len(scoring_models), len(tasks_cm)))
    for r, mdl in enumerate(scoring_models):
        for c, (_, pos, neg) in enumerate(tasks_cm):
            _, _, auroc = compute_roc(
                SCORE_MATRIX[mdl][pos], SCORE_MATRIX[mdl][neg])
            mat[r, c] = auroc if not np.isnan(auroc) else 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, vmin=0.4, vmax=1.0, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, label='AUROC')
    ax.set_xticks(range(len(tasks_cm)))
    ax.set_xticklabels([t[0] for t in tasks_cm], fontsize=9)
    ax.set_yticks(range(len(scoring_models)))
    ax.set_yticklabels(scoring_models, fontsize=10)
    ax.set_xlabel('Classification Task (Positive vs. Negative)')
    ax.set_ylabel('Scoring Model')
    ax.set_title('Cross-Model AUROC Matrix\n(rows = scoring model, columns = classification task)',
                 fontsize=12, fontweight='bold')
    for r in range(len(scoring_models)):
        for c in range(len(tasks_cm)):
            v = mat[r, c]
            color = 'white' if v < 0.55 or v > 0.85 else 'black'
            ax.text(c, r, f'{v:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    fig.tight_layout()
    path = f'{OUT_DIR}/fig7_auroc_heatmap.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')
    return mat, scoring_models, tasks_cm


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Figure C – Score distributions (violin/strip plots)
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_distributions():
    """
    For the Dark model: violin plots of per-sequence NLL for all 5 datasets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A – Dark model scores all datasets
    ax = axes[0]
    data_by_ds = []
    labels = []
    colors = []
    for ds in ORDER:
        sc = SCORE_MATRIX['Dark'][ds]
        sc = sc[~np.isnan(sc)]
        data_by_ds.append(sc)
        labels.append(ds)
        colors.append(PALETTE[ds])

    parts = ax.violinplot(data_by_ds, positions=range(len(ORDER)),
                          showmedians=True, showextrema=True, widths=0.7)
    for i, (pc, col) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    parts['cmedians'].set_colors('black'); parts['cmedians'].set_linewidth(2)
    parts['cmaxes'].set_colors('grey');   parts['cmins'].set_colors('grey')
    parts['cbars'].set_colors('grey')

    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=12, fontsize=9)
    ax.set_ylabel('Per-Sequence Mean NLL (nats)')
    ax.set_title('Dark Model: Score Distributions\nfor All Datasets')
    ax.invert_yaxis()   # lower NLL → top = model prefers these

    # Panel B – All models on Dark sequences
    ax = axes[1]
    data_by_model = []
    model_labels  = []
    model_colors  = []
    for mdl in ORDER:
        sc = SCORE_MATRIX[mdl]['Dark']
        sc = sc[~np.isnan(sc)]
        data_by_model.append(sc)
        model_labels.append(mdl)
        model_colors.append(PALETTE[mdl])

    parts2 = ax.violinplot(data_by_model, positions=range(len(ORDER)),
                           showmedians=True, showextrema=True, widths=0.7)
    for pc, col in zip(parts2['bodies'], model_colors):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    parts2['cmedians'].set_colors('black'); parts2['cmedians'].set_linewidth(2)
    parts2['cmaxes'].set_colors('grey');    parts2['cmins'].set_colors('grey')
    parts2['cbars'].set_colors('grey')

    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=12, fontsize=9)
    ax.set_ylabel('Per-Sequence Mean NLL (nats)')
    ax.set_title('Dark Sequences: Score Distributions\nby Scoring Model')
    ax.invert_yaxis()

    fig.suptitle('Per-Sequence Log-Likelihood Score Distributions',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = f'{OUT_DIR}/fig8_score_distributions.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Figure D – Cross-dataset score matrix heatmap (mean NLL)
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_matrix_heatmap():
    mean_mat = np.zeros((len(ORDER), len(ORDER)))
    for r, mdl in enumerate(ORDER):
        for c, ds in enumerate(ORDER):
            sc = SCORE_MATRIX[mdl][ds]
            mean_mat[r, c] = np.nanmean(sc)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mean_mat, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean NLL (nats)')
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=15, fontsize=9)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=10)
    ax.set_xlabel('Sequences Scored')
    ax.set_ylabel('Scoring Model')
    ax.set_title('Cross-Dataset Mean NLL Matrix\n(row = model, col = sequence source; '
                 'diagonal = native validation loss)', fontsize=11, fontweight='bold')
    for r in range(len(ORDER)):
        for c in range(len(ORDER)):
            v = mean_mat[r, c]
            color = 'white' if v > 3.1 or v < 2.6 else 'black'
            diag = ' ★' if r == c else ''
            ax.text(c, r, f'{v:.3f}{diag}', ha='center', va='center',
                    fontsize=9, color=color)
    fig.tight_layout()
    path = f'{OUT_DIR}/fig9_cross_score_heatmap.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')
    return mean_mat


# ══════════════════════════════════════════════════════════════════════════════
# 10.  Figure E – Aggregate bio vs. random AUROC by scoring model
# ══════════════════════════════════════════════════════════════════════════════

def plot_aggregate_roc():
    """
    Aggregate task: (White + Algae + Dark) vs. (Random Algae + Random Full)
    Score with each of the 5 models.
    """
    bio_classes  = ['White', 'Algae', 'Dark']
    rand_classes = ['Random Algae', 'Random Full']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A – ROC for each scoring model on aggregate bio vs random
    ax = axes[0]
    agg_results = []
    for mdl in ORDER:
        pos_scores = np.concatenate([
            SCORE_MATRIX[mdl][d][~np.isnan(SCORE_MATRIX[mdl][d])]
            for d in bio_classes
        ])
        neg_scores = np.concatenate([
            SCORE_MATRIX[mdl][d][~np.isnan(SCORE_MATRIX[mdl][d])]
            for d in rand_classes
        ])
        fpr, tpr, auroc = compute_roc(pos_scores, neg_scores)
        ci_lo, ci_hi = auroc_bootstrap_ci(pos_scores, neg_scores, n_resamples=2000)
        if fpr is not None:
            ax.plot(fpr, tpr, color=PALETTE[mdl], lw=2.5, label=f'{mdl} AUC={auroc:.3f}')
        agg_results.append({
            'Scoring Model': mdl, 'AUROC': round(auroc, 4),
            'CI lo': round(ci_lo, 4), 'CI hi': round(ci_hi, 4),
            'N bio': len(pos_scores), 'N rand': len(neg_scores),
        })
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('Aggregate: Biological vs. Random\n(all 5 scoring models)')
    ax.legend(fontsize=8, loc='lower right')

    # Panel B – AUROC bar chart by scoring model
    ax = axes[1]
    df_agg = pd.DataFrame(agg_results)
    colors = [PALETTE[m] for m in df_agg['Scoring Model']]
    err_lo = (df_agg['AUROC'] - df_agg['CI lo']).values
    err_hi = (df_agg['CI hi'] - df_agg['AUROC']).values
    bars = ax.bar(df_agg['Scoring Model'], df_agg['AUROC'],
                  color=colors, edgecolor='black', lw=0.8, alpha=0.85,
                  yerr=[err_lo, err_hi], capsize=5, error_kw={'lw': 1.5})
    ax.axhline(0.5, color='k', ls='--', lw=1, alpha=0.5, label='Chance')
    ax.set_ylim(0.4, 1.02); ax.set_ylabel('AUROC')
    ax.set_title('Aggregate Bio vs. Random AUROC\nby Scoring Model (95% CI)')
    ax.tick_params(axis='x', rotation=10)
    for bar, v in zip(bars, df_agg['AUROC']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle('Aggregate Biological vs. Random Sequence Discrimination',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = f'{OUT_DIR}/fig10_aggregate_roc.pdf'
    fig.savefig(path); fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved {path}')
    return df_agg


# ══════════════════════════════════════════════════════════════════════════════
# 11.  Print narrative summary
# ══════════════════════════════════════════════════════════════════════════════

ROC_CAPTION = """
─────────────────────────────────────────────────────────────────────────────
  FIGURE CAPTIONS — ROC/AUC FIGURES
─────────────────────────────────────────────────────────────────────────────

Figure 6.  ROC Curves for Six Binary Sequence-Classification Tasks.
Receiver operating characteristic (ROC) curves computed by using per-sequence
mean negative log-likelihood (NLL) as a discriminant score.  For each task,
the scoring model was the GPT-2 trained on the positive class (except the
'Dark vs White (White model)' panel).  Positive class = biological/lower-entropy
sequences; the model assigns lower NLL (higher log-likelihood) to sequences it
finds more probable.  Area under the curve (AUROC) and 95% bootstrap confidence
intervals (5 000 resamples, N=1 000 sequences per class) are reported.  The
dashed diagonal represents chance-level discrimination (AUROC = 0.5).

Figure 7.  Cross-Model AUROC Heatmap.
AUROC values for five classification tasks (columns) evaluated with each of the
five trained scoring models (rows).  Diagonal-adjacent cells (native model on
its target task) should show highest AUROC; off-diagonal cells reveal how well
structural/statistical features learned by one model generalise to discriminating
other corpora.  Colour scale: green = high discriminability; red = chance/below.

Figure 8.  Per-Sequence Score Distributions.
(Left) Violin plots of per-sequence mean NLL assigned by the dark-protein model
to held-out sequences from each of the five corpora.  Lower NLL (top) indicates
the model assigns higher probability to those sequences.  (Right) Violin plots
of per-sequence mean NLL assigned by each of the five models to dark-protein
validation sequences.  Inverted y-axis: higher position = lower loss = model
prefers the sequences.

Figure 9.  Cross-Dataset Mean NLL Scoring Matrix.
Mean per-sequence NLL (nats) for every (scoring model, sequence source) combination.
Diagonal entries (★) correspond to each model evaluated on its native validation
split, matching the reported final validation losses.  Off-diagonal entries measure
generalisation: low NLL for a biological model scoring random sequences would
indicate the model has not specialised; high NLL for random sequences relative to
biological sequences confirms learned specificity.

Figure 10.  Aggregate Biological vs. Random Discrimination.
(Left) ROC curves for the aggregate classification task (White + Algae + Dark as
positive class; Random Algae + Random Full as negative class) using each of the
five scoring models.  (Right) AUROC bar chart with 95% bootstrap confidence
intervals.  Models trained on biological data are expected to show higher AUROC
than random-trained models on this task.
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('\n═' * 35)
    print('ROC / AUC ANALYSIS')
    print('═' * 35)

    print('\n[A] Plotting primary ROC curves …')
    roc_df = plot_roc_curves()
    print(roc_df.to_string(index=False))

    print('\n[B] Cross-model AUROC heatmap …')
    mat, sm, tasks_cm = plot_auroc_heatmap()

    print('\n[C] Score distribution plots …')
    plot_score_distributions()

    print('\n[D] Cross-dataset mean NLL matrix …')
    mean_mat = plot_score_matrix_heatmap()
    print('Cross-dataset mean NLL matrix:')
    df_mat = pd.DataFrame(mean_mat, index=ORDER, columns=ORDER).round(4)
    print(df_mat.to_string())

    print('\n[E] Aggregate bio vs. random AUROC …')
    df_agg = plot_aggregate_roc()
    print(df_agg.to_string(index=False))

    # Save tables
    roc_df.to_csv(f'{OUT_DIR}/table5_roc_results.csv', index=False)
    df_agg.to_csv(f'{OUT_DIR}/table6_aggregate_roc.csv', index=False)
    df_mat.to_csv(f'{OUT_DIR}/table7_cross_nll_matrix.csv')
    print(f'\nTables saved to {OUT_DIR}/')

    print(ROC_CAPTION)
    print(f'\nAll ROC figures saved to: {OUT_DIR}/')


if __name__ == '__main__':
    main()
