"""
Compute F1 at threshold optimized on Kaggle cross-val folds,
then apply that threshold to FogAtHome held-out evaluation.
This mirrors the paper's approach: train on Kaggle defog, test on FogAtHome.
"""
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

CACHE = 'logs/comprehensive_eval_cache'

COMBOS = [
    ('lc', 'probe'), ('lc', 'finetune'), ('lc', 'supervised'),
    ('mc', 'probe'), ('mc', 'finetune'), ('mc', 'supervised'),
    ('sc', 'probe'), ('sc', 'finetune'), ('sc', 'supervised'),
]


def find_optimal_f1_threshold(y_true, y_score):
    """Find threshold maximising F1 using the full PR curve (O(n log n))."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # thresholds has len = len(precision) - 1; last P/R point is trivial (P=1, R=0)
    denom = precision[:-1] + recall[:-1]
    f1_scores = np.where(denom > 0, 2 * precision[:-1] * recall[:-1] / denom, 0.0)
    best_idx = f1_scores.argmax()
    return thresholds[best_idx], f1_scores[best_idx]


rows = []
for ctx, mtype in COMBOS:
    name = f'{ctx}_{mtype}'
    print(f'Processing {name}...', flush=True)

    # --- Kaggle cross-val: find optimal threshold ---
    kaggle_df = pd.read_parquet(f'{CACHE}/kaggle_{name}_frames.parquet')
    y_true_kag  = kaggle_df['native_label'].values.astype(int)
    y_score_kag = kaggle_df['pred_prob_fog'].values.astype(float)
    opt_thr, kaggle_f1 = find_optimal_f1_threshold(y_true_kag, y_score_kag)
    kaggle_auc = roc_auc_score(y_true_kag, y_score_kag)

    # --- FogAtHome: apply same threshold ---
    fa_df = pd.read_parquet(f'{CACHE}/fogathome_{name}_frames.parquet')
    y_true_fa  = fa_df['native_label'].values.astype(int)
    y_score_fa = fa_df['pred_prob_fog'].values.astype(float)
    y_pred_fa  = (y_score_fa >= opt_thr).astype(int)

    fa_f1        = f1_score(y_true_fa, y_pred_fa, zero_division=0)
    fa_precision = precision_score(y_true_fa, y_pred_fa, zero_division=0)
    fa_recall    = recall_score(y_true_fa, y_pred_fa, zero_division=0)
    fa_auc       = roc_auc_score(y_true_fa, y_score_fa)

    rows.append({
        'combo':         name,
        'context':       ctx,
        'model_type':    mtype,
        'opt_thr':       round(opt_thr, 4),
        'kaggle_F1':     round(kaggle_f1, 4),
        'kaggle_AUC':    round(kaggle_auc, 4),
        'FA_F1':         round(fa_f1, 4),
        'FA_Precision':  round(fa_precision, 4),
        'FA_Recall':     round(fa_recall, 4),
        'FA_AUC':        round(fa_auc, 4),
    })

results = pd.DataFrame(rows)

print('\n=== F1 at Kaggle-Optimal Threshold, Applied to FogAtHome ===\n')
print(results[['combo', 'opt_thr', 'kaggle_F1', 'FA_F1', 'FA_Precision', 'FA_Recall', 'FA_AUC']].to_string(index=False))

print('\n--- Best models (sorted by FA_F1) ---')
print(results.sort_values('FA_F1', ascending=False)[['combo', 'FA_F1', 'FA_AUC']].to_string(index=False))

print('\n--- Paper comparison ---')
print('Paper 3rd place (best F1):  AUC=0.803, F1=0.746')
print('Paper 4th place (best AUC): AUC=0.812, F1=0.734')
best = results.loc[results['FA_F1'].idxmax()]
print(f'Our best FA_F1:  {best["combo"]} -> F1={best["FA_F1"]}, AUC={best["FA_AUC"]}')
best_auc = results.loc[results['FA_AUC'].idxmax()]
print(f'Our best FA_AUC: {best_auc["combo"]} -> AUC={best_auc["FA_AUC"]}, F1={best_auc["FA_F1"]}')

out_path = 'logs/fogathome_optimal_threshold_f1.csv'
results.to_csv(out_path, index=False)
print(f'\nSaved to {out_path}')
