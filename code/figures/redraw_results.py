"""Regenerate statistical plots from exported CSV values; paper PDFs remain unchanged."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MODELS = ['metaifd', 'gat', 'sage', 'lightgbm', 'rf']
NAMES = ['Meta-IFD', 'GAT', 'GraphSAGE', 'LightGBM', 'RF']
ORANGE = '#ed7d31'
BLUE = '#209ccc'

def save(fig, output, name):
    fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(output / (name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

def main(repo, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    full = pd.read_csv(repo / 'results/main/main_VERIFIED_SUMMARY.csv')
    f = full[full.metric.eq('macro_f1')].set_index(['dataset', 'model', 'arm'])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1))
    for ax, ds in zip(axes, ['Phish', 'Ponzi']):
        for i, model in enumerate(MODELS):
            for off, arm, color, marker in [(-.11, 'X14', BLUE, 'o'), (.11, 'X14_SIFE44', ORANGE, 'D')]:
                row = f.loc[ds, model, arm]
                ax.errorbar(row.mean_percent, i+off, xerr=row.sd_across_split_means_pp,
                            fmt=marker, color=color, capsize=2, markersize=4,
                            label=arm if i == 0 else None)
        ax.set(yticks=range(5), yticklabels=NAMES, xlabel='Test Macro-F1 (%)', title=ds)
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, output, 'fig03_main_comparison')
    group = pd.read_csv(repo / 'results/ablation/PAIRED_GROUP_CONTRIBUTIONS.csv')
    styles = [('without_distribution20', 'distribution20', '#139d81'),
              ('without_flow8', 'flow8', ORANGE), ('without_counterparty8', 'counterparty8', '#cc69a0'),
              ('X14_Pure36', 'intensity8', '#39a7ef')]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8))
    for ax, ds in zip(axes, ['Phish', 'Ponzi']):
        for j, (arm, label, color) in enumerate(styles):
            rows = group[group.dataset.eq(ds) & group.removed_arm.eq(arm)].set_index('model').loc[MODELS]
            ax.errorbar(rows.mean_contribution_full_minus_removed_pp, np.arange(5)+(j-1.5)*.13,
                        xerr=rows.sd_contribution_pp, fmt='o', markersize=3, color=color, capsize=2, label=label)
        ax.axvline(0, color='black', lw=.7); ax.set(yticks=range(5), yticklabels=NAMES,
            xlabel='Macro-F1 contribution (pp): full - removed', title=ds); ax.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7)
    fig.tight_layout(rect=[0,.06,1,1]); save(fig, output, 'fig04_group_ablation')
    balance = pd.read_csv(repo / 'results/activity_matching/MATCH_COVERAGE_BALANCE.csv')
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for ax, ds in zip(axes, ['Phish', 'Ponzi']):
        rows = balance[balance.dataset.eq(ds)]
        for off, before, color, label in [(-.1, True, '#8a909b', 'Full TEST'), (.1, False, ORANGE, 'Matched')]:
            prefix = 'before_' if before else 'after_'
            for i, cov in enumerate(['count', 'value']):
                x = rows[prefix+'standardized_gap_'+cov].abs()
                ax.errorbar(x.mean(), i+off, xerr=[[x.mean()-x.min()], [x.max()-x.mean()]],
                            color=color, fmt='o', capsize=2, label=label if i == 0 else None)
        ax.axvline(.1, linestyle='--', color='gray'); ax.set(yticks=[0,1], yticklabels=['Count','Value'],
            xlabel='Absolute class-mean gap / TRAIN SD', title=ds); ax.invert_yaxis(); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, output, 'activity_balance_detail')
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.2))
    ax = axes[0, 0]
    for j, ds in enumerate(['Phish', 'Ponzi']):
        rows = balance[balance.dataset.eq(ds)]
        for off, prefix, color, label in [(-.08, 'before_', '#8a909b', 'Full TEST'), (.08, 'after_', ORANGE, 'Matched')]:
            for k, cov in enumerate(['count', 'value']):
                x = rows[prefix+'standardized_gap_'+cov].abs()
                ax.errorbar(x.mean(), 2*j+k+off, xerr=[[x.mean()-x.min()], [x.max()-x.mean()]],
                            fmt='o', color=color, capsize=2, label=label if j == k == 0 else None)
    ax.axvline(.1, linestyle='--', color='gray'); ax.set(yticks=range(4),
        yticklabels=['Phish count', 'Phish value', 'Ponzi count', 'Ponzi value'],
        xlabel='Absolute class-mean gap / TRAIN SD', title='(a) Activity balance')
    ax.invert_yaxis(); ax.legend(fontsize=7)
    ax = axes[0, 1]
    for i, ds in enumerate(['Phish', 'Ponzi']):
        rows = balance[balance.dataset.eq(ds)]; x = rows.positive_retention * 100
        balanced = ((rows.after_standardized_gap_count.abs() < .1) & (rows.after_standardized_gap_value.abs() < .1)).sum()
        ax.bar(i, x.mean(), color='#ffd3af', edgecolor=ORANGE, width=.55)
        ax.errorbar(i, x.mean(), yerr=[[x.mean()-x.min()], [x.max()-x.mean()]], color='gray', capsize=2)
        ax.text(i, x.mean()+5, f'{x.mean():.1f}%\n{balanced}/20 balanced', ha='center', fontsize=8)
    ax.set(xticks=[0,1], xticklabels=['Phish','Ponzi'], ylim=(0,122), ylabel='Fraud TEST retained (%)', title='(b) Coverage')
    for ax, ds in zip(axes[1], ['Phish', 'Ponzi']):
        for off, filename, color, label in [(-.09, 'FULL_PAIRED_STABILITY.csv', '#8a909b', 'Full TEST'),
                                             (.09, 'MATCHED_PAIRED_STABILITY.csv', ORANGE, 'Matched')]:
            values = pd.read_csv(repo / 'results/activity_matching' / filename)
            rows = values[values.dataset.eq(ds) & values.arm.eq('X14_SIFE44') & values.metric.eq('macro_f1')].set_index('model').loc[MODELS]
            ax.errorbar(rows.mean_delta_pp, np.arange(5)+off, xerr=rows.sample_sd_delta_pp,
                        fmt='o', markersize=4, color=color, capsize=2, label=label)
        ax.axvline(0, color='gray', lw=.8); ax.set(yticks=range(5), yticklabels=NAMES,
            xlabel='Macro-F1 gain over X14 (pp)', title=ds + (': balance not met' if ds == 'Phish' else ''))
        ax.invert_yaxis(); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, output, 'fig05_matching_overview')
    print('Recreated statistical plot content; supplied draw.io paper PDFs were not overwritten.')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=ROOT / 'generated/figures')
    a = p.parse_args(); main(a.repo.resolve(), a.output.resolve())
