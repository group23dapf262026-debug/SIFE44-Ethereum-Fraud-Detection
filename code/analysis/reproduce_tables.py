"""Rebuild seed-then-partition summaries and paired comparisons from frozen metrics."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = ['macro_f1', 'micro_f1', 'binary_f1', 'precision', 'recall', 'accuracy',
           'average_precision', 'roc_auc', 'fpr']

def summarize(frame):
    split = frame.groupby(['dataset', 'model', 'arm', 'split_seed'], sort=True)[METRICS].mean().reset_index()
    rows = []
    for (dataset, model, arm), group in split.groupby(['dataset', 'model', 'arm'], sort=True):
        assert len(group) == 20
        for metric in METRICS:
            values = group[metric].to_numpy() * 100
            rows.append(dict(dataset=dataset, model=model, arm=arm, metric=metric,
                             mean_percent=float(values.mean()), sd_across_split_means_pp=float(values.std(ddof=1))))
    return split, pd.DataFrame(rows)

def compare(actual, expected, keys, columns, atol=1e-10):
    a = actual.set_index(keys).sort_index(); b = expected.set_index(keys).sort_index()
    assert a.index.equals(b.index), 'Coverage differs from the recorded export'
    np.testing.assert_allclose(a[columns].to_numpy(float), b[columns].to_numpy(float), atol=atol, rtol=0)

def rebuild(repo, output):
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for stage, expected_count in [('main', 3800), ('ablation', 5000)]:
        frames = []
        for path in sorted((repo / 'results' / stage / 'runs').glob('*/metrics.csv')):
            frame = pd.read_csv(path, float_precision='round_trip')
            frames.append(frame.loc[frame.role.eq('test')].copy())
        assert len(frames) == 10
        all_fits = pd.concat(frames, ignore_index=True)
        identity = ['dataset', 'model', 'arm', 'split_seed', 'model_seed']
        assert not all_fits.duplicated(identity).any()
        assert len(all_fits) == expected_count
        assert all_fits.groupby(identity[:-1]).size().eq(5).all()
        split, summary = summarize(all_fits)
        split.to_csv(output / f'{stage}_split_means.csv', index=False)
        summary.to_csv(output / f'{stage}_summary.csv', index=False)
        if stage == 'main':
            expected = pd.read_csv(repo / 'results/main/main_VERIFIED_SUMMARY.csv')
            compare(summary, expected, ['dataset', 'model', 'arm', 'metric'],
                    ['mean_percent', 'sd_across_split_means_pp'])
            deltas = []
            for (ds, model), group in split.groupby(['dataset', 'model']):
                p = group.pivot(index='split_seed', columns='arm', values='macro_f1')
                for arm in p.columns:
                    if arm == 'X14': continue
                    d = (p[arm] - p.X14).to_numpy() * 100
                    deltas.append(dict(dataset=ds, model=model, arm=arm, delta_pp=d.mean(),
                                       sd_delta_pp=d.std(ddof=1), wins=int((d > 0).sum()),
                                       ties=int((d == 0).sum()), losses=int((d < 0).sum())))
            delta = pd.DataFrame(deltas)
            expected = pd.read_csv(repo / 'results/main/main_PAIRED_DELTAS.csv')
            compare(delta, expected, ['dataset', 'model', 'arm'], ['delta_pp', 'sd_delta_pp', 'wins', 'ties', 'losses'])
            delta.to_csv(output / 'main_paired_deltas.csv', index=False)
        else:
            contributions = []
            for (ds, model), group in split.groupby(['dataset', 'model']):
                p = group.pivot(index='split_seed', columns='arm', values='macro_f1')
                for arm in p.columns:
                    if arm == 'X14_SIFE44': continue
                    d = (p.X14_SIFE44 - p[arm]).to_numpy() * 100
                    contributions.append(dict(dataset=ds, model=model, removed_arm=arm,
                        mean_contribution_full_minus_removed_pp=d.mean(), sd_contribution_pp=d.std(ddof=1)))
            contribution = pd.DataFrame(contributions)
            expected = pd.read_csv(repo / 'results/ablation/PAIRED_GROUP_CONTRIBUTIONS.csv')
            compare(contribution, expected, ['dataset', 'model', 'removed_arm'],
                    ['mean_contribution_full_minus_removed_pp', 'sd_contribution_pp'])
            contribution.to_csv(output / 'group_contributions.csv', index=False)
        counts[stage] = dict(fits=len(all_fits), partitions=20, seeds_per_partition=5, verified_against_recorded_results=True)
        print(f'{stage}: {len(all_fits)} TEST fit records verified', flush=True)
    (output / 'TABLE_REPLAY.json').write_text(json.dumps(counts, indent=2) + '\n', encoding='utf-8')
    return counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'generated/tables')
    args = parser.parse_args()
    rebuild(args.repo.resolve(), args.output.resolve())
