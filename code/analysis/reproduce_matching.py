"""Replay activity matching from cohort records before opening any prediction file."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('frozen_posthoc', Path(__file__).with_name('original') / 'analyze.py')
original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def freeze_memberships(repo, output):
    """Only cohort, split membership and the fixed plan are read in this phase."""
    plan = json.loads((repo / 'configs/activity_matching.json').read_text(encoding='utf-8'))
    caliper = float(plan['activity_matching']['caliper'])
    subsets = {}; pairs = []; balances = []; scaling = []
    for ds in ['Phish', 'Ponzi']:
        folder = repo / 'results/cohorts' / ds.lower()
        with np.load(folder / 'roots.npz', allow_pickle=False) as raw:
            roots = {k: raw[k] for k in ['ids', 'labels', 'addresses', 'activity']}
        byid = {int(v): i for i, v in enumerate(roots['ids'])}
        for seed in range(2001, 2021):
            member = json.loads((folder / f'split_{seed}.json').read_text(encoding='utf-8'))
            train = np.array([byid[v['node_id']] for v in member['train']])
            test = np.array(sorted([byid[v['node_id']] for v in member['test']], key=lambda i: int(roots['ids'][i])))
            mu = roots['activity'][train].mean(axis=0)
            sd = roots['activity'][train].std(axis=0, ddof=0); sd = np.where(sd == 0, 1., sd)
            x = (roots['activity'] - mu) / sd
            positive = test[roots['labels'][test] == 1]; negative = test[roots['labels'][test] == 0]
            index, distance = original.optimal_pairs(x[positive], x[negative], caliper)
            pos = positive[index[:, 0]]; neg = negative[index[:, 1]]
            ids = set(roots['ids'][np.concatenate([pos, neg])].astype(int).tolist())
            assert len(ids) == 2 * len(pos)
            assert not ids.intersection(roots['ids'][train].tolist())
            subsets[ds, seed] = ids
            for pair, (i, j, value) in enumerate(zip(pos, neg, distance)):
                pairs.append(dict(dataset=ds, split_seed=seed, pair_number=pair,
                    positive_id=int(roots['ids'][i]), negative_id=int(roots['ids'][j]), distance=float(value)))
            row = dict(dataset=ds, split_seed=seed, test_positive=len(positive), test_negative=len(negative),
                       matched_pairs=len(pos), positive_retention=len(pos) / len(positive))
            for k, name in enumerate(['count', 'value']):
                row['before_standardized_gap_' + name] = float(x[positive, k].mean() - x[negative, k].mean())
                row['after_standardized_gap_' + name] = float(x[pos, k].mean() - x[neg, k].mean())
                scaling.append(dict(dataset=ds, split_seed=seed, variable=name, train_mean=mu[k], train_population_sd=sd[k]))
            balances.append(row)
    pairs = pd.DataFrame(pairs); pairs.to_csv(output / 'MATCHED_PAIRS.csv', index=False)
    balances = pd.DataFrame(balances); balances.to_csv(output / 'MATCH_COVERAGE_BALANCE.csv', index=False)
    pd.DataFrame(scaling).to_csv(output / 'TRAIN_ACTIVITY_SCALING.csv', index=False)
    lock = {'status': 'ALL40MATCH_MEMBERSHIPS_FROZEN_BEFORE_PREDICTION_READ',
            'members': {f'{ds}/{seed}': sorted(ids) for (ds, seed), ids in subsets.items()},
            'files': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob('*.csv')}}
    write_json(output / 'MATCH_LOCK.json', lock)
    expected_pairs = pd.read_csv(repo / 'results/activity_matching/MATCHED_PAIRS.csv')
    keys = ['dataset', 'split_seed', 'pair_number']
    a = pairs.set_index(keys).sort_index(); b = expected_pairs.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_array_equal(a[['positive_id', 'negative_id']], b[['positive_id', 'negative_id']])
    np.testing.assert_allclose(a.distance, b.distance, atol=1e-12, rtol=0)
    expected_balance = pd.read_csv(repo / 'results/activity_matching/MATCH_COVERAGE_BALANCE.csv')
    a = balances.set_index(['dataset', 'split_seed']).sort_index()
    b = expected_balance.set_index(['dataset', 'split_seed']).sort_index()
    np.testing.assert_allclose(a.to_numpy(float), b[a.columns].to_numpy(float), atol=1e-12, rtol=0)
    print('All 40 matching memberships and balance diagnostics verified before prediction access', flush=True)
    return subsets

def replay(repo, output):
    output.mkdir(parents=True, exist_ok=False)
    subsets = freeze_memberships(repo, output)
    lock_hash = hashlib.sha256((output / 'MATCH_LOCK.json').read_bytes()).hexdigest()
    records = []
    for path in sorted((repo / 'results/main/test_predictions').glob('*.csv.gz')):
        frame = pd.read_csv(path, float_precision='round_trip')
        assert frame.role.eq('test').all()
        for (ds, model, seed, model_seed, arm), group in frame.groupby(['dataset', 'model', 'split_seed', 'model_seed', 'arm'], sort=True):
            ids = subsets[ds, int(seed)]
            assert group.node_id.is_unique
            assert np.array_equal(group.prediction, (group.score > .5).astype(int))
            selected = group.loc[group.node_id.isin(ids)]
            assert set(selected.node_id) == ids
            assert selected.label.sum() * 2 == len(selected)
            records.append(dict(dataset=ds, model=model, split_seed=seed, model_seed=model_seed, arm=arm,
                                **original.metrics(selected)))
        assert hashlib.sha256((output / 'MATCH_LOCK.json').read_bytes()).hexdigest() == lock_hash
        print('Frozen matching scores replayed:', path.name, flush=True)
    per_fit = pd.DataFrame(records); assert len(per_fit) == 3800
    per_fit.to_csv(output / 'MATCHED_PER_FIT_METRICS.csv', index=False)
    split, summary = original.summarize(per_fit)
    split.to_csv(output / 'MATCHED_SPLIT_MEANS.csv', index=False)
    summary.to_csv(output / 'MATCHED_SUMMARY.csv', index=False)
    expected = pd.read_csv(repo / 'results/activity_matching/MATCHED_SUMMARY.csv')
    keys = ['dataset', 'model', 'arm', 'metric']
    a = summary.set_index(keys).sort_index(); b = expected.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a[['mean_percent', 'sd_across_split_means_pp']],
                               b[['mean_percent', 'sd_across_split_means_pp']], atol=1e-10, rtol=0)
    write_json(output / 'MATCHING_REPLAY.json', dict(partitions=40, frozen_test_fit_evaluations=3800,
               matching_and_summary_verified=True, training_performed=False, threshold_search=False))
    print('Verified 3,800 matched TEST fit evaluations against the recorded summary', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'generated/matching')
    args = parser.parse_args(); replay(args.repo.resolve(), args.output.resolve())
