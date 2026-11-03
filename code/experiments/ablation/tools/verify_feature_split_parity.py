"""Compare recomputed root features and all20partitions with the reviewed prior definition."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sife20.protocol import stratified_parts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reviewed-project',required=True);args=parser.parse_args()
    reference=Path(args.reviewed_project);sys.path.insert(0,str(reference))
    from sife.lr_study import stratified_parts as reviewed_parts
    reports=[]
    for dataset in ('Phish','Ponzi'):
        actual=ROOT/'validation/real_notebooks_release'/dataset/'rf/data/roots.npz'
        with np.load(actual,allow_pickle=False) as values:data={key:values[key] for key in values.files}
        with np.load(reference/'validation/real_feature_replay'/f'{dataset}_features.npz',allow_pickle=False) as values:
            old={key:values[key] for key in values.files}
        for name in ('ids','labels','source14'):np.testing.assert_array_equal(data[name],old[name])
        np.testing.assert_array_equal(data['addresses'].astype('U42'),old['addresses'].astype('U42'))
        np.testing.assert_allclose(data['sife44'],old['sife44'],rtol=1e-8,atol=1e-10)
        for seed in range(2001,2021):
            expected=reviewed_parts(data,seed);observed=stratified_parts(data,seed)
            for role in ('train','validation','test'):np.testing.assert_array_equal(observed[role],expected[role])
        reports.append(dict(dataset=dataset,strict_roots=len(data['ids']),all_twenty_split_memberships_identical=True,
            source14_exact=True,labels_addresses_exact=True,
            sife44_max_absolute_error=float(np.max(np.abs(data['sife44']-old['sife44'])))))
    report=dict(status='PASS',feature_code_sha256=hashlib.sha256((ROOT/'sife20/features.py').read_bytes()).hexdigest(),
                note='Feature selectors/groups changed; numeric extractor parity validated on actual roots.',
                reference_feature_sha256=hashlib.sha256((reference/'sife/features.py').read_bytes()).hexdigest(),datasets=reports)
    (ROOT/'validation/feature_split_parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
