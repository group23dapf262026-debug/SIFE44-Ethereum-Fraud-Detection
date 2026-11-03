"""Read both user-facing download formats and reject a changed result artifact."""
from pathlib import Path
import tempfile
import unittest
import zipfile
import hashlib
import json
from tools.collect_results import read_result

ROOT=Path(__file__).resolve().parents[1]

class CollectorFormatChecks(unittest.TestCase):
    def make_export(self,path):
        files={
            'protocol.json':json.dumps({'protocol':{'study':'sife20_group_ablation_five_fixed_v1','run_mode':'smoke'}}).encode(),
            'summary.csv':b'dataset,model,arm,metric,mean_percent\nPhish,rf,X14_SIFE44,macro_f1,80\n',
            'split_means.csv':b'dataset,model,arm,split_seed,macro_f1\nPhish,rf,X14_SIFE44,2001,0.8\n'}
        marker={'status':'COMPLETE','format_fixture_only':True,'files':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()}}
        with zipfile.ZipFile(path,'w') as z:
            for name,payload in files.items():z.writestr('results/smoke/'+name,payload)
            z.writestr('results/smoke/RESULT_MANIFEST.json',json.dumps(marker))

    def test_direct_and_outer_kaggle_zip_give_identical_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'direct.zip';self.make_export(source)
            original=read_result(source,True)
            outer=Path(folder)/'results.zip'
            with zipfile.ZipFile(source) as z,zipfile.ZipFile(outer,'w') as out:
                for name in z.namelist():out.writestr('kaggle_working/'+name,z.read(name))
            actual=read_result(outer,True)
            self.assertEqual(original[0],actual[0]);self.assertEqual(original[1],actual[1])
            self.assertTrue(original[2].equals(actual[2]));self.assertTrue(original[3].equals(actual[3]))

    def test_tampered_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'direct.zip';self.make_export(source)
            changed=Path(folder)/'changed.zip'
            with zipfile.ZipFile(source) as z,zipfile.ZipFile(changed,'w') as out:
                for name in z.namelist():
                    payload=z.read(name)
                    if name=='results/smoke/summary.csv':payload+=b'changed'
                    out.writestr(name,payload)
            with self.assertRaisesRegex(ValueError,'Changed artifact'):read_result(changed,True)

if __name__=='__main__':unittest.main()
