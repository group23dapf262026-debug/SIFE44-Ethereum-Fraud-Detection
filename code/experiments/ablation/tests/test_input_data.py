"""Exercise portable ZIP/folder input and raw float64 restoration with real files."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from sife20.data import resolve_input,open_raw_member,restore_float64_attributes,obtain_graph
from sife20.utilities import sha256 as file_hash


class InputChecks(unittest.TestCase):
    def test_zip_folder_and_locked_float64_restoration(self):
        content=b':START_ID,:END_ID,count,sum\na,b,1,0.5\nc,d,3,2.25\n'
        member='Meta-IFD data/Ponzi_unbalance/edge/call_ca_ca.csv'
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);raw=root/'raw'
            (raw/'Ponzi_unbalance/edge').mkdir(parents=True)
            (raw/'Ponzi_unbalance/CA_labeled.csv').write_text('Address\n')
            (raw/'Ponzi_unbalance/edge/call_ca_ca.csv').write_bytes(content)
            archive=root/'OneDrive_2026-09-18.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr(member,content)
            golden=root/'golden.npy';np.save(golden,np.array([[1.,.5],[3.,2.25]]))
            reference=dict(attribute_provenance=dict(relations={'CA__call__CA':dict(file='CA__call__CA_attrs.npy',rows=2,sha256=file_hash(golden))}),
                raw_members={member:dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())})
            self.assertEqual(resolve_input(raw,'Ponzi','raw'),raw.resolve())
            for i,source in enumerate([raw,archive]):
                with open_raw_member(source,member) as stream:self.assertEqual(stream.read(),content)
                output=root/f'restored{i}'
                restore_float64_attributes(source,output,reference)
                np.testing.assert_array_equal(np.load(output/'CA__call__CA_attrs.npy'),np.load(golden))
                restore_float64_attributes(source,output,reference)  # Verified cache reuse.
                (output/'CA__call__CA_attrs.npy').write_bytes(b'changed')
                with self.assertRaises(ValueError):restore_float64_attributes(source,output,reference)

    def test_processed_zip_and_file_hash_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'data.pt';source.write_bytes(b'known fixture, not deserialized')
            sha=file_hash(source)
            self.assertEqual(obtain_graph(source,root/'direct','Phish',sha),source)
            archive=root/'HEIG.zip'
            with zipfile.ZipFile(archive,'w') as z:z.write(source,arcname='Phish/processed/data.pt')
            actual=obtain_graph(archive,root/'extract','Phish',sha)
            self.assertEqual(actual.read_bytes(),source.read_bytes())
            with self.assertRaises(ValueError):obtain_graph(source,root/'wrong','Phish','0'*64)


if __name__=='__main__':unittest.main(verbosity=2)
