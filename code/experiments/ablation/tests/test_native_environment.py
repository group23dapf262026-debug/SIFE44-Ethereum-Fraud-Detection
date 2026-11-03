"""Reproduce the install-success stop bug and exercise real CPU heterogeneous sampling."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_environment',ROOT/'tools/native_environment.py')
environment=importlib.util.module_from_spec(spec);spec.loader.exec_module(environment)


def result(code=0,error=''):
    return subprocess.CompletedProcess(args=[],returncode=code,stdout='PASS' if code==0 else '',stderr=error)


class NativeEnvironmentTests(unittest.TestCase):
    def test_successful_install_continues_without_unconditional_restart(self):
        with patch.object(environment,'probe_native_subprocess',side_effect=[result(1,'missing native backend'),result()]), \
             patch.object(environment.subprocess,'check_call') as install, \
             patch.object(environment,'probe_native_current_kernel',return_value={'status':'PASS'}):
            report=environment.ensure_native_backend(True)
        self.assertEqual(report['status'],'PASS')
        command=install.call_args.args[0]
        self.assertIn('--only-binary=:all:',command);self.assertIn('--no-index',command)
        self.assertIn('--no-deps',command);self.assertIn('--force-reinstall',command)
        self.assertNotIn('torch',command)

    def test_failed_fresh_probe_after_install_is_not_reported_as_restart_fix(self):
        with patch.object(environment,'probe_native_subprocess',side_effect=[result(1,'missing'),result(1,'undefined binary symbol')]), \
             patch.object(environment.subprocess,'check_call'), \
             patch.object(environment,'probe_native_current_kernel') as current:
            with self.assertRaisesRegex(RuntimeError,'fresh interpreter after installation'):
                environment.ensure_native_backend(True)
        current.assert_not_called()

    def test_restart_only_when_fresh_interpreter_passes_but_kernel_fails(self):
        with patch.object(environment,'probe_native_subprocess',return_value=result()), \
             patch.object(environment.subprocess,'check_call') as install, \
             patch.object(environment,'probe_native_current_kernel',side_effect=ImportError('stale extension')):
            with self.assertRaisesRegex(RuntimeError,'works in a fresh interpreter but this kernel still fails'):
                environment.ensure_native_backend(True)
        install.assert_not_called()

    def test_missing_backend_in_offline_mode_does_not_install(self):
        with patch.object(environment,'probe_native_subprocess',return_value=result(1,'not installed')), \
             patch.object(environment.subprocess,'check_call') as install:
            with self.assertRaisesRegex(RuntimeError,'Enable package installation'):
                environment.ensure_native_backend(False)
        install.assert_not_called()

    def test_real_sampling_refreshes_cached_false_flags_without_install(self):
        import torch_geometric.typing
        # Recreate the cached capability flag in the screenshot's already-imported PyG.
        with patch.object(torch_geometric.typing,'WITH_TORCH_SPARSE',False), \
             patch.object(torch_geometric.typing,'WITH_PYG_LIB',False), \
             patch.object(environment.subprocess,'check_call',side_effect=AssertionError('Already installed; do not reinstall')):
            report=environment.ensure_native_backend(False)
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(report['target_types'],['EOA','CA']);self.assertEqual(report['relations'],6)

    def test_six_gnn_notebooks_contain_corrected_preflight_and_same_science(self):
        count=0
        for dataset in ('Phish','Ponzi'):
            for model in ('MetaIFD','GAT','GraphSAGE'):
                path=next((ROOT/dataset).glob('*_'+model+'_SIFE20_GROUP_ABLATION_RUN_ALL.ipynb'))
                notebook=json.loads(path.read_text(encoding='utf-8'))
                setup=next(''.join(c['source']) for c in notebook['cells'] if 'environment' in c.get('metadata',{}).get('tags',[]))
                self.assertIn('probe_native_current_kernel',setup)
                self.assertIn('NATIVE_ENVIRONMENT_REPORT=ensure_native_backend',setup)
                self.assertNotIn('Native wheels installed. Restart this kernel',setup)
                self.assertNotIn('import torch_geometric',setup[:setup.index('NATIVE_PROBE_SOURCE')])
                count+=1
        self.assertEqual(count,6)


if __name__=='__main__':unittest.main()
