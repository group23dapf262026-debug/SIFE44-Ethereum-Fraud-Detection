import importlib.util
import subprocess
import sys
INSTALL_MISSING_PACKAGES=True
requirements={'numpy':'numpy>=1.24,<3','pandas':'pandas>=2,<4','scipy':'scipy>=1.11,<2',
              'sklearn':'scikit-learn>=1.3,<2','joblib':'joblib>=1.3,<2',
              'threadpoolctl':'threadpoolctl>=3,<4','torch':'torch>=2.1',
              'torch_geometric':'torch-geometric>=2.4,<3'}
for module,requirement in requirements.items():
    if importlib.util.find_spec(module) is None:
        if not INSTALL_MISSING_PACKAGES:raise RuntimeError(f'Install {requirement} first')
        subprocess.check_call([sys.executable,'-m','pip','install',requirement])
import torch
print('Python',sys.version.split()[0],'| Torch',torch.__version__)

"""Literal notebook setup: install native wheels only when needed, then test sampling."""
import importlib
import json
import subprocess
import sys
import torch

NATIVE_PROBE_SOURCE = '''
import json
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import torch_geometric.typing
graph=HeteroData()
for nt in ('EOA','CA'):graph[nt].num_nodes=3
for et in [('CA','call','CA'),('EOA','call','CA'),('EOA','trans','CA'),
           ('CA','trans','CA'),('CA','trans','EOA'),('EOA','trans','EOA')]:
    graph[et].edge_index=torch.tensor([[0,1,2],[1,2,0]],dtype=torch.long)
for target in ('EOA','CA'):
    loader=NeighborLoader(graph,num_neighbors=[2,2],input_nodes=(target,torch.tensor([0,1])),
        batch_size=2,shuffle=False,replace=False,subgraph_type='directional',num_workers=0)
    batch=next(iter(loader))
    assert batch[target].batch_size==2
    assert torch.equal(batch[target].n_id[:2],torch.tensor([0,1]))
print(json.dumps(dict(status='PASS',torch=torch.__version__,cuda=torch.version.cuda,
    pyg_lib=torch_geometric.typing.WITH_PYG_LIB,torch_sparse=torch_geometric.typing.WITH_TORCH_SPARSE)))
'''


def probe_native_subprocess():
    """Check the installed binary in a fresh interpreter, avoiding stale notebook flags."""
    return subprocess.run([sys.executable,'-c',NATIVE_PROBE_SOURCE],
                          capture_output=True,text=True,timeout=90)


def probe_native_current_kernel():
    """Exercise the actual heterogeneous two-hop sampler for both dataset target types."""
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborLoader
    graph=HeteroData()
    for nt in ('EOA','CA'):graph[nt].num_nodes=3
    for et in [('CA','call','CA'),('EOA','call','CA'),('EOA','trans','CA'),
               ('CA','trans','CA'),('CA','trans','EOA'),('EOA','trans','EOA')]:
        graph[et].edge_index=torch.tensor([[0,1,2],[1,2,0]],dtype=torch.long)
    for target in ('EOA','CA'):
        loader=NeighborLoader(graph,num_neighbors=[2,2],input_nodes=(target,torch.tensor([0,1])),
            batch_size=2,shuffle=False,replace=False,subgraph_type='directional',num_workers=0)
        batch=next(iter(loader))
        if batch[target].batch_size!=2 or not torch.equal(batch[target].n_id[:2],torch.tensor([0,1])):
            raise RuntimeError('Native sampler returned incorrect root IDs')
    return dict(status='PASS',target_types=['EOA','CA'],fanouts=[2,2],relations=6)


def ensure_native_backend(allow_install=True):
    """Continue after a successful install/probe; restart only for a demonstrated stale kernel."""
    fresh=probe_native_subprocess()
    if fresh.returncode!=0:
        details=(fresh.stderr or fresh.stdout)[-3500:]
        if not allow_install:
            raise RuntimeError('Native PyG sampling is not ready. Enable package installation or install matching wheels.\n'+details)
        release='.'.join(torch.__version__.split('+')[0].split('.')[:2])+'.0'
        runtime='cpu' if torch.version.cuda is None else 'cu'+torch.version.cuda.replace('.','')
        wheel_index=f'https://data.pyg.org/whl/torch-{release}+{runtime}.html'
        print('Installing native sampling wheels for Torch',torch.__version__,'from',wheel_index,flush=True)
        # No PyPI fallback, no source compilation, no Torch/CUDA replacement.
        command=[sys.executable,'-m','pip','install','--only-binary=:all:','--no-deps','--no-index',
                 '--force-reinstall','torch_scatter','torch_sparse','-f',wheel_index]
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError as error:
            raise RuntimeError('Matching native wheel installation failed for '+torch.__version__+
                '. Check Internet access and the wheel listing '+wheel_index+
                '. Torch/CUDA were not replaced. Original sampler check:\n'+details) from error
        importlib.invalidate_caches()
        fresh=probe_native_subprocess()
        if fresh.returncode!=0:
            raise RuntimeError('Native sampling still fails in a fresh interpreter after installation. '
                'A kernel restart alone is not a verified fix.\n'+(fresh.stderr or fresh.stdout)[-3500:])
    # Import PyG after installation in a clean notebook. For an already-imported PyG,
    # refresh its Python capability flags without unloading compiled extension modules.
    import torch_geometric
    import torch_geometric.typing
    importlib.invalidate_caches()
    importlib.reload(torch_geometric.typing)
    try:
        report=probe_native_current_kernel()
    except Exception as error:
        raise RuntimeError('Native sampling works in a fresh interpreter but this kernel still fails. '
            'Restart Session/Kernel once, then Run All in this same session. '
            'No training has started. Kernel error: '+str(error)) from error
    report.update(torch=torch.__version__,pyg=torch_geometric.__version__,cuda=torch.version.cuda,
        pyg_lib=torch_geometric.typing.WITH_PYG_LIB,torch_sparse=torch_geometric.typing.WITH_TORCH_SPARSE)
    print('Native heterogeneous sampler ready:',json.dumps(report),flush=True)
    return report



NATIVE_ENVIRONMENT_REPORT=ensure_native_backend(INSTALL_MISSING_PACKAGES)
