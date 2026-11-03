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
