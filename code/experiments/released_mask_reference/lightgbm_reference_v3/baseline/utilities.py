import hashlib

import importlib.metadata

import json

import os

from pathlib import Path

import platform

import random

import numpy as np

import pandas as pd

import torch

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def runtime_information(device):
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'torch-geometric','joblib','threadpoolctl']
    versions = {name: importlib.metadata.version(name) for name in packages}
    try:
        versions['lightgbm'] = importlib.metadata.version('lightgbm')
    except importlib.metadata.PackageNotFoundError:
        versions['lightgbm'] = None
    for name in ['pyg-lib','torch-sparse','torch-scatter']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=versions,
                device=str(device), cuda=torch.version.cuda,
                cpu_threads=torch.get_num_threads(),
                gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
                sampler='none; X14 baseline',
                bitwise_cross_platform_equivalence_claimed=False)

def seal_files(root, files):
    root = Path(root)
    return {Path(p).relative_to(root).as_posix(): sha256(p) for p in files}

def verify_files(root, manifest):
    for relative, expected in manifest.items():
        require(sha256(Path(root)/relative) == expected, f'Frozen file changed: {relative}')

def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(value, temporary)
    os.replace(temporary, path)
