"""Deterministic, label-free permutation of complete SIFE vectors within role/type."""
from functools import lru_cache
from pathlib import Path
import numpy as np


def shuffle_positions(ids, node_type, role, split_seed):
    """Return a bijective donor index, stable under reordering of the same IDs.

    Ordinary random permutations may contain fixed points. Labels, predictions,
    X14, and other roles do not enter this mapping. Different roles/types/partitions
    use independent deterministic streams. This is one negative control per
    partition, shared across all training seeds, not a permutation significance test.
    """
    ids = np.asarray(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique integer IDs are required for shuffling')
    if node_type not in ('EOA','CA') or role not in ('train','validation','test'):
        raise ValueError('Known node type and role required')
    entropy = [44012, int(split_seed),
               ('train','validation','test').index(role), ('EOA','CA').index(node_type)]
    generator = np.random.default_rng(np.random.SeedSequence(entropy))
    canonical = np.argsort(ids, kind='stable')
    result = np.empty(len(ids), dtype=np.int64)
    result[canonical] = canonical[generator.permutation(len(ids))]
    result.setflags(write=False)
    return result


@lru_cache(maxsize=8)
def cached_shuffle_positions(keys_file, node_type, role, split_seed):
    keys = np.load(keys_file, mmap_mode='r', allow_pickle=False)
    return shuffle_positions(keys,node_type,role,split_seed)


def save_shuffle_maps(root, target_type, role, split_seed, tabular):
    """Save donor IDs before fitting/scoring this role; cache manifests hash them."""
    root = Path(root)
    if tabular:
        with np.load(root/'roots.npz',allow_pickle=False) as values:
            typed_ids = {target_type:values['ids'].copy()}
    else:
        typed_ids = {nt:np.load(root/'features'/(nt+'_ids.npy'),mmap_mode='r') for nt in ('EOA','CA')}
    for nt,ids in typed_ids.items():
        order = shuffle_positions(ids,nt,role,split_seed)
        np.save(root/f'shuffle_{nt}_donor_ids.npy',np.asarray(ids[order]))
