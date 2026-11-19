from .utilities import require, sha256, read_json, write_json, atomic_torch_save
from pathlib import Path

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import torch

from torch_geometric.data import HeteroData

from torch_geometric.loader import NeighborLoader

import torch_geometric.typing

def save_pyg_role_batches(graph, membership, target_type, role, output_dir,
                          sampler_seed, fanouts=(100,100), batch_size=512):
    """Cache actual PyG NeighborLoader samples; root IDs precede neighbors, as in author code."""
    require(torch_geometric.typing.WITH_PYG_LIB or torch_geometric.typing.WITH_TORCH_SPARSE,
            'NeighborLoader needs a working pyg-lib or torch-sparse installation for this Torch build; no sampler fallback is allowed')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sampling needs topology and node counts, not full features or edge attributes.
    topology = HeteroData()
    for nt in graph.node_types:
        topology[nt].num_nodes = graph[nt].num_nodes
    for et in graph.edge_types:
        topology[et].edge_index = graph[et].edge_index
    roots = torch.tensor(membership.node_id.to_numpy(copy=True), dtype=torch.long)
    records = []
    # Explicit per-role seeds are a reproducibility correction, not recovered author RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sampler_seed)
        loader = NeighborLoader(topology, num_neighbors=list(fanouts),
                    input_nodes=(target_type, roots), batch_size=batch_size,
                    shuffle=role != 'test', replace=False, subgraph_type='directional',
                    num_workers=0, filter_per_worker=False)
        for sampled in loader:
            size = int(sampled[target_type].batch_size)
            ids = {nt: sampled[nt].n_id.cpu().clone() for nt in graph.node_types}
            seed_ids = ids[target_type][:size]
            batch = dict(node_ids=ids,
                         edge_index={et: sampled[et].edge_index.cpu().clone() for et in graph.edge_types},
                         seed_ids=seed_ids, seed_labels=graph[target_type].y[seed_ids].cpu().clone(),
                         target_type=target_type)
            require(torch.equal(roots[sampled[target_type].input_id], seed_ids), 'PyG input/root identity mismatch')
            path = output_dir/f'batch_{len(records):04d}.pt'
            atomic_torch_save(batch, path)
            records.append(dict(file=path.name, sha256=sha256(path), seeds=seed_ids.tolist()))
    actual = [node for record in records for node in record['seeds']]
    require(len(actual) == len(set(actual)) == len(membership) and set(actual) == set(membership.node_id),
            'PyG sampled root coverage mismatch')
    write_json(output_dir/'index.json', dict(role=role, seed=sampler_seed,
        backend='pyg_neighbor_loader_directional_v1', fanouts=list(fanouts),
        batch_size=batch_size, records=records))
    return load_role_batches(output_dir)

def load_role_batches(output_dir):
    output_dir = Path(output_dir)
    index = read_json(output_dir/'index.json')
    records = []
    for row in index['records']:
        path = output_dir/row['file']
        require(sha256(path) == row['sha256'], 'Sampled topology cache changed')
        records.append(dict(row, path=str(path)))
    return records
