# Reproduction guide

## 1. Replay results first

The smallest environment supports only NumPy/Pandas/SciPy/scikit-learn and plotting:

```bash
python -m pip install -r requirements-analysis.txt
python code/verify_repository.py
python code/analysis/reproduce_tables.py
python code/analysis/reproduce_matching.py --output generated/matching_run_1
```

These commands inspect frozen artifacts and recompute summaries. They do not perform benchmark training. Matching refuses to overwrite an existing output directory so that its membership lock is not silently replaced.

## 2. Prepare training inputs and environment

Read [DATA.md](../DATA.md). Provide explicit raw and processed paths; the code accepts the verified release only. The full graph is much larger than the compact root evidence and can require substantial RAM/disk even for a smoke run. A reduced smoke budget still prepares/verifies the real input graph.

Create a separate training environment, install a Torch build suitable for the machine, and install the remaining requirements:

```bash
python -m pip install -r requirements-training.txt
```

The notebook environment cell probes a real heterogeneous NeighborLoader sample. For graph models, native `pyg-lib` or matching `torch-sparse`/`torch-scatter` wheels must match Torch/CUDA. The setup keeps installed Torch/CUDA and avoids source-compilation fallbacks; it requests a restart only for a demonstrated stale-kernel failure. If no matching binary exists, choose a compatible environment instead of bypassing the probe.

The actual completed-run versions are recorded in [observed_runtime_versions.json](../configs/observed_runtime_versions.json). Those are evidence, not a universal lockfile: an original CUDA/Linux wheel may not run on Windows/CPU. Bitwise equality across runtimes is not claimed. Broad installation ranges in requirements are not a claim that every possible combination was tested.

## 3. Run a notebook

All notebooks contain literal readable source cells, input preparation and the execution stages. They do not require a downloaded project-code mount. Options:

- Jupyter/Colab/local server: open a notebook, set the Configuration cell paths/device, and run cells in order.
- Kaggle: mount the two upstream inputs and use the existing defaults; choose a suitable graph accelerator when needed.
- Command line: use `code/run_notebook.py` with explicit paths.

POSIX example:

```bash
python code/run_notebook.py code/experiments/main/Phish/02_Phish_GAT_MAIN20_RUN_ALL.ipynb --raw /data/dataRaw --processed /data/dataPhish --work-root /work/sife --mode smoke
```

PowerShell example:

```powershell
python code/run_notebook.py code/experiments/main/Ponzi/04_Ponzi_LightGBM_MAIN20_RUN_ALL.ipynb --raw "D:/data/dataRaw" --processed "D:/data/dataPhish" --work-root "D:/work/sife" --mode smoke
```

The runner changes path and smoke/full/device settings in memory, not stored scientific cells. Smoke and full outputs are separated. Use `--no-install` to require dependencies to exist already. An original-mask LightGBM reference without a smoke branch requires explicit `--mode full`; the runner will not silently execute its full budget when smoke is requested.

For the actual study, change to `--mode full`: 20 partitions × 5 seeds × all supported arms for the selected task/model. Do not compare smoke values with the paper table. Do not edit feature formulas, threshold, seeds or configurations to improve TEST outcomes.

## 4. Resume

Keep the work root on persistent storage. Completed fits and artifacts are checked against their hash-bound identities before reuse. Exact ongoing-run continuation additionally needs sampled topology/views and compatible source, inputs and runtime. A compact results repository or a downloaded checkpoint alone is not an exact cached snapshot.

Meta-IFD ablation is especially expensive. The preserved sources include the reviewed session-budget/resume behavior. An interrupted or partial run is not a completed ablation, and TEST must remain gated until every requested fit is locked.

## 5. Validate changes

Run the lightweight feature invariants without graph training:

```bash
cd code/experiments/main
python -m unittest discover -s tests -p test_features.py -v
```

The full original source tests also exercise native graph training and require the complete training environment; they are not part of the analysis-only quick check. Historical source-audit helpers may expect their original release layout. Use the repository-level verifier and the documented runner for this packaged layout.

Do not call a smoke/unit test a completed 3,800- or 5,000-fit retraining. Read the current validation report for the exact checks performed when this artifact was prepared.
