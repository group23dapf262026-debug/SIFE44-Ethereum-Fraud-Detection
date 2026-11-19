# Direct and Shuffle follow-up experiments

The release contains 20 standalone notebooks for Phish and Ponzi. Each
experiment covers Meta-IFD, GAT, GraphSAGE, LightGBM, and Random Forest.
Both suites reuse the main study's 20 partitions and five training seeds.

| Experiment | Notebook directory | TEST fits | Paired comparisons |
|---|---|---:|---:|
| Direct comparison with adapted UFE12 | [Direct notebooks](notebooks/direct) | 5,000 | 40 |
| Aligned and shuffled SIFE44 | [Shuffle notebooks](notebooks/shuffle) | 3,000 | 30 |

The Direct suite compares Pure36, SIFE44, and their combinations with
UFE12 against the UFE12 control. Every arm retains X14. UFE12 is adapted
to the same processed incident records as SIFE44. The Shuffle suite
compares X14 and both SIFE44 augmentation arms. It permutes whole
descriptor vectors within partition, subset, and node type.

## Replay the reported comparisons

Run these commands from the repository root.

```bash
python -m pip install -r requirements-analysis.txt
python code/analysis/reproduce_followup.py
```

The script reads [saved TEST metrics](../../../results/followup), averages
the five seeds within each partition, and writes the replay to
`generated/followup`. It checks the replay against the saved summaries,
paired differences, pointwise 95% corrected-resampling intervals,
and Holm-adjusted p-values. Holm adjustment applies within each suite.

## Run the notebooks

See [data access](../../../DATA.md) for the original graph inputs.
Select a notebook, set the raw and processed input paths, and use
`RUN_MODE='smoke'` to check the environment. Use `RUN_MODE='full'` with
persistent working storage to reproduce a full task. The training runner
supports the same inputs on a local machine. From this directory:

```bash
python -m pip install -r requirements-training.txt
python run_notebook.py notebooks/direct/Phish/05_Phish_RF_DIRECT20_RUN_ALL.ipynb \
  --raw /data/raw --processed /data/HEIG --work-root /work/sife44 \
  --mode smoke --device cpu
```

Graph classifiers need compatible PyTorch Geometric native sampling
packages. The [analysis plan](configs/ANALYSIS_PLAN.json) defines the
arms and comparison families. The [source signatures](../../../results/followup/source_signatures.json)
record the notebook code-cell hashes checked against the completed runs.
See [third-party notices](THIRD_PARTY_NOTICES.md) and
[license scope](CODE_LICENSE_SCOPE.json) for component-specific terms.
