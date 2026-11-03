# Validation performed while preparing this artifact

The artifact was checked locally on 4 November 2026. The following checks were performed on the packaged files, not on an inferred description of the experiments.

| Check | Scope and result |
|---|---|
| Frozen experiment notebooks | All 30 notebook code-cell sequences match their handoff sources; executable cells were not translated or rewritten |
| Completed-run source identities | All 30 main/ablation/reference notebook definition signatures match their recorded run protocols; native notebooks use IPython history normalization and the LightGBM reference signs literal cells |
| Notebook format | All 30 notebooks pass nbformat schema validation; outputs and execution counts are cleared |
| Source syntax and language | Included Python files and notebook code cells parse; repository code/documentation is English, without residual Vietnamese prose or punctuation mojibake |
| Main result replay | All 3,800 TEST fit records rebuild the recorded metric summaries and paired Macro-F1 differences |
| Ablation replay | All 5,000 TEST fit records rebuild the group contributions using the ablation's own retrained full-input controls |
| Matching construction | All 40 memberships, matched pair identities/distances and balance/coverage diagnostics match the frozen exports before prediction access |
| Matching scores | All 3,800 matched TEST fit evaluations rebuild the recorded matched summaries |
| Feature formula tests | Eight existing tests pass, including zero-mass/empty channels, typed directions, input widths, invalid inputs, fixture parity and Pure36 rescaling invariance |
| Matching algorithm tests | Five existing tests pass, including exhaustive assignment comparisons, empty support, no replacement and deterministic duplicate values |
| Local command-line runner | A tiny notebook fixture verifies real IPython source history, explicit data paths and separation of smoke/full output roots |
| Statistical redraw | Result plot scripts execute and write separate generated outputs; supplied paper figure PDFs remain unchanged |
| Integrity | Repository-level SHA256 hashes, source mappings and original archive identities are recorded; local Markdown links are checked |

## Anonymous-mirror preparation checks

Main TEST prediction evidence was split into 40 whole-partition gzip shards. Every row and decimal score string was compared with its original compact export; the largest shard is below 2.3 MB. The five paper PDF exports were scanned for the author/institution identifiers supplied for this artifact; no such identifiers were found in extracted text or metadata.

An in-memory/local-directory simulation redacted only identity-bearing presentation files. The `--anonymous-view` verifier passed with all 30 frozen notebook signatures and scientific file hashes still required. A deliberately modified scientific Python file was then rejected by its checksum. This is a local test, not a claim that an online mirror has been created or independently reviewed.

## Limits of these checks

No fresh full graph training, new hyperparameter search, threshold tuning, or classifier inference from saved neural checkpoints was performed during packaging. The 3,800 and 5,000 fit counts refer to completed frozen artifacts from the prior runs, not new training in this packaging session.

The packaging host has no Torch/PyG installation. The full native GNN unit/smoke training suite was therefore not rerun here. Formula, assignment, source-identity and frozen-result replays were run. The original completed run environments are recorded separately in `configs/observed_runtime_versions.json`.

The local checks do not certify every Python/Torch/CUDA/OS combination. GitHub Actions configuration is included, but it has not run on GitHub because this artifact has not been published. Original archive manifests can reference excluded checkpoints; only `MANIFEST.sha256` defines this compact repository's file inventory.

A scoped MIT grant now covers the listed author-owned contributions. Upstream redistribution permission for the mixed Meta-IFD components remains unconfirmed; the license update is not a whole-repository permission clearance. No public repository URL or paper DOI is asserted.
