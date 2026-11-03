# Data and compact result evidence

## Upstream benchmark inputs

Phish and Ponzi come from the Meta-IFD benchmark release. Follow the access instructions in the [authors' repository](https://github.com/GISec-Team/Meta-IFD), with reference commit `439c6daa936f67f19486be76642e690a2153d0c5` recorded in the experiment sources. The associated peer-reviewed article is [DOI: 10.1109/TIFS.2024.3521611](https://doi.org/10.1109/TIFS.2024.3521611).

Training consumes two inputs: the raw data archive/directory (`dataRaw`) and the processed graph release (`dataPhish`/the HEIG processed input), as explicit paths. Inputs may be ZIPs or extracted directories where supported by the notebook resolver. Paths on Kaggle are defaults, not a requirement; use the command-line runner elsewhere.

The original graphs and raw transaction tables are not redistributed in this compact artifact. Obtain them from the original source and retain the corresponding provenance/usage conditions. Do not substitute a similarly named graph without checking bytes and field alignment.

## Task and cohort definitions

| Statistic | Phish | Ponzi |
|---|---:|---:|
| Target type | EOA | CA |
| Released labeled rows | 2,763 | 1,342 |
| Unique labeled addresses | 2,762 | 1,342 |
| Main cohort | 2,761 | 1,342 |
| Main fraud / normal | 1,205 / 1,556 | 191 / 1,151 |

The main cohort removes every supervised row associated with duplicated labeled addresses. For Phish this excludes both conflicting rows of one address; Ponzi is unchanged. Graph topology remains intact. Original-mask references use their original supervised cohorts separately. Address-disjointness holds within each partition; addresses recur across the 20 repeated partitions.

Descriptors use processed incident edge records, not individual raw blockchain transactions. Edge count and value fields retain the supplied meaning; no ETH unit is inferred. For Ponzi, the notebooks restore all six float64 count/value arrays from the raw source while verifying order and byte hashes. Casting those arrays to float32 before feature extraction changes the numerical definition used for replay.

## Checksums and reference records

The frozen notebook assets `code/experiments/main/assets/release_reference_Phish.json` and `release_reference_Ponzi.json` specify the graph hashes and address/label reference information. Original `protocol.json` files under `results/*/runs/*` retain data-manifest and runtime records. Graph bytes are checked before deserialization, and restored raw arrays are checked independently.

`results/cohorts/phish/roots.npz` and `results/cohorts/ponzi/roots.npz` are compact labeled-root evidence, not the full graph. The archive contains the exported root fields, including source attributes, SIFE columns, labels, addresses and log-activity summaries. The matching code loads only `ids`, `labels`, `addresses` and `activity`. The activity columns are already log-transformed: do **not** apply another log transform.

`split_2001.json` through `split_2020.json` hold the train/validation/test membership for each task. Membership contents were checked across all five main model exports before deduplicating them into one copy per task.

## Predictions and replay scope

`results/main/test_predictions/*_partNN.csv.gz` retains the original TEST CSV rows for every main-study fit, including node IDs, labels, addresses, scores, decisions, split seed, training seed and arm. Decimal score strings were retained without rounding. Every source prediction file was checked against its original run-manifest hash before TEST rows were selected.

The provenance record `provenance/main_prediction_source_hashes.csv` maps these rows to the original source file hashes. No TRAIN/validation predictions or group-ablation prediction files are shipped. All stages include their original compact fit metrics, so the ablation table replay does not require ablation predictions.

Scores must use the frozen decision rule `score > 0.5`, with exact ties assigned normal. Matching is deterministic 1:1 fraud-to-normal matching without replacement, using TRAIN-root scaling and a fixed 0.5 distance caliper. It is not propensity-score matching.

## Resume and redistribution

The large original output ZIPs remain in the authors' private working archive; their SHA256 hashes are recorded. They contain fitted artifacts not included here. A cached exact resume needs the original work root, compatible source/runtime/data identities, sampled batches and generated views. A saved metric table or checkpoint alone is not that work root.

Do not commit newly downloaded full graphs or working caches to Git. The provided `.gitignore` excludes common raw-data, cache and checkpoint locations.

## Anonymous-mirror file sizes

Prediction CSVs are divided into four gzip shards per model/task, with five whole partitions per shard. `provenance/prediction_shards.json` maps original compact files to shards and records hashes, rows and split seeds. Original row strings and score precision are unchanged.
