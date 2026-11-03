# Provenance and preservation

The artifact was assembled from the reviewed `SIFE_PHISH_PONZI_MAIN_EXPERIMENTS_20261030` handoff, final metric CSVs, and the five final author-supplied draw.io PDF exports.

## Source code

`provenance/source_files.json` records original and artifact file hashes. Python implementation files are byte-identical copies. All 30 notebook code-cell sequences are unchanged; outputs/execution metadata were removed and legacy punctuation was repaired in Markdown prose only. `provenance/notebook_code_identity.json` records those code-cell sequence fingerprints. Renaming a directory or clearing outputs does not establish cross-platform bitwise numerical equivalence.

The independently frozen main, ablation and released-mask implementations are kept in separate directories. They are not consolidated into a new shared trainer because that would obscure the exact code underlying completed runs. Some source modules retain historical compatibility branches; the supplied main/ablation dispatchers and notebook matrix contain only the five selected classifiers.

## Original results versus compact evidence

Thirty original output archives remain unchanged in the authors' working archive. Their hashes, protocol hashes and original status are recorded in `original_result_archives.json`. Original `RESULT_MANIFEST.json` and locks are preserved for traceability and can reference omitted checkpoints or caches. They are not a list of files promised by the compact repository.

The artifact's own `MANIFEST.sha256` enumerates included files. It is distinct from the original scientific fit/test locks. TEST-only prediction rows were selected from manifest-verified source CSVs without rounding score strings; the source-file mapping is `main_prediction_source_hashes.csv`.

`.gitattributes` disables automatic line-ending conversion so that a Git checkout preserves these hash-bound source bytes on different operating systems.

Main metric replay uses all main TEST fit records. Group-removal replay uses all ablation TEST records and their own controls. Matching reconstruction reads cohort/membership inputs before opening any prediction file and verifies its results against the frozen pair identities and balance exports.

## Figures

The five `figures/paper` PDFs are unchanged copies of the manuscript exports. Figure 1 and Figure 2 are author-maintained schematics; their editable draw.io source is not inferred from the PDF. The statistical redraw script recreates data content for Figures 3–5 in a separate output directory; it does not claim identical draw.io styling or generate missing schematic source.

## Scope

Packaging and replay add no training, hyperparameter search, threshold selection, best-seed filtering, changed cohort, or new scientific results. Dates in folder names identify historical handoffs, not acceptance or publication. This artifact was prepared locally; creation of a public GitHub repository or an anonymous mirror is a separate action by the authors.

## License metadata update

The authors selected a scoped MIT grant for their original contributions. The full-file and notebook-cell lists are recorded in `CODE_LICENSE_SCOPE.json`. No experiment implementation, notebook executable cell, numeric decoder tensor, prediction or result value was modified to add this license. No upstream-derived component was relicensed. Upstream distribution permission remains a separate unresolved provenance item.

## Review-mirror preparation

Main TEST evidence was partition-sharded to keep each file below 4 MB without changing rows or decimal score strings. Anonymous-view verification skips only hashes of explicitly listed identity-bearing presentation files; all scientific source and data checks remain mandatory. Actual creation of an online anonymous mirror was not performed during this preparation.
