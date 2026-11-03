# Column-group ablation

This is a post-hoc follow-up to the completed main study. The same five model families and both datasets are retained. No arm/model/dataset is selected because its prior improvement was favorable.

The44SIFE input columns are divided into four disjoint groups:

| Group | Columns | Meaning |
|---|---:|---|
| distribution20 |20| Entropy,normalized entropy,HHI,JSD of frequency/value,and maximum value share for four typed/directional channels |
| flow8 |8| In-minus-out entropy,out-minus-in HHI,record-degree asymmetry and value-flow asymmetry for call/transfer |
| counterparty8 |8| CA partner record share and count-weight share for four channels |
| intensity8 |8| Mean value and standard deviation of log1p(value) for four channels |

`GROUP_COLUMNS.csv` specifies every zero-based index,name and group. Each SIFE column occurs exactly once. Ordering is preserved after deletion. No deleted column is replaced by zero or by another computed feature.

| Active arm | X14 columns | Kept SIFE columns | Total |
|---|---:|---:|---:|
| X14_SIFE44 |14|44|58|
| without_distribution20 |14|24|38|
| without_flow8 |14|36|50|
| without_counterparty8 |14|36|50|
| X14_Pure36 |14|36|50|

The last arm is the deletion of intensity8. It has the same input definition as X14+Pure36 in the main study. Pure36 alone and X14 alone are not active arms here. All five models,including MetaIFD,support all five arms because X14 is always retained.

The comparison is **full control minus removed-group arm**,after averaging five model seeds within each partition. A positive contribution means the full control performed better; a negative contribution means the deletion performed better. Report decreases and variability as well as increases. Mean/sampleSD over20overlapping partitions is descriptive evidence,not100independent replications.

## Scope of the intervention

This removes input columns,not all possible information about a concept. Removing distribution20 still leaves distribution-derived funnel contrasts in flow8. Removing counterparty8 leaves EOA/CA graph types. Removing intensity8 leaves all original X14 monetary attributes. Therefore neither deletion is a causal intervention or proof of independence from money/type/flow information.

Group size differs and network input widths/parameter counts change. Report parameter_count and fit_seconds from each fit; do not call this capacity-matched. No feature selection,HPO,threshold tuning or comparison-driven recipe changes are permitted in this run.

## Pairing and preservation

Fit a fresh full58control within each ablation notebook. All arms use the same strict membership,model seeds,per-split sampled topology and Meta decoder views. Preserve unchanged feature formulas,backbones,model-specific preprocessing,fixed configs and decision score>.5. Compatibility helpers for source-only/Pure-only backbone tests do not add experimental arms.

The completed main-study classifiers/caches are not imported as fitted ablation controls: their scientific identities and potentially runtime/sample draws differ. Keep both result sets and do not overwrite them. All500requested fits in each model/dataset notebook must finish and be hash-locked before any TEST sampling/scoring.

## Budget

20split seeds2001–2020 ×5model seeds12–16 ×5arms =500fits per notebook. Five models ×two datasets =10notebooks,5000fits total. Smoke uses1split,1seed,5arms; GNNs use2epochs/fanout2. Smoke scores are not publishable benchmark results.
