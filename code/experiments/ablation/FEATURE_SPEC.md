# SIFE44 and Pure36

Feature version: `sife44_distribution_v9_formulas_v1`. The formulas match the frozen version-9 definition on finite, nonnegative records. This implementation rejects invalid incident count/sum weights instead of silently replacing them. Source14 and labels do not enter `extract_sife44`.

For one typed relation and direction, let N be the number of incident processed records, C the sum of count, V the sum of the released value field. Positive-mass distributions are p_j=count_j/C and q_j=value_j/V. When C or V is zero in a nonempty channel, the corresponding distribution is uniform1/N; absent channels have zero statistics. This is a declared convention for undefined distributions, not an observed uniform movement of money.

`FEATURE_COLUMNS.csv` lists all44indices, formulas and Pure36 membership. Ordering:7statistics for each of(call-in,call-out,transfer-in,transfer-out)=28;4directional contrasts for each relation=8;CA record/count shares for each channel=8.

Entropy uses natural logarithms. Normalized entropy divides by ln(N), including zero-value records in N. JSD uses natural logs and lies in[0,ln2]. Standard deviation is over log1p(value) of processed counterpart records, not individual transaction amounts. Mean value is log1p(V/C), with zero when C=0. CA ratios use typed counterpart identity and count weights; edge directions are not symmetrized.

The four contrasts per relation are entropy_in - entropy_out, HHI_out - HHI_in, (N_in - N_out)/(N_in+N_out), and (V_in - V_out)/(V_in+V_out). Zero denominators return0.

Pure36 removes mean-value and log-value-standard-deviation in each of the four channels:8columns. Under positive global multiplicative rescaling of count and value, with topology and record support fixed, its36values are unchanged. This is not statistical independence from monetary information: entropy/concentration and flow ratios retain information about the distribution of values. X14+Pure36 also retains source14.

Node IDs must be sorted, unique and in range. Edge attributes must align with edge_index and contain count,sum in that order. Full graph pair uniqueness and raw endpoint alignment belong to the prepared-data audit. Each requested node aggregates all its incident records; classifier neighbor sampling does not change this definition.

The active group-ablation inputs have58/38/50/50/50columns; see GROUP_ABLATION_SPEC.md. `select_inputs` also keeps14/36column compatibility helpers for backbone validation; these are not active study arms. Pure36 does not require a source14 matrix. Meta-IFD Pure36-only is rejected because the existing generator consumes source14. GNN neighbor nodes use the same extractor as target roots; exhaustive replay of all neighbors is not part of the labelled-root checks in the feature-definition stage.

The data are static and contain no timestamp basis for forecasting or early-detection claims. The physical unit of the released value field is not changed or inferred here.

## Precision required for replay

For Phish, use released graph attributes at their released precision. For Ponzi, use the audited float64 count/sum sidecars restored from raw; casting them to float32 before extracting features changes some columns by small rounding amounts. The first replay against a previously prepared float32 sidecar produced a maximum absolute error5.235472055886703e-8 and correctly failed the strict comparison. This is an input-precision difference, not a reason to alter formulas or enlarge tolerance. Float64 restoration verifies each full raw CSV hash and each full output NPY hash against the reviewed provenance.

## Invariance argument

Let count weights be multiplied by a positive constant a and value weights by a positive constant b throughout the input graph, with graph topology and record support fixed. Each nonempty count/value probability remains unchanged; zero-mass branches stay zero-mass and keep their uniform convention. Therefore entropy, normalized entropy, HHI, JSD and maximum share are unchanged. Record-degree contrasts do not depend on weights; common value scaling cancels from the inflow/outflow ratio; count scaling cancels from the CA count share. All36selected features are invariant under this transformation. The two discarded statistics per channel are not generally invariant. This property does not cover node-specific or direction-specific rescaling, changes of partners/records, or a model that also receives source14.
