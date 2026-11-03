# Experimental protocol

## Released-mask references

The X14-only reference notebooks use the released masks and original labeled cohorts. They evaluate an implementation comparison and document residual deviations; they are not pooled with the strict repeated-partition study and are not an exact reproduction claim for every published model score. Native graph/Random Forest references and the LightGBM reference retain separate source directories because their implementations were frozen independently.

## Main study

Tasks: Phish and Ponzi. Models: Meta-IFD, GAT, GraphSAGE, LightGBM, Random Forest. Partition seeds 2001–2020; training seeds 12–16; stratified 60/20/20. Arms: X14, Pure36, X14_Pure36, X14_SIFE44. Meta-IFD excludes Pure36-only. The 3,800 fits are 2 tasks × 20 partitions × 5 seeds × (3 + 4 + 4 + 4 + 4 arms across models).

This is a static, transductive study: held-out structure and attributes remain graph context, while held-out labels are not classifier inputs. Validation selects neural checkpoints; TEST only scores locked fits. The lock covers one model–dataset run, not an untouched external confirmation of choices made before the study.

### Fixed settings

| Model | Settings |
|---|---|
| Meta-IFD | Width 128, 3 views, 2 message-passing layers, learning rate 0.001, contrast weight 0.1 |
| GAT | 2 layers, 4 heads/layer, width 32/head, averaged output heads, ELU, no dropout, learning rate 0.01 |
| GraphSAGE | Linear/tanh encoder, projected mean layers of width 32, ReLU, learning rate 0.01 |
| LightGBM | 300 trees, 31 leaves, minimum child samples 20, learning rate 0.01, no class weighting |
| Random Forest | 100 trees, maximum depth 10, square-root feature sampling, bootstrap, minimum leaf size 1 |

Neural training: Adam, weight decay 0.0005, unweighted cross-entropy plus Meta-IFD's contrast term; at most 1,000 epochs; latest checkpoint wins validation Macro-F1 ties; stop when staleness exceeds 50 epochs. Batch size 512; two-hop directional PyG sampling, up to 100 neighbors per relation per hop. Sampling caches are reused within each model across arms, training seeds and epochs. The source records any seed/traversal corrections relative to the upstream implementation.

GAT applies signed-log to X14, then standardizes active inputs using unique typed TRAIN sampled nodes. GraphSAGE and Meta-IFD standardize added descriptors only, leaving the original X14 source inputs unchanged. Tree inputs are unscaled. TRAIN moments can include unlabeled transductive neighbors. No hyperparameter or decision-threshold search is performed in the fixed study.

Configuration values reflect exploratory development before the fixed runs. Historical origin text in the preserved original protocols is provenance, not evidence of a preregistered protocol. The main model scope was reduced from an earlier six-model exploratory setup; historical LR runs are outside this five-model artifact and do not support a claim about linear-model coverage.

## Group removal

Five arms retain X14: full X14_SIFE44, without_distribution20, without_flow8, without_counterparty8, and X14_Pure36 (without intensity8). There are 5,000 fits. Every deletion is compared with a full-input control **retrained in that ablation run**, not with the main-study full-input mean.

Removing columns does not erase all related information from X14, other derived columns or typed graph structure. Input widths and parameter counts differ. The intervention is not a causal or capacity-matched information-removal experiment. Decreases and negative group contributions remain in the results.

## Metrics and aggregation

Fraud is class 1. The primary metric is the mean of class-0 F1 and class-1 F1. Secondary metrics include precision, recall, binary F1, FPR, AP, AUC, micro-F1 and accuracy. Undefined precision/recall/F1 are zero. Decision: class-1 score > 0.5; exact ties are normal.

Five seeds are averaged within each partition before computing the mean and sample SD across the 20 partition means. Paired differences are within-partition percentage-point differences. Wins/20 is descriptive. Overlapping holdouts are not independent datasets; no best-seed selection is performed.

## Exploratory matching

The two covariates are exported log total incident count and log total incident value. TRAIN-root means/population SDs define scaling. In TEST, matching maximizes eligible 1:1 pairs under a Euclidean-distance caliper of 0.5 before minimizing distance, without replacement. The subset membership is frozen before prediction reads and shared across models/arms/seeds.

Balance is the absolute fraud–normal mean gap divided by the corresponding TRAIN-root SD; both covariates must be below 0.1. Report positive-retention coverage alongside balance. The matched population has 50% fraud prevalence. Successful measured balance is not independence from all activity variables or evidence of a causal effect.

The optional corrected-resampling inference CSV is an assumption-dependent sensitivity analysis, not the paper's primary descriptive evidence. Graph dependence and non-additive Macro-F1 remain limitations.
