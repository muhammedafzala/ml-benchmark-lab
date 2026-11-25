# Results

Published benchmarks were regenerated with v1.0.0, including Newton-CG for Logistic Regression and one OpenMP thread. The supplied configs write new runs to `results/tmp/`. See the [published output notices](published/README.md) for dataset attribution and reproduction information.

Write local runs to `results/tmp/`, which Git ignores. Put selected CSV reports, charts, and benchmark summaries in `results/published/` when you want to share them. Include the experiment config and dataset version in each published summary so readers can reproduce it.

## CarDekho exploratory charts

The [EDA notebook](../notebooks/cardekho_eda.ipynb) generates these charts from the original CSV. Source data by Nehal Birla and collaborators, [Vehicle dataset, Kaggle version 4](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) and [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/). See the [data guide](../data/README.md) for the exact file and checksum.

- [Selling price distribution](published/cardekho-price-distribution.png)
- [Median price by vehicle year and distance driven](published/cardekho-price-year-distance.png)
- [Median price by vehicle and seller categories](published/cardekho-price-categories.png)

## CarDekho regression benchmark

Gradient Boosting ranked first by five-fold validation MAE and had the lowest reserved-test MAE. The leaderboard is ordered by validation MAE. All test scores were calculated after selection, using the same reserved split and fixed model settings.

- [Leaderboard](published/cardekho-benchmark/leaderboard.csv), [validation scores by fold](published/cardekho-benchmark/cv_fold_scores.csv), and [run metadata](published/cardekho-benchmark/run_metadata.json)
- [Validation MAE](published/cardekho-benchmark/cv-mae.png), [test MAE and RMSE](published/cardekho-benchmark/test-errors.png), and [test R²](published/cardekho-benchmark/test-r2.png)
- [Training and prediction time](published/cardekho-benchmark/timing.png), [actual versus predicted prices](published/cardekho-benchmark/actual-vs-predicted.png), and [residuals](published/cardekho-benchmark/residuals.png)

Timing is one wall-clock measurement per pipeline for fitting on all 2,861 training rows and one for predicting all 716 test rows. Both include preprocessing. Loading, cross-validation, scoring, and chart writing are excluded. Millisecond differences can vary with system load.

This benchmark uses six inputs and excludes the raw vehicle name, which contains many model and variant strings. Exact repeats were removed, but nearby matches or the same listing with edited fields may cross the split. The full file was explored before benchmarking, and the random holdout has no listing dates to support a forward-in-time test. Prices are listed amounts without confirmation of completed sales. [Dataset source and license](../data/README.md)

The release audit found that 334 of the 716 test rows share all six model inputs with at least one training row. Matching features can describe different cars. The CSV has no listing IDs to establish whether the same listing crossed the split. The reported metrics combine rows with seen and unseen feature combinations, so they do not isolate performance on unseen combinations.

## California Housing regression benchmark

The benchmark uses all eight numeric inputs from 20,640 census block groups. The shuffled split with seed 42 produced 16,512 training rows and 4,128 test rows. Random Forest ranked first across five shared training folds and had a test MAE of 0.329 and R² of 0.804. MAE and RMSE are in units of USD 100,000.

- [Leaderboard](published/california-housing-benchmark/leaderboard.csv), [validation scores by fold](published/california-housing-benchmark/cv_fold_scores.csv), and [run metadata](published/california-housing-benchmark/run_metadata.json)
- [Validation MAE](published/california-housing-benchmark/cv-mae.png), [test MAE and RMSE](published/california-housing-benchmark/test-errors.png), and [test R²](published/california-housing-benchmark/test-r2.png)
- [Training and prediction time](published/california-housing-benchmark/timing.png)

Redistribution permission for individual values could not be established. The release includes aggregate evaluations only. Local runs still generate the actual-versus-predicted and residual plots in the ignored results directory.

The target reaches the USD 500,001 cap in 965 rows. The data come from the 1990 Census, and nearby block groups can appear on both sides of the split. Treat the scores as performance on this random holdout of historical block groups. [Dataset source and attribution](../data/README.md#california-housing)

## Breast Cancer Wisconsin Diagnostic classification benchmark

Logistic Regression ranked first by mean malignant-class F1 across five shared stratified folds. The reserved test set contained 42 malignant and 72 benign cases. Positive-class precision, recall, and F1 treat scikit-learn label `0` as malignant. ROC-AUC uses the predicted probability for that label. Prediction timing measures the batch of class labels and excludes the separate probability call.

- [Leaderboard](published/breast-cancer-benchmark/leaderboard.csv), [validation scores by fold](published/breast-cancer-benchmark/cv_fold_scores.csv), and [run metadata](published/breast-cancer-benchmark/run_metadata.json)
- [Validation F1](published/breast-cancer-benchmark/cv-f1.png), [test metrics](published/breast-cancer-benchmark/test-metrics.png), and [confusion matrix](published/breast-cancer-benchmark/confusion-matrix.png)
- [ROC curves](published/breast-cancer-benchmark/roc-curves.png) and [training and prediction time](published/breast-cancer-benchmark/timing.png)

This is an educational benchmark. It has no clinical validation. [Dataset source, attribution, and limitations](../data/README.md#breast-cancer-wisconsin-diagnostic)
