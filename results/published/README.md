# Reference benchmark snapshots

The benchmark files were regenerated from actual v1.0.0 runs so the results and metadata reflect the final implementation. The runs used Newton-CG for Logistic Regression and one OpenMP thread. The CarDekho K-Nearest Neighbors test MAE changed from about INR 197,250 to INR 197,467 under the one-thread setting. All three model rankings were preserved. The CarDekho exploratory charts were generated separately by the EDA notebook.

The repository's MIT license applies to the software. The datasets have separate terms. New runs from the shipped configs write to `results/tmp/`.

## CarDekho

The exploratory charts and CarDekho benchmark outputs contain information from the [Vehicle dataset, version 4](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), published by Nehal Birla and collaborators and available under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/) and [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).

The [data guide](../../data/README.md#cardekho-used-cars) identifies the exact source file and checksum. The [EDA notebook](../../notebooks/cardekho_eda.ipynb) and the [preprocessing code](../../src/ml_benchmark/preprocessing.py) with the [experiment config](../../configs/used_car_prices.toml) provide the method for deriving the analyzed data, including exact-row deduplication, feature selection, and fold-local transformations. Any derivative CarDekho database is offered on the same ODbL terms through these reproducible instructions.

## California Housing

The California Housing outputs use the 1990 U.S. Census block-group data distributed through StatLib and scikit-learn. Credit R. Kelley Pace and Ronald Barry, *Sparse Spatial Autoregressions* (1997). The upstream archive and dataset documentation provide no explicit redistribution license that we could verify. The public release includes aggregate benchmark results only. The source CSV and charts showing individual block-group values remain local. [Original paper](https://doi.org/10.1016/S0167-7152(96)00140-X) · [Data guide](../../data/README.md#california-housing)

## Breast Cancer Wisconsin Diagnostic

The classification outputs use Wolberg, Mangasarian, N. Street, and W. Street's *Breast Cancer Wisconsin (Diagnostic)* dataset (1993), available through the UCI Machine Learning Repository under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The experiment uses scikit-learn's numeric labels and applies fold-local preprocessing. These artifacts are for education and have no clinical validation. [UCI dataset](https://doi.org/10.24432/C5DW2B) · [Data guide](../../data/README.md#breast-cancer-wisconsin-diagnostic)
