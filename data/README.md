# Data

## CarDekho used cars

This project uses `CAR DETAILS FROM CAR DEKHO.csv` from **Vehicle dataset**, version 4, published by Nehal Birla and collaborators. Kaggle describes the data as used-car information from websites. It does not document the collection dates or confirm completed sales. [Kaggle data card](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

Kaggle lists Open Database License for the database and Database Contents License for its contents. [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) · [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/)

The CSV has no currency field, and the data card omits the units. We use Indian rupees (INR), consistent with the University of South Carolina course example, which cites the Kaggle dataset and converts its raw `selling_price` to USD using an INR exchange rate. [STAT 540 course notes](https://people.stat.sc.edu/gregorkb/STAT_540_fa_2025/SAS_04_stat_methods_2.pdf)

### Download

1. Open the [Kaggle dataset page](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) and select version 4. Download the archive through Kaggle. If Kaggle prompts you to sign in, follow its sign-in flow.
2. Extract only `CAR DETAILS FROM CAR DEKHO.csv` to `data/raw/`. Create the folder if needed. Git ignores it.
3. To check that you have the same file, run `shasum -a 256 'data/raw/CAR DETAILS FROM CAR DEKHO.csv'` from the repository root. The expected checksum is `454fe4ea4f6d2af973493287abf0631006a32bd4f462103e4dfcffa75d721edf`.

The [official version 4 download endpoint](https://www.kaggle.com/api/v1/datasets/download/nehalbirla/vehicle-dataset-from-cardekho?datasetVersionNumber=4) also served the archive publicly during this analysis.

### Columns

| CSV header | Meaning |
|---|---|
| `name` | Vehicle make, model, and variant as recorded. |
| `year` | Vehicle year. |
| `selling_price` | Listed selling price in INR. |
| `km_driven` | Odometer distance in kilometres. |
| `fuel` | Fuel type. |
| `seller_type` | Individual, dealer, or Trustmark Dealer. |
| `transmission` | Manual or automatic. |
| `owner` | Ownership history category, including test-drive cars. |

## California Housing

The scikit-learn fetcher uses the California Housing data from StatLib, derived from the 1990 U.S. Census. Each row summarizes a California census block group. Credit R. Kelley Pace and Ronald Barry, *Sparse Spatial Autoregressions*, 1997. [scikit-learn dataset description](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) · [Original paper](https://doi.org/10.1016/S0167-7152(96)00140-X)

From the repository root, run the download script. It calls `fetch_california_housing(as_frame=True)`, keeps the scikit-learn cache in `data/raw/sklearn-cache/`, and exports `data/raw/california_housing.csv`. Git ignores both paths. The first run requires network access to the archive used by scikit-learn. [Fetcher source for scikit-learn 1.6.1](https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/datasets/_california_housing.py)

```bash
conda run -n ml-benchmark python scripts/fetch_california_housing.py
```

If the default download host is unavailable, download `cal_housing.tgz` from the [Figshare file API](https://api.figshare.com/v2/file/download/5976036), place it in `data/raw/sklearn-cache/`, and rerun the script. The expected SHA-256 is `aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`. Scikit-learn verifies the archive and converts it into its local cache.

| Column | Meaning |
|---|---|
| `MedInc` | Median income in the block group |
| `HouseAge` | Median house age in years |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block-group population |
| `AveOccup` | Average people per household |
| `Latitude` | Latitude in degrees |
| `Longitude` | Longitude in degrees |
| `MedHouseVal` | Median house value in units of USD 100,000. This is the target. |

The target is capped at `5.00001`, equivalent to USD 500,001. The observed values cannot distinguish expensive block groups above the cap. The data describe 1990 block groups and cannot establish current or individual home values. Random splitting can put nearby block groups in both training and test sets, so the scores describe a random holdout and can overstate performance in entirely new areas.

## Breast Cancer Wisconsin Diagnostic

The benchmark uses scikit-learn's bundled copy of the Breast Cancer Wisconsin (Diagnostic) dataset. Credit Wolberg, Mangasarian, N. Street, and W. Street (1993). UCI lists the dataset under CC BY 4.0. [UCI dataset and attribution](https://doi.org/10.24432/C5DW2B) · [Scikit-learn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

The 30 numeric inputs summarize cell nuclei in digitized images from fine-needle aspirates of breast masses. Each of ten measurements appears as a mean, a standard error, and a worst-value summary. The measurements are radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. The bundled copy omits the UCI record ID. [UCI feature descriptions](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

| Diagnosis | UCI label | Scikit-learn label | Full | Train | Test |
|---|---|---:|---:|---:|---:|
| Malignant (positive) | M | 0 | 212 | 170 | 42 |
| Benign (negative) | B | 1 | 357 | 285 | 72 |
| Total | | | 569 | 455 | 114 |

From the repository root, export the bundled data to the Git-ignored `data/raw/breast_cancer.csv`. The export works offline.

```bash
conda run -n ml-benchmark python scripts/prepare_breast_cancer.py
```

This historical dataset contains 569 records with measurements derived from a specific specimen and imaging process. An internal random holdout gives no evidence of performance at other sites, across patient subgroups, or under different acquisition practices. The benchmark has no clinical validation and is for education only.
