# Zomato Bangalore Restaurant Analysis

This repository contains reusable Python modules and unit tests for inspecting, cleaning, exploring, and clustering restaurant listing data. It is prepared as a code-only project: no source dataset or dataset-derived analysis output is distributed here.

The code is designed for a local copy of the Zomato Bangalore Restaurants dataset. It supports data cleaning, exploratory analysis, correlation summaries, visualizations, and descriptive K-Means clustering. The clustering workflow describes groups in the supplied data; it is not a predictor of restaurant success, revenue, or profitability.

## Dataset access and licensing

The referenced dataset is [Zomato Bangalore Restaurants on Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants), listed by the Kaggle account `himanshupoddar` (Himanshu Poddar).

The exact license, reuse conditions, and identities of the underlying rights holders have not been verified. Available listing metadata credits the data files to “Original Authors,” which does not itself grant reuse or redistribution rights. A public Kaggle download is not evidence that the dataset is openly licensed.

Obtain the dataset independently from its source and check the applicable terms and permissions before using it. The dataset is not included in this repository. No dataset-derived charts, summary tables, reports, statistics, or restaurant records are distributed here.

For local execution, place an authorized copy at:

```text
data/raw/zomato.csv
```

The cleaning module raises a clear file-not-found error if this local input is unavailable. This repository does not download the dataset automatically.

## Repository structure

```text
.
├── src/
│   ├── data_cleaning.py  # Loading, inspection, cleaning, and cleaning-report functions
│   ├── eda.py            # Reusable descriptive summaries, correlations, and chart functions
│   ├── clustering.py     # Clustering preparation, evaluation, profiling, and chart functions
│   └── utils.py          # Project paths, directory setup, and input validation
├── tests/
│   ├── test_data_cleaning.py
│   ├── test_eda.py
│   └── test_clustering.py
├── requirements.txt
├── .gitignore
└── README.md
```

The analysis notebook and generated files remain local and are excluded from publication because the notebook contains dataset-derived summaries and references to derived figures. Raw and processed data, reports, charts, and local environment/cache files are also ignored by Git. Running the modules may create those outputs locally; they are not part of this code-only repository.

## Tools

- Python 3.10 or newer
- Pandas and NumPy for data loading and preparation
- Matplotlib and Seaborn for local visualizations
- scikit-learn for scaling, K-Means, silhouette evaluation, and PCA
- pytest for unit tests

## Local setup

From the repository root, create and activate a virtual environment, then install the requirements:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

After obtaining the dataset independently and confirming that your use is permitted, put `zomato.csv` in `data/raw/`. The data loading helpers use this location by default; several functions also accept explicit input paths or DataFrames.

Run the unit tests from the repository root:

```bash
python -m pytest -q
```

The reusable workflows are implemented as functions in `src/data_cleaning.py`, `src/eda.py`, and `src/clustering.py`. The unit tests use small in-memory examples and do not require the full dataset. Dataset-dependent functions require the local CSV and create local outputs under `data/processed/` and `reports/figures/`; those paths are intentionally ignored for publication.

## Project scope and limitations

The code performs descriptive analysis on observational restaurant listings. Correlations and cluster assignments describe patterns in the input data and do not establish cause and effect or measure sales, demand, or profitability. Results depend on the data version, missing-value patterns, selected fields, and analysis choices. No analysis results are published in this repository because dataset reuse rights remain unverified.

## Attribution

Dataset source reference: [Zomato Bangalore Restaurants, Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants), posted under `himanshupoddar`. The underlying data rights holders and license remain unverified. This attribution identifies the listing used as a reference; it does not imply endorsement or permission to redistribute the dataset.
