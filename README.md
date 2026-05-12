# Real Estate Market Analyzer

> An end-to-end learning project that walks through the full data science lifecycle on real housing data — from ingestion, through ML modeling and time-series forecasting, to a Power BI dashboard.

This repo is built **incrementally**, one phase at a time. Each phase ships code **and** an educational markdown doc in [docs/](docs/) explaining what was built, why we chose the approach, and the concepts behind it.

---

## What this project does

1. **Ingests** two public housing datasets (Ames + Zillow ZHVI).
2. **Cleans** them — handling missing values, outliers, and inconsistent types.
3. **Explores** with statistical summaries and visualizations (EDA).
4. **Engineers features** like price-per-sqft, home age, location clusters.
5. **Trains** three regression models — Linear Regression, Random Forest, XGBoost — and tunes them.
6. **Forecasts** neighborhood-level price trends over time.
7. **Explains** predictions with SHAP / permutation importance.
8. **Calculates ROI** for prospective real estate investors (cap rate, cash-on-cash, IRR).
9. **Exports** clean CSVs and an SQLite database ready for a Power BI dashboard.

---

## Project layout

```
real-estate-analyzer/
├── data/
│   ├── raw/         # Original downloaded files (gitignored)
│   ├── processed/   # Cleaned, model-ready datasets
│   └── outputs/     # Files for Power BI to consume
├── notebooks/       # Jupyter notebooks for each learning phase
├── src/             # Production Python modules
├── docs/            # Educational walkthroughs (read these!)
├── tests/
├── requirements.txt
└── README.md
```

---

## Quickstart (Windows / PowerShell)

> Prerequisite: **Python 3.11+**. Run `python --version` to confirm.

```powershell
# 1. From the project root, create an isolated virtual environment.
#    Why: keeps these packages out of your global Python install.
python -m venv .venv

# 2. Activate it.
.venv\Scripts\Activate.ps1
# If PowerShell blocks the script with an execution-policy error, run:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# then activate again.

# 3. Upgrade pip and install dependencies.
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Download the raw datasets into data/raw/.
python -m src.data_loader

# 5. Launch Jupyter to step through the first notebook.
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Quickstart (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data_loader
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## The seven phases

| Phase | Theme                        | Outputs                                | Docs       |
| ----- | ---------------------------- | -------------------------------------- | ---------- |
| **1** | Setup & Data Acquisition     | folder skeleton, `data_loader.py`      | 01, 02     |
| **2** | Data Cleaning & EDA          | `preprocessing.py`, `eda.py`           | 03, 04     |
| **3** | Feature Engineering          | engineered feature columns             | 05         |
| **4** | ML Model Training            | `models.py`, saved best model          | 06, 07, 08, 09 |
| **5** | Time-Series Forecasting      | `forecasting.py`, per-ZIP forecasts    | 10         |
| **6** | Interpretability & ROI       | SHAP plots, `roi_calculator.py`        | 11, 14     |
| **7** | Power BI Export              | CSV bundle + SQLite database           | 12, 13     |

Each phase **stops at a learning checkpoint** before moving on, so you can absorb the concepts before adding more.

---

## Datasets

This project uses two complementary, freely-available public datasets:

1. **Ames Housing** (Dean De Cock, 2011) — 2,930 home sales in Ames, Iowa with ~80 descriptive features. The gold-standard dataset for learning regression because every variable is documented and the feature richness lets us practice realistic preprocessing.
2. **Zillow ZHVI** (Zillow Home Value Index, ZIP-level) — monthly smoothed home values per US ZIP code from 2000 to present. Provides the time-series and geographic dimension Ames lacks.

`data_loader.py` downloads both automatically. If the URLs change or you're offline, the script prints clear manual-download instructions. See [docs/02_understanding_the_data.md](docs/02_understanding_the_data.md) for the full data dictionary.

---

## How to use the educational docs

The [docs/](docs/) folder is the **learning core** of this repo. Read them **in order**, alongside the matching notebook. Each doc follows the same template:

1. What the code does (plain English)
2. Why we chose this approach
3. How it works (line-by-line for tricky parts)
4. Concepts introduced (with analogies)
5. Math/statistics behind it (intuitive)
6. Common pitfalls
7. Further reading
8. Self-test quiz at the end

If you can answer the quiz at the bottom of each doc without peeking, you're ready for the next phase.

---

## License & attribution

Educational project. Datasets are credited to their respective sources (Dean De Cock for Ames, Zillow Research for ZHVI). All code MIT-licensed.
