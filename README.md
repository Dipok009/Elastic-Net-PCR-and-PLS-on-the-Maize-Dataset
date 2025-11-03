# Elastic-Net-PCR-and-PLS-on-the-Maize-Dataset
This repository contains a single Python script/notebook that trains and evaluates three regression approaches—**Elastic Net**, **Principal Components Regression (PCR)**, and **Partial Least Squares (PLS)**—to predict the target variable `DtoA` from a maize dataset. It includes data cleaning, feature encoding, cross‐validation with model selection, and rich visualizations of tuning curves and coefficient paths. Logs are captured to a file for reproducibility.

---

## 📦 What’s Included

* Data loading and cleaning (drop missing rows, strip column names)
* Label encoding for the categorical column `Geno_Code`
* Train/test split with `random_state=42`
* **ElasticNetCV** with grid over `alpha` and `l1_ratio` (5-fold CV)
* **PCR** via `Pipeline(StandardScaler → PCA → LinearRegression)` with GridSearchCV over number of components
* **PLS** with GridSearchCV over number of components and a final fitted PLS model for inspection
* Saved plots:
  * `data_dist.png` — histogram + KDE of `DtoA`
  * `mse_vs_alpha_plot.png` — CV-MSE vs `log10(alpha)` at best `l1_ratio`
  * `cv_mse_vs_l1_ratio_plot_from_path.png` — CV-MSE vs `l1_ratio` at best `alpha`
  * `cv_mse_contour_plot.png` — 2D CV-MSE surface over (`alpha`, `l1_ratio`)
  * `coefficient_plot.png` — Elastic Net coefficient shrinkage path
* A text log with all printed outputs: `elasticnet_log.txt`

---

## 📁 Repository Structure

```
.
├── README.md                  # This file
├── main.py or notebook.ipynb  # Your code (choose one style)
├── maize.csv                  # Place your dataset here OR update the path in code
└── outputs/                   # (optional) Where plots & logs can be written
```

## 🧰 Requirements

* Python ≥ 3.9
* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn

You can create a virtual environment and install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate         # On Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pandas scikit-learn matplotlib seaborn
```
---

## 🔢 Data Assumptions

* File path to data: `~/sta5703/maize.csv`
* Target column: `DtoA`
* Categorical feature: `Geno_Code` (label encoded). The script auto-strips whitespace from column names and drops any rows containing missing values.
---

## ▶️ How to Run

### Option A — As a script (`main.py`)

1. Copy your current code into a file named `main.py` at the project root.
2. (Recommended) Update all save paths to a local `outputs/` directory:

   ```python
   LOG_DIR = Path('outputs')
   LOG_DIR.mkdir(parents=True, exist_ok=True)
   log_path = LOG_DIR / 'elasticnet_log.txt'
   ```
3. Run:

   ```bash
   python main.py
   ```



## 🔍 Methods & Tuning Details

### Elastic Net
* **Scaler:** `StandardScaler` fitted on training data, applied to train/test
* **Cross-validation:** `KFold(n_splits=5, shuffle=True, random_state=42)`
* **Grid:**
  * `alphas = logspace(-3, 3, 50)`
  * `l1_ratio = np.arange(0.00, 1.00, 0.02)`
* **Model:** `ElasticNetCV(max_iter=10000, cv=cv, n_jobs=-1, random_state=42)`
* **Outputs:** active vs zeroed coefficients, best `alpha` & `l1_ratio`, CV-MSE curves and contour, coefficient path plot

### PCR (Principal Components Regression)

* **Pipeline:** `StandardScaler → PCA → LinearRegression`
* **GridSearchCV:** `pca__n_components = 1..50`, `cv=5`, scoring = `neg_mean_squared_error`
* **Outputs:** best `k`, best CV-MSE, error bar plot (MSE vs `k`), explained variance per PC, top absolute loadings for PC1

### PLS (Partial Least Squares)

* **GridSearchCV:** `n_components = 1..29`, `cv=5`, scoring = `neg_mean_squared_error`
* **Final model for inspection:** `PLSRegression(n_components=1, scale=True)`
* **Outputs:** best `n_components`, CV-MSE curve, final test metrics

---

## 📊 Saved Figures

After a successful run, you should have:

* `data_dist.png`
* `mse_vs_alpha_plot.png`
* `cv_mse_vs_l1_ratio_plot_from_path.png`
* `cv_mse_contour_plot.png`
* `coefficient_plot.png`
  
---

## ✅ Reproducibility Notes

* Determinism is aided by `random_state=42` in the KFold and models.
* Train/test split: `test_size=0.2`, `random_state=42`.

## 🙏 Acknowledgements

* Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [Seaborn](https://seaborn.pydata.org/).

---

