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

> **Note:** In the provided code, plots and logs are saved to your home folder under `~/sta5703/`. You can keep this or change to a local `outputs/` directory (recommended for a repo).

---

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

(Optional) create a `requirements.txt`:

```txt
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

## 🔢 Data Assumptions

* File path to data: `~/sta5703/maize.csv` (you may change this)
* Target column: `DtoA`
* Categorical feature: `Geno_Code` (label encoded). The script auto-strips whitespace from column names and drops any rows containing missing values.

> If your file is in a different location, update the line:
>
> ```python
> df = pd.read_csv('~/sta5703/maize.csv')
> ```
>
> to point to your actual CSV path (e.g., `data/maize.csv`).

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

### Option B — As a Jupyter notebook

* Keep your current cells in `notebook.ipynb` and run them interactively. Plots will render inline and will also save to disk due to the `plt.savefig(...)` calls.

---

## 📝 Logging

The script redirects `stdout` to `elasticnet_log.txt` so you have a persistent record of:

* Data summaries and shapes
* Counts of missing values
* Elastic Net tuning results (best `alpha`, `l1_ratio`, active features, coefficients)
* PCR GridSearch best number of components and CV-MSE
* PLS GridSearch best number of components and CV-MSE
* Final test metrics for each approach (R², MSE, RMSE, MAE) and timing

In the original code, the log file is saved to:

```
~/sta5703/elasticnet_log.txt
```

You can change this to `outputs/elasticnet_log.txt` to keep everything in-repo.

---

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
* **Final model for inspection:** `PLSRegression(n_components=1, scale=True)` (prints intercept and top coefficients)
* **Outputs:** best `n_components`, CV-MSE curve, final test metrics

---

## 📊 Saved Figures

After a successful run, you should have:

* `data_dist.png`
* `mse_vs_alpha_plot.png`
* `cv_mse_vs_l1_ratio_plot_from_path.png`
* `cv_mse_contour_plot.png`
* `coefficient_plot.png`

Include these in your README or repo’s **Assets/Images** section if you’d like.

---

## ✅ Reproducibility Notes

* Determinism is aided by `random_state=42` in the KFold and models.
* Train/test split: `test_size=0.2`, `random_state=42`.
* Be mindful that OS, library versions, and BLAS backends can produce tiny numeric differences.

---

## 🧪 Example Results (replace with your actual numbers)

Elastic Net (test set):

```
R² = 0.87
MSE = 0.1234
RMSE = 0.3513
MAE = 0.2745
Best alpha = 0.0316
Best l1_ratio = 0.62
Active features = 18 / 42
```

PCR (test set):

```
Best k = 14
R² = 0.84
MSE = 0.1456
RMSE = 0.3816
MAE = 0.2988
```

PLS (test set):

```
Best n_components (CV) = 7
R² = 0.85
MSE = 0.1380
RMSE = 0.3715
MAE = 0.2920
```

> Replace the placeholders above using values from `elasticnet_log.txt` after your run.

---

## 🧹 Common Tweaks

* **Change output locations**: point `log_path` and `plt.savefig(...)` to `outputs/`
* **Headless servers**: add `matplotlib.use('Agg')` at the very top to avoid display backend issues
* **Different dataset path**: modify the `pd.read_csv(...)` line
* **Add CLI args**: wrap code in a `main()` and use `argparse` for `--data`, `--outdir`, etc.

---

## 🛡️ License

Add a license file such as **MIT** or **Apache-2.0** at the project root. Example `LICENSE` is recommended for open-source use.

---

## 🙏 Acknowledgements

* Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [Seaborn](https://seaborn.pydata.org/).

---

## 🔗 Citation

If you use this code in academic work, please cite scikit-learn and any relevant references for Elastic Net, PCA/PCR, and PLS.

```bibtex
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in Python},
  author={Pedregosa, Fabian and others},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
```

---

## 📣 Contributing

Issues and pull requests are welcome! Please open an issue describing bugs or feature requests, and feel free to submit PRs with improvements (tests, CLI, docs, refactors).
