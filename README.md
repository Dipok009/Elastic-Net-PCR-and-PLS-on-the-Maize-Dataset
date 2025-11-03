# Elastic-Net-PCR-and-PLS-on-the-Maize-Dataset
This repository contains a single Python script/notebook that trains and evaluates three regression approaches—**Elastic Net**, **Principal Components Regression (PCR)**, and **Partial Least Squares (PLS)**—to predict the target variable `DtoA` from a maize dataset. It includes data cleaning, feature encoding, cross‐validation with model selection, and rich visualizations of tuning curves and coefficient paths. Logs are captured to a file for reproducibility.

---

## 📦 What’s Included

* Data loading and cleaning (drop missing rows, strip column names)
* Label encoding for the categorical column `Geno_Code`
* Train/test split with `random_state=42`
* **ElasticNetCV** with grid over `alpha` and `l1_ratio` (5-fold CV)
* **PCR** via `Pipeline(StandardScaler → PCA → LinearRegression)` with GridSearchCV over number of components
* **PLS** with GridSearchCV over the number of components and a final fitted PLS model for inspection
* Saved plots
* A text log with all printed outputs: `elasticnet_log.txt`

---

## 📁 Repository Structure

```
├── README.md                  
├── main.py 
├── maize.csv                 

```

## 🧰 Requirements

* Python ≥ 3.9
* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn


## 🔍 Methods & Tuning Details

### Elastic Net
* **Scaler:** `StandardScaler` fitted on training data, applied to train/test
* **Cross-validation:** `KFold(n_splits=5, shuffle=True, random_state=42)`
* **Grid:**
  * `alphas = logspace(-3, 3, 50)`
  * `l1_ratio = np.arange(0.00, 1.00, 0.02)`
* **Model:** `ElasticNetCV(max_iter=10000, cv=cv, n_jobs=-1, random_state=42)`
* **Outputs:** active vs zeroed coefficients, best `alpha` & `l1_ratio`, CV-MSE curves, coefficient path plot

### PCR (Principal Components Regression)

* **Pipeline:** `StandardScaler → PCA → LinearRegression`
* **GridSearchCV:** `pca__n_components = 1..50`, `cv=5`, scoring = `neg_mean_squared_error`
* **Outputs:** best `k`, best CV-MSE, error bar plot (MSE vs `k`), explained variance per PC, top absolute loadings for PC1

### PLS (Partial Least Squares)

* **GridSearchCV:** `n_components = 1..29`, `cv=5`, scoring = `neg_mean_squared_error`
* **Final model for inspection:** `PLSRegression(n_components=1, scale=True)`
* **Outputs:** best `n_components`, CV-MSE curve, final test metrics

---

## ✅ Reproducibility Notes

* Determinism is aided by `random_state=42` in the KFold and models.
* Train/test split: `test_size=0.2`, `random_state=42`.

## 🙏 Acknowledgements

* Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [Seaborn](https://seaborn.pydata.org/).

---

