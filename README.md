# Online Shoppers Purchasing Intention — ML Classification Project

Binary classification project predicting whether an e-commerce session results in a purchase (`Revenue = True`), using the [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

**Authors:** Bartalucci, Parisi

---

## Results at a Glance

| Model | Nested CV Macro F1 |
|---|---|
| Random Forest | 0.8019 |
| SVM | 0.7988 |
| **XGBoost (Champion)** | **0.8118** |

- **Hold-out Test Macro F1: 0.80** (validated on a locked 20% split, never touched during training)
- Benchmark from Sakar et al. (2018): 0.61 — our pipeline achieves a **+31% improvement**

---

## Repository Structure

```
Data-Mining-ML-project/
├── online_shoppers_intention.csv        # Raw dataset (12,330 sessions, 18 columns: 17 predictors + target)
├── requirements.txt                     # Exact Python package versions
├── .python-version                      # Python runtime marker: 3.10
├── runtime.txt                          # Deployment/runtime marker: python-3.10
├── README.md
├── CHANGELOG_AND_REFACTORING_REPORT.md  # Full changelog and technical report
│
├── jupyter/
│   ├── 1_Exploratory_Data_Analysis.ipynb          # EDA, chi-square, mutual information, LOF profiling
│   ├── 2_Nested_CV_and_Strict_Pipelines.ipynb     # Model training: RF vs SVM vs XGBoost (Nested CV)
│   ├── 3_Evaluation_Ablation_and_XAI.ipynb        # Final evaluation, ablation studies, ROC/PR, SHAP, LIME
│   ├── 4_Class_Balancing_Experiments.ipynb        # SMOTE ratio experiment (justifies sampling_strategy=0.7)
│   └── utils/
│       └── lof_sampler.py                         # Shared LOF_Sampler custom imbalanced-learn class
│
├── models/
│   ├── final_best_pipeline.pkl                    # Serialised champion XGBoost pipeline (joblib)
│   └── model_metadata.json                        # Champion params, nested CV scores for all 3 models
│
├── figures/
│   └── fig_01 … fig_10.png                        # Pre-generated notebook/report figures (150 dpi)
│
└── 2_locked_test_data/
    ├── X_test_locked.csv                          # Hold-out feature matrix (sealed by Notebook 2)
    └── y_test_locked.csv                          # Hold-out labels (sealed by Notebook 2)
```

---

## Environment Setup

This project targets **Python 3.10** with dependencies installed from `requirements.txt`.

The notebooks intentionally use the portable Jupyter kernel metadata:

```text
display_name: Python 3
name: python3
```

Do **not** require a local Conda environment name in the project files. Any correctly configured Python 3.10 environment with the packages from `requirements.txt` is valid.

The repository also includes:
- `.python-version` — local version-manager marker set to `3.10`
- `runtime.txt` — deployment/runtime marker set to `python-3.10`

`ipykernel` is included in `requirements.txt` because it is the runtime bridge that allows the selected Python environment to execute Jupyter notebook cells as a **Python 3** kernel.

### Option A — Conda (recommended)

```bash
conda create -n online-shoppers-ml python=3.10 -y
conda activate online-shoppers-ml
pip install -r requirements.txt
```

### Option B — pip (any Python 3.10 environment)

```bash
pip install -r requirements.txt
```

> **Note on SHAP + XGBoost compatibility:** Notebook 3 includes an in-notebook compatibility wrapper for `shap==0.49.1` with `xgboost==3.2.0`, so a clean `pip install -r requirements.txt` is sufficient. No manual editing of installed packages is required.

---

## How to Run

Notebooks **must be executed in order** (each notebook depends on outputs from the previous one):

| Step | Notebook | Outputs produced | Runtime (approx.) |
|---|---|---|---|
| 1 | `1_Exploratory_Data_Analysis.ipynb` | EDA plots, chi-square table | ~2 min |
| 2 | `2_Nested_CV_and_Strict_Pipelines.ipynb` | `models/`, `2_locked_test_data/` | ~15–20 min |
| 3 | `3_Evaluation_Ablation_and_XAI.ipynb` | Evaluation plots, SHAP charts, LIME explanation | ~5 min |
| 4 | `4_Class_Balancing_Experiments.ipynb` | SMOTE comparison chart | ~3 min |

### Launch Jupyter

```bash
conda activate online-shoppers-ml
jupyter notebook
```

Then open each notebook from the `jupyter/` folder, select the generic **Python 3** kernel, and run **Kernel → Restart & Run All**.

---

## Methodology Overview

### Pipeline Architecture
Each model is wrapped in an `imblearn.Pipeline` with the following sequential steps:

```
LOF_Sampler → StandardScaler → SMOTE → Classifier
```

1. **LOF_Sampler** — `LocalOutlierFactor`-based custom sampler that removes outliers only within each training fold (prevents data leakage). `n_neighbors=100`, `contamination=0.05`.
2. **StandardScaler** — Feature standardisation (required for SVM; harmless for tree-based models).
3. **SMOTE** — Synthetic oversampling of the minority class. `sampling_strategy=0.7` (determined empirically in Notebook 4).
4. **Classifier** — The model under evaluation (RF / SVM / XGBoost).

### Validation Strategy: Nested Cross-Validation
- **Outer loop:** `StratifiedKFold(n_splits=5)` — unbiased performance estimate.
- **Inner loop:** `GridSearchCV(cv=3)` — hyperparameter optimisation.
- **Metric:** Macro F1-score (equally weights both classes; robust to imbalance).
- The test set is **fully isolated** from training — locked to CSV after the initial split in Notebook 2.

### Feature Engineering
- Categorical encoding: `pd.get_dummies` on `Month` and `VisitorType`.
- Boolean conversion: `Weekend` and `Revenue` cast to `int`.
- **`Region` dropped** — a train-only chi-square test (Notebook 1, using the same stratified split seed as Notebook 2) found p = 0.6825, indicating statistical independence from `Revenue` (alpha = 0.05). The raw `Region` column is removed; after one-hot encoding, the model uses 25 feature columns.
