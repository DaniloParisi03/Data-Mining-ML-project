# Online Shoppers Purchasing Intention — ML Classification Project

Binary classification project predicting whether an e-commerce session results in a purchase (`Revenue = True`), using the [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

**Authors:** Bartalucci, Parisi

---

## Results at a Glance

| Model | Nested CV Macro F1 |
|---|---|
| Random Forest | 0.8019 |
| SVM (RBF) | 0.7986 |
| **XGBoost (Champion)** | **0.8118** |

- **Hold-out Test Macro F1: 0.80** (validated on a locked 20% split, never touched during training)
- Benchmark from Sakar et al. (2018): 0.61 — our pipeline achieves a **+31% improvement**

---

## Repository Structure

```
Data-Mining-ML-project/
├── online_shoppers_intention.csv        # Raw dataset (12,330 sessions, 18 features)
├── requirements.txt                     # Exact Python package versions
├── README.md
├── CHANGELOG_AND_REFACTORING_REPORT.md  # Full changelog and technical report
│
├── jupyter/
│   ├── 1_Exploratory_Data_Analysis.ipynb          # EDA, chi-square, mutual information, LOF profiling
│   ├── 2_Nested_CV_and_Strict_Pipelines.ipynb     # Model training: RF vs SVM vs XGBoost (Nested CV)
│   ├── 3_Evaluation_Ablation_and_XAI.ipynb        # Final evaluation, ablation studies, ROC/PR, SHAP
│   ├── 4_Class_Balancing_Experiments.ipynb        # SMOTE ratio experiment (justifies sampling_strategy=0.7)
│   └── utils/
│       └── lof_sampler.py                         # Shared LOF_Sampler custom imbalanced-learn class
│
├── models/
│   ├── final_best_pipeline.pkl                    # Serialised champion XGBoost pipeline (joblib)
│   └── model_metadata.json                        # Champion params, nested CV scores for all 3 models
│
└── 2_locked_test_data/
    ├── X_test_locked.csv                          # Hold-out feature matrix (sealed by Notebook 2)
    └── y_test_locked.csv                          # Hold-out labels (sealed by Notebook 2)
```

---

## Environment Setup

This project was developed with **Python 3.10** in a dedicated Conda environment named `DMML_AY2526`.

### Option A — Conda (recommended)

```bash
conda create -n DMML_AY2526 python=3.10 -y
conda activate DMML_AY2526
pip install -r requirements.txt
```

### Option B — pip (any Python 3.10 environment)

```bash
pip install -r requirements.txt
```

> **Note on SHAP + XGBoost compatibility:** `shap==0.49.1` and `xgboost==3.2.0` require a one-line patch to `shap/explainers/_tree.py` (see `CHANGELOG_AND_REFACTORING_REPORT.md` for details). Without this patch, Notebook 3's SHAP cells will raise a `ValueError` when loading the serialised XGBoost model.

---

## How to Run

Notebooks **must be executed in order** (each notebook depends on outputs from the previous one):

| Step | Notebook | Outputs produced | Runtime (approx.) |
|---|---|---|---|
| 1 | `1_Exploratory_Data_Analysis.ipynb` | EDA plots, chi-square table | ~2 min |
| 2 | `2_Nested_CV_and_Strict_Pipelines.ipynb` | `models/`, `2_locked_test_data/` | ~15–20 min |
| 3 | `3_Evaluation_Ablation_and_XAI.ipynb` | Evaluation plots, SHAP charts | ~5 min |
| 4 | `4_Class_Balancing_Experiments.ipynb` | SMOTE comparison chart | ~3 min |

### Launch Jupyter

```bash
conda activate DMML_AY2526
jupyter notebook
```

Then open each notebook from the `jupyter/` folder and run **Kernel → Restart & Run All**.

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
- **`Region` dropped** — chi-square test (Notebook 1) found p = 0.32, indicating statistical independence from `Revenue` (α = 0.05).
