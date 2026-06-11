# Online Shoppers Purchasing Intention — ML Classification Project

Binary classification project predicting whether an e-commerce session results in a purchase (`Revenue = True`), using the [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

**Authors:** Bartalucci, Parisi

---

## 1. Project Purpose and Methodology

This project studies the UCI Online Shoppers Purchasing Intention dataset and builds a machine learning pipeline to predict whether an online browsing session ends in a purchase (`Revenue = True`).

The task is a binary classification problem, but it is not a perfectly balanced one: most sessions do not lead to a purchase. For this reason, the project uses **Macro F1-score** as the primary metric, because it gives balanced importance to both purchase and non-purchase classes.

The full workflow is notebook-based and covers the main stages of a data mining project:

- Exploratory data analysis, including class imbalance, correlations, chi-square tests, mutual information, and LOF-based outlier profiling.
- Leakage-safe preprocessing and model training with `imblearn` pipelines.
- Nested cross-validation to compare Random Forest, SVM, and XGBoost.
- Class-balancing experiments with undersampling, oversampling, SMOTE, and ADASYN.
- Final evaluation on a locked hold-out test set.
- Explainability and critical analysis using SHAP, LIME, ablation studies, threshold diagnostics, and error profiling.

The core modeling pipeline is:

```text
LOF_Sampler -> StandardScaler -> SMOTE -> Classifier
```

The most important methodological rule is strict train/test isolation: the locked test set is created once in Notebook 2 and is used only for final evaluation and diagnostics. Preprocessing, outlier removal, and SMOTE are fitted inside cross-validation folds so validation and test data never influence training decisions. The final champion model is **XGBoost**.

---

## 2. Repository Structure

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

## 3. Results

| Model | Nested CV Macro F1 |
|---|---|
| Random Forest | 0.8019 |
| SVM | 0.7988 |
| **XGBoost (Champion)** | **0.8118** |

- **Hold-out Test Macro F1:** 0.80, validated on the locked 20% test split.
- **ROC-AUC:** 0.9260.
- **Average Precision:** 0.7225.
- **Benchmark from Sakar et al. (2018):** 0.61 Macro F1.

---

## 4. How to Test the Project Yourself

The project targets **Python 3.10** and uses the generic Jupyter kernel metadata:

```text
display_name: Python 3
name: python3
```

No machine-specific Jupyter kernel name is required.

### Step 1 — Download and enter the repository

```powershell
git clone https://github.com/DaniloParisi03/Data-Mining-ML-project.git
cd Data-Mining-ML-project
```

### Step 2 — Create and activate a Python 3.10 environment

Using a local virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively, with Conda:

```bash
conda create -n online-shoppers-ml python=3.10 -y
conda activate online-shoppers-ml
pip install -r requirements.txt
```

Notebook 3 includes an in-notebook SHAP/XGBoost compatibility wrapper, so no manual editing of installed packages is required.

### Step 3 — Run the notebooks in order

Launch Jupyter:

```powershell
python -m jupyter notebook
```

Open the `jupyter/` folder, select the **Python 3** kernel, and execute these notebooks with **Kernel -> Restart & Run All**:

| Order | Notebook | What it verifies |
|---|---|---|
| 1 | `1_Exploratory_Data_Analysis.ipynb` | Dataset checks, imbalance, EDA, feature-selection evidence |
| 2 | `2_Nested_CV_and_Strict_Pipelines.ipynb` | Locked test split, leakage-safe pipelines, nested CV, saved model |
| 3 | `3_Evaluation_Ablation_and_XAI.ipynb` | Final locked-test evaluation, ablations, SHAP, LIME, error analysis |
| 4 | `4_Class_Balancing_Experiments.ipynb` | Class-balancing comparison and SMOTE ratio justification |

Notebook 2 is the slow step because it runs full nested cross-validation and grid searches.

### Optional — Terminal execution instead of VS Code/Jupyter

If you prefer to execute the notebooks directly from the terminal, run the following commands after activating the environment and installing `requirements.txt`:

```powershell
cd jupyter
python -m jupyter nbconvert --to notebook --execute --inplace "1_Exploratory_Data_Analysis.ipynb" --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=7200
python -m jupyter nbconvert --to notebook --execute --inplace "2_Nested_CV_and_Strict_Pipelines.ipynb" --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=7200
python -m jupyter nbconvert --to notebook --execute --inplace "3_Evaluation_Ablation_and_XAI.ipynb" --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=7200
python -m jupyter nbconvert --to notebook --execute --inplace "4_Class_Balancing_Experiments.ipynb" --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=7200
```

`kernel_name=python3` forces the generic project kernel, while `timeout=7200` gives the long nested-CV cells enough time to finish.
