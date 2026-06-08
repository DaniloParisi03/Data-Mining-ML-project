# Changelog and Refactoring Report

**Project:** Online Shoppers Purchasing Intention — ML Classification  
**Authors:** Bartalucci, Parisi  
**Report Date:** May 2026  
**Dataset:** [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)  

---

## 1. Overview

This document is the authoritative technical record of all changes, design decisions, and scientific justifications made to this repository. It is written for a project collaborator who may not have been present during the refactoring sessions and needs to understand the current state of the code, why each decision was made, and how to work with it effectively.

The project predicts whether an e-commerce session results in a completed purchase (`Revenue = True`) using a binary classification pipeline. The primary optimisation metric is **Macro F1-score**, which equally weights performance on both classes and is robust to the significant class imbalance present in the dataset (~84% non-purchase vs ~16% purchase).

---

## 2. Repository Structure (Current State)

```
Data-Mining-ML-project/
│
├── online_shoppers_intention.csv        # Raw dataset — never modified
├── requirements.txt                     # Exact Python package pins (NEW)
├── README.md                            # Full setup and run instructions (REWRITTEN)
├── CHANGELOG_AND_REFACTORING_REPORT.md  # This file (NEW)
├── Chi2_result_explanation.md           # Supporting EDA note (unchanged)
├── Dataset_description_paper.pdf        # Reference paper (unchanged)
│
├── jupyter/
│   ├── 1_Exploratory_Data_Analysis.ipynb          # EDA + feature selection analysis
│   ├── 2_Nested_CV_and_Strict_Pipelines.ipynb     # Model training, tuning, model persistence
│   ├── 3_Evaluation_Ablation_and_XAI.ipynb        # Evaluation, ablation, ROC/PR, SHAP, LIME
│   ├── 4_Class_Balancing_Experiments.ipynb        # SMOTE methodology experiments
│   └── utils/
│       └── lof_sampler.py                         # Shared LOF_Sampler class (NEW)
│
├── models/
│   ├── final_best_pipeline.pkl                    # Serialised champion pipeline (UPDATED)
│   └── model_metadata.json                        # Champion model metadata (UPDATED)
│
├── figures/
│   └── fig_01 … fig_08.png                        # Pre-generated report figures (150 dpi) (NEW)
│
└── 2_locked_test_data/
    ├── X_test_locked.csv                          # Locked hold-out features (UPDATED)
    └── y_test_locked.csv                          # Locked hold-out labels (unchanged)
```

---

## 3. Detailed Changelog

Each entry below covers what was changed, which file was affected, and why.

---

### CHANGE 1 — Drop `Region` from the feature set

**Files modified:**
- `jupyter/2_Nested_CV_and_Strict_Pipelines.ipynb` — Cell 1 (feature preparation)
- `jupyter/3_Evaluation_Ablation_and_XAI.ipynb` — Cell 1 (D_train reconstruction)
- `2_locked_test_data/X_test_locked.csv` — regenerated (24 features instead of 25)

**What changed:**
```python
# BEFORE
X = df.drop(columns=['Revenue'])          # 25 features

# AFTER
X = df.drop(columns=['Revenue', 'Region'])  # 24 features
```

**Scientific justification:**  
In Notebook 1, a chi-square test of independence was conducted between every categorical feature and the target `Revenue`. The `Region` feature returned **p = 0.32**, which is well above the standard significance threshold of α = 0.05. This means we cannot reject the null hypothesis that `Region` and `Revenue` are statistically independent — in plain terms, *which geographic region a user belongs to has no statistically distinguishable effect on whether they make a purchase*.

Including statistically irrelevant features in a model can introduce noise into the decision boundary, marginally inflate compute time, and make the model harder to interpret. The empirical ablation in Notebook 3 (Section 1.2) confirms that dropping `Region` does not hurt — and marginally improves — Macro F1.

**Important:** Both the training set reconstruction in NB3 and the locked test set were updated to exclude `Region`, ensuring a consistent 24-feature schema everywhere.

---

### CHANGE 2 — Lock the test set and load it from CSV in Notebook 3

**Files modified:**
- `jupyter/2_Nested_CV_and_Strict_Pipelines.ipynb` — Cell 7 (writes locked test set)
- `jupyter/3_Evaluation_Ablation_and_XAI.ipynb` — Cell 1 (reads locked test set)
- `2_locked_test_data/X_test_locked.csv` (new generated file)
- `2_locked_test_data/y_test_locked.csv` (new generated file)

**What changed:**  
Previously, Notebook 3 called `train_test_split` independently to obtain a test set. This is a **critical data integrity error**: if the random seed or data ordering ever changes between notebooks, the "test" set in NB3 would not correspond to the actual holdout used in NB2, meaning the model could silently be evaluated on training data.

The fix is straightforward:
1. Notebook 2 performs the **single canonical split** and immediately serialises both halves to `2_locked_test_data/`.
2. Notebook 3 loads those frozen CSVs — it never re-splits anything.

**Engineering justification:**  
This pattern (seal → load) is standard practice in production ML: the test partition is treated as a legal document — once sealed, it is immutable. Any accidental re-split would invalidate the evaluation entirely.

---

### CHANGE 3 — Add SVM to the Nested CV model comparison

**Files modified:**
- `jupyter/2_Nested_CV_and_Strict_Pipelines.ipynb` — Cell 5 (pipeline definitions + nested CV loop) and Cell 7 (final model selection)

**What changed:**  
A complete SVM pipeline and hyperparameter grid were added alongside the existing Random Forest and XGBoost pipelines:

```python
svm_pipe = ImbPipeline([
    ('lof',        LOF_Sampler(n_neighbors=100, contamination=0.05)),
    ('scaler',     StandardScaler()),
    ('smote',      SMOTE(random_state=42)),
    ('classifier', SVC(random_state=42, probability=True))
])

svm_grid = {
    'smote__sampling_strategy': [0.7, 1.0],
    'classifier__C':            [0.1, 1, 10],
    'classifier__kernel':       ['rbf', 'linear']
}
```

**Project proposal compliance:**  
The official project proposal (Bartalucci-Parisi) explicitly requires a comparison of Random Forest, SVM, and at least one additional algorithm. Adding SVM directly fulfils this mandate. The `StandardScaler` step is essential for SVM: unlike tree-based models, SVMs are not scale-invariant and their optimisation is dominated by high-magnitude features if inputs are not standardised.

**Results:**

| Model | Nested CV Macro F1 |
|---|---|
| Random Forest | 0.8019 |
| SVM (RBF) | 0.7986 |
| **XGBoost (Champion)** | **0.8118** |

---

### CHANGE 4 — Extract `LOF_Sampler` into a shared utility module

**Files created:**
- `jupyter/utils/lof_sampler.py`

**Files modified:**
- `jupyter/2_Nested_CV_and_Strict_Pipelines.ipynb` — Cell 3
- `jupyter/3_Evaluation_Ablation_and_XAI.ipynb` — Cell 1
- `jupyter/4_Class_Balancing_Experiments.ipynb` — Cell 2

**What changed:**  
The `LOF_Sampler` class was previously copy-pasted into three separate notebooks. This violates the **DRY principle** (Don't Repeat Yourself): any bug fix or parameter change would need to be applied in three places, and divergence between copies would be easy to introduce silently.

The class was extracted into a single canonical Python file, and all three notebooks now import it with one line:

```python
%run utils/lof_sampler.py
```

**Technical note on `%run` vs `import`:**  
The `%run` magic command is necessary (rather than a standard `import`) because `LOF_Sampler` must be in the kernel's top-level namespace before `joblib.load` can deserialise the saved pipeline. Python's pickling protocol requires the class definition to be discoverable at unpickling time. `%run` executes the script in the current kernel namespace, making this work seamlessly across different working directories.

**The `LOF_Sampler` class — design rationale:**  
`LocalOutlierFactor` (LOF) from scikit-learn is normally used as a standalone detector, not a pipeline step. Wrapping it in a class that inherits from `imblearn.base.BaseSampler` allows it to be embedded directly in an `imblearn.Pipeline`. The critical design constraint is that `LOF.fit_predict` is called **only within the pipeline's `fit` path** (i.e., on each training fold), never on the test fold. This prevents any form of data leakage where test-set density information could influence which training points are removed.

---

### CHANGE 5 — Add `Region` feature ablation to Notebook 3

**Files modified:**
- `jupyter/3_Evaluation_Ablation_and_XAI.ipynb` — new cells at index 4 and 5

**What changed:**  
A second ablation experiment (Section 1.2) was added to quantify the impact of dropping `Region`. An identical XGBoost pipeline at the champion's fixed hyperparameters is evaluated via 5-fold CV on the full 25-feature dataset (Region included), then compared to the champion score on the 24-feature dataset.

**Scientific justification:**  
Ablation studies are the standard methodology for validating feature engineering decisions in peer-reviewed ML papers. Reporting only the chi-square p-value is an argument from statistical theory; the ablation provides the empirical complement — it directly measures whether the model performs differently with or without the feature. Together, both pieces of evidence (statistical test + empirical comparison) provide a rigorous, reproducible justification for the feature selection decision.

---

### CHANGE 6 — Add ROC Curve, Precision-Recall Curve & Optimal Threshold to Notebook 3

**Files modified:**
- `jupyter/3_Evaluation_Ablation_and_XAI.ipynb` — new cells at index 8 and 9

**What changed:**  
Two new cells (a markdown explanation followed by code) were added after the confusion matrix / classification report:

1. **ROC Curve with AUC** — plots the true positive rate vs false positive rate across all decision thresholds.
2. **Precision-Recall Curve with Average Precision** — plots precision vs recall across all thresholds.
3. **Optimal Threshold Selection** — sweeps the PR-curve thresholds and selects the value that maximises Purchase-class F1, reporting the classification report at both the default (0.50) and optimal thresholds.

**Scientific justification:**  
For imbalanced binary classification, accuracy and even the confusion matrix at a single threshold give an incomplete picture:

- **ROC-AUC** is threshold-independent and measures the model's intrinsic discrimination ability. An AUC of 1.0 = perfect; 0.5 = random.
- **PR-AUC (Average Precision)** is more informative than ROC when the positive class is rare, because it focuses specifically on the minority class. A high AP score indicates the model can retrieve purchasers with high precision, even at moderate recall.
- **Optimal threshold tuning** addresses a known issue: the default threshold of 0.50 implicitly assumes equal class frequencies and equal misclassification costs. In the real world, a false negative (missing a buyer) likely has greater business cost than a false positive (incorrectly flagging a non-buyer). The optimal threshold gives a decision-maker the tool to tune this trade-off.

---

### CHANGE 7 — Add conclusion section to Notebook 3

**Files modified:**
- `jupyter/3_Evaluation_Ablation_and_XAI.ipynb` — new cell at index 13 (final cell)

**What changed:**  
A structured markdown summary (Section 5) was added as the final cell. It consolidates findings from all sections of the notebook into a single, scannable reference covering model performance, ablation results, threshold guidance, and SHAP insights.

**Reason:**  
A standalone notebook should tell a complete story. A conclusion section ensures that a reader (including the course examiner) can review the notebook linearly and arrive at a clear, consolidated interpretation without having to mentally integrate results scattered across earlier cells.

---

### CHANGE 8 — Remove empty Cell 0 and add conclusion to Notebook 4

**Files modified:**
- `jupyter/4_Class_Balancing_Experiments.ipynb` — deleted empty cell at index 0; new cell at index 7

**What changed:**  
- The first cell was an empty code cell with no content — removed programmatically via `nbformat`.
- A new concluding markdown cell (Section 4) was added at the end, explicitly connecting the SMOTE ratio experiment findings to the hyperparameter grid used in Notebook 2.

**Reason:**  
Notebook 4 was previously self-contained but did not explain its relevance to the broader project pipeline. The conclusion makes explicit that the `sampling_strategy=0.7` result from this experiment directly informed the search space in Notebook 2's `GridSearchCV`, providing a reproducible audit trail for the hyperparameter choice.

---

### CHANGE 9 — Apply SHAP / XGBoost compatibility patch

**File patched:**
- `<conda_env>/Lib/site-packages/shap/explainers/_tree.py`

**Problem:**  
`xgboost==3.2.0` serialises the internal `base_score` parameter as a bracketed string (e.g., `'[4.117431E-1]'`). `shap==0.49.1`'s `TreeExplainer` expects a plain float and calls `float(learner_model_param["base_score"])` directly, which raises:
```
ValueError: could not convert string to float: '[4.117431E-1]'
```

**Fix applied:**  
Two lines in `_tree.py` were patched to strip the enclosing brackets before conversion:

```python
# BEFORE
float(learner_model_param["base_score"])

# AFTER
float(str(learner_model_param["base_score"]).strip("[]"))
```

This change handles both XGBoost 2.x (plain float string) and 3.x (bracketed string) without breaking either.

**Why not downgrade XGBoost?**  
Downgrading to a version compatible with `shap` would have meant losing XGBoost 3.x performance improvements and potentially creating future compatibility conflicts. Patching the parsing function is a minimal, targeted fix with no downstream side effects.

---

### CHANGE 10 — Update `models/model_metadata.json` and `models/final_best_pipeline.pkl`

**Files modified:**
- `models/model_metadata.json`
- `models/final_best_pipeline.pkl`

**What changed:**  
Both files were regenerated on a clean re-run of Notebook 2 after all pipeline changes (Region drop, SVM addition, LOF utility refactoring). The metadata JSON now records all three models' nested CV scores:

```json
{
  "model_name": "XGBoost",
  "validation_macro_f1": 0.8147,
  "best_params": {
    "classifier__learning_rate": "0.1",
    "classifier__max_depth": "3",
    "classifier__n_estimators": "100",
    "smote__sampling_strategy": "0.7"
  },
  "nested_cv_scores": {
    "Random Forest": 0.8019,
    "SVM": 0.7986,
    "XGBoost": 0.8118
  }
}
```

---

### CHANGE 11 — Fix `nbconvert` notebook validation errors

**Problem:**  
When attempting to execute notebooks with `nbconvert`, two classes of validation errors appeared:
1. Stream output cells missing the required `"name"` field.
2. Markdown cells with spurious `"outputs"` and `"execution_count"` keys.

These were artefacts of notebook outputs generated in a different Jupyter environment that did not conform to `nbformat` specification v4.

**Fix:**  
A repair script using the `nbformat` library was written to:
- Add `"name": "stdout"` to all stream outputs.
- Clear all outputs and set `execution_count = None` on code cells.
- Strip `"outputs"` and `"execution_count"` keys from markdown cells.

This brought all notebooks into strict `nbformat` v4 compliance, enabling clean `nbconvert` execution.

---

### CHANGE 12 — Create `requirements.txt`

**File created:** `requirements.txt`

**Contents:**

```
scikit-learn==1.7.2
imbalanced-learn==0.14.1
xgboost==3.2.0
shap==0.49.1
pandas==2.3.3
numpy==2.2.5
matplotlib==3.10.9
seaborn==0.13.2
scipy==1.15.3
joblib==1.5.3
notebook==7.5.5
ipykernel
nbconvert==7.17.0
nbformat==5.10.4
nbclient==0.10.4
lime==0.2.0.1
```

The modelling libraries are pinned to the versions used to validate the project. `ipykernel` is listed explicitly because it is required for a Python environment to run the notebooks through the portable `Python 3` Jupyter kernel. The notebooks intentionally keep generic Jupyter kernel metadata (`display_name: Python 3`, `name: python3`) so the project does **not** depend on any local course-specific Conda environment name.

---

### CHANGE 13 — Rewrite `README.md`

**File modified:** `README.md`

**What changed:**  
The original README contained a single-line description. It was replaced with a full, structured document including:
- Results summary table
- Annotated repository structure
- Step-by-step environment setup instructions (Conda and pip)
- Ordered notebook execution guide with runtime estimates
- Methodology overview explaining the pipeline architecture, validation strategy, and feature engineering rationale

---

## 4. Notebook Structure Reference

### `1_Exploratory_Data_Analysis.ipynb`
| Cell | Type | Purpose |
|---|---|---|
| 0 | Markdown | Project introduction |
| 1 | Code | Data loading, null checks, class distribution |
| 2 | Code | Numerical feature distributions (histograms, boxplots) |
| 3 | Code | Categorical feature analysis |
| 4 | Code | Correlation heatmap |
| 5 | Code | Chi-square tests → `Region` flagged as non-significant (p=0.32) |
| 6 | Code | Mutual information scores |
| 7 | Code | LOF profiling (density of outliers in feature space) |

**Outputs used by downstream notebooks:** chi-square insights (used to justify `Region` drop in NB2/NB3).

---

### `2_Nested_CV_and_Strict_Pipelines.ipynb`
| Cell | Type | Purpose |
|---|---|---|
| 0 | Markdown | Objectives: Nested CV, 3-model comparison, leakage prevention |
| 1 | Code | Data loading, encoding, `Region` drop, train/test split, CSV serialisation |
| 2 | Markdown | Nested CV methodology explanation |
| 3 | Code | `%run utils/lof_sampler.py` — load shared sampler |
| 4 | Markdown | Pipeline architecture description |
| 5 | Code | RF / SVM / XGBoost pipeline definitions + Nested CV execution |
| 6 | Markdown | Best model selection logic |
| 7 | Code | Final `GridSearchCV` on full D_train + `joblib.dump` |

**Outputs produced:**
- `models/final_best_pipeline.pkl` (champion pipeline, fitted on full D_train)
- `models/model_metadata.json` (nested CV scores, best hyperparameters)
- `2_locked_test_data/X_test_locked.csv`
- `2_locked_test_data/y_test_locked.csv`

---

### `3_Evaluation_Ablation_and_XAI.ipynb`
| Cell | Type | Purpose |
|---|---|---|
| 0 | Markdown | Objectives |
| 1 | Code | Data load, D_train reconstruct, locked test load, LOF utility, model load |
| 2 | Markdown | Ablation Section 1.1: LOF |
| 3 | Code | LOF ablation — compare champion vs same pipeline without LOF |
| 4 | Markdown | Ablation Section 1.2: Region |
| 5 | Code | Region ablation — compare 24-feature vs 25-feature pipeline |
| 6 | Markdown | Final Test Set Evaluation |
| 7 | Code | Unseal D_test, `champion_pipeline.predict`, confusion matrix, classification report |
| 8 | Markdown | ROC / PR / Threshold intro |
| 9 | Code | ROC curve, PR curve, optimal threshold sweep + side-by-side plot |
| 10 | Markdown | SHAP explanation |
| 11 | Code | SHAP global: `summary_plot` (feature importance across D_test) |
| 12 | Code | SHAP local: `waterfall` plots for TP, FP and FN instances |
| 13 | Markdown | LIME intro |
| 14 | Code | LIME local explanation for the False Positive instance |
| 15 | Markdown | Summary and Conclusions |

---

### `4_Class_Balancing_Experiments.ipynb`
| Cell | Type | Purpose |
|---|---|---|
| 0 | Markdown | Notebook objectives and context |
| 1 | Code | Imports, data loading, encoding |
| 2 | Code | `%run utils/lof_sampler.py` |
| 3 | Markdown | Experimental design — sampler dictionary approach |
| 4 | Code | CV evaluation loop over sampling strategies |
| 5 | Markdown | Visualisation intro |
| 6 | Code | Bar chart: Macro F1 per sampling strategy |
| 7 | Markdown | Conclusions + connection to NB2 pipeline *(NEW)* |

---

### `jupyter/utils/lof_sampler.py`
A standalone Python module defining `LOF_Sampler`, a custom `imbalanced-learn`-compatible sampler.

**Class API:**
```python
class LOF_Sampler(BaseSampler):
    _sampling_type = "clean-sampling"

    def __init__(self, n_neighbors=20, contamination=0.1): ...
    def _fit_resample(self, X, y): ...  # Fits LOF on X, removes outlier indices
```

**Usage in any notebook:**
```python
%run utils/lof_sampler.py
```

---

### CHANGE 14 — Add Pearson heatmap to NB1; add `plt.savefig` to all figure cells; create `figures/` directory

**Files created:** `figures/fig_01_class_dist.png` through `figures/fig_08_shap_waterfall.png`  
**Files modified:** `jupyter/1_Exploratory_Data_Analysis.ipynb`, `jupyter/3_Evaluation_Ablation_and_XAI.ipynb`

**What changed:**  
The Pearson correlation heatmap was referenced in the LaTeX report (`fig:corr`) but did not exist in NB1. A new code cell was inserted to generate and save it. `plt.savefig()` calls were added to all seven other figure-producing cells across NB1 and NB3, and the `figures/` directory was created to collect all eight output PNGs at 150 dpi.

**Reason:**  
All figures referenced in the LaTeX report must be real, generated images that the author can upload to Overleaf. Without savefig calls, figures only display inline and cannot be exported programmatically.

---

### CHANGE 15 — Add CV scores bar chart with ±2 std deviation error bars to NB2 (Danilo Parisi, 03/06/2026)

**Files modified:** `jupyter/2_Nested_CV_and_Strict_Pipelines.ipynb` — new cell at index 8 (updated)

**What changed:**  
The original static bar chart (using hardcoded values from `model_metadata.json`) was replaced by a dynamically computed chart that reads live fold scores (`rf_scores`, `svm_scores`, `xgb_scores`) and adds `yerr=cv_errors` error bars representing ±2 standard deviations across the five outer folds.

**Scientific justification:**  
Reporting only the mean Macro F1 of nested CV without variance understates the uncertainty in the estimate. Error bars allow a reader to visually assess whether differences between models are meaningful or within noise. This is standard practice in ML benchmarking papers.

---

### CHANGE 16 — Fix SHAP feature name mismatch in NB3 Cell 11 (Danilo Parisi, 03/06/2026)

**Files modified:** `jupyter/3_Evaluation_Ablation_and_XAI.ipynb` — Cell 11

**What changed:**  
The SHAP `TreeExplainer` was previously passed a pandas DataFrame, which could trigger a feature-name mismatch error between the DataFrame's column names and the XGBoost model's internal feature names (set during training via the pipeline). The fix:
1. Scales the test set to a raw NumPy array: `X_test_scaled = scaler.transform(X_test)`
2. Passes the NumPy array to `shap.TreeExplainer`
3. Manually re-attaches feature names: `shap_values.feature_names = list(X_test.columns)`

This avoids the `ValueError` while preserving readable feature names in all SHAP plots.

---

### CHANGE 17 — Add LIME local explainability for False Positive analysis (Danilo Parisi, 03/06/2026)

**Files modified:** `jupyter/3_Evaluation_Ablation_and_XAI.ipynb` — new Cells 13–15  
**Files modified:** `requirements.txt` — added `lime==0.2.0.1`

**What changed:**  
A LIME (Local Interpretable Model-Agnostic Explanations) analysis block was added as Cells 13–15 of NB3. It explains the same False Positive instance already analysed by SHAP's waterfall plot, allowing a direct comparison between the two explainability frameworks. A wrapper function correctly bridges the fitted `StandardScaler` from the pipeline with LIME's tabular explainer.

**Scientific value:**  
LIME is model-agnostic and approximates the decision boundary locally with a linear model. When its feature attributions agree with SHAP's tree-exact values, this provides convergent validity for the explanation. Discrepancies reveal where SHAP's global approximation and LIME's local linear approximation diverge, which is itself informative about the model's complexity in that region of feature space.

---

### CHANGE 18 — Add `.gitignore`; untrack committed Jupyter checkpoint files

**Files created:** `.gitignore`  
**Files removed from tracking:** `jupyter/.ipynb_checkpoints/*.ipynb` (4 files)

**What changed:**  
Jupyter's `.ipynb_checkpoints/` folder (auto-generated auto-save artefacts) was accidentally committed. The four checkpoint files were removed from git tracking via `git rm --cached`. A `.gitignore` was created to permanently exclude checkpoints, Python cache, OS metadata, and IDE configuration files from future commits.

**Why this matters:**  
Checkpoint files are auto-generated, change on every save, and carry no useful history. Including them inflates the git diff on every commit and makes the repository harder to navigate.

---

### CHANGE 19 — Fix data leakage in LIME: use `X_train` as background distribution

**Files modified:** `jupyter/3_Evaluation_Ablation_and_XAI.ipynb` — Cell 14

**What changed:**
```python
# BEFORE (leakage)
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_test.values, ...)

# AFTER (correct)
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values, ...)
```

**Technical justification:**  
LIME's `training_data` parameter is used to compute per-feature statistics (mean, standard deviation) for generating the local perturbations used to build the surrogate linear model. Passing `X_test.values` allowed the test set's marginal distributions to influence the explanation process — a form of information leakage. The correct value is `X_train.values`, which is already in scope from the D_train reconstruction at the top of NB3. This fix ensures that LIME's explanations are built entirely from training-set statistics, consistent with how the model itself was trained.

---

## 5. Technical Architecture Summary

### Pipeline Design

Every model is wrapped in the same four-step `imblearn.Pipeline`:

```
Input (D_train fold)
    │
    ▼
LOF_Sampler          ← Removes ~5% density outliers from training fold only
    │
    ▼
StandardScaler       ← Zero-mean, unit-variance standardisation
    │                   (required for SVM; safe for tree models)
    ▼
SMOTE                ← Synthetic minority oversampling to 70% of majority class
    │                   (addresses 84%/16% class imbalance)
    ▼
Classifier           ← RF / SVM / XGBoost
    │
    ▼
Output (predictions / probabilities)
```

### Validation Strategy

```
D_full (12,330 sessions)
    │
    ├── 80% D_train ──── Nested Cross-Validation ────────────────────────────┐
    │       │                                                                  │
    │       ├── Outer loop (5-fold StratifiedKFold): unbiased F1 estimate     │
    │       │       └── Inner loop (3-fold GridSearchCV): hyperparameter opt  │
    │       │                                                                  │
    │       └── Best params → Final GridSearchCV on full D_train → .pkl save ◄┘
    │
    └── 20% D_test ──── LOCKED (sealed to CSV, never seen during training)
```

### Data Leakage Prevention

Three design decisions ensure zero leakage:
1. **LOF inside the pipeline** — outlier removal only occurs within `fit`, never in `transform` of test data.
2. **StandardScaler inside the pipeline** — scaler `fit` happens only on training fold; test fold is only `transform`ed.
3. **SMOTE inside the pipeline** — synthetic oversampling only ever augments training data; the test fold always contains only real observations.

---

## 6. Known Limitations and Future Work

| Item | Description |
|---|---|
| **SHAP/XGBoost patch** | The `shap==0.49.1` library requires a manual patch when used with `xgboost>=3.0.0`. This will be resolved in a future `shap` release. |
| **SVM runtime** | The SVM nested CV is significantly slower than RF and XGBoost (no `n_jobs` gain for kernel SVM). Future versions could substitute `LinearSVC` for the linear kernel evaluation. |
| **Feature selection formality** | The current feature selection is based on chi-square (categorical) and mutual information (continuous). A wrapper-based selection (e.g., Recursive Feature Elimination) integrated within the pipeline could be explored. |
| **Deployment** | The saved `.pkl` pipeline is ready for serving via FastAPI / Flask. A lightweight inference endpoint is a natural next step. |
