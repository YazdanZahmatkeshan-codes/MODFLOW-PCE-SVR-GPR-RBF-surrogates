"""
Surrogate Modeling: Support Vector Regression (SVR)
====================================================
Trains an RBF-kernel SVR surrogate on each of N bootstrap training sets and
evaluates predictive performance against a fixed reference dataset.

Workflow
--------
1. Load the reference (test) dataset.
2. For each bootstrap sample (CSV file):
   a. Scale input features and output values with StandardScaler.
   b. Fit an SVR model with RBF kernel.
   c. Select optimal hyperparameters (C, ε, γ) via 5-fold cross-validation.
   d. Predict on the reference set and compute R², RMSE, and NRMSE.
3. Aggregate metrics across all bootstrap models and save to Excel.

File naming convention
----------------------
Reference  : Reference.csv
Training   : bootstrap_sample_1.csv … bootstrap_sample_N.csv

Both files must have the same column layout:
  [param_1, param_2, …, param_P, output]

Dependencies
------------
  pip install numpy pandas scikit-learn openpyxl
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────────────────────────
REFERENCE_FILE     = "Reference.csv"
TRAIN_FILE_PATTERN = "bootstrap_sample_{i}.csv"
N_MODELS           = 50
OUTPUT_FILE        = "SVR_surrogate_metrics.xlsx"

# Hyperparameter search space
PARAM_GRID = {
    "svr__regressor__C":       [0.1, 1, 10, 100, 1000],  # Regularisation strength
    "svr__regressor__epsilon": [0.01, 0.1, 0.5],          # ε-insensitive tube width
    "svr__regressor__gamma":   ["scale", "auto"],          # RBF kernel bandwidth
}
CV_FOLDS = 5


# ── Helper: Compute Prediction Metrics ─────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_range: float) -> tuple[float, float, float]:
    """
    Compute R², RMSE, and NRMSE between true and predicted values.

    Parameters
    ----------
    y_true  : Ground-truth output values.
    y_pred  : Model-predicted output values.
    y_range : Output range (max − min) used for NRMSE normalisation.

    Returns
    -------
    (r2, rmse, nrmse) as floats.
    """
    r2    = r2_score(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / y_range
    return r2, rmse, nrmse


# ── Load Reference Dataset ─────────────────────────────────────────────────────
ref_df  = pd.read_csv(REFERENCE_FILE)
X_ref   = ref_df.iloc[:, :-1].values
y_ref   = ref_df.iloc[:, -1].values
y_range = y_ref.max() - y_ref.min()

print(f"Reference file: {X_ref.shape[0]} samples, {X_ref.shape[1]} parameters")
print("\n" + "═" * 60)
print("Support Vector Regression (SVR)  —  RBF kernel")
print("═" * 60)


# ── Main Loop: Train and Evaluate One SVR per Bootstrap Sample ─────────────────
results = []

for i in range(1, N_MODELS + 1):
    print(f"  Model {i:>2d}/{N_MODELS}", end=" ... ")

    train_df = pd.read_csv(TRAIN_FILE_PATTERN.format(i=i))
    X_train  = train_df.iloc[:, :-1].values
    y_train  = train_df.iloc[:, -1].values

    # Pipeline:
    #   x_scaler  – normalises input features (mean 0, unit variance)
    #   svr       – SVR wrapped in TransformedTargetRegressor to also
    #               scale the output variable before fitting
    inner = TransformedTargetRegressor(
        regressor=SVR(kernel="rbf"),
        transformer=StandardScaler(),
    )
    pipeline = Pipeline([
        ("x_scaler", StandardScaler()),
        ("svr",      inner),
    ])

    # Select hyperparameters by minimising cross-validated MSE
    grid_search = GridSearchCV(
        pipeline, PARAM_GRID,
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate on the reference set
    y_pred          = best_model.predict(X_ref)
    r2, rmse, nrmse = compute_metrics(y_ref, y_pred, y_range)

    print(
        f"R²={r2:.4f}  RMSE={rmse:.6f}  NRMSE={nrmse:.4f}  "
        f"[C={best_params['svr__regressor__C']}, "
        f"ε={best_params['svr__regressor__epsilon']}, "
        f"γ={best_params['svr__regressor__gamma']}]"
    )

    results.append({
        "Model":        f"Model_{i}",
        "Best_C":       best_params["svr__regressor__C"],
        "Best_Epsilon": best_params["svr__regressor__epsilon"],
        "Best_Gamma":   best_params["svr__regressor__gamma"],
        "R2":           r2,
        "RMSE":         rmse,
        "NRMSE":        nrmse,
    })


# ── Save Results ────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
summary_df = results_df[["R2", "RMSE", "NRMSE"]].agg(["mean", "std", "min", "max"])

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="All_Models", index=False)
    summary_df.to_excel(writer, sheet_name="Summary")

print("\n" + "═" * 60)
print(f"Results saved to '{OUTPUT_FILE}'")
print("\nSummary across all bootstrap models:")
print(summary_df.round(4))