"""
Surrogate Modeling: Radial Basis Function (RBF) Interpolation
=============================================================
Trains an RBF surrogate on each of N bootstrap training sets and evaluates
predictive performance against a fixed reference dataset.

Workflow
--------
1. Load the reference (test) dataset.
2. For each bootstrap sample (CSV file):
   a. Standardise input features and output values with StandardScaler.
   b. Select the optimal smoothing parameter via k-fold cross-validation.
   c. Fit the final RBF interpolant on all training data.
   d. Predict on the reference set and compute R², RMSE, and NRMSE.
3. Aggregate metrics across all bootstrap models and save to Excel.

RBF kernel
----------
Thin-plate splines ("thin_plate") are used as the radial basis function
because they provide good approximation quality for scattered data with
no explicit length-scale parameter to tune.

Smoothing parameter
-------------------
The smoothing value s controls the trade-off between exact interpolation
(s = 0) and a least-squares fit that tolerates noise (s > 0).  The best s
is chosen from SMOOTHING_GRID via 5-fold cross-validation.

File naming convention
----------------------
Reference  : Reference.csv
Training   : bootstrap_sample_1.csv … bootstrap_sample_N.csv

Both files must have the same column layout:
  [param_1, param_2, …, param_P, output]

Dependencies
------------
  pip install numpy pandas scikit-learn scipy openpyxl
"""

import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────────────────────────
REFERENCE_FILE     = "Reference.csv"
TRAIN_FILE_PATTERN = "bootstrap_sample_{i}.csv"
N_MODELS           = 50
RBF_KERNEL         = "thin_plate"                      # Radial basis function type
SMOOTHING_GRID     = [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]  # Candidate smoothing values
CV_FOLDS           = 5
RANDOM_STATE       = 42
OUTPUT_FILE        = "RBF_surrogate_metrics.xlsx"


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


# ── Helper: Cross-Validate Smoothing Parameter ─────────────────────────────────
def select_smoothing(X: np.ndarray, y: np.ndarray,
                     candidates: list[float], n_splits: int = 5,
                     random_state: int = 42) -> float:
    """
    Choose the RBF smoothing value that minimises k-fold cross-validated MSE.

    Parameters
    ----------
    X          : Standardised input matrix (n_samples × n_params).
    y          : Standardised output vector.
    candidates : List of smoothing values to evaluate.
    n_splits   : Number of cross-validation folds.
    random_state : Seed for fold shuffling.

    Returns
    -------
    Best smoothing value from `candidates`.
    """
    kf        = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_s    = candidates[0]
    best_mse  = np.inf

    for s in candidates:
        fold_mses = []
        for train_idx, val_idx in kf.split(X):
            try:
                # Rbf expects each dimension as a separate positional argument.
                # *X[train_idx].T unpacks the matrix columns into (x1, x2, ..., xP).
                rbf  = Rbf(*X[train_idx].T, y[train_idx],
                           function=RBF_KERNEL, smooth=s)
                pred = rbf(*X[val_idx].T)
                fold_mses.append(np.mean((y[val_idx] - pred) ** 2))
            except Exception:
                # Degenerate fold (e.g., singular system) — penalise this s
                fold_mses.append(np.inf)

        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse, best_s = mean_mse, s

    return best_s


# ── Load Reference Dataset ─────────────────────────────────────────────────────
ref_df  = pd.read_csv(REFERENCE_FILE)
X_ref   = ref_df.iloc[:, :-1].values
y_ref   = ref_df.iloc[:, -1].values
y_range = y_ref.max() - y_ref.min()

print(f"Reference file: {X_ref.shape[0]} samples, {X_ref.shape[1]} parameters")
print("\n" + "═" * 60)
print(f"Radial Basis Function (RBF)  —  kernel: {RBF_KERNEL}")
print("═" * 60)


# ── Main Loop: Train and Evaluate One RBF per Bootstrap Sample ─────────────────
results = []

for i in range(1, N_MODELS + 1):
    print(f"  Model {i:>2d}/{N_MODELS}", end=" ... ")

    train_df = pd.read_csv(TRAIN_FILE_PATTERN.format(i=i))
    X_train  = train_df.iloc[:, :-1].values
    y_train  = train_df.iloc[:, -1].values

    # Standardise input features
    x_scaler   = StandardScaler()
    X_train_sc = x_scaler.fit_transform(X_train)
    X_ref_sc   = x_scaler.transform(X_ref)

    # Standardise output (zero mean, unit variance) for numerical stability
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_sc    = (y_train - y_mean) / y_std

    # Select optimal smoothing parameter via cross-validation
    best_s = select_smoothing(X_train_sc, y_train_sc, SMOOTHING_GRID,
                               n_splits=CV_FOLDS, random_state=RANDOM_STATE)

    # Fit the final RBF interpolant on all training data
    rbf = Rbf(*X_train_sc.T, y_train_sc, function=RBF_KERNEL, smooth=best_s)

    # Predict on the reference set and invert the output standardisation
    y_pred_sc       = rbf(*X_ref_sc.T)
    y_pred          = y_pred_sc * y_std + y_mean
    r2, rmse, nrmse = compute_metrics(y_ref, y_pred, y_range)

    print(f"R²={r2:.4f}  RMSE={rmse:.6f}  NRMSE={nrmse:.4f}"
          f"  [smoothing={best_s}]")

    results.append({
        "Model":          f"Model_{i}",
        "Best_Smoothing": best_s,
        "R2":             r2,
        "RMSE":           rmse,
        "NRMSE":          nrmse,
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