"""
Surrogate Modeling: Gaussian Process Regression (GPR)
======================================================
Trains a GPR surrogate on each of N bootstrap training sets and evaluates
predictive performance against a fixed reference dataset.

Workflow
--------
1. Load the reference (test) dataset.
2. For each bootstrap sample (CSV file):
   a. Standardise input features with StandardScaler.
   b. Define a composite kernel: ConstantKernel × Matérn(ν=2.5) + WhiteKernel.
   c. Fit the GPR model (kernel hyperparameters optimised via log-marginal
      likelihood maximisation with multiple random restarts).
   d. Predict on the reference set and compute R², RMSE, and NRMSE.
3. Aggregate metrics across all bootstrap models and save to Excel.

Kernel choice
-------------
ConstantKernel × Matérn(ν=2.5) is a popular choice for physical surrogates
because it produces once-differentiable sample paths, balancing smoothness and
flexibility.  WhiteKernel accounts for observation noise / numerical error.

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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────────────────────────
REFERENCE_FILE     = "Reference.csv"
TRAIN_FILE_PATTERN = "bootstrap_sample_{i}.csv"
N_MODELS           = 50
N_RESTARTS         = 3      # Number of random restarts for kernel optimisation
RANDOM_STATE       = 42     # Seed for reproducibility
OUTPUT_FILE        = "GPR_surrogate_metrics.xlsx"


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
print("Gaussian Process Regression (GPR)  —  Matérn kernel (ν=2.5)")
print("═" * 60)


# ── Main Loop: Train and Evaluate One GPR per Bootstrap Sample ─────────────────
results = []

for i in range(1, N_MODELS + 1):
    print(f"  Model {i:>2d}/{N_MODELS}", end=" ... ")

    train_df = pd.read_csv(TRAIN_FILE_PATTERN.format(i=i))
    X_train  = train_df.iloc[:, :-1].values
    y_train  = train_df.iloc[:, -1].values

    # Standardise input features (GPR is sensitive to input scale)
    x_scaler   = StandardScaler()
    X_train_sc = x_scaler.fit_transform(X_train)
    X_ref_sc   = x_scaler.transform(X_ref)

    # Composite kernel definition:
    #   ConstantKernel  – overall output scale
    #   Matérn(ν=2.5)   – smooth, physically meaningful covariance
    #   WhiteKernel     – noise / numerical jitter term
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,              # Subtract and divide by y mean/std
        n_restarts_optimizer=N_RESTARTS,
        random_state=RANDOM_STATE,
    )
    gpr.fit(X_train_sc, y_train)

    # Predict mean only (no uncertainty needed for deterministic benchmarking)
    y_pred          = gpr.predict(X_ref_sc)
    r2, rmse, nrmse = compute_metrics(y_ref, y_pred, y_range)

    print(f"R²={r2:.4f}  RMSE={rmse:.6f}  NRMSE={nrmse:.4f}")

    results.append({
        "Model":            f"Model_{i}",
        "Optimized_Kernel": str(gpr.kernel_),   # Log optimised kernel parameters
        "R2":               r2,
        "RMSE":             rmse,
        "NRMSE":            nrmse,
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