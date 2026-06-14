"""
Surrogate Modeling: Polynomial Chaos Expansion (PCE) with LASSO Regression
===========================================================================
Trains a PCE surrogate on each of N bootstrap training sets and evaluates
predictive performance against a fixed reference dataset.

Workflow
--------
1. Load the reference (test) dataset.
2. For each bootstrap sample (CSV file):
   a. Define uniform marginal distributions from data bounds.
   b. Build a multivariate polynomial basis via Stieltjes recurrence.
   c. Evaluate the basis on all training points to form the design matrix.
   d. Fit a LASSO regressor with cross-validated regularisation strength α.
   e. Predict on the reference set and compute R², RMSE, and NRMSE.
3. Aggregate metrics across all bootstrap models and save to Excel.

File naming convention
----------------------
Reference  : Reference.csv
Training   : bootstrap_sample_1.csv … bootstrap_sample_N.csv

Both files must have the same column layout:
  [param_1, param_2, …, param_P, output]

Dependencies
------------
  pip install numpy pandas scikit-learn chaospy openpyxl
"""

import warnings
import numpy as np
import pandas as pd
import chaospy as cp
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Configuration ───────────────────────────────────────────────────────────────
REFERENCE_FILE     = "Reference.csv"
TRAIN_FILE_PATTERN = "bootstrap_sample_{i}.csv"
N_MODELS           = 50          # Number of bootstrap training files
POLY_ORDER         = 2           # PCE polynomial order
ALPHAS             = np.logspace(-5, 1, 50)   # LASSO regularisation candidates
CV_FOLDS           = 5           # Cross-validation folds for α selection
OUTPUT_FILE        = "PCE_surrogate_metrics.xlsx"


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
X_ref   = ref_df.iloc[:, :-1].values   # Input parameters
y_ref   = ref_df.iloc[:, -1].values    # Simulation output
y_range = y_ref.max() - y_ref.min()    # For NRMSE normalisation

n_ref_samples, n_params = X_ref.shape
print(f"Reference file: {n_ref_samples} samples, {n_params} parameters")


# ── Main Loop: Train and Evaluate One PCE per Bootstrap Sample ─────────────────
results = []

for i in range(1, N_MODELS + 1):
    print(f"\n── Model {i:>2d}/{N_MODELS} " + "─" * 35)

    # Load bootstrap training data
    train_df = pd.read_csv(TRAIN_FILE_PATTERN.format(i=i))
    X_train  = train_df.iloc[:, :-1].values
    y_train  = train_df.iloc[:, -1].values
    n_train  = X_train.shape[0]
    print(f"   Training samples : {n_train}")

    # ── Step 1: Define Marginal Distributions ──────────────────────────────────
    # Uniform distributions bounded by the union of training and reference ranges.
    # A small epsilon prevents boundary artefacts in the polynomial evaluation.
    distributions = []
    for j in range(n_params):
        lo  = min(X_train[:, j].min(), X_ref[:, j].min())
        hi  = max(X_train[:, j].max(), X_ref[:, j].max())
        eps = 1e-6 * max(abs(hi - lo), 1.0)
        distributions.append(cp.Uniform(lo - eps, hi + eps))

    joint_dist = cp.J(*distributions)

    # ── Step 2: Build Polynomial Basis ────────────────────────────────────────
    # Stieltjes' method constructs orthogonal polynomials w.r.t. the joint dist.
    expansion = cp.expansion.stieltjes(POLY_ORDER, joint_dist)
    n_basis   = len(expansion)
    print(f"   Basis terms      : {n_basis}  (order={POLY_ORDER}, dim={n_params})")

    if n_basis > n_train:
        print(f"   [NOTE] Basis size ({n_basis}) > training samples ({n_train}). "
              "LASSO will handle the underdetermined system via regularisation.")

    # ── Step 3: Evaluate Basis (Design Matrix) ─────────────────────────────────
    # Rows = training realisations, columns = polynomial basis terms.
    Psi_train = np.array(expansion(*X_train.T), dtype=float).T   # (n_train, n_basis)
    Psi_ref   = np.array(expansion(*X_ref.T),   dtype=float).T   # (n_ref,   n_basis)

    # ── Step 4: LASSO with Cross-Validated Regularisation ─────────────────────
    # StandardScaler normalises columns of the design matrix before regression.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso",  Lasso(max_iter=100_000, tol=1e-4)),
    ])

    grid_search = GridSearchCV(
        pipeline,
        param_grid={"lasso__alpha": ALPHAS},
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(Psi_train, y_train)

    best_model = grid_search.best_estimator_
    best_alpha = grid_search.best_params_["lasso__alpha"]
    print(f"   Best LASSO α     : {best_alpha:.2e}")

    # ── Step 5: Evaluate on Reference Set ─────────────────────────────────────
    y_pred          = best_model.predict(Psi_ref)
    r2, rmse, nrmse = compute_metrics(y_ref, y_pred, y_range)

    print(f"   R²    = {r2:.4f}")
    print(f"   RMSE  = {rmse:.6f}")
    print(f"   NRMSE = {nrmse:.4f}")

    results.append({
        "Model":      f"Model_{i}",
        "Best_Alpha": round(best_alpha, 8),
        "R2":         r2,
        "RMSE":       rmse,
        "NRMSE":      nrmse,
    })


# ── Save Results ────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
summary_df = results_df[["R2", "RMSE", "NRMSE"]].agg(["mean", "std", "min", "max"])

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="All_Models", index=False)
    summary_df.to_excel(writer, sheet_name="Summary")

print("\n" + "═" * 55)
print(f"Results saved to '{OUTPUT_FILE}'")
print("\nSummary across all bootstrap models:")
print(summary_df.round(4))