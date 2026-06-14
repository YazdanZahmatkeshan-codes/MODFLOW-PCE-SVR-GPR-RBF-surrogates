"""
MODFLOW Groundwater Flow Simulation with Multiple Sampling Strategies
=====================================================================
Generates hydraulic conductivity (hk) samples using four methods, runs a
1D steady-state heterogeneous MODFLOW-2005 model for each sample, and records simulated
heads at a specified observation point.

Model setup
-----------
- Domain  : 1 layer · 1 row · N_CELLS columns
- Boundaries : constant-head at first and last cells
- Stresses   : uniform recharge + two pumping wells
- Solver     : PCG (Preconditioned Conjugate Gradient)

Sampling methods
----------------
LHS    – Latin Hypercube Sampling  (scipy.stats.qmc)
Halton – Scrambled Halton sequence (scipy.stats.qmc)
Sobol  – Scrambled Sobol sequence  (scipy.stats.qmc)
Random – Independent uniform random draws (numpy)

Outputs
-------
One CSV file per sampling method:  inputs_outputs_<METHOD>.csv
  Columns: hk_cell_1 … hk_cell_N, head_at_500m

Dependencies
------------
  pip install flopy numpy scipy pandas
  MODFLOW-2005 executable (mf2005) must be on PATH or in the working directory.
"""

import numpy as np
import flopy.modflow as fpm
import flopy.utils as fpu
from scipy.stats import qmc
import pandas as pd
import timeit

# ── User-Configurable Parameters ───────────────────────────────────────────────
SAMPLE_SIZE      = 32       # Number of realisations per method
HK_MIN           = 1.0     # Lower bound of hydraulic conductivity [m/d]
HK_MAX           = 20.0    # Upper bound of hydraulic conductivity [m/d]
N_CELLS          = 21      # Number of model columns
DELR             = 100.0   # Column width [m]  →  total domain = N_CELLS × DELR
RECHARGE         = 0.001   # Areal recharge rate [m/d]
WELL_RATE        = -1.6    # Extraction rate per well [m³/d]  (negative = pumping)
TOP              = 50.0    # Top elevation of the aquifer [m]
BOT              = 0.0     # Bottom elevation of the aquifer [m]
STRT             = 20.0    # Initial / boundary head [m]
OBS_COL          = 5       # 0-based column index of the observation point

SAMPLING_METHODS = ['LHS', 'Halton', 'Sobol', 'Random']

# ── Helper: Generate Hydraulic Conductivity Samples ───────────────────────────
def generate_samples(method: str, n_samples: int, n_dim: int,
                     low: float, high: float) -> np.ndarray:
    """
    Return an (n_samples × n_dim) array of hk values scaled to [low, high].

    Parameters
    ----------
    method   : One of 'LHS', 'Halton', 'Sobol', or 'Random'.
    n_samples: Number of realisations.
    n_dim    : Number of uncertain parameters (= number of model cells).
    low, high: Bounds of the uniform distribution.
    """
    if method == 'LHS':
        raw = qmc.LatinHypercube(d=n_dim).random(n_samples)
    elif method == 'Halton':
        raw = qmc.Halton(d=n_dim, scramble=True).random(n=n_samples)
    elif method == 'Sobol':
        raw = qmc.Sobol(d=n_dim, scramble=True).random(n=n_samples)
    elif method == 'Random':
        return np.random.uniform(low, high, size=(n_samples, n_dim))
    else:
        raise ValueError(f"Unknown sampling method: '{method}'. "
                         f"Choose from {SAMPLING_METHODS}.")

    return qmc.scale(raw, low, high)


# ── Helper: Build and Run a Single MODFLOW Realisation ────────────────────────
def run_modflow(model_name: str, hk_array: np.ndarray) -> float:
    """
    Construct a MODFLOW model with the given hk field, run it, and return the
    simulated head at OBS_COL.  Returns np.nan if the model does not converge.

    Parameters
    ----------
    model_name : Unique identifier used for all FloPy file I/O.
    hk_array   : 1-D array of hydraulic conductivity values (length = N_CELLS).
    """
    model = fpm.Modflow(modelname=model_name, exe_name='mf2005')

    # -- Discretisation (DIS) --------------------------------------------------
    fpm.ModflowDis(
        model,
        nlay=1, nrow=1, ncol=N_CELLS,
        delr=DELR, delc=1,
        top=TOP, botm=BOT,
    )

    # -- Basic package (BAS): IBOUND and starting heads ------------------------
    # IBOUND = -1 → constant-head cell;  1 → active cell
    ibound = np.ones((1, 1, N_CELLS), dtype=int)
    ibound[0, 0, 0]  = -1   # left boundary
    ibound[0, 0, -1] = -1   # right boundary
    fpm.ModflowBas(model, ibound=ibound, strt=STRT)

    # -- Recharge (RCH): spatially uniform --
    fpm.ModflowRch(model, rech=RECHARGE)

    # -- Well (WEL): two extraction wells ------------------------------------
    well_data = {0: [[0, 0,  5, WELL_RATE],
                     [0, 0, 15, WELL_RATE]]}
    fpm.ModflowWel(model, stress_period_data=well_data)

    # -- Layer Property Flow (LPF): heterogeneous hk -------------------------
    fpm.ModflowLpf(model, hk=hk_array.reshape(1, 1, N_CELLS), laytyp=1)

    # -- Solver and output control -------------------------------------------
    fpm.ModflowPcg(model)
    fpm.ModflowOc(model)

    # Run the model
    model.write_input()
    success, _ = model.run_model(silent=True)

    if not success:
        return np.nan

    # Read the binary head file and extract the head at the observation column
    hds  = fpu.HeadFile(model_name + '.hds')
    head = hds.get_data(totim=1.0)   # shape: (nlay, nrow, ncol)
    return float(head[0, 0, OBS_COL])


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    wall_start = timeit.default_timer()

    for method in SAMPLING_METHODS:
        print(f"\n{'─' * 60}")
        print(f"Sampling method: {method}  ({SAMPLE_SIZE} realisations)")
        print(f"{'─' * 60}")

        samples    = generate_samples(method, SAMPLE_SIZE, N_CELLS, HK_MIN, HK_MAX)
        heads_obs  = []
        hk_records = []

        for i in range(SAMPLE_SIZE):
            model_name = f'gwmodel_{method}_{i}'
            head_val   = run_modflow(model_name, samples[i])

            if np.isnan(head_val):
                print(f"  [WARNING] Realisation {i:>3d} did not converge — skipped.")

            heads_obs.append(head_val)
            hk_records.append(samples[i])

        # ── Save to CSV ──────────────────────────────────────────────────────
        col_names = [f'hk_cell_{j + 1}' for j in range(N_CELLS)]
        df = pd.DataFrame(hk_records, columns=col_names)
        df['head_at_500m'] = heads_obs
        out_path = f'inputs_outputs_{method}.csv'
        df.to_csv(out_path, index=False)

        # ── Summary statistics ───────────────────────────────────────────────
        valid = [h for h in heads_obs if not np.isnan(h)]
        print(f"  Converged: {len(valid)}/{SAMPLE_SIZE}")
        print(f"  Head at obs. cell — mean: {np.mean(valid):.4f} m, "
              f"var: {np.var(valid):.6f} m²")
        print(f"  Results saved to: {out_path}")

    elapsed = timeit.default_timer() - wall_start
    print(f"\nTotal elapsed time: {elapsed:.1f} s")