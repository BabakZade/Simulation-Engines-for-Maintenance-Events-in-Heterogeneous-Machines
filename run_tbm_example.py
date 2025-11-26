"""
Example script for running the Time-Based Maintenance (TBM) simulation engine.

This script demonstrates:
(1) How to supply fixed and dynamic covariates
(2) How to define a custom covariate update function
(3) How to call the TBM engine and inspect results
"""

import numpy as np
import pandas as pd

from costleap.time_based import simulate_all_machines


# -----------------------------
# 1. USER-SUPPLIED INPUTS
# -----------------------------

# (A) Number of machines and observation time
n_machines = 5
t_obs = 5.0
m = 1000
delta_t = t_obs / m

# (B) Preventive maintenance intervals (user supplies these)
T_value = 1.0
T_machines = {i: T_value for i in range(1, n_machines + 1)}

# (C) Fixed covariates (user supplies — here we generate a simple example)
np.random.seed(4)
fixed_covs = np.random.binomial(1, 0.5, size=(n_machines, 4))

# (D) Dynamic covariates (user supplies initial trajectories)
n_dynamic_features = 1
base_dyn = np.zeros((m+1, n_dynamic_features))
machines_dynamic_covs = {i: base_dyn.copy() for i in range(1, n_machines + 1)}

# (E) User-defined dynamic covariate update rule
def example_cov_update(dynamic_covs, failure_type, valid_indices, machine_id):
    """Set covariate 0 = 1 after minor failure type 3 (just an example)."""
    if failure_type == 3:
        dynamic_covs[valid_indices+1:, 0] = 1
        return dynamic_covs.copy(), True
    return dynamic_covs, False


# -----------------------------
# 2. Model PARAMETERS
# -----------------------------

# Hazard model — user-selected values
include_minor = True
include_catas = True

model_type_minor = "weibull"
shape_minor = 2.0
scale_minor = 2.5
intercept_minor = None

model_type_catas = "weibull"
shape_catas = 2.0
scale_catas = 5.0
intercept_catas = None
with_covariates_catas = False

# Covariate effects (user supplies)
beta_fixed = np.array([-0.2, 0.3, 0.4, -0.1])
beta_dynamic = np.array([0.1])

beta_multinom_fixed = np.array([
    [0.9, 0.9],
    [0.4, 0.5],
    [0.1, 0.0],
    [0.0, 0.2]
])
beta_multinom_dynamic = np.array([[0.1, 0.2]])

n_minor_types = 3


# -----------------------------
# 3. COST PARAMETERS
# -----------------------------

# Catastrophic cost
shape_cat = 8.0
scale_cat = 8.0
loc_fixed_cat = 50.0
gamma_coeffs_cat_fixed = np.zeros(4)
gamma_coeffs_cat_dynamic = np.zeros(1)

# Minor cost (one set per minor type)
shape_minor_list = [6.0, 5.0, 4.0]
scale_minor_list = [5.0, 5.0, 6.0]
loc_fixed_minor_list = [30.0, 20.0, 25.0]

gamma_coeffs_minor_fixed_list = [np.zeros(4) for _ in range(n_minor_types)]
gamma_coeffs_minor_dynamic_list = [np.zeros(1) for _ in range(n_minor_types)]

# Composite type mapping (optional)
minor_combo_map = {3: (1, 2)}

theta_copula = {3: 2.0}

# PM cost
shape_pm = 2.0
scale_pm = 10.0
loc_fixed_pm = 30.0
gamma_coeffs_pm_fixed = np.zeros(4)
gamma_coeffs_pm_dynamic = np.zeros(1)


# -----------------------------
# 4. RUN SIMULATION
# -----------------------------

results_df, dyn_covs_out = simulate_all_machines(
    n_machines=n_machines,
    t_obs=t_obs,
    m=m,
    n_dynamic_features=n_dynamic_features,
    delta_t=delta_t,
    T_machines=T_machines,
    push=0.5,
    include_minor=include_minor,
    model_type_minor=model_type_minor,
    shape_minor=shape_minor,
    scale_minor=scale_minor,
    intercept_minor=intercept_minor,
    with_covariates_minor=True,
    include_catas=include_catas,
    model_type_catas=model_type_catas,
    shape_catas=shape_catas,
    scale_catas=scale_catas,
    intercept_catas=intercept_catas,
    with_covariates_catas=with_covariates_catas,
    fixed_covs=fixed_covs,
    machines_dynamic_covs=machines_dynamic_covs,
    beta_fixed=beta_fixed,
    beta_dynamic=beta_dynamic,
    beta_multinom_fixed=beta_multinom_fixed,
    beta_multinom_dynamic=beta_multinom_dynamic,
    n_minor_types=n_minor_types,
    cov_update_fn=example_cov_update,
    gamma_coeffs_cat_fixed=gamma_coeffs_cat_fixed,
    gamma_coeffs_cat_dynamic=gamma_coeffs_cat_dynamic,
    gamma_coeffs_minor_fixed_list=gamma_coeffs_minor_fixed_list,
    gamma_coeffs_minor_dynamic_list=gamma_coeffs_minor_dynamic_list,
    theta_copula=theta_copula,
    shape_cat=shape_cat,
    scale_cat=scale_cat,
    loc_fixed_cat=loc_fixed_cat,
    shape_minor_list=shape_minor_list,
    scale_minor_list=scale_minor_list,
    loc_fixed_minor_list=loc_fixed_minor_list,
    use_covariates=False,
    minor_combo_map=minor_combo_map,
    gamma_coeffs_pm_fixed=gamma_coeffs_pm_fixed,
    gamma_coeffs_pm_dynamic=gamma_coeffs_pm_dynamic,
    shape_pm=shape_pm,
    scale_pm=scale_pm,
    loc_fixed_pm=loc_fixed_pm
)

print("\n--- Simulation completed ---")
print(results_df.head())

