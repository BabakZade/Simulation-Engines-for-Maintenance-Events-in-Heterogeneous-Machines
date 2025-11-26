"""
Cost simulation module.

This module handles:
- Gamma distribution cost generation with covariates
- Frank Copula for correlated costs
- Cost simulation for all failure types
- PM cost generation
"""

import numpy as np
from scipy.stats import gamma

def generate_gamma_cost(fixed_covs, dynamic_cov_t, gamma_coeffs_fixed, gamma_coeffs_dynamic,
                        machine_id, a, b, loc_fixed, use_covariates):
    """
    Generate cost from Gamma distribution with optional covariate effects
    
    Parameters:
    -----------
    gamma_coeffs_fixed : array-like
        Fixed covariate coefficients for location parameter
    gamma_coeffs_dynamic : array-like
        Dynamic covariate coefficients for location parameter
    a : float
        Gamma distribution shape parameter
    b : float
        Gamma distribution scale parameter
    loc_fixed : float
        Fixed location parameter when not using covariates
    use_covariates : bool
        Whether to use covariates to affect location parameter
    
    Returns:
    --------
    float : Generated cost from Gamma distribution
    """
    if use_covariates:
        loc = 0.0
        if fixed_covs is not None and gamma_coeffs_fixed is not None and len(gamma_coeffs_fixed) > 0:
            loc += np.dot(fixed_covs[machine_id - 1], gamma_coeffs_fixed)
        if dynamic_cov_t is not None and gamma_coeffs_dynamic is not None and len(gamma_coeffs_dynamic) > 0:
            loc += np.dot(dynamic_cov_t, gamma_coeffs_dynamic)
        loc = float(loc)
    else:
        loc = loc_fixed

    cost = gamma.rvs(a, loc=loc, scale=b)
    return cost

# Frank Copula sampling function
def frank_copula_sample(theta, u1):
    """
    Sample from Frank Copula given u1 (more robust implementation)
    
    Parameters:
    -----------
    theta : float
        Frank Copula dependence parameter
    u1 : float
        First uniform random variable in [0,1]
    
    Returns:
    --------
    float : Second uniform random variable u2 in [0,1] dependent on u1
    """
    if theta == 0:
        return np.random.uniform()

    w = np.random.uniform()
    e_neg_theta = np.exp(-theta)
    denom = np.exp(-theta * u1)

    numerator = w * (e_neg_theta - 1.0)
    denom_A = denom - w * (denom - 1.0)

    eps = 1e-14
    denom_A = np.maximum(denom_A, eps)
    inside = 1.0 + numerator / denom_A
    inside = np.maximum(inside, eps)

    u2 = -np.log(inside) / theta
    u2 = np.clip(u2, 0.0, 1.0)
    return u2

# Correlated cost generation using Frank Copula
def get_failure_costs_with_frank_copula(machine_id, dynamic_cov_t, fixed_covs, 
    gamma_coeffs_y1_fixed, gamma_coeffs_y1_dynamic, 
    gamma_coeffs_y2_fixed, gamma_coeffs_y2_dynamic, 
    theta, a1=2.0, a2=2.0, b1=1.0, b2=1.0, 
    loc_fixed1=0.0, loc_fixed2=0.0, use_covariates=True):
    """
    Generate two correlated failure costs using Frank Copula
    Supports both fixed and dynamic covariates
    
    Parameters:
    -----------
    gamma_coeffs_y1_fixed, gamma_coeffs_y2_fixed : array-like
        Fixed covariate coefficients for cost 1 and cost 2
    gamma_coeffs_y1_dynamic, gamma_coeffs_y2_dynamic : array-like
        Dynamic covariate coefficients for cost 1 and cost 2
    theta : float
        Frank Copula dependence parameter
    a1, a2 : float
        Gamma distribution shape parameters for cost 1 and 2
    b1, b2 : float
        Gamma distribution scale parameters for cost 1 and 2
    loc_fixed1, loc_fixed2 : float
        Fixed location parameters when not using covariates
    use_covariates : bool
        Whether to use covariates to affect location parameters
    
    Returns:
    --------
    tuple : (cost1, cost2) - Two correlated costs
    """
    # Calculate location parameters
    if use_covariates:
        loc1 = 0.0
        loc2 = 0.0
        
        if fixed_covs is not None and gamma_coeffs_y1_fixed is not None and len(gamma_coeffs_y1_fixed) > 0:
            loc1 += np.dot(fixed_covs[machine_id - 1], gamma_coeffs_y1_fixed)
        if dynamic_cov_t is not None and gamma_coeffs_y1_dynamic is not None and len(gamma_coeffs_y1_dynamic) > 0:
            loc1 += np.dot(dynamic_cov_t, gamma_coeffs_y1_dynamic)
            
        if fixed_covs is not None and gamma_coeffs_y2_fixed is not None and len(gamma_coeffs_y2_fixed) > 0:
            loc2 += np.dot(fixed_covs[machine_id - 1], gamma_coeffs_y2_fixed)
        if dynamic_cov_t is not None and gamma_coeffs_y2_dynamic is not None and len(gamma_coeffs_y2_dynamic) > 0:
            loc2 += np.dot(dynamic_cov_t, gamma_coeffs_y2_dynamic)
            
        loc1 = float(loc1)
        loc2 = float(loc2)
    else:
        loc1 = loc_fixed1
        loc2 = loc_fixed2

    # Copula sampling
    u1 = np.random.uniform()
    u2 = frank_copula_sample(theta, u1)

    cost1 = gamma.ppf(u1, a=a1, loc=loc1, scale=b1)
    cost2 = gamma.ppf(u2, a=a2, loc=loc2, scale=b2)

    return cost1, cost2

def simulate_failure_costs(
    machine_id, failure_index, failure_time, failure_type,
    fixed_covs, dynamic_covs, n_minor_types,
    gamma_coeffs_cat_fixed, gamma_coeffs_cat_dynamic,
    gamma_coeffs_minor_fixed_list,gamma_coeffs_minor_dynamic_list,
    theta_copula,
    shape_cat=2.0, scale_cat=1.0, loc_fixed_cat=0.0,
    shape_minor_list=None,  scale_minor_list=None, loc_fixed_minor_list=None,
    use_covariates=True, minor_combo_map=None
):
    """
    Generate maintenance costs per event.

    Rules
    -----
    - ftype == -1 : catastrophic → single Gamma cost.
    - ftype == 0  : censored     → cost = 0.
    - ftype in [1..n_minor_types]:
        * If `ftype` is in `minor_combo_map` as a PAIR (a, b), interpret the event as
          types `a` and `b` occurring simultaneously → generate TWO correlated Gamma
          costs via a Frank copula and sum them.
        * Otherwise, a single minor-type Gamma cost.

    Notes
    -----
    - Only PAIR combinations are supported in `minor_combo_map`. If a mapping has more
      or fewer than 2 components, a ValueError is raised.
    - `theta_copula` can be:
        * a single float (used for all pairs), or
        * a dict providing pair-specific values. The lookup will try:
            1) the composite type id (ftype), then
            2) the sorted component tuple, e.g. (min(a,b), max(a,b)).
      If neither key exists in the dict, a ValueError is raised.
    - `dynamic_covs[idx]` is used to fetch the dynamic covariate vector at the event time.
      If `idx` is out of bounds (e.g., censored rows), a zero-like vector is used.

    Expected shapes
    ---------------
    - fixed_covs:  (n_machines, n_fixed_features)
    - dynamic_covs: (n_time_steps, n_dynamic_features) or N-D with time on axis 0
    - *_list arguments: lists of length n_minor_types
    """
    # Defaults
    if shape_minor_list is None:
        shape_minor_list = [2.0] * n_minor_types
    if scale_minor_list is None:
        scale_minor_list = [1.0] * n_minor_types
    if loc_fixed_minor_list is None:
        loc_fixed_minor_list = [0.0] * n_minor_types
    if minor_combo_map is None:
        minor_combo_map = {}

    # Validate lengths
    if len(gamma_coeffs_minor_fixed_list) != n_minor_types:
        raise ValueError("len(gamma_coeffs_minor_fixed_list) must equal n_minor_types.")
    if len(gamma_coeffs_minor_dynamic_list) != n_minor_types:
        raise ValueError("len(gamma_coeffs_minor_dynamic_list) must equal n_minor_types.")
    if len(shape_minor_list) != n_minor_types:
        raise ValueError("len(shape_minor_list) must equal n_minor_types.")
    if len(scale_minor_list) != n_minor_types:
        raise ValueError("len(scale_minor_list) must equal n_minor_types.")
    if len(loc_fixed_minor_list) != n_minor_types:
        raise ValueError("len(loc_fixed_minor_list) must equal n_minor_types.")

    # Validate `minor_combo_map` keys and enforce pair-only combos
    for combo_type, comps in minor_combo_map.items():
        if not (1 <= combo_type <= n_minor_types):
            raise ValueError(f"minor_combo_map key out of range: {combo_type} not in 1..{n_minor_types}")
        if not isinstance(comps, (tuple, list)) or len(comps) != 2:
            raise ValueError(f"minor_combo_map[{combo_type}] must be a 2-tuple/list, e.g. (1, 3)")
        a, b = int(comps[0]), int(comps[1])
        if a == b:
            raise ValueError(f"minor_combo_map[{combo_type}] components must differ: ({a}, {b})")
        if not (1 <= a <= n_minor_types) or not (1 <= b <= n_minor_types):
            raise ValueError(f"minor_combo_map[{combo_type}] components out of range: ({a}, {b})")
   
    costs = []

    for j, ftype in enumerate(failure_type):
        idx = failure_index[j]     
        dynamic_cov_t = dynamic_covs[idx] if dynamic_covs is not None else None 


        if ftype == -1:  # catastrophic failure
            cost = generate_gamma_cost(
                fixed_covs, dynamic_cov_t, gamma_coeffs_cat_fixed, gamma_coeffs_cat_dynamic, machine_id,
                a=shape_cat, b=scale_cat, loc_fixed=loc_fixed_cat, use_covariates=use_covariates
            )
            costs.append(cost)
            continue

        if ftype == 0:  # truncated
            costs.append(0.0)
            continue

        # Composite minor type → PAIR only via minor_combo_map
        if ftype in minor_combo_map:
            a_type, b_type = minor_combo_map[ftype]
            i1 = int(a_type) - 1
            i2 = int(b_type) - 1

            # Resolve theta:
            # - if dict → must be keyed by composite ftype (int)
            # - if float → use the same value for all pairs
            if isinstance(theta_copula, dict):
                if ftype not in theta_copula:
                    raise ValueError(
                        f"No theta_copula provided for composite type {ftype}. "
                        f"Provide theta_copula as a float or a dict keyed by composite ftype, e.g. {{4: 1.5}}."
                    )
                theta_for_pair = float(theta_copula[ftype])
            else:
                theta_for_pair = float(theta_copula)

            # Two correlated Gamma costs via Frank copula, then sum
            c1, c2 = get_failure_costs_with_frank_copula(
                machine_id=machine_id,
                dynamic_cov_t=dynamic_cov_t,  # flattened internally
                fixed_covs=fixed_covs,
                gamma_coeffs_y1_fixed=gamma_coeffs_minor_fixed_list[i1],
                gamma_coeffs_y1_dynamic=gamma_coeffs_minor_dynamic_list[i1],
                gamma_coeffs_y2_fixed=gamma_coeffs_minor_fixed_list[i2],
                gamma_coeffs_y2_dynamic=gamma_coeffs_minor_dynamic_list[i2],
                theta=theta_for_pair,
                a1=shape_minor_list[i1], a2=shape_minor_list[i2],
                b1=scale_minor_list[i1], b2=scale_minor_list[i2],
                loc_fixed1=loc_fixed_minor_list[i1], loc_fixed2=loc_fixed_minor_list[i2],
                use_covariates=use_covariates
            )
            costs.append(float(c1 + c2))
        else:
            # Single minor type
            i0 = ftype - 1
            c = generate_gamma_cost(
                fixed_covs, dynamic_cov_t,
                gamma_coeffs_minor_fixed_list[i0], gamma_coeffs_minor_dynamic_list[i0],
                machine_id,
                a=shape_minor_list[i0], b=scale_minor_list[i0], loc_fixed=loc_fixed_minor_list[i0],
                use_covariates=use_covariates
            )
            costs.append(float(c))      
            
    return costs

# =========================================
# PM costs generation
# =========================================
def simulate_periodical_pm(t_obs, T, delta_t, machine_id, fixed_covs, dynamic_covs,
                           gamma_coeffs_pm_fixed, gamma_coeffs_pm_dynamic,
                           a=2.0, b=1.0, loc_fixed=0.0, use_covariates=True):
    """
    Simulate the complete process of periodical preventive maintenance (PM)
    
    Parameters:
    -----------
    T : float
        PM period (time between consecutive PMs)
    gamma_coeffs_pm_fixed, gamma_coeffs_pm_dynamic : array-like
        Coefficients for PM cost
    a : float
        Gamma shape parameter for PM cost
    b : float
        Gamma scale parameter for PM cost
    loc_fixed : float
        Fixed location parameter for PM cost
    use_covariates : bool
        Whether to use covariates
    
    Returns:
    --------
    tuple : (pm_index, pm_times, pm_costs)
        - pm_index: list of PM time indices
        - pm_times: list of PM occurrence times
        - pm_costs: list of PM costs
    """
    pm_times = []
    pm_costs = []
    pm_index = []

    # Perform PM every T time units until simulation ends
    t = T
    while t <= t_obs:
        # Get dynamic covariates at this time
        index = int(t / delta_t)
        pm_index.append(index)
        dynamic_cov_t = dynamic_covs[index, :] if dynamic_covs is not None else None

        # Calculate PM cost
        cost = generate_gamma_cost(
            fixed_covs, dynamic_cov_t, 
            gamma_coeffs_pm_fixed, gamma_coeffs_pm_dynamic,
            machine_id, a, b, loc_fixed, use_covariates
        )

        pm_times.append(t)
        pm_costs.append(float(cost))

        t += T

    return pm_index, pm_times, pm_costs
