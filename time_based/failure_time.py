"""
Failure time generation module.

This module contains the getFailureTime function that determines when 
failures occur based on cumulative hazard functions.
"""

import numpy as np
from .hazard import compute_cumulative_integrals


def getFailureTime(
    s, cumulative_integrals_minor, cumulative_integrals_catas, cumulative_integrals,
    dynamic_covs_changed, machine_id, valid_indices, m, delta_t,
    # Minor failure parameters
    include_minor=True,
    model_type_minor="linear", shape_minor=None, scale_minor=None, intercept_minor=None,
    fixed_covs=None, dynamic_covs=None, beta_fixed=None, beta_dynamic=None,
    with_covariates_minor=True, T=None, push=0.0,
    # Catastrophic failure parameters
    include_catas=True,
    model_type_catas="linear", shape_catas=None, scale_catas=None, intercept_catas=None,
    with_covariates_catas=False
):
    """
    General version: supports only-minor / only-catastrophic / both failure types
    
    Parameters:
    -----------
    s : float
        Cumulative hazard threshold (transformed from uniform random variable)
    cumulative_integrals_minor : list
        Cumulative integrals for minor failures
    cumulative_integrals_catas : list
        Cumulative integrals for catastrophic failures
    cumulative_integrals : list
        Combined cumulative integrals
    dynamic_covs_changed : bool
        Flag indicating if dynamic covariates have changed
    machine_id : int
        Machine identifier
    valid_indices : int
        Current time index
    m : int
        Total number of time intervals
    delta_t : float
        Time interval length
    include_minor : bool
        Whether to include minor failures (pm_affects=True)
    include_catas : bool
        Whether to include catastrophic failures (pm_affects=False)
    model_type_minor, model_type_catas : str
        Hazard model types
    shape_minor, scale_minor, intercept_minor : float
        Minor failure hazard parameters
    shape_catas, scale_catas, intercept_catas : float
        Catastrophic failure hazard parameters
    fixed_covs, dynamic_covs : array-like
        Covariate arrays
    beta_fixed, beta_dynamic : array-like
        Covariate coefficients
    with_covariates_minor, with_covariates_catas : bool
        Whether to use covariates for each failure type
    T : float
        PM interval
    push : float
        PM effectiveness
    
    Returns:
    --------
    tuple : 
        - valid_indices: time index of failure
        - ft: failure time
        - s: updated cumulative hazard threshold
        - cumulative_integrals_minor: updated minor integrals
        - cumulative_integrals_catas: updated catastrophic integrals
        - cumulative_integrals: updated combined integrals
        - is_censored: whether observation is censored
    """
    u = np.random.uniform(0, 1)
    s = s - np.log(u)

    # Step 1: Update cumulative integrals
    if (len(cumulative_integrals_minor) == 0 or dynamic_covs_changed or 
        len(cumulative_integrals_catas) == 0):

        if include_minor:
            cumulative_integrals_minor = compute_cumulative_integrals(
                cumulative_integrals_minor, machine_id, valid_indices, m, delta_t,
                shape=shape_minor, scale=scale_minor, intercept=intercept_minor,
                fixed_covs=fixed_covs, dynamic_covs=dynamic_covs, 
                beta_fixed=beta_fixed, beta_dynamic=beta_dynamic,
                model_type=model_type_minor, with_covariates=with_covariates_minor,
                pm_affects=True, T=T, push=push
            )

        if include_catas:
            cumulative_integrals_catas = compute_cumulative_integrals(
                cumulative_integrals_catas, machine_id, valid_indices, m, delta_t,
                shape=shape_catas, scale=scale_catas, intercept=intercept_catas,
                fixed_covs=fixed_covs, dynamic_covs=dynamic_covs, 
                beta_fixed=beta_fixed, beta_dynamic=beta_dynamic,
                model_type=model_type_catas, with_covariates=with_covariates_catas,
                pm_affects=False, T=None, push=0.0
            )

        # Step 2: Combine (only add existing components)
        if include_minor and include_catas:
            cumulative_integrals = [
                mi + ca for mi, ca in zip(cumulative_integrals_minor, cumulative_integrals_catas)
            ]
        elif include_minor:
            cumulative_integrals = cumulative_integrals_minor
        elif include_catas:
            cumulative_integrals = cumulative_integrals_catas
        else:
            raise ValueError("At least one of include_minor or include_catas must be True.")

    # Step 3: Find failure time
    search_start = max(0, valid_indices + 1)
    candidates = np.where(np.array(cumulative_integrals[search_start:]) >= s)[0]

    if len(candidates) == 0:
        # Censored observation
        is_censored = True
        return (valid_indices, -1, s, cumulative_integrals_minor, 
                cumulative_integrals_catas, cumulative_integrals, is_censored)

    valid_indices = candidates[0] + search_start
    ft = valid_indices * delta_t
    is_censored = False

    return (valid_indices, ft, s, cumulative_integrals_minor, 
            cumulative_integrals_catas, cumulative_integrals, is_censored)
