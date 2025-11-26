"""
Cost Module

This module provides classes and functions for computing maintenance costs
based on different maintenance types and covariate effects.
"""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from scipy.stats import gamma as gamma_dist


@dataclass
class CostParams:
    """
    Cost model parameters.
    
    Attributes:
        pm_shape: Gamma shape parameter for perfect PM cost
        pm_scale: Gamma scale parameter for perfect PM cost
        cm_shape: Gamma shape parameter for CM cost
        cm_scale: Gamma scale parameter for CM cost (should be much higher than PM)
        c_0: Base cost coefficient for repair effectiveness in imperfect PM
        gamma_coeffs: Coefficients for c_fix calculation (γ in exp(γ'W)) in imperfect PM
        epsilon_std: Standard deviation of noise term ε in imperfect PM
    """
    # Perfect PM (gamma distribution)
    pm_shape: float = 2.0
    pm_scale: float = 50.0
    
    # CM (gamma distribution) - should be much higher than PM
    cm_shape: float = 2.0
    cm_scale: float = 200.0  # 4x PM cost
    
    # Imperfect PM parameters
    c_0: float = 100.0
    gamma_coeffs: np.ndarray = None
    epsilon_std: float = 5.0


def compute_maintenance_cost(
    maintenance_type: str,
    cost_params: CostParams,
    cost_covariates: np.ndarray = None,
    cost_covariate_effects: Dict[str, np.ndarray] = None,
    repair_effectiveness: float = None
) -> float:
    """
    Compute maintenance cost based on maintenance type and covariates.
    
    Cost Models:
    - Perfect PM: Cost ~ Gamma(shape, scale) + location, where location = β'W
    - CM: Cost ~ Gamma(shape, scale) + location, where location = β'W
    - Imperfect PM: Cost = c_fix + c_0*u + ε, where c_fix = exp(γ'W), u is repair effectiveness
    
    Args:
        maintenance_type: Type of maintenance ('perfect_pm', 'cm', 'imperfect_pm')
        cost_params: Cost parameter object
        cost_covariates: Cost covariate vector W
        cost_covariate_effects: Covariate effects on cost parameters
            Example: {"pm_location": np.array([0.1, 0.2]), "cm_location": np.array([0.3, -0.1])}
        repair_effectiveness: Repair effectiveness u = (y_before - y_after) / y_before 
                            (only used for imperfect_pm)
    
    Returns:
        Maintenance cost
    
    Notes:
        - Uses 3-parameter gamma distribution for perfect PM and CM
        - Shape and scale parameters are fixed
        - Location parameter is affected by covariates: loc(W) = β'W (linear effect)
        - Final cost: Cost = Gamma(shape, scale) + loc(W)
        - Covariates directly shift the entire distribution
    """
    
    if maintenance_type == 'perfect_pm':
        # Perfect PM cost follows 3-parameter gamma distribution
        shape = cost_params.pm_shape
        scale = cost_params.pm_scale
        
        # Compute location parameter from covariates (linear effect)
        location = 0.0
        if cost_covariates is not None and cost_covariate_effects is not None:
            if 'pm_location' in cost_covariate_effects:
                beta = cost_covariate_effects['pm_location']
                location = np.dot(beta, cost_covariates)
        
        # Generate from 3-parameter gamma: shape, scale, location
        cost = gamma_dist.rvs(a=shape, scale=scale, loc=location)
        return cost
    
    elif maintenance_type == 'cm':
        # CM cost follows 3-parameter gamma distribution (higher than PM)
        shape = cost_params.cm_shape
        scale = cost_params.cm_scale
        
        # Compute location parameter from covariates (linear effect)
        location = 0.0
        if cost_covariates is not None and cost_covariate_effects is not None:
            if 'cm_location' in cost_covariate_effects:
                beta = cost_covariate_effects['cm_location']
                location = np.dot(beta, cost_covariates)
        
        # Generate from 3-parameter gamma: shape, scale, location
        cost = gamma_dist.rvs(a=shape, scale=scale, loc=location)
        return cost
    
    elif maintenance_type == 'imperfect_pm':
        # Imperfect PM: C_IPM = c_fix + c_0*u + ε
        if repair_effectiveness is None:
            raise ValueError("repair_effectiveness must be provided for imperfect_pm")
        
        # Calculate c_fix = exp(γ'W)
        c_fix = 1.0  # default if no covariates
        if cost_covariates is not None and cost_params.gamma_coeffs is not None:
            if len(cost_params.gamma_coeffs) != len(cost_covariates):
                raise ValueError("Length of gamma_coeffs must match length of cost_covariates")
            c_fix = np.exp(np.dot(cost_params.gamma_coeffs, cost_covariates))
        
        # Calculate total cost
        c_effectiveness = cost_params.c_0 * repair_effectiveness
        epsilon = np.random.normal(0, cost_params.epsilon_std)
        
        cost = c_fix + c_effectiveness + epsilon
        
        # Ensure cost is non-negative
        cost = max(0.0, cost)
        
        return cost
    
    else:
        raise ValueError(f"Unknown maintenance type: {maintenance_type}")
