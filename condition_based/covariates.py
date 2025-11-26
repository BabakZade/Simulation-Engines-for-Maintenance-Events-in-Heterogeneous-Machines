"""
Covariate Management Module

This module provides classes for defining and managing covariates in degradation processes.
Covariates can be fixed, time-dependent, path-dependent, or discrete series.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass


@dataclass
class CovariateSpec:
    """
    Define covariate specifications.
    
    Attributes:
        name: Covariate name
        type: Type of covariate - 'fixed', 'time_dependent', 'path_dependent', 'discrete_series'
        time_function: Function of time t for time-dependent covariates
        path_function: Function of degradation level x for path-dependent covariates
        noise_std: Standard deviation of additive noise
        values: List of predefined values for discrete_series type
        probs: Probabilities for categorical variables
    """
    name: str
    type: str  # 'fixed', 'time_dependent', 'path_dependent', 'discrete_series'
    initial_value: Any = 0.0
    time_function: Optional[Callable] = None
    path_function: Optional[Callable] = None
    noise_std: float = 0.0
    values: Optional[List[float]] = None
    probs: Optional[List[float]] = None


class CovariateManager:
    """
    Manage covariates over time.
    This class maintains the state of all covariates and updates them according to their specifications.
    """
    
    def __init__(self, covariate_specs: List[CovariateSpec]):
        """
        Initialize covariate manager.
        
        Args:
            covariate_specs: List of covariate specifications
        """
        # Convert list to dictionary for easy lookup by covariate name
        self.specs = {spec.name: spec for spec in covariate_specs}
        self.current_values = {}
        # Dictionary to store historical values of each covariate
        self.histories = {name: [] for name in self.specs.keys()}
        self.current_step = 0

        # Initialize covariate values
        for name, spec in self.specs.items():
            if spec.type == 'fixed':
                # If initial_value is a list, randomly pick one value
                if isinstance(spec.initial_value, (list, np.ndarray)):
                    self.current_values[name] = np.random.choice(spec.initial_value)
                # If initial_value is a dictionary with "values" key, randomly pick with probabilities
                elif isinstance(spec.initial_value, dict) and "values" in spec.initial_value:
                    self.current_values[name] = np.random.choice(
                        spec.initial_value["values"], 
                        p=spec.initial_value.get("probs", None)
                    )
                # Use the initial_value directly
                else:
                    self.current_values[name] = spec.initial_value
            else:
                self.current_values[name] = spec.initial_value

    def update_covariates(self, t: float, x: float, dt: float = 0.01):
        """
        Update all covariates based on current time and degradation level.
        
        Args:
            t: Current time
            x: Current degradation level
            dt: Time step (not used currently, kept for compatibility)
        """
        for name, spec in self.specs.items():
            if spec.type == 'fixed':
                # Fixed covariates don't change
                pass
            
            elif spec.type == 'time_dependent' and spec.time_function:
                # Time-dependent: update based on time function
                new_value = spec.time_function(t)
                if spec.noise_std > 0:
                    new_value += np.random.normal(0, spec.noise_std)
                self.current_values[name] = new_value
            
            elif spec.type == 'path_dependent' and spec.path_function:
                # Path-dependent: update based on degradation level
                new_value = spec.path_function(x)
                if spec.noise_std > 0:
                    new_value += np.random.normal(0, spec.noise_std)
                self.current_values[name] = new_value
            
            elif spec.type == 'discrete_series' and spec.values is not None:
                # Discrete series: use predefined values based on step
                if self.current_step < len(spec.values):
                    self.current_values[name] = spec.values[self.current_step]
                else:
                    # If beyond series length, maintain last value
                    self.current_values[name] = spec.values[-1]

        # Record current values in history
        for name in self.specs.keys():
            self.histories[name].append(self.current_values[name])
        
        self.current_step += 1

    def get_covariate_vector(self) -> np.ndarray:
        """
        Get current covariate values as a numpy array.
        
        Returns:
            Array of current covariate values
        """
        return np.array(list(self.current_values.values()))

    def get_covariate_names(self) -> List[str]:
        """
        Get list of covariate names.
        
        Returns:
            List of covariate names in order
        """
        return list(self.specs.keys())
