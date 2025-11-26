import numpy as np

def no_covariate_update(dynamic_covs, failure_type, valid_indices, machine_id):
    """Default covariate update function: does nothing."""
    return dynamic_covs, False
