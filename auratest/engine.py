import numpy as np
from typing import Iterator

def generate_incremental_perturbations(
    data: np.ndarray, 
    feature_index: int, 
    steps: int = 5, 
    step_size: float = 1.0
) -> Iterator[np.ndarray]:
    """
    Yields perturbed copies of the data incrementally.
    Uses a generator to prevent Out-Of-Memory (OOM) errors on massive datasets.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("AuraTest requires input data to be a NumPy array.")
        
    for step in range(steps):
        # Create a deep copy to avoid memory leakage/mutation
        modified_data = data.copy()
        # Increment the specific feature
        modified_data[:, feature_index] += (step * step_size)
        yield modified_data

def perturb_feature_random(
    data: np.ndarray, 
    feature_index: int, 
    noise_std: float = 1.0
) -> np.ndarray:
    """
    Generates a copy of the data where the target feature is randomized.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("AuraTest requires input data to be a NumPy array.")
        
    modified_data = data.copy()
    noise = np.random.normal(0, noise_std, size=modified_data.shape[0])
    modified_data[:, feature_index] += noise
    return modified_data

def add_global_noise(
    data: np.ndarray, 
    noise_std: float = 0.01
) -> np.ndarray:
    """
    Adds Gaussian noise to all features.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("AuraTest requires input data to be a NumPy array.")
        
    noise = np.random.normal(0, noise_std, size=data.shape)
    return data + noise
