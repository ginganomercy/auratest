import numpy as np
from typing import Callable

from .engine import perturb_feature_incremental, perturb_feature_random, add_global_noise
from .exceptions import MonotonicityError, InvarianceError, RobustnessError

def assert_monotonic(
    predict_fn: Callable[[np.ndarray], np.ndarray], 
    data: np.ndarray, 
    feature_index: int, 
    direction: str = "increasing"
) -> None:
    """
    Asserts that the model's output strictly follows a monotonic direction 
    (increasing or decreasing) when a specific feature is incrementally increased.
    """
    if direction not in ["increasing", "decreasing"]:
        raise ValueError("direction must be either 'increasing' or 'decreasing'")
        
    # Generate data: 5 incremental steps
    steps = 5
    n_samples = data.shape[0]
    perturbed_data = perturb_feature_incremental(data, feature_index, steps=steps, step_size=1.0)
    
    # Predict all at once (Vectorized for speed)
    predictions = predict_fn(perturbed_data)
    
    # Reshape predictions back to (steps, n_samples)
    predictions = predictions.reshape(steps, n_samples)
    
    # Check monotonicity across steps
    for step in range(1, steps):
        diff = predictions[step] - predictions[step - 1]
        
        if direction == "increasing":
            # Predictions must not go down
            violations = np.where(diff < 0)[0]
            if len(violations) > 0:
                raise MonotonicityError(f"Monotonicity violated! Expected output to increase or stay flat, but it decreased for sample index {violations[0]}.")
        else:
            # Predictions must not go up
            violations = np.where(diff > 0)[0]
            if len(violations) > 0:
                raise MonotonicityError(f"Monotonicity violated! Expected output to decrease or stay flat, but it increased for sample index {violations[0]}.")

def assert_invariance(
    predict_fn: Callable[[np.ndarray], np.ndarray], 
    data: np.ndarray, 
    feature_index: int, 
    tolerance: float = 1e-5
) -> None:
    """
    Asserts that altering a specific feature (e.g. protected attribute) 
    does NOT change the model's prediction beyond a given tolerance.
    """
    original_predictions = predict_fn(data)
    
    perturbed_data = perturb_feature_random(data, feature_index, noise_std=5.0)
    perturbed_predictions = predict_fn(perturbed_data)
    
    diff = np.abs(original_predictions - perturbed_predictions)
    violations = np.where(diff > tolerance)[0]
    
    if len(violations) > 0:
        max_diff = np.max(diff[violations])
        raise InvarianceError(
            f"Invariance violated! Changing feature {feature_index} caused the output to change by {max_diff:.6f}, which exceeds tolerance {tolerance}."
        )

def assert_robustness(
    predict_fn: Callable[[np.ndarray], np.ndarray], 
    data: np.ndarray, 
    noise_std: float = 0.01, 
    tolerance: float = 0.1
) -> None:
    """
    Asserts that adding small Gaussian noise to the input doesn't drastically swing the output.
    """
    original_predictions = predict_fn(data)
    
    noisy_data = add_global_noise(data, noise_std=noise_std)
    noisy_predictions = predict_fn(noisy_data)
    
    diff = np.abs(original_predictions - noisy_predictions)
    violations = np.where(diff > tolerance)[0]
    
    if len(violations) > 0:
        max_diff = np.max(diff[violations])
        raise RobustnessError(
            f"Robustness violated! Small input noise ({noise_std}) caused output deviation of {max_diff:.6f}, exceeding tolerance {tolerance}."
        )
