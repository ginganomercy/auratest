import numpy as np
from typing import Callable, Any

from .engine import generate_incremental_perturbations, perturb_feature_random, add_global_noise
from .exceptions import MonotonicityError, InvarianceError, RobustnessError

def _safe_predict(predict_fn: Callable, data: np.ndarray) -> np.ndarray:
    """
    Safely calls the predict function and ensures the output is a NumPy array.
    This prevents crashes when models return PyTorch Tensors or Pandas Series.
    """
    preds = predict_fn(data)
    return np.asarray(preds)

def assert_monotonic(
    predict_fn: Callable[[np.ndarray], Any], 
    data: np.ndarray, 
    feature_index: int, 
    direction: str = "increasing",
    steps: int = 5,
    step_size: float = 1.0,
    strict: bool = False
) -> None:
    """
    Asserts that the model's output strictly follows a monotonic direction.
    """
    if direction not in ["increasing", "decreasing"]:
        raise ValueError("direction must be either 'increasing' or 'decreasing'")
        
    generator = generate_incremental_perturbations(
        data, feature_index, steps=steps, step_size=step_size
    )
    
    prev_predictions = None
    
    for step, perturbed_data in enumerate(generator):
        current_predictions = _safe_predict(predict_fn, perturbed_data)
        
        if prev_predictions is not None:
            diff = current_predictions - prev_predictions
            
            if direction == "increasing":
                # If strict, diff must be > 0. Else, diff must be >= 0.
                if strict:
                    violations = np.where(diff <= 0)[0]
                else:
                    violations = np.where(diff < 0)[0]
                    
                if len(violations) > 0:
                    raise MonotonicityError(f"Monotonicity violated! Expected output to strictly increase, but it failed for sample index {violations[0]}.")
            else:
                if strict:
                    violations = np.where(diff >= 0)[0]
                else:
                    violations = np.where(diff > 0)[0]
                    
                if len(violations) > 0:
                    raise MonotonicityError(f"Monotonicity violated! Expected output to strictly decrease, but it failed for sample index {violations[0]}.")
                    
        prev_predictions = current_predictions

def assert_invariance(
    predict_fn: Callable[[np.ndarray], Any], 
    data: np.ndarray, 
    feature_index: int, 
    tolerance: float = 1e-5,
    noise_std: float = 1.0
) -> None:
    """
    Asserts that altering a specific feature does NOT change the prediction beyond tolerance.
    """
    original_predictions = _safe_predict(predict_fn, data)
    
    perturbed_data = perturb_feature_random(data, feature_index, noise_std=noise_std)
    perturbed_predictions = _safe_predict(predict_fn, perturbed_data)
    
    diff = np.abs(original_predictions - perturbed_predictions)
    violations = np.where(diff > tolerance)[0]
    
    if len(violations) > 0:
        max_diff = np.max(diff[violations])
        raise InvarianceError(
            f"Invariance violated! Changing feature {feature_index} caused the output to change by {max_diff:.6f}, exceeding tolerance {tolerance}."
        )

def assert_robustness(
    predict_fn: Callable[[np.ndarray], Any], 
    data: np.ndarray, 
    noise_std: float = 0.01, 
    tolerance: float = 0.1
) -> None:
    """
    Asserts that adding small Gaussian noise doesn't drastically swing the output.
    """
    original_predictions = _safe_predict(predict_fn, data)
    
    noisy_data = add_global_noise(data, noise_std=noise_std)
    noisy_predictions = _safe_predict(predict_fn, noisy_data)
    
    diff = np.abs(original_predictions - noisy_predictions)
    violations = np.where(diff > tolerance)[0]
    
    if len(violations) > 0:
        max_diff = np.max(diff[violations])
        raise RobustnessError(
            f"Robustness violated! Small input noise ({noise_std}) caused output deviation of {max_diff:.6f}, exceeding tolerance {tolerance}."
        )
