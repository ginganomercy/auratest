import pytest
import numpy as np
from auratest import assert_monotonic, assert_invariance, assert_robustness
from auratest.exceptions import MonotonicityError, InvarianceError, RobustnessError

# --- DUMMY MODELS ---

def good_credit_model(X: np.ndarray) -> np.ndarray:
    """
    A logical model. 
    Feature 0: Income (Positive weight)
    Feature 1: Age (Positive weight)
    Feature 2: Gender (Zero weight - Fair model)
    """
    weights = np.array([2.0, 1.5, 0.0])
    return np.dot(X, weights)

def bad_credit_model(X: np.ndarray) -> np.ndarray:
    """
    A flawed model.
    Feature 0: Income (Negative weight! Violates monotonicity)
    Feature 1: Age (Positive weight)
    Feature 2: Gender (High weight! Violates invariance/fairness)
    """
    weights = np.array([-1.5, 1.5, 5.0])
    return np.dot(X, weights)

def fragile_model(X: np.ndarray) -> np.ndarray:
    """
    A fragile model that swings wildly with small noise.
    """
    # Exaggerates any noise by 1000x
    return np.sum(X * 1000, axis=1)

# --- TESTS ---

def test_monotonicity_passes_on_good_model():
    data = np.random.rand(10, 3)
    # Income (Feature 0) should strictly increase output
    assert_monotonic(good_credit_model, data, feature_index=0, direction="increasing")

def test_monotonicity_fails_on_bad_model():
    data = np.random.rand(10, 3)
    # Income (Feature 0) has a negative weight in bad_model, so increasing it will decrease output!
    with pytest.raises(MonotonicityError):
        assert_monotonic(bad_credit_model, data, feature_index=0, direction="increasing")

def test_invariance_passes_on_fair_model():
    data = np.random.rand(10, 3)
    # Gender (Feature 2) has zero weight, so changing it should not affect output
    assert_invariance(good_credit_model, data, feature_index=2, tolerance=0.01)

def test_invariance_fails_on_biased_model():
    data = np.random.rand(10, 3)
    # Gender (Feature 2) has a high weight, so randomizing it will drastically change the output
    with pytest.raises(InvarianceError):
        assert_invariance(bad_credit_model, data, feature_index=2, tolerance=0.01)

def test_robustness_passes_on_good_model():
    data = np.random.rand(10, 3)
    # Good model is linear with small weights, so small noise = small change
    assert_robustness(good_credit_model, data, noise_std=0.01, tolerance=0.5)

def test_robustness_fails_on_fragile_model():
    data = np.random.rand(10, 3)
    # Fragile model amplifies noise by 1000x, it will easily exceed tolerance
    with pytest.raises(RobustnessError):
        assert_robustness(fragile_model, data, noise_std=0.01, tolerance=0.5)
