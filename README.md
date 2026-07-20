<div align="center">
  <h1>AuraTest 🧪</h1>
  <p><strong>Test-Driven Development (TDD) framework for AI and Probabilistic Models.</strong></p>
  
  [![PyPI version](https://badge.fury.io/py/auratest.svg)](https://badge.fury.io/py/auratest)
  ![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
  ![License](https://img.shields.io/badge/License-MIT-green)
</div>

---

## The Problem: Fragile AI Deployments
Data Scientists rarely write Unit Tests because AI models are probabilistic. You cannot easily `assert output == 5`. Because of this, biased, illogical, and fragile models often leak into production undetected.

## The Solution: AuraTest
**AuraTest** allows you to test the *mathematical invariants* of your model rather than point outputs. It ensures your AI obeys the laws of physics, logic, and fairness before it is ever deployed.

---

## Installation

Install AuraTest easily via pip:

```bash
pip install auratest
```
*(Requires `numpy` to be installed in your environment).*

---

## Quick Start (pytest)

```python
import numpy as np
from auratest import assert_monotonic, assert_invariance

def test_credit_model_is_logical():
    # 1. Your trained model's predict function
    def predict_risk(X):
        return my_model.predict(X)
        
    sample_data = np.random.rand(10, 5) # 10 patients, 5 features
    
    # 2. Ensure that as Feature 2 (Age) increases, Risk strictly increases.
    assert_monotonic(predict_risk, sample_data, feature_index=2, direction="increasing")
    
def test_credit_model_is_fair():
    # 3. Ensure that altering Feature 0 (Gender) does NOT change predictions by > 1%
    assert_invariance(predict_risk, sample_data, feature_index=0, tolerance=0.01)
```

## How It Works (Production Features)
AuraTest acts as an independent testing engine wrapping your model's outputs. 

*   **Anti-OOM Generator Engine:** Instead of hoarding memory, the perturbation engine uses batch generators. You can test gigabytes of synthetic data without crashing your CI/CD runners.
*   **Safe Model Adapters:** Automatically intercepts and coerces arbitrary model outputs (whether it is a Pandas DataFrame, a PyTorch Tensor, or a Scikit-Learn Multiclass Probability array) into pure NumPy formats to ensure zero crashes during shape operations.
*   **Strict & Customizable Math Bounds:** Exposes deep parameters (`steps`, `step_size`, `noise_std`) and features a `strict` monotonicity mode to cater to rigorous compliance and audit requirements.

## Support This Project

AuraTest is an open-source project built out of passion. If it has saved you from deploying a biased or broken model into production, consider supporting the creator by following on Instagram!

[![Follow on Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/galaxy_scream)

---

## Contributing & Testing

We welcome PRs! To run the test suite locally and verify your changes:
```bash
# Clone the repository
git clone https://github.com/ginganomercy/auratest.git
cd auratest

# Install with development dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v
```
