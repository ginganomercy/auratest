<div align="center">
  <h1>AuraTest 🧪</h1>
  <p><strong>Test-Driven Development (TDD) framework for AI and Probabilistic Models.</strong></p>
  
  [![PyPI version](https://badge.fury.io/py/auratest.svg)](https://badge.fury.io/py/auratest)
  ![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
  ![License](https://img.shields.io/badge/License-MIT-green)
</div>

---

## 🎯 The Problem
Data Scientists rarely write Unit Tests because AI models are probabilistic. You cannot easily `assert output == 5`. Because of this, biased, illogical, and fragile models often leak into production undetected.

## 🛠️ The Solution: AuraTest
**AuraTest** allows you to test the *mathematical invariants* of your model rather than point outputs. It ensures your AI obeys the laws of physics, logic, and fairness before it is ever deployed.

### Installation
```bash
pip install auratest
```

### Quick Start (pytest)
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
