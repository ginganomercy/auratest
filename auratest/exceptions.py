class AuraTestError(AssertionError):
    """
    Base exception for AuraTest failures. 
    Inherits from AssertionError so it automatically fails test runners like Pytest.
    """
    pass

class MonotonicityError(AuraTestError):
    """Raised when a model violates monotonic constraints."""
    pass

class InvarianceError(AuraTestError):
    """Raised when a model output changes unexpectedly (violates fairness/invariance)."""
    pass

class RobustnessError(AuraTestError):
    """Raised when a model is overly sensitive to noise."""
    pass
