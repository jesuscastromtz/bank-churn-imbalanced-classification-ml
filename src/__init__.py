"""
Core module for bank churn prediction

Exports only essential functions for class balancing and model evaluation.
"""

from .core import error_count, evaluate_model, upsample, downsample

__all__ = ['error_count', 'evaluate_model', 'upsample', 'downsample']