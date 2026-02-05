"""
Test básico de funciones core del proyecto
Verifica que las funciones de balanceo y evaluación funcionan correctamente.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src import upsample, downsample, error_count


def test_upsample():
    """Verifica que upsampling duplica la clase minoritaria correctamente"""
    # Dataset sintético desbalanceado
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                                weights=[0.8, 0.2], random_state=42)
    
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    
    # Test
    X_up, y_up = upsample(X_df, y_series, repeat=5)
    
    ratio_original = (y_series == 1).sum() / len(y_series)
    ratio_upsampled = (y_up == 1).sum() / len(y_up)
    
    # Esperado: ratio_upsampled ≈ 0.5 (más balanceado que original)
    assert ratio_upsampled > ratio_original, "Upsampling no balanceó correctamente"
    assert len(X_up) > len(X_df), "Upsampling no aumentó el dataset"
    
    print("✓ test_upsample PASÓ")


def test_downsample():
    """Verifica que downsampling reduce la clase mayoritaria"""
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                                weights=[0.8, 0.2], random_state=42)
    
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    
    # Test
    X_down, y_down = downsample(X_df, y_series, fraction=0.2)
    
    assert len(X_down) < len(X_df), "Downsampling no redujo el dataset"
    assert (y_down == 1).sum() > 0, "Se perdió la clase minoritaria"
    
    print("✓ test_downsample PASÓ")


def test_error_count():
    """Verifica que error_count cuenta incorrecciones"""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    errors = error_count(y_true, y_pred)
    
    assert errors == 1, f"Error: esperado 1 error, obtuvo {errors}"
    
    print("✓ test_error_count PASÓ")


if __name__ == "__main__":
    print("Ejecutando tests básicos...")
    test_error_count()
    test_upsample()
    test_downsample()
    print("\n✅ Todos los tests pasaron")
