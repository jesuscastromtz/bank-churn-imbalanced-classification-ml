"""
Módulo core de funciones para clasificación de churn bancario

Contiene funciones auxiliares para:
- Evaluación de modelos
- Balanceo de clases (upsampling, downsampling)

Autor: Jesús Castro Martínez
Versión: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def error_count(answers, predictions):
    """
    Cuenta el número de predicciones incorrectas.
    
    Parámetros:
    -----------
    answers : array-like
        Valores reales
    predictions : array-like
        Valores predichos
    
    Retorna:
    --------
    int : Cantidad de errores
    """
    return np.sum(np.array(answers) != np.array(predictions))


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Entrena un modelo y evalúa su rendimiento con múltiples métricas.
    
    Parámetros:
    -----------
    model : estimador sklearn
        Modelo de clasificación con métodos fit, predict, predict_proba
    X_train : DataFrame
        Features de entrenamiento
    y_train : Series
        Target de entrenamiento
    X_val : DataFrame
        Features de validación
    y_val : Series
        Target de validación
    
    Retorna:
    --------
    tuple : (matriz_confusión, accuracy, precision, recall, f1, auc_roc)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score, recall_score, roc_auc_score, precision_score,
        f1_score, confusion_matrix
    )
    
    # Entrenamiento y predicción
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Cálculo de métricas
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    matrix = confusion_matrix(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_proba)
    
    # Impresión de métricas
    print("Métricas de Evaluación")
    print("-------------------------------")
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad (Recall): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}\n")
    
    # Visualización de la matriz de confusión
    plt.figure(figsize=(4, 2))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['Real 0', 'Real 1'])
    plt.title('Matriz de Confusión', fontsize=14)
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.show()
    
    return matrix, accuracy, precision, recall, f1, roc_auc


def upsample(features, target, repeat):
    """
    Sobre-muestrea (sobremuestreo) la clase minoritaria.
    
    Parámetros:
    -----------
    features : DataFrame
        Features del dataset
    target : Series
        Variable objetivo
    repeat : int
        Número de veces a repetir la clase minoritaria (1)
    
    Retorna:
    --------
    tuple : (features_upsampled, target_upsampled)
        Dataset balanceado
    """
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled


def downsample(features, target, fraction):
    """
    Submuestrea (submuestreo) la clase mayoritaria.
    
    Parámetros:
    -----------
    features : DataFrame
        Features del dataset
    target : Series
        Variable objetivo
    fraction : float
        Fracción de la clase mayoritaria (0) a mantener (0.0-1.0)
    
    Retorna:
    --------
    tuple : (features_downsampled, target_downsampled)
        Dataset balanceado
    """
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled
