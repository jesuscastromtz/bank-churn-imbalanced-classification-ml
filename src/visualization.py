import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score, roc_curve, confusion_matrix
import seaborn as sns


# =============================================================================
# Función para graficar importancia de features
# =============================================================================
def plot_feature_importance(models_dict, feature_names, top_n=15, figsize=(15, 10)):
    """
    Grafica la importancia de features para múltiples modelos
    
    Parameters:
    - models_dict: diccionario con nombre del modelo y el modelo entrenado
    - feature_names: lista con los nombres de las features
    - top_n: número de features más importantes a mostrar
    - figsize: tamaño de la figura
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    # Colores diferentes para cada modelo
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (model_name, model) in enumerate(models_dict.items()):
        ax = axes[idx]
        
        # Obtener importancia de features según el tipo de modelo
        if hasattr(model, 'feature_importances_'):
            # Para modelos con feature_importances_ (Random Forest, Gradient Boosting, XGBoost)
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Para modelos con coeficientes (Logistic Regression)
            importances = np.abs(model.coef_[0])
        else:
            print(f"Modelo {model_name} no tiene atributo de importancia de features")
            continue
        
        # Crear DataFrame con importancias
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Ordenar y tomar las top_n features
        feature_imp_df = feature_imp_df.sort_values('importance', ascending=False).head(top_n)
        
        # Crear gráfico de barras
        bars = ax.barh(range(len(feature_imp_df)), 
                        feature_imp_df['importance'], 
                        color=colors[idx % len(colors)],
                        alpha=0.7)
        
        ax.set_yticks(range(len(feature_imp_df)))
        ax.set_yticklabels(feature_imp_df['feature'])
        ax.set_xlabel('Importancia')
        ax.set_title(f'Importancia de Features - {model_name}', fontsize=12, fontweight='bold')
        
        # Añadir valores en las barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        ax.grid(axis='x', alpha=0.3)
    
    # Eliminar ejes vacíos si hay menos de 4 modelos
    for idx in range(n_models, 4):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Función para gráfico comparativo de todas las importancias
# =============================================================================
def plot_comparative_feature_importance(models_dict, feature_names, top_n=10, figsize=(12, 8)):
    """
    Gráfico comparativo de las importancias de features entre todos los modelos
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colores para cada modelo
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Obtener las top_n features más importantes en promedio
    all_importances = []
    
    for model_name, model in models_dict.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            continue
        
        all_importances.append(importances)
    
    if all_importances:
        # Calcular importancia promedio
        avg_importance = np.mean(all_importances, axis=0)
        
        # Crear DataFrame con importancias promedio
        avg_imp_df = pd.DataFrame({
            'feature': feature_names,
            'avg_importance': avg_importance
        })
        
        # Tomar las top_n features más importantes en promedio
        top_features = avg_imp_df.nlargest(top_n, 'avg_importance')['feature'].tolist()
        
        # Graficar importancias para cada modelo de las top features
        bar_width = 0.2
        x_pos = np.arange(len(top_features))
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                continue
            
            # Obtener importancias para las top features
            model_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            model_imp_df = model_imp_df[model_imp_df['feature'].isin(top_features)]
            model_imp_df = model_imp_df.set_index('feature').reindex(top_features)
            
            ax.bar(x_pos + idx * bar_width, model_imp_df['importance'].values, 
                    bar_width, label=model_name, color=colors[idx], alpha=0.8)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Importancia')
        ax.set_title('Comparación de Importancia de Features entre Modelos', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + bar_width * (len(models_dict) - 1) / 2)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# Función para gráfico comparativo de todas las importancias
# =============================================================================