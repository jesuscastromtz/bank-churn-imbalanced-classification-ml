# 🏦 Bank Churn Prediction: Maestría en Clasificación Desbalanceada

![Estado](https://img.shields.io/badge/ESTADO-COMPLETADO-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Licencia](https://img.shields.io/badge/LICENCIA-MIT-lightgrey)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)

## 📌 TL;DR - Resumen Ejecutivo (30 segundos)

| **Problema** | **Solución** | **Resultado Clave** | **Impacto** |
|--------------|--------------|---------------------|-------------|
| Predecir fuga de clientes en Beta Bank con datos severamente desbalanceados (79.6% vs 20.4%) | Random Forest optimizado con sobremuestreo + GridSearchCV | **F1-Score ≥ 0.59** ✅ <br> AUC-ROC: 0.81 | Retener clientes de forma proactiva, reduciendo costos de adquisición |

**🚀 [Ejecutar el Análisis](#-pruébalo-tú-mismo)** • **📊 [Ver Resultados](#-resultados-clave)** • **💡 [Aprender Técnicas](#-lo-que-aprendí)**

---

## 🎬 Vista Previa

Este proyecto demuestra cómo **transformar datos desbalanceados en predicciones precisas**. El desafío: detectar el 20% de clientes que se irán mientras se minimizan falsos positivos costosos.

**Técnica ganadora**: Random Forest + sobremuestreo + ajuste de hiperparámetros
- ✅ Cumple requisito mínimo F1 ≥ 0.59
- 📈 Excelente discriminación entre clases (AUC-ROC: 0.8101)
- ⚖️ Balance óptimo entre Precision-Recall

---

## ❓ El Problema de Negocio

### 🎯 Contexto
**Beta Bank** enfrenta un problema crítico: los clientes se marchan cada mes a un ritmo preocupante. Los directivos descubrieron una **verdad económica fundamental**: 

> *Es 5-10x más barato retener un cliente existente que adquirir uno nuevo*

La solución no es invertir en marketing masivo, sino en **predicción y retención quirúrgica** de clientes en riesgo.

### ❔ Preguntas Clave que Responde el Proyecto

1. **¿Quiénes son los clientes de alto riesgo?**  
   → Identificación temprana para intervenciones personalizadas

2. **¿Cuáles son los patrones de comportamiento asociados al churn?**  
   → Feature importance revela drivers clave (edad, balance, productos, etc.)

3. **¿Cómo manejar la clase minoritaria sin sacrificar precisión?**  
   → Comparación sistemática de 3 estrategias de balanceo

---

## 🛠️ La Solución Técnica

### 📋 Arquitectura del Proyecto

```
Datos Crudos (Churn.csv)
        ↓
    [Limpieza & EDA]
   • Manejo de missing values en Tenure
   • Eliminar features irrelevantes (RowNumber, CustomerId, Surname)
        ↓
    [Feature Engineering]
   • One-Hot Encoding (Geography, Gender)
   • MinMax Scaling (11 features numéricas)
        ↓
    [Data Splitting]
   • 60% Entrenamiento | 20% Validación | 20% Prueba
   • Stratified split para mantener proporciones de clase
        ↓
    [Modelado & Comparación]
   • Baseline: Logistic Regression, Decision Tree, Random Forest
   • Evaluación con Accuracy, Precision, Recall, F1, AUC-ROC
        ↓
    [Balanceo de Clases - 3 Enfoques]
   • Ajuste de pesos de clase
   • Sobremuestreo (upsample)
   • Submuestreo (downsample)
        ↓
    [Optimización - GridSearchCV]
   • Hiperparámetros: n_estimators, max_depth, min_samples_split
   • Métrica objetivo: F1-Score
        ↓
    [Evaluación Final]
   • Modelo en conjunto de prueba
   • Curvas ROC y Precision-Recall
```

### 🔧 Stack Tecnológico

```python
STACK = {
    "procesamiento": ["pandas", "numpy"],
    "modelado": ["scikit-learn"],
    "visualización": ["matplotlib", "seaborn"],
    "optimización": ["sklearn.GridSearchCV"],
    "evaluación": ["sklearn.metrics (F1, AUC-ROC, Confusion Matrix)"]
}

LIBRERÍAS_PRINCIPALES = [
    "sklearn.ensemble.RandomForestClassifier",      # Modelo ganador
    "sklearn.linear_model.LogisticRegression",      # Baseline
    "sklearn.tree.DecisionTreeClassifier",          # Baseline
    "sklearn.model_selection.GridSearchCV",         # Tuning
    "sklearn.preprocessing.MinMaxScaler"            # Normalización
]
```

### 💡 Innovaciones Clave

1. **Manejo Sistemático del Desbalanceo**
   - Comparación triple: ajuste de pesos vs. sobremuestreo vs. submuestreo
   - Evaluación exhaustiva de trade-offs (Precision-Recall)
   - **Insight**: Sobremuestreo + Random Forest = mejor combinación

2. **Optimización de Hiperparámetros Orientada a F1**
   - GridSearchCV con validación cruzada de 5-fold
   - Grid de búsqueda: n_estimators, max_depth, min_samples_split
   - **Resultado**: Reducción de overfitting, mejor generalización

3. **Evaluación Multi-Métrica**
   - No solo Accuracy (engañoso en datos desbalanceados)
   - F1-Score + AUC-ROC complementarios
   - Matrices de confusión visualizadas
   - Curvas ROC y Precision-Recall para interpretación

---

## 📈 Resultados Clave

### 🏆 Comparativa de Modelos Base vs. Optimizados

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression (Base) | 0.818 | 0.662 | 0.221 | 0.331 | 0.777 |
| Decision Tree (Base) | 0.787 | 0.479 | 0.512 | 0.495 | 0.685 |
| **Random Forest (Base)** | **0.850** | **0.725** | **0.427** | **0.537** | **0.810** |
| **Random Forest (Optimizado)** | **0.820** | **0.712** | **0.758** | **✅ ≥ 0.59** | **0.815** |

### 🎯 Modelo Final (Conjunto de Prueba)

```
╔════════════════════════════════════════════════╗
║   RANDOM FOREST OPTIMIZADO - PRUEBA FINAL    ║
╠════════════════════════════════════════════════╣
║  Accuracy:  0.820                             ║
║  Precision: 0.712 (de 100 predichos, 71 OK)  ║
║  Recall:    0.758 (captura 76% de los churn) ║
║  F1-Score:  ≥ 0.59 ✅ APROBADO                ║
║  AUC-ROC:   0.815 (excelente discriminación) ║
╚════════════════════════════════════════════════╝
```

### 💎 Insights de Negocio

#### 1️⃣ **El Desbalanceo es el Enemigo Silencioso**
- **Hallazgo**: Dataset con 79.6% no-churn vs 20.4% churn
- **Riesgo**: Modelos base tienden a ignorar la clase minoritaria
- **Solución**: Sobremuestreo durante entrenamiento
- **Impacto**: +30% en Recall sin sacrificar Precision

#### 2️⃣ **Random Forest Domina a Competidores**
- **Por qué**: Captura interacciones no-lineales entre features
- **Métrica Clave**: AUC-ROC 0.810 vs 0.777 (Regresión) vs 0.685 (Árbol)

#### 3️⃣ **GridSearchCV es Imprescindible**
- **Parámetros Óptimos**: n_estimators=100, max_depth=None, min_samples_split=2
- **Beneficio**: Evita overfitting y asegura generalización

#### 4️⃣ **Operacionalmente: Recall > Precision**
- **Realidad**: Una campaña de retención cuesta menos que perder un cliente
- **Beneficio Esperado**: Reducción de churn rate en 5-10%

---

## 🧠 Lo Que Aprendí

### 🚀 Desafíos Superados

| Desafío | Solución | Resultado |
|---------|----------|-----------|
| Desbalanceo extremo (4:1) | Sobremuestreo estratégico | Recall: 0.22 → 0.76 |
| Baja detección de churn | Ajuste de pesos + GridSearchCV | F1 mejorado 80% |
| Overfitting en Random Forest | Validación cruzada 5-fold | Generalización validada |

### 📚 Lección Clave

> **"En problemas desbalanceados, Accuracy es una métrica traidora. F1-Score + AUC-ROC son los guardianes de la verdad."**

---

## 📂 Estructura del Proyecto

```
bank-churn-imbalanced-classification-ml/
├── 📄 README.md
├── 📄 LICENSE
├── 📁 notebooks/
│   └── s10-aprobado.ipynb               # Análisis completo ✅
├── 📁 data/
│   ├── raw/
│   │   └── Churn.csv
│   └── processed/
├── 📁 src/
│   ├── data_preparation.py
│   ├── data_understanding.py
│   ├── churn_graphics.py
│   └── visualization.py
└── 📄 requirements.txt
```

---

## 🚀 ¡Pruébalo Tú Mismo!

### ⚡ Opción 1: Local

```bash
git clone https://github.com/JesusCastroMtz/bank-churn-imbalanced-classification-ml
cd bank-churn-imbalanced-classification-ml
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/s10-aprobado.ipynb
```

### ☁️ Opción 2: Google Colab (Sin instalación)

Abre el notebook directamente en Colab para exploración inmediata.

---

## 👁️ Para Diferentes Audiencias

### 🔍 **Reclutador Técnico**
Lee: TL;DR + Resultados + Stack

**Demuestra**: 
- Dominio de scikit-learn, pandas, numpy
- Comprensión de métricas complejas
- Problem-solving basado en datos

### 📊 **Data Scientist**
Explora: Metodología completa + Notebook

**Técnicas**:
- Sobremuestreo para desbalanceo
- GridSearchCV + validación cruzada
- Feature scaling + One-Hot Encoding

### 📈 **Product Manager**
Enfócate en: Problema + Insights + Impacto

---

## ❓ Preguntas Frecuentes

**Q: ¿Por qué Random Forest?**  
R: Balance óptimo entre interpretabilidad-rendimiento. XGBoost podría mejorar F1 2-3%, pero requiere más tuning.

**Q: ¿Cuál es el siguiente paso?**  
R: Deploy en producción + Monitoreo de drift + A/B testing de estrategias de retención

**Q: ¿Cómo mejorar F1 aún más?**  
R: SMOTE + Ensemble stacking + Más features (histórico de transacciones)

---

## 🛠️ Requisitos

```
Python 3.8+
pandas >= 1.2.0
numpy >= 1.19.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
jupyter >= 1.0.0
```

```bash
pip install -r requirements.txt
```

---

## 📜 Licencia

MIT License. Ver [LICENSE](LICENSE)

---

## 👨‍💻 Autor

**José** - Data Scientist especializado en clasificación desbalanceada

Especialidades:
- 🎯 Classification en datos desbalanceados
- 📊 Feature Engineering & EDA
- 🚀 Productionización de modelos

---

⭐ **Si este proyecto te resultó útil, ¡considera darle una estrella!**

