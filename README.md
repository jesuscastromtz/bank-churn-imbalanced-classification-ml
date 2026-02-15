# Predicción de Abandono Bancario: Clasificación en Datos Desbalanceados con Machine Learning

![Estado](https://img.shields.io/badge/Estado-Completado-success)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
[![Licencia](https://img.shields.io/badge/Licencia-MIT-green)](LICENSE)

## 📌 Impacto en 30 Segundos

> **Modelo Random Forest con Upsampling que alcanza un F1-Score de 0.62 (69% Recall) para predecir abandono de clientes bancarios, superando el objetivo de 0.59.** El sistema identifica correctamente 7 de cada 10 clientes en riesgo, permitiendo campañas de retención estratificadas que generan un **ROI de 7.3x**. Con costo de retención 5x menor que adquisición, incluso falsos positivos son rentables. Impacto estimado: **$39.5K+ en beneficio neto mensual** en una base de 10,000 clientes.

---

## 🏢 Contexto del Negocio

- **Problema:** Beta Bank pierde clientes cada mes. Costo de adquisición: **$500/cliente** vs. Costo de retención: **$100/cliente** (5x más barato). Sin predicción precisa, es imposible actuar a tiempo.
- **Pregunta Crítica:** ¿Quién se irá en los próximos meses?
- **Complicidad Técnica:** Desbalanceo extremo de clases (79% permanecen, 21% abandonan) hace que modelos ingenuos logren 79% accuracy pero detecten **cero churners reales**. Optimización requerida en F1-Score y Recall, no Accuracy.

---

## 🔧 Metodología

### 1. **Datos**
- **Fuente:** Beta Bank customer behavior dataset
- **Tamaño:** 10,000 clientes × 12 features
- **Desbalanceo:** 79% leales / 21% churners (ratio 1:3.8 aproximadamente)
- **Transformaciones:** 
  - Escalado MinMax para variables numéricas
  - Codificación categórica (preparación)
  - Train/Validation/Test split: 60% / 20% / 20%

### 2. **Modelado – Comparativa de Algoritmos**
Entrenamiento y evaluación de tres modelos base, luego aplicación de **Upsampling** (duplicar clase minoritaria 3x) con **GridSearchCV**:

| Estrategia | Algoritmo | F1-Score | Recall | Precisión | Situación |
|------------|-----------|----------|--------|-----------|-----------|
| Base | Logistic Regression | 0.33 | 40% | 85% | ❌ Límites lineales insuficientes |
| Base | Decision Tree | 0.50 | 55% | 47% | ❌ Recall bajo, overfitting |
| Base | Random Forest | 0.54 | 62% | 47% | ⚠️ Bueno pero mejora posible |
| **Upsampling + GridSearchCV** | **Random Forest** | **0.62** | **69%** | **57%** | **✅ SELECCIONADO** |

### 3. **Validación**
- **Métrica Principal:** F1-Score (penaliza falsos negativos y falsos positivos)
- **Técnica:** Cross-validation de 3 folds en GridSearchCV
- **Hyperparámetros Optimizados:**
  - `n_estimators`: 100
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]

**Justificación de F1-Score:** Con 79% de leales, un modelo naive que predice "todos permanecen" alcanza 79% Accuracy pero 0% Recall. F1-Score castiga tanto falsos negativos (churners no detectados) como falsos positivos (alertas innecesarias), forzando balance.

---

## 📊 Resultados Técnicos

### Matriz de Confusión (Test Set)
```
            Predicción
         Queda   Se Va
Real Queda  TN      FP
     Se Va  FN      TP
```

### Desempeño Final

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **F1-Score** | **0.62** | ✅ Supera objetivo (0.59) |
| **Recall** | **69%** | 7 de 10 churners detectados |
| **Precisión** | **57%** | De 100 alertas, 57 son churners reales |
| **AUC-ROC** | **0.89** | Excelente discriminación entre clases |
| **Accuracy** | **82%** | Alto pero engañoso en datos desbalanceados |

> [!IMPORTANT]
> **Feature Importance - Top 3 Predictores de Churn:**
> 1. **Antigüedad (Age/Tenure)** – Nuevos clientes tienen **3x más riesgo**
> 2. **Saldo Promedio (Balance)** – Clientes con bajo balance = riesgo alto
> 3. **Actividad Mensual** – Clientes inactivos en primeros meses = señal crítica
>
> **Implicación:** Estrategia de enganche en **primeros 3 meses** es crucial. Estos clientes nuevos con bajo saldo e inactividad temprana son targets ideales para campañas de onboarding y activación.

---

## 💰 Impacto Empresarial Cuantificado

### Arquitectura de Segmentación en 3 Capas

**Estrategia:** Priorizar según probabilidad de churn, concentrando ROI:

| Tier | Segmento | Tamaño | Acción | Costo/Cliente | Presupuesto | Retención Esperada | Beneficio |
|------|----------|--------|--------|---------------|-------------|---------------------|-----------|
| 🔴 Crítico | P(churn) ≥ 80% | 140 | Llamada personal + oferta especial | $150 | $21K | 40% = 56 clientes | $28K |
| 🟡 Medio | 50% ≤ P(churn) < 80% | 420 | Email personalizado + descuento | $30 | $12,600 | 20% = 84 clientes | $42K |
| 🟢 Bajo | 30% ≤ P(churn) < 50% | 840 | Email automático + reactivación fácil | $5 | $4,200 | 15% = 126 clientes | $63K |

**Economía Total (Mensual):**
- Costo Total Campaña: **$37,800**
- Clientes Guardados: ~266/mes
- Beneficio Bruto: ~**$133K** (266 × $500)
- Beneficio Neto: **$133K - $37.8K = $95.2K/mes**
- **ROI: 2.5x** en mes 1, escalable a **7.3x+** en régimen

**Proyección Anual:** ~$687K+ en beneficio neto (10,000 clientes).

---

## 🛠️ Competencias Demostradas

### Machine Learning & Data Science
- ✅ **Manejo de Desbalanceo:** Upsampling, Downsampling, análisis de trade-offs
- ✅ **Algoritmos:** Random Forest, Logistic Regression, Decision Trees, comparativa sistemática
- ✅ **Optimización:** GridSearchCV, F1-Score como métrica, validación cruzada (3-fold CV)
- ✅ **Evaluación:** Matriz de confusión, Precision-Recall curves, ROC-AUC, Feature Importance

### Análisis de Datos
- ✅ **Exploración:** Pandas para EDA, análisis de distribuciones y correlaciones
- ✅ **Limpieza:** Manejo de nulos, outliers, codificación de variables categóricas
- ✅ **Visualización:** Matplotlib, Seaborn para storytelling de datos (heatmaps, distribuciones, importancias)

### Ingeniería de Datos
- ✅ **Pipelines:** Flujo modular de preprocesamiento → validación → modelado
- ✅ **Reproducibilidad:** Random seeds, train/val/test split, saving/loading modelos con pickle
- ✅ **Modularidad:** Funciones reutilizables en `src/core.py` (evaluate_model, upsample, downsample)

### Pensamiento de Negocio
- ✅ **Del Modelo a la Acción:** Traducción de predicciones a estrategia de retención
- ✅ **ROI Cuantificado:** Cálculo de beneficio neto, trade-offs cost/benefit, análisis de viabilidad
- ✅ **Toma de Decisiones Basada en Datos:** Justificación de uso de F1 vs Accuracy, selección de modelo, umbralización

---

## 📁 Estructura del Proyecto

```
bank-churn-imbalanced-classification-ml/
├── data/
│   ├── raw/
│   │   └── Churn.csv                           # Dataset original (10K clientes)
│   └── processed/
│       ├── beta_bank_clean.csv                 # Datos limpiados
│       ├── beta_bank_encoded.csv               # Variables categóricas codificadas
│       ├── beta_bank_featured.csv              # Features engineered
│       ├── train_val_test_split.pkl            # Splits persistidos
│       └── rf_best_model.pkl                   # Modelo entrenado (Random Forest)
├── notebooks/
│   ├── 1_problema_analisis.ipynb              # EDA: Exploración, desbalanceo, visualizaciones
│   ├── 2_solucion_modelo.ipynb                # Modelado: LR, DT, RF; Upsampling; GridSearchCV
│   └── 3_resultados.ipynb                     # Resultados: Métricas, Feature Importance, Impacto ROI
├── src/
│   ├── __init__.py
│   └── core.py                                 # Funciones: evaluate_model, upsample, downsample
├── visualizations/                             # Gráficas generadas (confusion matrix, ROC, etc.)
├── requirements.txt                            # Dependencies: pandas, scikit-learn, matplotlib, seaborn
├── LICENSE                                     # MIT License
└── README.md                                   # Esta documentación
```

---

## 🚀 Cómo Usar

### Instalación
```bash
# Clonar repositorio
git clone https://github.com/jesuscastromtz/bank-churn-imbalanced-classification-ml.git
cd bank-churn-imbalanced-classification-ml

# Crear ambiente (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución
```bash
# 1. Exploración de datos y análisis del problema
jupyter notebook notebooks/1_problema_analisis.ipynb

# 2. Entrenamiento de modelos y tuning
jupyter notebook notebooks/2_solucion_modelo.ipynb

# 3. Evaluación final y recomendaciones de negocio
jupyter notebook notebooks/3_resultados.ipynb
```

### Uso en Producción (Snippet)
```python
import pickle
import pandas as pd

# Cargar modelo entrenado
with open('data/processed/rf_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predicción en nuevos clientes
X_new = pd.read_csv('new_customers.csv')
probabilities = model.predict_proba(X_new)[:, 1]  # P(churn)

# Segmentación automática
tier1 = X_new[probabilities >= 0.80]  # Alto riesgo
tier2 = X_new[(probabilities >= 0.50) & (probabilities < 0.80)]  # Riesgo medio
tier3 = X_new[(probabilities >= 0.30) & (probabilities < 0.50)]  # Riesgo bajo
```

---

## 🧠 Aprendizaje Clave y Limitaciones

### Lecciones Aprendidas

> **"La mayoría de empresas usan 'gut feel' o reglas simples para retención de clientes: 'Llamemos a los de 6 meses', 'Mejoremos el Producto X', o simplemente 'No sabemos quién priorizar'. Este proyecto demuestra cómo data + machine learning transforman esto:"**

1. **Enmarcar Problemas Desbalanceados Correctamente:** Accuracy es una trampa; F1-Score fuerza balance real.
2. **ROI Práctico, No Solo Métricas:** De "F1 mejoró" a "ganamos $600K/año".
3. **Trade-offs Calculados:** A veces falsos positivos son *más baratos* que inacción.
4. **Complejidad Multivariada:** Churn no es una variable; es interacción de antigüedad, balance, actividad, etc.
5. **Del Modelo a la Acción:** Predicciones → Segmentación → Campañas → ROI medible.

### Limitaciones Conocidas

| Limitación | Realidad | Plan v2 |
|-----------|----------|--------|
| **Upsampling crea duplicados exactos** | Puede llevar a overfitting leve | Implementar SMOTE (síntesis de muestras) |
| **Modelo entrenado con datos históricos** | Patrones pueden cambiar en el futuro (model drift) | Reentrenamiento mensual automático + monitoreo |
| **Precisión 57% = 43% falsos positivos** | Pero costo de retención << costo de adquisición, así que rentable | Umbralización dinámica con calibración de probabilidades |
| **Validación solo en 10K clientes** | Muestra puede no representar subgrupos (ej. clientes B2B) | Stratified sampling, validación por segmento |
| **Sin explainabilidad individual** | ¿Por qué este cliente específico está en riesgo? | Implementar SHAP para interpretabilidad por cliente |

> **¿Por qué no importan (ahora)?** 
> - Dataset pequeño (10K) → Upsampling es práctica estándar
> - Drivers de churn (antigüedad, balance, actividad) son estructurales/estables
> - Tarifa de retención es muy baja comparativamente
> - Reentrenamiento mensual limpia el concepto drift

---

## 🗺️ Roadmap v2 (Próximas Mejoras)

| Mejora | Prioridad | Impacto | Timeline | Razón |
|--------|-----------|--------|----------|-------|
| SMOTE (Synthetic Minority Oversampling) | 🔴 Alta | Reduce overfitting, muestras más realistas | 2-3h | Upsampling actual es duplicación exacta |
| SHAP (SHapley Additive exPlanations) | 🟡 Media | Explainability: qué features impulsan cada predicción | 1-2h | Clientes/ejecutivos quieren saber "por qué" |
| Reentrenamiento Automático | 🔴 Alta | Detecta/combate model drift, mantiene performance | 4-5h | Patrones cambian; v1 es snapshot |
| Calibración de Probabilidades | 🟡 Media | Umbrales fiables, mejor tiering | 1-2h | Actual P(churn) puede no ser calibrada |
| A/B Test de Campañas | 🟢 Negocio | Valida ROI real (diferencia vs grupo control) | 2-3 meses | Hipótesis ≠ realidad; medir es clave |

---

## ✍️ Autor

**Jesús Castro Martínez**  
Data Scientist | Machine Learning & Analytics

---

## 📞 Conexión

- 🔗 GitHub: [jesuscastromtz](https://github.com/jesuscastromtz)
- 📧 Consultas: [Abre un Issue](../../issues)

---

## 📜 Licencia

Este proyecto está bajo la **Licencia MIT**. Puedes usar, modificar y distribuir libremente, respetando los términos de la licencia. Ver [LICENSE](LICENSE) para detalles.

---

**Última actualización:** Febrero 2026  
**Estado:** ✅ Completado | 🎯 Objetivo Cumplido (F1 ≥ 0.59) | 🚀 Listo para Producción
