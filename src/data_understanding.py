import pandas as pd
import numpy as np

# =============================================================================
# Funcion para usar display/print
# =============================================================================
def show_dataframe(df, name=""):
    """
    Función inteligente que usa display en notebooks y print en otros entornos
    """
    try:
        # Intenta usar display (funciona en Jupyter)
        from IPython.display import display
        if name:
            print(f"\n{name}:")
        display(df)
        return True  # Indica que se usó display
    except ImportError:
        # Fallback a print si no está en Jupyter
        if name:
            print(f"\n{name}:")
        print(df.to_string())
        return False  # Indica que se usó print

# =============================================================================
# Funcion para cargar los datos
# =============================================================================
def load_data(data_path, name_dataset=""):
    """
    Función para cargar los datos
    """
    df = pd.read_csv(data_path)
    
    print(f" DATOS CARGADOS - {name_dataset.upper()}")
    print("=" * 75)
    
    print("\n DIMENSIONES Y ESTRUCTURA:")
    print("-" * 50)
    print(f"  - Filas: {df.shape[0]:,}")
    print(f"  - Columnas: {df.shape[1]}")
    print(f"  - Total de elementos: {df.size:,}")
    
    return df

# Ejemplo de uso:
# df = load_data('ruta', 'nombre')

# =============================================================================
# Funcion para análisis exploratorio
# =============================================================================
def explore_data(df, name_dataset="", show_samples=False, random_state=42):
    """
    Realiza el análisis exploratorio a un DataFrame.
    """
    print(f" ANÁLISIS EXPLORATORIO - {name_dataset.upper()}")
    print("=" * 75)
    
    print("\n PRIMERAS FILAS:")
    print("-" * 50)
    show_dataframe(df.head())
    
    print("\n INFORMACIÓN GENERAL:")
    print("-" * 50)
    show_dataframe(df.info())
    
    # Opción para mostrar dataframe
    if show_samples:    
        print("\n MUESTRA DE DATOS:")
        print("-" * 50)
        
        print("\n  - Últimas 3 filas:")
        show_dataframe(df.tail(3))
        
        print("\n  - Muestra aleatoria de 5 filas:")
        show_dataframe(df.sample(min(5, len(df)), random_state=random_state))

# Ejemplo de uso:
# explore_data(df, name_dataset="Dataset")
# explore_data(df, name_dataset="Dataset", show_samples=True)

# =============================================================================
# Funcion para análisis descriptivo
# =============================================================================
def describe_data(df, name_dataset="", max_unique_values=10):
    """
    Función personalizada para análisis exploratorio de datos
    que incluye análisis de valores únicos para variables categóricas
    
    Parámetros:
    - df: DataFrame a analizar
    - mostrar_dataframe_func: función para mostrar DataFrames
    - max_valores_unicos: máximo número de valores únicos a mostrar
    """
    
    print(f"ANÁLISIS DESCRIPTIVO - {name_dataset.upper()}")
    print("=" * 75)
    
    # Variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nVARIABLES NUMÉRICAS ({len(numeric_cols)}):")
        print("-" * 50)
        numeric_summary = df.describe()
        show_dataframe(numeric_summary)
        
    else:
        print("\nNo hay variables numéricas en el dataset")
    
    # Variables categóricas
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        print(f"\nVARIABLES CATEGÓRICAS ({len(categorical_cols)}):")
        print("-" * 50)
        categorical_summary = df.describe(include=["object", "category"])
        show_dataframe(categorical_summary) 
        
        # Análisis adicional: distribución de categorías para variables con pocos valores únicos
        print(f"\nDISTRIBUCIÓN DE CATEGORÍAS (con ≤ {max_unique_values} valores únicos):")
        print("-" * 50)
        
        for col in categorical_cols:
            unique_values = df[col].nunique()
            if unique_values <= max_unique_values and unique_values > 0:
                print(f"\n- Variable: {col}")
                distribution = df[col].value_counts(dropna=False)
                df_distribution = pd.DataFrame({
                    'Categoría': distribution.index,
                    'Frecuencia': distribution.values,
                    'Porcentaje': (distribution.values / len(df) * 100).round(2)
                })
                show_dataframe(df_distribution)
                
    else:
        print("\nNo hay variables categóricas en el dataset")

# Ejemplo de uso:
# describe_data(df, name_dataset="Dataset", max_unique_values=10)

# =============================================================================
# Funcion para análisis de calidad
# =============================================================================
def qualify_data(df, name_dataset=""):
    """
    Realiza el análisis de calidad a un DataFrame.
    """
    print(f" ANÁLISIS DE CALIDAD - {name_dataset.upper()}")
    print("-" * 75)
    
    # Análisis de valores
    print("\n ANÁLISIS DE VALORES NULOS")
    print("-" * 50)
    null_analysis = df.isnull().sum()
    null_percentage = (df.isnull().sum() / len(df)) * 100
    
    null_summary = pd.DataFrame({
        'Valores_Nulos': null_analysis,
        'Porcentaje_Nulos': null_percentage.round(2)
    })
    
    # Mostrar solo columnas con valores nulos
    columnas_con_nulos = null_summary[null_summary['Valores_Nulos'] > 0]
    if len(columnas_con_nulos) > 0:
        print(f"  - Columnas con valores nulos ({len(columnas_con_nulos)}):")
        show_dataframe(columnas_con_nulos)
    else:
        print("  - No hay valores nulos en el dataset")
    
    # Análisis de duplicados
    print("\n ANÁLISIS DE DUPLICADOS")
    print("-" * 50)
    duplicates = df.duplicated().sum()
    
    if duplicates > 0:
        print(f"  - Filas duplicadas: {duplicates}")
        print(f"  - Porcentaje de duplicados: {(duplicates/len(df)*100):.2f}%")
        print("  - Se recomienda revisar y eliminar duplicados")
    else:
        print("  - No hay filas duplicadas")
    
    # Calificación de calidad de datos
    calidad = calculate_data_quality(df)
    print(f"\n  - Calificación de datos: {calidad}/10")

# Ejemplo de uso:
# qualify_data(df, name_dataset="Dataset")

# =============================================================================
# Funcion para calcular la calidad de datos
# =============================================================================
def calculate_data_quality(df):
    """
    Calcula una puntuación de calidad de datos (0-10)
    """
    puntuacion = 10
    
    # Penalizar por valores nulos
    nulos_porcentaje = (df.isnull().sum().sum() / df.size) * 100
    if nulos_porcentaje > 20:
        puntuacion -= 4
    elif nulos_porcentaje > 10:
        puntuacion -= 2
    elif nulos_porcentaje > 5:
        puntuacion -= 1
    
    # Penalizar por duplicados
    duplicados_porcentaje = (df.duplicated().sum() / len(df)) * 100
    if duplicados_porcentaje > 10:
        puntuacion -= 3
    elif duplicados_porcentaje > 5:
        puntuacion -= 2
    elif duplicados_porcentaje > 0:
        puntuacion -= 1
    
    # Penalizar si no hay variables numéricas
    if len(df.select_dtypes(include=[np.number]).columns) == 0:
        puntuacion -= 1
    
    return max(0, puntuacion)

print("FUNCIONES DE EXPLORACIÓN: LISTAS")