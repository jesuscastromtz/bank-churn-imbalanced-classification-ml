import pandas as pd
import re

# =============================================================================
#  Funcion para renombrar columnas
# =============================================================================
def pascal_to_snake(name):
    # Inserta _ antes de las mayúsculas (excepto la primera) y convierte a minúsculas
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    # Maneja siglas como "ID", "USD", etc.
    name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1_\2', name)
    return name.lower()

def rename_columns_to_snake(df, excluded_columns=None, inplace=False):
    """
    Renombra las columnas de un DataFrame de PascalCase a snake_case
    
    Args:
        df (pd.DataFrame): DataFrame cuyas columnas se renombrarán
        excluded_columns (list): Lista de columnas a excluir del renombrado
        inplace (bool): Si modificar el DataFrame original
    
    Returns:
        pd.DataFrame: DataFrame con columnas renombradas
    """
    # Si excluded_columns es None, crear lista vacía
    excluded_columns = excluded_columns or []
    
    # Crear diccionario de mapeo
    column_mapping = {}
    for col in df.columns:
        if col in excluded_columns:
            # Mantener la columna original si está excluida
            column_mapping[col] = col
        else:
            # Aplicar la conversión pascal_to_snake
            column_mapping[col] = pascal_to_snake(col)
    
    # Aplicar el renombrado
    if inplace:
        df.rename(columns=column_mapping, inplace=True)
        return df
    else:
        return df.rename(columns=column_mapping)

# =============================================================================
#  Funcion para normalizacion
# =============================================================================
def to_snake_case(text):
    if pd.isna(text):
        return text
    # Reemplazar múltiples caracteres por underscores y eliminar paréntesis
    #text = text.replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '')
    # Convertir a minúsculas
    return text.lower()

# df['col'] = df['col'].apply(to_snake_case)
