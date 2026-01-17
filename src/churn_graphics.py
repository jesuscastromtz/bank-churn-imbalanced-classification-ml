"""
Librería de análisis de churn (abandono de clientes)

Este módulo proporciona funciones para analizar patrones de abandono
en datasets de clientes, con visualizaciones automáticas y estadísticas.

Funciones principales:
- analysis_churn(): Análisis de churn para variables numéricas
- analysis_churn_category(): Análisis de churn con variable categórica

Autor: Jesús Castro Martínez
Versión: 1.0
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# =============================================================================
# FUNCIONES AUXILIARES GENERALES
# =============================================================================

def _format_bin_labels(bin_group, max_right_value, label_format_style='auto'):
    """
    Formatea las etiquetas de los bins según el estilo especificado
    
    Parámetros:
    -----------
    bin_group : Interval
        Intervalo del bin
    max_right_value : float
        Valor máximo del lado derecho de todos los bins
    label_format_style : str
        Estilo de formato ('auto' o 'simple')
    
    Retorna:
    --------
    str : Etiqueta formateada
    """
    if pd.isna(bin_group):
        return 'N/A'
    
    left, right = bin_group.left, bin_group.right
    
    if label_format_style == 'auto':
        # Lógica automática basada en magnitud
        if left >= 1000000:
            return f"{left/1000000:.1f}M-{right/1000000:.1f}M"
        elif left >= 1000:
            return f"{left/1000:.0f}K-{right/1000:.0f}K"
        else:
            # Formato simple para números pequeños
            left, right = int(left), int(right)
            return f"{left}+" if right == max_right_value else f"{left}-{right-1}"
    
    elif label_format_style == 'simple':
        # Formato original sin abreviaciones
        left, right = int(left), int(right)
        return f"{left}+" if right == max_right_value else f"{left}-{right-1}"
    
    return f"{left:.1f}-{right:.1f}"

def _create_bins_and_stats(df, numeric_column, category_column, target_column, 
                          bins_n, label_format_style):
    """
    Crea bins y calcula estadísticas para una o múltiples categorías
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame con datos
    numeric_column : str
        Columna numérica a analizar
    category_column : str or None
        Columna categórica (None si no hay)
    target_column : str
        Columna objetivo
    bins_n : int
        Número de bins
    label_format_style : str
        Estilo de formato de etiquetas
    
    Retorna:
    --------
    DataFrame : Estadísticas agrupadas
    """
    df_copy = df.copy()
    
    # Crear bins
    bins = pd.cut(df_copy[numeric_column], bins=bins_n, include_lowest=True)
    df_copy['bin_group'] = bins
    
    # Obtener el valor right más alto de todos los bins
    # Usar unique() y filtrar valores NaN
    unique_intervals = [interval for interval in df_copy['bin_group'].unique() if pd.notna(interval)]
    if not unique_intervals:
        max_right_value = 0
    else:
        max_right_value = max(interval.right for interval in unique_intervals)
    
    # Definir columnas para agrupar
    groupby_cols = ['bin_group']
    if category_column:
        groupby_cols.append(category_column)
    
    # Calcular estadísticas
    group_stats = df_copy.groupby(groupby_cols, observed=False).agg(
        total=(target_column, 'count'),
        churn_count=(target_column, 'sum')
    ).reset_index()
    
    group_stats['churn_rate'] = (group_stats['churn_count'] / group_stats['total']) * 100
    group_stats['bin_label'] = group_stats['bin_group'].apply(
        lambda x: _format_bin_labels(x, max_right_value, label_format_style)
    )
    
    return group_stats

# =============================================================================
# FUNCIÓN PRINCIPAL analysis_churn + SUS SUBFUNCIONES
# =============================================================================

def analysis_churn(df, target_column='exited', 
                   analysis_column='age', column_title_numeric=None,
                   bins_n=14, palette=None, label_format_style='auto'):
    """
    Análisis univariado completo de churn para variables numéricas
    
    Presenta un análisis univariado de una variable numérica mostrando su 
    distribución general, distribución con respecto a churn, boxplot comparativo
    y tasa de abandono por grupos.
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame con los datos
    analysis_column : str
        Columna de la variable numérica
    target_column : str
        Columna objetivo de abandono (default 'exited')
    bins_n : int
        Número de bins para histogramas (default 14)
    palette : dict
        Paleta de colores para abandono (default: '0'=verde, '1'=rojo)
    column_title_numeric : str
        Título personalizado para la columna
    label_format_style : str
        Estilo de formato de etiquetas ('auto' o 'simple')
    
    Retorna:
    --------
    tuple
        (fig, axes, group_stats) - Figura matplotlib, ejes y estadísticas por grupos
    
    Ejemplos:
    ---------
    >>> import pandas as pd
    >>> # Análisis básico de edad vs abandono
    >>> fig, axes, group_stats = analysis_churn(
    ...     df=df_eda, target_column='exited', 
    ...     analysis_column='age', column_title_numeric='Edad',
    ...     bins_n=14, palette={'0': 'green', '1': 'red'}
    ... )
    """
    # 1. Preparar título
    num_title = column_title_numeric or analysis_column.replace('_', ' ').title()
    
    # 2. Crear paleta
    if palette is None:
        palette = {'0': '#2E8B57', '1': '#DC143C'}
    else:
        palette = {str(k): v for k, v in palette.items()}
        
    # 3. Preparar DataFrame
    df_viz = df.copy()
    df_viz[target_column] = df_viz[target_column].astype(str)
    
    # 4. Calcular estadísticas por grupos usando función auxiliar
    group_stats = _create_bins_and_stats(
        df, analysis_column, None, target_column, bins_n, label_format_style
    )
    
    # 5. Crear figura con 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(20, 18))
    
    # Título principal
    fig.suptitle(f'Análisis de {num_title} vs Abandono', 
                fontsize=22, fontweight='bold', y=0.98)
    
    # 6. Llamar a las subfunciones
    _plot_distribution_general(
        axes[0], df_viz, analysis_column, num_title, bins_n
    )
    _plot_distribution_by_churn(
        axes[1], df_viz, analysis_column, target_column, num_title, bins_n, palette
    )
    _plot_boxplots_comparison(
        axes[2], df_viz, analysis_column, target_column, num_title, palette
    )
    _plot_churn_rate_by_groups(axes[3], group_stats, num_title, palette)
    
    # 7. Añadir líneas de referencia
    _add_churn_reference_lines(axes[3], df, target_column)
    
    # 8. Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, top=0.94)
    plt.show()
    
    # 9. Mostrar resumen estadístico
    _show_churn_summary(df, analysis_column, target_column, num_title)
    
    return fig, axes, group_stats

# SUB-FUNCIONES PARA analysis_churn
# =============================================================================

def _plot_distribution_general(ax, df, analysis_column, num_title, bins_n):
    """Distribución total de la variable numérica"""
    # Distribución total
    sns.histplot(
        data=df, x=analysis_column, stat='count', bins=bins_n, 
        kde=True, color='steelblue', alpha=0.7, ax=ax)
    
    ax.set_title(f'Distribución de {num_title} (Total Clientes)', 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(num_title, fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Añadir estadísticas en el gráfico
    mean_val = df[analysis_column].mean()
    median_val = df[analysis_column].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, alpha=0.8, 
               label=f'Media: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle=':', linewidth=1, alpha=0.8,
               label=f'Mediana: {median_val:.1f}')
    
    # Añadir texto con estadísticas
    stats_text = f'N: {len(df):,}\nMin: {df[analysis_column].min():.1f}\nMax: {df[analysis_column].max():.1f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(loc='upper right', fontsize=9)

def _plot_distribution_by_churn(ax, df, analysis_column, target_column, num_title, bins_n, palette):
    """Distribución por estado de abandono"""
    # Asegurar que palette tenga las claves correctas
    if not all(str(k) in palette for k in df[target_column].unique()):
        unique_values = sorted(df[target_column].unique())
        colors = sns.color_palette("Set2", len(unique_values))
        palette = dict(zip(unique_values, colors))
    
    # Histograma por estado de abandono
    sns.histplot(
        data=df, x=analysis_column, bins=bins_n, kde=True,
        hue=target_column, palette=palette, alpha=0.6, 
        multiple='layer', element='step', ax=ax)
    
    ax.set_title(f'Distribución de {num_title} por Estado de Abandono', 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(num_title, fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Personalizar leyenda
    handles, labels = ax.get_legend_handles_labels()
    nuevas_etiquetas = []
    for label in labels:
        if label == '0':
            count = (df[target_column] == '0').sum()
            nuevas_etiquetas.append(f'No Abandono (n={count})')
        elif label == '1':
            count = (df[target_column] == '1').sum()
            nuevas_etiquetas.append(f'Abandono (n={count})')
        else:
            nuevas_etiquetas.append(f'{label} (n={(df[target_column] == label).sum()})')
    
    ax.legend(handles=handles, labels=nuevas_etiquetas, title='Estado',
              fontsize=10, loc='upper right')

def _plot_boxplots_comparison(ax, df, analysis_column, target_column, num_title, palette):
    """Comparación boxplot por estado de abandono"""
    # Crear DataFrame para boxplot
    df_boxplot = df.copy()
    
    # Mapear valores a etiquetas más legibles
    mapeo_etiquetas = {'0': 'No Abandono', '1': 'Abandono'}
    df_boxplot['estado_abandono'] = df_boxplot[target_column].map(mapeo_etiquetas)
    
    # Crear paleta para boxplot
    palette_boxplot = {}
    for k, v in palette.items():
        if k in mapeo_etiquetas:
            palette_boxplot[mapeo_etiquetas[k]] = v
        else:
            palette_boxplot[k] = v
    
    # Boxplot
    sns.boxplot(data=df_boxplot, x=analysis_column, y='estado_abandono',
                palette=palette_boxplot, orient='h', ax=ax)
    
    ax.set_title(f'Comparación de {num_title} por Estado de Abandono', 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(num_title, fontsize=12)
    ax.set_ylabel('')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Añadir estadísticas en el gráfico
    for i, estado in enumerate(['No Abandono', 'Abandono']):
        if estado in df_boxplot['estado_abandono'].values:
            subset = df_boxplot[df_boxplot['estado_abandono'] == estado][analysis_column]
            median_val = subset.median()
            ax.axvline(median_val, ymin=i/len(palette_boxplot), ymax=(i+1)/len(palette_boxplot), 
                      color='red', linestyle='--', alpha=0.5, linewidth=0.8)

def _plot_churn_rate_by_groups(ax, group_stats, num_title, palette):
    """Crea gráfico de barras de tasa de abandono por grupos"""
    # Usar un color para todas las barras (primero de la paleta)
    bar_color = list(palette.values())[1] if '1' in palette else '#DC143C'
    
    sns.barplot(data=group_stats, x='bin_label', y='churn_rate',
                color=bar_color, alpha=0.8, ax=ax,
                order=sorted(group_stats['bin_label'].unique()))
    
    ax.set_title(f'Tasa de Abandono por Grupo de {num_title}', 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Tasa de Abandono (%)', fontsize=12)
    ax.set_xlabel(f'Rango de {num_title}', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en barras
    _add_bar_values(ax, group_stats, bar_color)
    
    # Remover leyenda automática si existe
    if ax.get_legend():
        ax.get_legend().remove()

def _add_bar_values(ax, group_stats, bar_color):
    """Añade valores de porcentaje sobre las barras"""
    for i, bin_label in enumerate(sorted(group_stats['bin_label'].unique())):
        subset = group_stats[group_stats['bin_label'] == bin_label]
        
        if not subset.empty and subset.iloc[0]['total'] >= 5:  # Solo si muestra suficiente
            row = subset.iloc[0]
            # Añadir porcentaje
            ax.text(i, row['churn_rate'] + 0.5, 
                    f"{row['churn_rate']:.1f}%",
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=bar_color)
            
            # Añadir tamaño de muestra si es pequeño
            if row['total'] < 100:
                ax.text(i, row['churn_rate']/2, 
                        f"n={row['total']}",
                        ha='center', va='center', fontsize=7,
                        color='white', fontweight='bold')

def _add_churn_reference_lines(ax, df, target_column):
    """Añade líneas de referencia de tasa de abandono promedio"""
    # Versión más robusta que maneja tanto strings como números
    try:
        # Si la columna es string ('0', '1'), convertir a int
        if df[target_column].dtype == 'object' or df[target_column].dtype == 'str':
            avg_churn = (df[target_column].astype(int).sum() / len(df)) * 100
        else:
            # Si ya es numérica
            avg_churn = (df[target_column].sum() / len(df)) * 100
    except Exception as e:
        # Si hay error, intentar conversión segura
        try:
            numeric_values = pd.to_numeric(df[target_column], errors='coerce')
            avg_churn = (numeric_values.sum() / numeric_values.notna().sum()) * 100
        except:
            print(f"Error al calcular tasa de abandono: {e}")
            return
    
    # Línea promedio general
    ax.axhline(y=avg_churn, color='red', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Promedio general: {avg_churn:.1f}%')
    
    ax.legend(title='Referencia', fontsize=9, loc='upper left')

def _show_churn_summary(df, analysis_column, target_column, num_title):
    """Muestra resumen estadístico en consola"""
    print("\n" + "="*80)
    print(f"RESUMEN DE ANÁLISIS DE ABANDONO POR {num_title.upper()}")
    print("="*80)
    
    # Estadísticas generales
    total_clientes = len(df)
    total_abandonos = df[target_column].astype(int).sum()
    tasa_abandono = (total_abandonos / total_clientes) * 100
    
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"   Total de clientes: {total_clientes:,}")
    print(f"   Total de abandonos: {total_abandonos:,}")
    print(f"   Tasa de abandono general: {tasa_abandono:.2f}%")
    
    # Estadísticas por variable numérica
    print(f"\n📈 ESTADÍSTICAS DE {num_title.upper()}:")
    print(f"   Mínimo: {df[analysis_column].min():.2f}")
    print(f"   Máximo: {df[analysis_column].max():.2f}")
    print(f"   Media: {df[analysis_column].mean():.2f}")
    print(f"   Mediana: {df[analysis_column].median():.2f}")
    print(f"   Desviación estándar: {df[analysis_column].std():.2f}")
    
    # Correlación con abandono
    if df[analysis_column].nunique() > 1:
        correlation = df[[analysis_column, target_column]].corr().iloc[0, 1]
        print(f"\n🔗 CORRELACIÓN CON ABANDONO:")
        print(f"   Coeficiente de correlación: {correlation:.4f}")
        
        if abs(correlation) > 0.3:
            print("   → Correlación moderada a fuerte")
        elif abs(correlation) > 0.1:
            print("   → Correlación débil")
        else:
            print("   → Correlación muy débil o nula")
    
    print("="*80)

# =============================================================================
# FUNCIÓN PRINCIPAL analysis_churn_category + SUS SUBFUNCIONES
# =============================================================================

def analysis_churn_category(df, numeric_column=None, column_title_numeric=None,
                            category_column=None, column_title_category=None, 
                            target_column=None, bins_n=14, palette=None,
                            label_format_style='auto'):
    """
    Análisis bivariado de variable numérica y categórica en casos de churn
    
    Presenta un análisis bivariado de una variable numérica de los clientes
    que abandonaron vs una variable categórica, mostrando histograma de 
    distribución y gráfico de barras con tasas de abandono por categoría.
    
    Parámetros:
    -----------
    df : DataFrame con datos
    numeric_column : str, columna numerica a analizar
    column_title_numeric : str, titulo personalizado numerica
    category_column : str, columna categórica a analizar
    target_column : str, columna objetivo 
    column_title_category : str, título personalizado
    bins_n : int, número de bins para edad
    palette : dict, paleta de colores (opcional)
    
    Retorna:
    --------
    tuple
        (fig, axes, None, group_stats, averages) - Figura, ejes, estadísticas y promedios
    
    Ejemplos:
    ---------
    # Ejemplo: Tasa de abandono por genero
    palette_gender = {
        'female': qualitative_palette[6],
        'male': qualitative_palette[9]
    }
    
    fig, axes, summary, stats, averages = analysis_churn_category(
        df=df_eda, numeric_column='age', column_title_numeric='Edad',
        category_column='gender', column_title_category='Genero', 
        target_column='exited', bins_n=14, palette=palette_gender,
        label_format_style='auto'
    )
    """
    
    # Configuración inicial
    num_title = column_title_numeric or numeric_column.replace('_', ' ').title()
    cat_title = column_title_category or category_column.replace('_', ' ').title()
    
    # Preparar datos
    df, df_churn, categories = _prepare_data(df, category_column, target_column)
    
    # Crear paleta
    palette = palette or _create_palette(categories)
    
    # Crear figura
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    fig.suptitle(f'Análisis de Abandono por {num_title} y {cat_title}', 
                fontsize=20, fontweight='bold', y=1.02)
    
    # Calcular estadísticas usando función auxiliar
    group_stats = _create_bins_and_stats(
        df, numeric_column, category_column, target_column, bins_n, label_format_style
    )
    
    # Crear gráficos
    _create_histogram(axes[0], df_churn, numeric_column, num_title, category_column, cat_title, palette, bins_n)
    _create_bars(axes[1], group_stats, num_title, category_column, palette)
    
    # Calcular y mostrar averages
    averages = _calculate_averages(df, category_column, target_column, categories)
    avg_churn = (df[target_column].sum() / len(df)) * 100
    
    # Añadir líneas de referencia
    _add_reference_lines(axes[1], avg_churn, averages, palette)
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    
    # Mostrar resumen
    _show_summary(df, category_column, target_column, cat_title, avg_churn, categories, averages)
    
    return fig, axes, None, group_stats, averages

# SUB-FUNCIONES PARA analysis_churn_category
# =============================================================================

def _prepare_data(df, category_column, target_column):
    """Prepara y limpia los datos"""
    df = df.copy()
    df_churn = df[df[target_column] == 1].copy()
    
    # Convertir binarios a string
    unique_values = df[category_column].dropna().unique()
    if set(unique_values).issubset({0, 1}):
        df[category_column] = df[category_column].astype(str)
        df_churn[category_column] = df_churn[category_column].astype(str)
    
    categories = sorted(df[category_column].dropna().unique())
    return df, df_churn, categories

def _create_palette(categories):
    """Crea paleta de colores"""
    if len(categories) == 2:
        return {categories[0]: '#FF6B6B', categories[1]: '#4ECDC4'}
    return dict(zip(categories, sns.color_palette("husl", len(categories))))

def _create_histogram(ax, df_churn, numeric_column, num_title, category_column, cat_title, palette, bins_n):
    """Crea histograma de distribución"""
    sns.histplot(data=df_churn, x=numeric_column, bins=bins_n, kde=True, stat='count', 
                hue=category_column, palette=palette, alpha=0.7, multiple="stack", ax=ax)
    
    ax.set_title('Distribución en Casos de Abandono', fontsize=14, fontweight='bold')
    ax.set_xlabel(num_title, fontsize=12)
    ax.set_ylabel('Número de Abandonos', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Crear leyenda personalizada
    _create_histogram_legend(ax, df_churn, category_column, palette, cat_title)

def _create_histogram_legend(ax, df_churn, category_column, palette, cat_title):
    """Crea leyenda personalizada para histograma"""
    categories = sorted(df_churn[category_column].unique())
    handles, labels = [], []
    
    for cat in categories:
        count = (df_churn[category_column] == cat).sum()
        pct = (count / len(df_churn) * 100) if len(df_churn) > 0 else 0
        
        labels.append(f"{cat} (n={count}, {pct:.1f}%)")
        handles.append(Patch(facecolor=palette[cat], alpha=0.7))
    
    ax.legend(handles, labels, title=f'{cat_title} - Abandonos', 
                fontsize=10, loc='upper right')

def _create_bars(ax, group_stats, num_title, category_column, palette):
    """Crea gráfico de barras de tasa de abandono"""
    sns.barplot(data=group_stats, x='bin_label', y='churn_rate', hue=category_column,
                palette=palette, alpha=0.8, ax=ax,
                order=sorted(group_stats['bin_label'].unique()))
    
    ax.set_title(f'Tasa de Abandono por Grupo de {num_title}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Tasa de Abandono (%)', fontsize=12)
    ax.set_xlabel(f'Rango de {num_title}', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en barras
    _add_values_bars(ax, group_stats, category_column, palette)
    
    # Remover leyenda automática
    if ax.get_legend():
        ax.get_legend().remove()

def _add_values_bars(ax, group_stats, category_column, palette):
    """Añade valores de porcentaje sobre las barras"""
    categories = sorted(group_stats[category_column].unique())
    
    for i, bin_label in enumerate(sorted(group_stats['bin_label'].unique())):
        for j, cat in enumerate(categories):
            subset = group_stats[
                (group_stats['bin_label'] == bin_label) & 
                (group_stats[category_column] == cat)
            ]
            
            if not subset.empty and subset.iloc[0]['total'] >= 10:  # Solo si muestra suficiente
                row = subset.iloc[0]
                offset = (j - len(categories)/2 + 0.5) * 0.15
                ax.text(i + offset, row['churn_rate'] + 0.3, 
                        f"{row['churn_rate']:.1f}%",
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=palette[cat])

def _calculate_averages(df, category_column, target_column, categories):
    """Calcula promedio de abandono por categoría"""
    averages = {}
    for cat in categories:
        mask = df[category_column] == cat
        if mask.any():
            averages[cat] = (df.loc[mask, target_column].sum() / mask.sum()) * 100
    return averages

def _add_reference_lines(ax, avg_churn, averages, palette):
    """Añade líneas de referencia con averages"""
    # Línea promedio general
    ax.axhline(y=avg_churn, color='red', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Promedio general: {avg_churn:.1f}%')
    
    # Líneas por categoría
    for cat, valor in averages.items():
        ax.axhline(y=valor, color=palette[cat], linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f'{cat}: {valor:.1f}%')
    
    ax.legend(title='Límites de Tasa de Abandono', fontsize=9, loc='upper left')

def _show_summary(df, category_column, target_column, cat_title, avg_churn, categories, averages):
    """Muestra resumen estadístico en consola"""
    print("\n" + "="*80)
    print(f"RESUMEN DE TASAS DE ABANDONO POR {cat_title.upper()}")
    print("="*80)
    
    summary_data = []
    total_clientes = len(df)
    total_abandonos = df[target_column].sum()
    
    # Total general
    summary_data.append({
        'Categoría': 'TOTAL GENERAL',
        'Tasa Abandono (%)': f"{avg_churn:.2f}",
        'Clientes': total_clientes,
        'Abandonos': total_abandonos
    })
    
    # Por categoría
    for cat in sorted(categories):
        mask = df[category_column] == cat
        total_cat = mask.sum()
        abandonos_cat = df.loc[mask, target_column].sum()
        tasa_cat = averages.get(cat, 0)
        
        summary_data.append({
            'Categoría': str(cat),
            'Tasa Abandono (%)': f"{tasa_cat:.2f}",
            'Clientes': total_cat,
            'Abandonos': abandonos_cat
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, justify='center'))
    print("="*80)
    
    # Comparación si hay 2 categorías
    if len(categories) == 2:
        cat1, cat2 = categories
        tasa1, tasa2 = averages.get(cat1, 0), averages.get(cat2, 0)
        print(f"\nComparación: {cat1} ({tasa1:.1f}%) vs {cat2} ({tasa2:.1f}%)")
        print(f"Diferencia: {abs(tasa1 - tasa2):.1f} puntos porcentuales")
        print("="*80)


"""
# Ejemplo:
fig, axes, group_stats = analysis_churn(
    df=df_eda, target_column='exited', 
    analysis_column='age', column_title_numeric='Edad',
    bins_n=14, palette={'0': 'green', '1': 'red'}
    )

# Ejemplo: Tasa de abandono por genero
palette_gender = {
    'female': qualitative_palette[6],
    'male': qualitative_palette[9],
}

fig, axes, summary, stats, averages = analysis_churn_category(
    df=df_eda, numeric_column='age', column_title_numeric='Edad',
    category_column='gender', column_title_category='Genero', 
    target_column='exited', bins_n=14, palette=palette_gender,
    label_format_style='auto'
)


"""