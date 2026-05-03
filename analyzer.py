"""
analyzer.py
Lógica de negocio para análisis de Share of Shelf (SoS) en perchas de supermercado.

Convenciones de datos:
- Cada fila = un SKU o producto observado en percha
- Columnas requeridas: Marca, Nivel, Facings
- Nivel: número entero (1 = suelo, aumenta hacia arriba) o etiqueta string
- Facings: número de frentes visibles del producto en percha
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


REQUIRED_COLUMNS = {"Marca", "Nivel", "Facings"}
COLUMN_ALIASES = {
    # Español
    "marca": "Marca", "brand": "Marca",
    "nivel": "Nivel", "level": "Nivel", "shelf": "Nivel", "fila": "Nivel", "piso": "Nivel",
    "facings": "Facings", "frentes": "Facings", "facing": "Facings", "unidades": "Facings",
    "qty": "Facings", "cantidad": "Facings",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas independientemente de mayúsculas o idioma."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Valida que el DataFrame tenga la estructura mínima requerida.
    Retorna (es_valido, mensaje).
    """
    df = normalize_columns(df)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return False, (
            f"Faltan columnas requeridas: **{', '.join(missing)}**.\n\n"
            f"El archivo debe tener: `Marca`, `Nivel`, `Facings` "
            f"(también acepta: brand/marca, level/nivel/shelf/fila, facings/frentes/facing).\n\n"
            f"Columnas encontradas: {', '.join(df.columns.tolist())}"
        )

    # Validar que Facings sea numérico
    df["Facings"] = pd.to_numeric(df["Facings"], errors="coerce")
    n_invalid = df["Facings"].isna().sum()
    if n_invalid > 0:
        return False, (
            f"La columna `Facings` tiene {n_invalid} valores no numéricos. "
            f"Todos los facings deben ser números enteros positivos."
        )

    if (df["Facings"] <= 0).any():
        return False, "La columna `Facings` contiene valores menores o iguales a cero."

    if df["Marca"].isna().any() or df["Nivel"].isna().any():
        return False, "Existen valores vacíos en las columnas `Marca` o `Nivel`."

    return True, "OK"


def compute_sos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Share of Shelf (%) por Marca y Nivel.
    
    SoS = Facings(marca, nivel) / Total_Facings(nivel) × 100
    
    Retorna un DataFrame pivotado:
    - Filas: Marcas
    - Columnas: Niveles
    - Valores: SoS en %
    """
    df = normalize_columns(df.copy())
    df["Facings"] = pd.to_numeric(df["Facings"])
    
    # Agrupar: suma de facings por (Marca, Nivel)
    grouped = df.groupby(["Marca", "Nivel"], as_index=False)["Facings"].sum()
    
    # Total facings por nivel (denominador correcto para SoS)
    total_por_nivel = grouped.groupby("Nivel")["Facings"].transform("sum")
    grouped["SoS"] = (grouped["Facings"] / total_por_nivel * 100).round(2)
    
    # Pivotar: filas = marcas, columnas = niveles
    pivot = grouped.pivot_table(
        index="Marca",
        columns="Nivel",
        values="SoS",
        aggfunc="sum",
        fill_value=0
    )
    
    # Ordenar niveles numéricamente si es posible
    try:
        pivot = pivot[sorted(pivot.columns, key=lambda x: float(str(x)))]
        pivot.columns = [f"Nivel {int(float(str(c)))}" if str(c).replace('.','').isdigit() else str(c)
                        for c in pivot.columns]
    except (ValueError, TypeError):
        pivot.columns = [str(c) for c in pivot.columns]
    
    # Ordenar marcas por SoS total descendente
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False).drop(columns="_total")
    
    return pivot


def compute_summary(df: pd.DataFrame) -> dict:
    """Genera métricas de resumen para el dashboard."""
    df = normalize_columns(df.copy())
    df["Facings"] = pd.to_numeric(df["Facings"])
    
    total_facings = df["Facings"].sum()
    n_marcas = df["Marca"].nunique()
    n_niveles = df["Nivel"].nunique()
    n_skus = len(df)
    
    # Marca dominante global
    marca_facings = df.groupby("Marca")["Facings"].sum()
    marca_lider = marca_facings.idxmax()
    sos_lider = (marca_facings.max() / total_facings * 100).round(1)
    
    # Nivel con más facings
    nivel_facings = df.groupby("Nivel")["Facings"].sum()
    nivel_top = nivel_facings.idxmax()
    
    return {
        "total_facings": int(total_facings),
        "n_marcas": n_marcas,
        "n_niveles": n_niveles,
        "n_skus": n_skus,
        "marca_lider": marca_lider,
        "sos_lider": sos_lider,
        "nivel_top": nivel_top,
    }


def compute_facings_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Pivota facings absolutos (no SoS) para vista alternativa."""
    df = normalize_columns(df.copy())
    df["Facings"] = pd.to_numeric(df["Facings"])
    
    grouped = df.groupby(["Marca", "Nivel"], as_index=False)["Facings"].sum()
    pivot = grouped.pivot_table(
        index="Marca", columns="Nivel", values="Facings",
        aggfunc="sum", fill_value=0
    )
    
    try:
        pivot = pivot[sorted(pivot.columns, key=lambda x: float(str(x)))]
        pivot.columns = [f"Nivel {int(float(str(c)))}" if str(c).replace('.','').isdigit() else str(c)
                        for c in pivot.columns]
    except (ValueError, TypeError):
        pivot.columns = [str(c) for c in pivot.columns]
    
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False).drop(columns="_total")
    
    return pivot


def get_template_csv() -> str:
    """Retorna un CSV de ejemplo con datos ficticios realistas."""
    data = """Marca,Nivel,Facings
Marca A,1,8
Marca A,2,12
Marca A,3,6
Marca A,4,4
Marca B,1,10
Marca B,2,6
Marca B,3,8
Marca B,4,3
Marca C,1,4
Marca C,2,5
Marca C,3,7
Marca C,4,6
Marca D,1,3
Marca D,2,4
Marca D,3,2
Marca D,4,5
Marca E,1,2
Marca E,2,1
Marca E,3,3
Marca E,4,2
Marca F,1,5
Marca F,2,2
Marca F,3,4
Marca F,4,0
"""
    return data
