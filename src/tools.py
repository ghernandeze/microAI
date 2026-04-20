from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import get_settings
from .data_loader import normalize_text


@dataclass
class ToolResult:
    title: str
    content: str


def find_department_row(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    target = normalize_text(name)

    exact = df[df["departamento_norm"] == target]
    if not exact.empty:
        return exact.iloc[0]

    partial = df[df["departamento_norm"].str.contains(target, regex=False)]
    if not partial.empty:
        return partial.iloc[0]

    return None


def get_top_departments(df: pd.DataFrame, n: int = 10) -> ToolResult:
    n = max(1, int(n))
    view = df[["departamento", "indice_oportunidad", "nivel"]].head(n).copy()
    return ToolResult(
        title=f"Top {n} departamentos",
        content=view.to_string(index=False),
    )


def explain_department(df: pd.DataFrame, department_name: str) -> ToolResult:
    row = find_department_row(df, department_name)
    if row is None:
        return ToolResult(
            title="Departamento no encontrado",
            content=f"No se encontró un departamento que coincida con '{department_name}'.",
        )

    factors = {
        "necesidad económica": float(row["pobreza_n"]),
        "brecha de microcrédito": float(row["microcredito_n"]),
        "brecha de inclusión financiera": float(row["productos_n"]),
        "baja infraestructura financiera": float(row["atm_n"]),
        "viabilidad digital": float(row["internet_n"]),
    }
    strongest = max(factors, key=factors.get)

    content = (
        f"Departamento: {row['departamento']}\n"
        f"Índice de oportunidad: {row['indice_oportunidad']:.3f}\n"
        f"Nivel: {row['nivel']}\n\n"
        f"Factor dominante: {strongest}.\n"
        f"Valores normalizados:\n"
        f"- pobreza_n: {row['pobreza_n']:.3f}\n"
        f"- microcredito_n: {row['microcredito_n']:.3f}\n"
        f"- productos_n: {row['productos_n']:.3f}\n"
        f"- atm_n: {row['atm_n']:.3f}\n"
        f"- internet_n: {row['internet_n']:.3f}"
    )
    return ToolResult(title=f"Explicación de {row['departamento']}", content=content)


def recommend_strategy(df: pd.DataFrame, top_n: int = 5) -> ToolResult:
    top_n = max(1, int(top_n))
    top = df.head(top_n)
    names = ", ".join(top["departamento"].tolist())

    content = (
        f"Recomendación inicial: priorizar estos {top_n} departamentos: {names}.\n\n"
        "La recomendación se apoya en los resultados del índice, que combina necesidad económica, "
        "brecha de microcrédito, brecha de inclusión financiera, infraestructura financiera y viabilidad digital."
    )
    return ToolResult(title="Recomendación estratégica", content=content)


def simulate_weights(
    df: pd.DataFrame,
    peso_pobreza: float,
    peso_micro: float,
    peso_productos: float,
    peso_atm: float,
    peso_internet: float,
    top_n: int = 10,
) -> ToolResult:
    total = peso_pobreza + peso_micro + peso_productos + peso_atm + peso_internet
    if total <= 0:
        raise ValueError("La suma de los pesos debe ser mayor que cero.")

    weights = {
        "pobreza_n": peso_pobreza / total,
        "microcredito_n": peso_micro / total,
        "productos_n": peso_productos / total,
        "atm_n": peso_atm / total,
        "internet_n": peso_internet / total,
    }

    temp = df.copy()
    temp["indice_simulado"] = (
        temp["pobreza_n"] * weights["pobreza_n"]
        + temp["microcredito_n"] * weights["microcredito_n"]
        + temp["productos_n"] * weights["productos_n"]
        + temp["atm_n"] * weights["atm_n"]
        + temp["internet_n"] * weights["internet_n"]
    )
    temp = temp.sort_values("indice_simulado", ascending=False).reset_index(drop=True)

    view = temp[["departamento", "indice_simulado"]].head(max(1, int(top_n)))
    content = (
        "Pesos normalizados usados:\n"
        f"- pobreza_n: {weights['pobreza_n']:.3f}\n"
        f"- microcredito_n: {weights['microcredito_n']:.3f}\n"
        f"- productos_n: {weights['productos_n']:.3f}\n"
        f"- atm_n: {weights['atm_n']:.3f}\n"
        f"- internet_n: {weights['internet_n']:.3f}\n\n"
        "Top resultante:\n"
        f"{view.to_string(index=False)}"
    )
    return ToolResult(title="Simulación de pesos", content=content)


def get_top_by_variable(df: pd.DataFrame, column: str, n: int = 10) -> ToolResult:
    LABELS = {
        "pobreza_n": "mayor pobreza",
        "microcredito_n": "mayor brecha de microcrédito",
        "productos_n": "mayor brecha de inclusión financiera",
        "atm_n": "menor infraestructura financiera (ATM)",
        "internet_n": "mayor viabilidad digital",
    }
    if column not in df.columns:
        return ToolResult(title="Variable no encontrada", content=f"La variable '{column}' no existe.")
    view = (
        df[["departamento", column, "indice_oportunidad", "nivel"]]
        .sort_values(column, ascending=False)
        .head(max(1, int(n)))
    )
    label = LABELS.get(column, column)
    return ToolResult(
        title=f"Top {n} por {label}",
        content=view.to_string(index=False),
    )


def filter_by_level(df: pd.DataFrame, level: str) -> ToolResult:
    valid = {"Alto", "Medio", "Bajo"}
    level = level.capitalize()
    if level not in valid:
        return ToolResult(title="Nivel inválido", content=f"Usa: Alto, Medio o Bajo.")
    view = df[df["nivel"] == level][["departamento", "indice_oportunidad", "nivel"]]
    if view.empty:
        return ToolResult(title=f"Nivel {level}", content="No hay departamentos con ese nivel.")
    return ToolResult(
        title=f"Departamentos de nivel {level}",
        content=view.to_string(index=False),
    )


def emitir_decision_formal(row):
    indice = row["indice_oportunidad"]
    internet = row["internet_n"]
    micro = row["microcredito_n"]

    if indice >= 0.75 and internet >= 0.50:
        return "Recomendado para despliegue inmediato"
    elif indice >= 0.55:
        return "Prioridad media, requiere mitigación previa"
    else:
        return "No recomendado para inversión inmediata"
    
    
def accion_requerida(row):
    if row["internet_n"] < 0.40:
        return "Fortalecer viabilidad digital antes del despliegue."
    if row["microcredito_n"] > 0.70:
        return "Diseñar estrategia de entrada temprana y validación comercial."
    if row["productos_n"] > 0.60:
        return "Desarrollar acciones de educación financiera y acompañamiento local."
    return "Validar condiciones operativas y ejecutar piloto controlado."


_GEOJSON_MAP = {
    "BOGOTA D.C.": "SANTAFE DE BOGOTA D.C",
    "SAN ANDRES Y PROVIDENCIA": "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA",
}


def _norm_geojson(name: str) -> str:
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", str(name).upper())
        if unicodedata.category(c) != "Mn"
    )
    return _GEOJSON_MAP.get(sin_tildes, sin_tildes)


def _build_choropleth(
    df: pd.DataFrame,
    value_col: str,
    legend_name: str,
    output_path,
    geo_data: dict,
) -> None:
    import folium
    map_df = df[["departamento", value_col]].copy()
    map_df["departamento"] = map_df["departamento"].apply(_norm_geojson)
    m = folium.Map(location=[4, -74], zoom_start=5)
    folium.Choropleth(
        geo_data=geo_data,
        data=map_df,
        columns=["departamento", value_col],
        key_on="feature.properties.NOMBRE_DPT",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend_name,
    ).add_to(m)
    m.save(str(output_path))


def _load_geo(settings) -> tuple:
    try:
        import folium  # noqa: F401
    except ImportError:
        return None, ToolResult(
            title="Error: folium no instalado",
            content="Instala folium con: pip install folium",
        )
    geo_path = settings.project_root / "data" / "geo" / "Colombia.geo.json"
    if not geo_path.exists():
        return None, ToolResult(
            title="Error: GeoJSON no encontrado",
            content=f"No se encontró el archivo {geo_path}",
        )
    with open(geo_path, "r", encoding="utf-8") as f:
        geo_data = json.load(f)
    return geo_data, None


def generate_map(df: pd.DataFrame) -> ToolResult:
    settings = get_settings()
    geo_data, err = _load_geo(settings)
    if err:
        return err
    output_path = settings.project_root / "reports" / "mapa_oportunidad.html"
    output_path.parent.mkdir(exist_ok=True)
    _build_choropleth(df, "indice_oportunidad", "Índice de oportunidad", output_path, geo_data)
    return ToolResult(
        title="Mapa generado",
        content=f"Mapa coroplético guardado en:\n{output_path}\n\nAbre ese archivo en tu navegador para verlo.",
    )


def generate_map_with_weights(
    df: pd.DataFrame,
    peso_pobreza: float,
    peso_micro: float,
    peso_productos: float,
    peso_atm: float,
    peso_internet: float,
) -> ToolResult:
    total = peso_pobreza + peso_micro + peso_productos + peso_atm + peso_internet
    if total <= 0:
        raise ValueError("La suma de los pesos debe ser mayor que cero.")
    w = {
        "pobreza_n":    peso_pobreza  / total,
        "microcredito_n": peso_micro  / total,
        "productos_n":  peso_productos / total,
        "atm_n":        peso_atm      / total,
        "internet_n":   peso_internet / total,
    }
    temp = df.copy()
    temp["indice_simulado"] = sum(temp[col] * w[col] for col in w)

    settings = get_settings()
    geo_data, err = _load_geo(settings)
    if err:
        return err
    output_path = settings.project_root / "reports" / "mapa_oportunidad.html"
    output_path.parent.mkdir(exist_ok=True)
    _build_choropleth(temp, "indice_simulado", "Índice simulado", output_path, geo_data)
    return ToolResult(
        title="Mapa con pesos personalizados generado",
        content=f"Mapa generado con pesos: pobreza={w['pobreza_n']:.2f}, microcrédito={w['microcredito_n']:.2f}, "
                f"productos={w['productos_n']:.2f}, ATM={w['atm_n']:.2f}, internet={w['internet_n']:.2f}\n"
                f"Guardado en:\n{output_path}",
    )