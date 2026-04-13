from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

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
