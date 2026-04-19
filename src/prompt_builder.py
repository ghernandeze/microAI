from __future__ import annotations

from textwrap import dedent

import pandas as pd

from .intent_router import RouteDecision
from .tools import explain_department, get_top_departments, recommend_strategy


SYSTEM_PROMPT = dedent(
    """
    Eres un asistente de apoyo a decisiones para un proyecto de priorización geográfica
    de microcrédito digital en Colombia.

    Reglas:
    - Usa únicamente la información del contexto entregado.
    - No inventes datos ni uses conocimiento externo para afirmar resultados del modelo.
    - Si la evidencia no es suficiente, dilo claramente.
    - Responde en español.
    - Mantén un tono técnico pero claro, como un estudiante explicando su análisis.
    - Cuando recomiendes una decisión, justifícala con las variables del contexto.
    - Si el usuario saluda, responde cordialmente y ofrece ayuda sobre el proyecto.
    - Si el usuario pregunta algo fuera del alcance del proyecto, responde amablemente que no tienes conocimiento suficiente sobre ese tema.
    - Responde como un asistente de decisión, no como un analista descriptivo.
    - Siempre que sea posible, entrega una decisión explícita.
    - Usa etiquetas como:
    - Decisión:
    - Prioridad:
    - Acción requerida:
    - Riesgo:
    Si el usuario pregunta por un departamento, indica si es:
    - Recomendado para despliegue inmediato
    - Prioridad media con mitigación previa
    - No recomendado para inversión inmediata
    - Evita expresiones vagas como “se recomienda considerar”.
    - Prefiere expresiones operativas como “acción requerida”, “mitigación previa” o “no recomendado”.
    """
).strip()


def build_context(df: pd.DataFrame, user_query: str, decision: RouteDecision) -> str:
    top = get_top_departments(df, 10)
    rec = recommend_strategy(df, 5)

    if decision.department:
        extra_result = explain_department(df, decision.department)
        dept_section = f"3. Información del departamento consultado:\n{extra_result.content}"
    else:
        dept_section = "3. Departamento consultado: ninguno específico."

    context = dedent(
        f"""
        CONTEXTO DEL MODELO

        1. Top general actual:
        {top.content}

        2. Recomendación base:
        {rec.content}

        {dept_section}

        4. Variables disponibles en el dataset:
        departamento, indice_oportunidad, nivel, pobreza_n, microcredito_n, productos_n, atm_n, internet_n

        5. Significado resumido de variables:
        - pobreza_n: necesidad económica
        - microcredito_n: brecha de microcrédito
        - productos_n: brecha de inclusión financiera general
        - atm_n: menor infraestructura financiera física
        - internet_n: viabilidad digital

        6. Consulta del usuario:
        {user_query}
        """
    ).strip()

    return context


def build_messages(df: pd.DataFrame, user_query: str, decision: RouteDecision) -> list[dict[str, str]]:
    context = build_context(df, user_query, decision)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Responde usando solo el contexto dado. Si recomiendas algo, explica el porqué "
                "a partir de las variables del modelo.\n\n"
                f"{context}"
            ),
        },
    ]
