from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from .tools import (
    explain_department,
    filter_by_level,
    generate_map,
    generate_map_with_weights,
    get_top_by_variable,
    get_top_departments,
    recommend_strategy,
    simulate_weights,
)

if TYPE_CHECKING:
    from .llm_client import LLMClient


AGENT_SYSTEM_PROMPT = """Eres un asistente de apoyo a decisiones para un proyecto de priorización geográfica de microcrédito digital en Colombia (datos 2024).

REGLAS IMPORTANTES:
- Si el usuario saluda o hace una pregunta casual, responde brevemente y ofrece ayuda. NO llames herramientas para saludos.
- SIEMPRE llama la herramienta correspondiente cuando el usuario pregunte sobre datos, departamentos, niveles o rankings. NUNCA respondas preguntas de datos desde el contexto de la conversación — siempre consulta la herramienta aunque ya hayas mostrado información similar antes.
- NUNCA inventes valores numéricos — úsalos solo si vienen de una herramienta.
- OBLIGATORIO: Si el usuario pregunta sobre noticias, seguridad, iniciativas del gobierno, situación económica, contexto regional, o cualquier información que NO esté en el dataset, DEBES llamar search_web inmediatamente. NUNCA respondas ese tipo de preguntas diciendo que no tienes información — usa search_web y presenta los resultados.

Variables del modelo (todas normalizadas 0-1):
- pobreza_n: necesidad económica (1 = máxima pobreza)
- microcredito_n: brecha de microcrédito (1 = menos acceso)
- productos_n: brecha de inclusión financiera general
- atm_n: carencia de infraestructura ATM (1 = menos cajeros)
- internet_n: viabilidad digital (1 = mejor conectividad)

Responde en español. Cuando hagas recomendaciones usa: Decisión:, Prioridad:, Riesgo:, Acción requerida:"""


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_top_departments",
            "description": "Retorna los N departamentos con mayor índice de oportunidad para microcrédito digital.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Número de departamentos a retornar. Por defecto 10."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_department",
            "description": "Explica en detalle los factores del índice de oportunidad de un departamento específico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department_name": {"type": "string", "description": "Nombre del departamento (ej: Chocó, La Guajira)."},
                },
                "required": ["department_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_weights",
            "description": "Recalcula el ranking de departamentos asignando pesos personalizados a cada variable. Los pesos se normalizan automáticamente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "peso_pobreza":   {"type": "number", "description": "Peso para pobreza."},
                    "peso_micro":     {"type": "number", "description": "Peso para brecha de microcrédito."},
                    "peso_productos": {"type": "number", "description": "Peso para inclusión financiera."},
                    "peso_atm":       {"type": "number", "description": "Peso para infraestructura ATM."},
                    "peso_internet":  {"type": "number", "description": "Peso para viabilidad digital."},
                    "top_n":          {"type": "integer", "description": "Cuántos departamentos mostrar. Por defecto 10."},
                },
                "required": ["peso_pobreza", "peso_micro", "peso_productos", "peso_atm", "peso_internet"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_by_variable",
            "description": "Top de departamentos ordenados por una variable específica del modelo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "enum": ["pobreza_n", "microcredito_n", "productos_n", "atm_n", "internet_n"],
                        "description": "Variable por la cual ordenar.",
                    },
                    "n": {"type": "integer", "description": "Número de departamentos. Por defecto 10."},
                },
                "required": ["variable"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_level",
            "description": "Lista los departamentos filtrados por nivel de oportunidad: Alto, Medio o Bajo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["Alto", "Medio", "Bajo"],
                        "description": "Nivel de oportunidad.",
                    },
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_strategy",
            "description": "Genera una recomendación estratégica de despliegue basada en los top departamentos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "Número de departamentos a incluir. Por defecto 5."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Busca información actualizada en internet. Usar para contexto externo al dataset: noticias, situación económica, riesgos operativos, geografía, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Consulta de búsqueda en español o inglés."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_map",
            "description": "Genera el mapa coroplético de Colombia con el índice de oportunidad BASE (sin modificar pesos).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_map_with_weights",
            "description": "Genera el mapa coroplético de Colombia recalculando el índice con pesos personalizados. Usar cuando el usuario pide el mapa con pesos diferentes al default.",
            "parameters": {
                "type": "object",
                "properties": {
                    "peso_pobreza":   {"type": "number"},
                    "peso_micro":     {"type": "number"},
                    "peso_productos": {"type": "number"},
                    "peso_atm":       {"type": "number"},
                    "peso_internet":  {"type": "number"},
                },
                "required": ["peso_pobreza", "peso_micro", "peso_productos", "peso_atm", "peso_internet"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict, df: pd.DataFrame) -> str:
    if name == "get_top_departments":
        return get_top_departments(df, arguments.get("n", 10)).content
    if name == "explain_department":
        return explain_department(df, arguments["department_name"]).content
    if name == "simulate_weights":
        return simulate_weights(
            df,
            arguments["peso_pobreza"],
            arguments["peso_micro"],
            arguments["peso_productos"],
            arguments["peso_atm"],
            arguments["peso_internet"],
            top_n=arguments.get("top_n", 10),
        ).content
    if name == "get_top_by_variable":
        return get_top_by_variable(df, arguments["variable"], arguments.get("n", 10)).content
    if name == "filter_by_level":
        return filter_by_level(df, arguments["level"]).content
    if name == "recommend_strategy":
        return recommend_strategy(df, arguments.get("top_n", 5)).content
    if name == "generate_map":
        return generate_map(df).content
    if name == "generate_map_with_weights":
        return generate_map_with_weights(
            df,
            arguments["peso_pobreza"],
            arguments["peso_micro"],
            arguments["peso_productos"],
            arguments["peso_atm"],
            arguments["peso_internet"],
        ).content
    if name == "search_web":
        return _search_tavily(arguments["query"])
    return f"Herramienta '{name}' no reconocida."


def _search_tavily(query: str) -> str:
    import json as _json
    from urllib import error as urllib_error
    from urllib import request as urllib_request

    from .config import get_settings

    key = get_settings().tavily_api_key
    if not key:
        return "Búsqueda web no disponible (TAVILY_API_KEY no configurada en .env)."

    payload = _json.dumps({
        "api_key": key,
        "query": query,
        "max_results": 4,
        "search_depth": "basic",
    }).encode("utf-8")

    req = urllib_request.Request(
        "https://api.tavily.com/search",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib_request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as e:
        return f"Error en búsqueda web (HTTP {e.code})."
    except urllib_error.URLError as e:
        return f"No se pudo conectar a Tavily: {e}."

    results = data.get("results", [])
    if not results:
        return "Sin resultados para esa búsqueda."
    lines = [f"- {r.get('title', '')}: {r.get('content', '')[:300]}" for r in results]
    return "\n".join(lines)


def run_agent(
    user_query: str,
    df: pd.DataFrame,
    llm_client: "LLMClient",
    history: list[dict] | None = None,
    max_iterations: int = 5,
) -> tuple[str, bool]:
    """
    Agentic loop: the LLM decides which tools to call, we execute them,
    and the LLM formulates the final response.
    Returns (response_text, has_map).
    Falls back to text-only mode if the model/provider doesn't support tool calling.
    """
    messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    if history:
        for msg in history[-6:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_query})

    has_map = False

    for _ in range(max_iterations):
        try:
            response = llm_client.call(messages, tools=TOOL_SCHEMAS)
        except RuntimeError as e:
            if "tool use" in str(e).lower() or "tool_use" in str(e).lower() or "404" in str(e):
                return _run_text_fallback(user_query, df, llm_client, history), False
            raise

        if not response.get("tool_calls"):
            return response.get("text", "Sin respuesta."), has_map

        messages.append(response["raw_assistant_message"])

        for tc in response["tool_calls"]:
            if tc["name"] in ("generate_map", "generate_map_with_weights"):
                has_map = True
            try:
                result = execute_tool(tc["name"], tc["arguments"], df)
            except Exception as e:
                result = f"Error al ejecutar {tc['name']}: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    messages.append({"role": "user", "content": "Resume los resultados obtenidos."})
    final = llm_client.call(messages)
    return final.get("text", "Sin respuesta."), has_map


def _run_text_fallback(
    user_query: str,
    df: pd.DataFrame,
    llm_client: "LLMClient",
    history: list[dict] | None = None,
) -> str:
    """Text-only fallback: embeds data context in the prompt instead of using tools."""
    from .prompt_builder import build_messages
    from .intent_router import route_query

    decision = route_query(user_query)
    messages = build_messages(df, user_query, decision, history=history)
    return llm_client.generate(messages)
