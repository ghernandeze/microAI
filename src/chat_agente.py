from __future__ import annotations

from .agent_tools import run_agent
from .data_loader import load_ranking_data
from .intent_router import route_query
from .llm_client import LLMClient
from .tools import (
    explain_department, filter_by_level, generate_map,
    get_top_by_variable, get_top_departments, recommend_strategy, simulate_weights,
)


HELP_TEXT = """Puedes escribir preguntas libres, por ejemplo:
- ¿Cuáles son los top departamentos?
- Explícame por qué Chocó aparece arriba
- ¿Qué departamento priorizarías para una prueba piloto?
- ¿Qué riesgos ves en La Guajira?
- Simular 0.30 0.30 0.15 0.10 0.15
- ayuda
- salir
"""


def handle_local_fallback(df, decision, raw_query: str) -> str:
    if decision.intent == "help":
        return HELP_TEXT

    if decision.intent == "top":
        return get_top_departments(df, decision.top_n or 10).content

    if decision.intent == "explain_department" and decision.department:
        return explain_department(df, decision.department).content

    if decision.intent == "recommend":
        return recommend_strategy(df, 5).content

    if decision.intent == "simulate":
        try:
            return simulate_weights(df, *decision.weights, top_n=10).content
        except ValueError as e:
            return f"Error: {e}"

    if decision.intent == "simulate_missing":
        return "No pude identificar 5 pesos. Usa algo como: simular 0.30 0.30 0.15 0.10 0.15"

    if decision.intent == "department_question" and decision.department:
        return explain_department(df, decision.department).content

    if decision.intent == "greeting":
        return (
            "Hola. Puedo ayudarte a analizar qué departamentos tienen mayor potencial "
            "para el despliegue de microcrédito digital en Colombia. "
            "Puedes preguntarme, por ejemplo, cuáles son los top departamentos, "
            "por qué un departamento aparece alto en el ranking, "
            "o qué recomendaría para una prueba piloto."
        )
        
    if decision.intent == "out_of_scope":
        return (
            "Puedo ayudarte con consultas relacionadas con el proyecto de microcrédito digital, "
            "como ranking de departamentos, explicación de resultados, recomendaciones "
            "y simulación de pesos. Sobre ese tema no tengo contexto suficiente."
        )

    if decision.intent == "map":
        return generate_map(df).content

    if decision.intent == "top_by_variable" and decision.weights:
        return get_top_by_variable(df, decision.weights[0], decision.top_n or 10).content

    if decision.intent == "filter_level" and decision.weights:
        return filter_by_level(df, decision.weights[0]).content

    if decision.intent == "weight_adjust":
        return (
            "Para redistribuir los pesos usa el formato: simular 0.30 0.30 0.15 0.10 0.15 "
            "(pobreza, microcrédito, productos, ATM, internet). "
            "O activa el LLM para hacerlo con lenguaje natural."
        )

    return (
        "No pude responder esa consulta en modo local. "
        "Si activas el API, el agente podrá responder preguntas más abiertas."
    )


def main() -> None:
    print("=" * 70)
    print("AGENTE CONVERSACIONAL DE PRIORIZACIÓN GEOGRÁFICA DE MICROCRÉDITO")
    print("=" * 70)
    
    print(HELP_TEXT)

    try:
        df = load_ranking_data()
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return

    llm_client = LLMClient()

    while True:
        query = input("\nTú: ").strip()
        if not query:
            continue

        decision = route_query(query)

        if decision.intent == "exit":
            print("\nAgente: Sesión finalizada.")
            break

        try:
            if llm_client.is_enabled():
                answer, _ = run_agent(query, df, llm_client)
            else:
                answer = handle_local_fallback(df, decision, query)
        except Exception as e:
            answer = f"Ocurrió un error al procesar la consulta: {e}"

        print(f"\nAgente:\n{answer}")


if __name__ == "__main__":
    main()
