from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .data_loader import normalize_text


@dataclass
class RouteDecision:
    intent: str
    department: Optional[str] = None
    top_n: Optional[int] = None
    weights: Optional[list[float]] = None
    use_llm: bool = True



KNOWN_DEPARTMENTS = [
    "antioquia", "atlantico", "bogota", "bolivar", "boyaca", "caldas",
    "caqueta", "cauca", "cesar", "choco", "cordoba", "cundinamarca",
    "huila", "la guajira", "magdalena", "meta", "narino",
    "norte de santander", "quindio", "risaralda", "santander",
    "sucre", "tolima", "valle del cauca",
]


def detect_department(query: str) -> Optional[str]:
    q = normalize_text(query)
    for dept in sorted(KNOWN_DEPARTMENTS, key=len, reverse=True):
        if dept in q:
            return dept
    return None


def route_query(query: str) -> RouteDecision:
    q = normalize_text(query)

    if q in {"ayuda", "help"}:
        return RouteDecision(intent="help", use_llm=False)

    if q in {"salir", "exit", "quit"}:
        return RouteDecision(intent="exit", use_llm=False)
    
    if q in {"hola", "buenas", "buenos dias", "buenas tardes", "buenas noches", "hey", "holi"}:
        return RouteDecision(intent="greeting", use_llm=False)

    if "simular" in q:
        numbers = re.findall(r"[-+]?\d*\.?\d+", q)
        if len(numbers) >= 5:
            weights = list(map(float, numbers[:5]))
            return RouteDecision(intent="simulate", weights=weights, use_llm=False)
        return RouteDecision(intent="simulate_missing", use_llm=False)

    m_top = re.search(r"top\s+(\d+)", q)
    if m_top:
        return RouteDecision(intent="top", top_n=int(m_top.group(1)), use_llm=False)

    if any(word in q for word in ["ranking", "top departamentos", "mejores departamentos"]):
        return RouteDecision(intent="top", top_n=10, use_llm=True)

    dept = detect_department(query)
    if dept and any(word in q for word in ["explica", "explicame", "por que", "porque", "analiza"]):
        return RouteDecision(intent="explain_department", department=dept, use_llm=True)

    if any(word in q for word in ["recomienda", "priorizar", "prioridad", "pilot", "piloto"]):
        return RouteDecision(intent="recommend", use_llm=True)

    if dept:
        return RouteDecision(intent="department_question", department=dept, use_llm=True)

    OUT_OF_SCOPE_HINTS = [
    "chiste", "receta", "pelicula", "novia", "futbol", "clima",
    "astrologia", "videojuego", "tarea de historia"
    ]

    if any(word in q for word in OUT_OF_SCOPE_HINTS):
        return RouteDecision(intent="out_of_scope", use_llm=False)

    if any(word in q for word in ["mapa", "mapa de colombia", "visualizar", "visualizacion", "ver mapa", "genera el mapa", "genera mapa"]):
        return RouteDecision(intent="map", use_llm=False)

    # Top by single variable
    VAR_KEYWORDS = {
        "pobreza": "pobreza_n",
        "microcredito": "microcredito_n",
        "microcrédito": "microcredito_n",
        "productos": "productos_n",
        "inclusion": "productos_n",
        "inclusión": "productos_n",
        "atm": "atm_n",
        "infraestructura": "atm_n",
        "internet": "internet_n",
        "digital": "internet_n",
    }
    if any(word in q for word in ["top por", "ranking por", "ordena por", "mayor", "menos acceso", "mas acceso", "más acceso"]):
        for keyword, col in VAR_KEYWORDS.items():
            if keyword in q:
                m = re.search(r"top\s+(\d+)", q)
                n = int(m.group(1)) if m else 10
                return RouteDecision(intent="top_by_variable", top_n=n, weights=[col], use_llm=False)

    # Filter by level
    if any(word in q for word in ["nivel alto", "nivel medio", "nivel bajo", "solo alto", "solo medio", "solo bajo"]):
        if "alto" in q:
            return RouteDecision(intent="filter_level", weights=["Alto"], use_llm=False)
        if "medio" in q:
            return RouteDecision(intent="filter_level", weights=["Medio"], use_llm=False)
        if "bajo" in q:
            return RouteDecision(intent="filter_level", weights=["Bajo"], use_llm=False)

    # Weight adjustment via natural language → let LLM interpret and recalculate
    WEIGHT_HINTS = ["peso", "ponderacion", "ponderación", "importancia", "dale mas", "dale más",
                    "reduce el peso", "aumenta el peso", "cambia el peso", "ajusta"]
    if any(word in q for word in WEIGHT_HINTS):
        return RouteDecision(intent="weight_adjust", use_llm=True)

    return RouteDecision(intent="general_question", use_llm=True)
