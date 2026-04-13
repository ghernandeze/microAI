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
    "amazonas", "antioquia", "arauca", "atlantico", "bogota", "bolivar", "boyaca",
    "caldas", "caqueta", "casanare", "cauca", "cesar", "choco", "cordoba",
    "cundinamarca", "guainia", "guaviare", "huila", "la guajira", "magdalena",
    "meta", "narino", "norte de santander", "putumayo", "quindio", "risaralda",
    "san andres", "san andres y providencia", "santander", "sucre", "tolima",
    "valle del cauca", "vaupes", "vichada"
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

    return RouteDecision(intent="general_question", use_llm=True)
