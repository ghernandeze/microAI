from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.agent_tools import run_agent
from src.chat_agente import handle_local_fallback
from src.data_loader import load_ranking_data
from src.intent_router import route_query
from src.llm_client import LLMClient

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agente Microcrédito Colombia",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    try:
        st.session_state.df = load_ranking_data()
    except FileNotFoundError:
        st.error(
            "No se encontró ranking_oportunidad.csv. "
            "Ejecuta los notebooks 02 y 03 antes de correr la app."
        )
        st.stop()

if "llm" not in st.session_state:
    st.session_state.llm = LLMClient()

df: pd.DataFrame = st.session_state.df
llm: LLMClient = st.session_state.llm

# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate_upload(file) -> None:
    try:
        tmp = pd.read_excel(file)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        return

    cols_lower = [c.lower().strip() for c in tmp.columns]
    has_dep = any("depart" in c or "dpto" in c for c in cols_lower)
    numeric_cols = tmp.select_dtypes(include="number").columns.tolist()

    if not has_dep:
        st.error("El archivo no tiene columna de departamento.")
        return
    if not numeric_cols:
        st.error("El archivo no tiene columnas numéricas.")
        return

    known = set(df["departamento"].str.upper())
    uploaded_deps = set(tmp.iloc[:, 0].astype(str).str.upper())
    unrecognized = uploaded_deps - known

    st.success(f"Archivo válido · {len(tmp)} filas · indicadores: {', '.join(numeric_cols)}")
    if unrecognized:
        st.warning(
            f"Departamentos no reconocidos (serán ignorados): {', '.join(sorted(unrecognized))}"
        )
    st.info(
        "Para incluir estos datos en el análisis coloca el archivo en data/raw/ "
        "y re-ejecuta los notebooks 02 y 03."
    )


def _render_map() -> None:
    map_path = Path("reports/mapa_oportunidad.html")
    if map_path.exists():
        components.html(map_path.read_text(encoding="utf-8"), height=520, scrolling=False)
    else:
        st.warning("El archivo del mapa no existe. Pide al agente que lo genere primero.")


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("has_map"):
            _render_map()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Microcrédito Digital")
    st.caption("Agente de priorización geográfica · Colombia 2024")
    st.divider()

    if llm.is_enabled():
        st.success("LLM activo (tool calling)")
    else:
        st.warning("Modo local (sin LLM)")
        st.caption("Define MODEL_API_URL y MODEL_API_KEY en .env para activar el LLM.")

    st.divider()
    st.subheader("Cargar nuevos datos")
    uploaded = st.file_uploader("Sube un Excel (.xlsx)", type=["xlsx"])
    if uploaded:
        _validate_upload(uploaded)

    st.divider()
    st.caption("Ejemplos: top 5 · mapa · explícame Chocó · dale más peso a pobreza · ¿qué está pasando en La Guajira?")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("Agente de Priorización Geográfica de Microcrédito")

# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    _render_message(msg)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Escribe tu consulta aquí…")
if not user_input:
    st.stop()

user_input = user_input.strip()
st.session_state.messages.append({"role": "user", "content": user_input})
with st.chat_message("user"):
    st.markdown(user_input)

# ── Routing ───────────────────────────────────────────────────────────────────
decision = route_query(user_input)

if decision.intent == "exit":
    reply = "Sesión finalizada."
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.stop()

has_map = False

try:
    if llm.is_enabled():
        history = [
            m for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]
        reply, has_map = run_agent(user_input, df, llm, history=history)
    else:
        reply = handle_local_fallback(df, decision, user_input)
except Exception as e:
    reply = f"Ocurrió un error: {e}"

# ── Show response ─────────────────────────────────────────────────────────────
st.session_state.messages.append({"role": "assistant", "content": reply, "has_map": has_map})
with st.chat_message("assistant"):
    st.markdown(reply)
    if has_map:
        _render_map()
