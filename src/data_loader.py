from __future__ import annotations

import unicodedata
from typing import Iterable

import pandas as pd

from .config import get_settings


REQUIRED_COLUMNS = {
    "departamento",
    "indice_oportunidad",
    "nivel",
    "pobreza_n",
    "microcredito_n",
    "productos_n",
    "atm_n",
    "internet_n",
}


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def validate_columns(columns: Iterable[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(
            "El archivo ranking_oportunidad.csv no tiene todas las columnas requeridas. "
            f"Faltan: {sorted(missing)}"
        )


def load_ranking_data() -> pd.DataFrame:
    settings = get_settings()
    path = settings.ranking_file

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {path}. "
            "Verifica que ya hayas ejecutado el notebook 03_indice_oportunidad."
        )

    df = pd.read_csv(path)
    validate_columns(df.columns)

    df = df.copy()
    df["departamento"] = df["departamento"].astype(str).str.strip()
    df["departamento_norm"] = df["departamento"].apply(normalize_text)
    df = df.sort_values("indice_oportunidad", ascending=False).reset_index(drop=True)
    return df
