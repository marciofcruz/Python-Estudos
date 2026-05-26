import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Carregando dados…")
def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Lê o CSV da estação meteorológica e retorna um DataFrame limpo.

    Parâmetros
    ----------
    caminho : str
        Caminho para o arquivo CSV (separador ';', encoding cp1252, decimais ',').

    Retorna
    -------
    pd.DataFrame com coluna DATETIME e tipos numéricos corrigidos.
    """
    df = pd.read_csv(
        caminho,
        sep=";",
        encoding="cp1252",
        decimal=",",
    )

    # ── Renomeia para facilitar acesso ────────────────────────────────────────
    df.rename(columns={
        "Radia\xe7\xe3o Solar": "Radiacao_Solar",   # Radiação Solar
        "Chuva Acumulada":      "Chuva_Acumulada",
        "Press\xe3o_Atm":       "Pressao_Atm",       # Pressão_Atm
    }, inplace=True)

    # ── Combina DATE + TIME em um único campo datetime ────────────────────────
    df["DATETIME"] = pd.to_datetime(
        df["DATE"] + " " + df["TIME"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    # ── Remove linhas sem data válida ─────────────────────────────────────────
    df.dropna(subset=["DATETIME"], inplace=True)

    # ── Ordena cronologicamente ───────────────────────────────────────────────
    df.sort_values("DATETIME", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna estatísticas descritivas das colunas numéricas."""
    cols_num = [
        "Radiacao_Solar", "Temp_solo", "Chuva_Acumulada",
        "Pressao_Atm", "Temperatura.Ar", "Umidade.Relativa",
        "Vel.Vento_REG1", "Dir.Vento_REG1",
    ]
    return df[cols_num].describe().T
