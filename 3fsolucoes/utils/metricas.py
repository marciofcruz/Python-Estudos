"""
utils/metricas.py
Cards de metricas usando st.components.v1.html()
que renderiza em iframe e nao sofre sanitizacao do Streamlit.
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd


def _card_html(icone: str, label: str, unidade: str,
               minv: float, medv: float, maxv: float,
               fmt: str, cor: str) -> str:
    return f"""
    <div style="flex:1; min-width:180px; background:#ffffff;
                border-radius:12px; border-left:5px solid {cor};
                padding:14px 12px;
                box-shadow:0 2px 8px rgba(0,0,0,0.10);">
        <div style="font-size:20px;">{icone}</div>
        <div style="font-size:11px; font-weight:700; color:#555;
                    text-transform:uppercase; letter-spacing:.05em;
                    margin:4px 0 10px 0;">{label}</div>
        <div style="display:flex; gap:6px;">
            <div style="flex:1; text-align:center; background:#f0f5ff;
                        border-radius:8px; padding:6px 2px;">
                <div style="font-size:9px; color:#999; font-weight:700;">MIN</div>
                <div style="font-size:15px; font-weight:700; color:#1565c0;">{minv:{fmt}}</div>
                <div style="font-size:9px; color:#aaa;">{unidade}</div>
            </div>
            <div style="flex:1; text-align:center; background:#f0fff4;
                        border-radius:8px; padding:6px 2px;
                        border:1px solid {cor}55;">
                <div style="font-size:9px; color:#999; font-weight:700;">MEDIA</div>
                <div style="font-size:18px; font-weight:800; color:{cor};">{medv:{fmt}}</div>
                <div style="font-size:9px; color:#aaa;">{unidade}</div>
            </div>
            <div style="flex:1; text-align:center; background:#fff0f0;
                        border-radius:8px; padding:6px 2px;">
                <div style="font-size:9px; color:#999; font-weight:700;">MAX</div>
                <div style="font-size:15px; font-weight:700; color:#c62828;">{maxv:{fmt}}</div>
                <div style="font-size:9px; color:#aaa;">{unidade}</div>
            </div>
        </div>
    </div>"""


def cards_metricas(df: pd.DataFrame):
    """Exibe grid 4x2 de cards min/media/max via iframe (components.html)."""

    variaveis = [
        ("Temperatura.Ar",   "Temp. do Ar",      "C",     ".1f", "#e53935", "🌡️"),
        ("Temp_solo",        "Temp. do Solo",     "C",     ".1f", "#fb8c00", "🌱"),
        ("Umidade.Relativa", "Umidade Relativa",  "%",     ".1f", "#1e88e5", "💧"),
        ("Radiacao_Solar",   "Radiacao Solar",    "W/m2",  ".0f", "#f9a825", "☀️"),
        ("Pressao_Atm",      "Pressao Atm.",      "hPa",   ".1f", "#6a1b9a", "🔵"),
        ("Vel.Vento_REG1",   "Vel. do Vento",     "m/s",   ".2f", "#00897b", "💨"),
        ("Dir.Vento_REG1",   "Direcao do Vento",  "graus", ".0f", "#546e7a", "🧭"),
        ("Chuva_Acumulada",  "Chuva Acumulada",   "mm",    ".1f", "#1565c0", "🌧️"),
    ]

    # monta todas as linhas de cards
    linhas_html = ""
    for i in range(0, len(variaveis), 4):
        linha = variaveis[i:i+4]
        cards = "".join(
            _card_html(icone, label, unidade,
                       df[campo].dropna().min(),
                       df[campo].dropna().mean(),
                       df[campo].dropna().max(),
                       fmt, cor)
            for campo, label, unidade, fmt, cor, icone in linha
        )
        linhas_html += f"""
        <div style="display:flex; gap:12px; margin-bottom:14px; flex-wrap:wrap;">
            {cards}
        </div>"""

    html_completo = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ margin:0; padding:4px; font-family: 'Trebuchet MS', sans-serif;
                   background:transparent; }}
        </style>
    </head>
    <body>
        {linhas_html}
    </body>
    </html>"""

    components.html(html_completo, height=340, scrolling=False)
