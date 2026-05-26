"""
utils/graficos.py
Funções reutilizáveis de série temporal com Plotly.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


# ── Layout padrão compartilhado ───────────────────────────────────────────────
LAYOUT_BASE = dict(
    hovermode="x unified",
    plot_bgcolor="#f8f9fb",
    paper_bgcolor="#ffffff",
    font=dict(family="Trebuchet MS", size=13),
    legend=dict(orientation="h", y=-0.25, x=0),
    margin=dict(t=50, b=60, l=60, r=30),
    xaxis=dict(
        showgrid=True,
        gridcolor="#e0e0e0",
        rangeslider=dict(visible=True, thickness=0.06),
        rangeselector=dict(
            buttons=[
                dict(count=1,  label="1d",  step="day",  stepmode="backward"),
                dict(count=7,  label="7d",  step="day",  stepmode="backward"),
                dict(count=14, label="14d", step="day",  stepmode="backward"),
                dict(step="all", label="Tudo"),
            ],
            bgcolor="#e8eaf6",
            activecolor="#3949ab",
        ),
    ),
    yaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
)


def _aplicar_layout(fig, titulo: str, altura: int = 480, **kwargs):
    layout = {**LAYOUT_BASE, "title": titulo, "height": altura, **kwargs}
    fig.update_layout(**layout)
    return fig


def resample(df: pd.DataFrame, coluna: str, freq: str) -> pd.Series:
    """Reamostra uma coluna por frequência ('10min', '1h', '1D')."""
    return df.set_index("DATETIME")[coluna].resample(freq).mean()


# ── Página 1: Temperatura & Umidade ──────────────────────────────────────────
def fig_temperatura(df: pd.DataFrame, freq: str = "10min") -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    tar  = resample(df, "Temperatura.Ar",   freq)
    tsol = resample(df, "Temp_solo",        freq)
    umi  = resample(df, "Umidade.Relativa", freq)

    fig.add_trace(go.Scatter(
        x=tar.index, y=tar,
        name="Temp. Ar (°C)",
        line=dict(color="#e53935", width=1.8),
        hovertemplate="Temp. Ar: <b>%{y:.1f} °C</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=tsol.index, y=tsol,
        name="Temp. Solo (°C)",
        line=dict(color="#fb8c00", width=1.4, dash="dot"),
        hovertemplate="Temp. Solo: <b>%{y:.1f} °C</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=umi.index, y=umi,
        name="Umidade (%)",
        line=dict(color="#1e88e5", width=1.8),
        opacity=0.85,
        hovertemplate="Umidade: <b>%{y:.1f} %</b><extra></extra>",
    ), secondary_y=True)

    _aplicar_layout(fig, "Temperatura do Ar, Solo e Umidade Relativa")
    fig.update_yaxes(title_text="Temperatura (°C)", secondary_y=False,
                     gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Umidade (%)", secondary_y=True,
                     showgrid=False)
    return fig


# ── Página 2: Radiação Solar & Pressão Atmosférica ───────────────────────────
def fig_radiacao(df: pd.DataFrame, freq: str = "10min") -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    rad  = resample(df, "Radiacao_Solar", freq)
    pres = resample(df, "Pressao_Atm",   freq)

    fig.add_trace(go.Scatter(
        x=rad.index, y=rad,
        name="Radiação Solar (W/m²)",
        fill="tozeroy",
        fillcolor="rgba(255,193,7,0.18)",
        line=dict(color="#f9a825", width=1.8),
        hovertemplate="Radiação: <b>%{y:.0f} W/m²</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=pres.index, y=pres,
        name="Pressão Atm. (hPa)",
        line=dict(color="#6a1b9a", width=1.8),
        hovertemplate="Pressão: <b>%{y:.2f} hPa</b><extra></extra>",
    ), secondary_y=True)

    _aplicar_layout(fig, "Radiação Solar e Pressão Atmosférica")
    fig.update_yaxes(title_text="Radiação (W/m²)", secondary_y=False,
                     gridcolor="#e0e0e0")
    fig.update_yaxes(title_text="Pressão (hPa)", secondary_y=True,
                     showgrid=False)
    return fig


# ── Página 3: Velocidade do Vento ────────────────────────────────────────────
def fig_vento(df: pd.DataFrame, freq: str = "10min") -> go.Figure:
    fig = go.Figure()

    vel = resample(df, "Vel.Vento_REG1", freq)

    fig.add_trace(go.Scatter(
        x=vel.index, y=vel,
        name="Vel. Vento (m/s)",
        fill="tozeroy",
        fillcolor="rgba(30,136,229,0.15)",
        line=dict(color="#1e88e5", width=1.6),
        hovertemplate="Vento: <b>%{y:.2f} m/s</b><extra></extra>",
    ))

    _aplicar_layout(fig, "Velocidade do Vento")
    fig.update_yaxes(title_text="m/s")
    return fig


# ── Página 3: Chuva Acumulada ────────────────────────────────────────────────
def fig_chuva(df: pd.DataFrame, freq: str = "10min") -> go.Figure:
    fig = go.Figure()

    chuva = resample(df, "Chuva_Acumulada", freq)

    fig.add_trace(go.Scatter(
        x=chuva.index, y=chuva,
        name="Chuva Acumulada (mm)",
        line=dict(color="#00897b", width=2),
        hovertemplate="Chuva: <b>%{y:.1f} mm</b><extra></extra>",
    ))

    _aplicar_layout(fig, "Chuva Acumulada", altura=360)
    fig.update_yaxes(title_text="mm")
    return fig
