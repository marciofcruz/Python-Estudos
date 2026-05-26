import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from utils.header import cabecalho
from utils.metricas import cards_metricas
from utils.graficos import LAYOUT_BASE, resample

cabecalho("Vento e Chuva")

df = st.session_state.get("df")
if df is None:
    st.warning("Volte a pagina inicial para carregar os dados.")
    st.stop()

# ── Controle de resolucao ─────────────────────────────────────────────────────
freq = st.radio(
    "Resolucao temporal",
    options=["10min", "1h", "1D"],
    format_func=lambda x: {"10min": "10 minutos", "1h": "Horaria", "1D": "Diaria"}[x],
    horizontal=True,
)

# ════════════════════════════════════════════════════════════════════════
# LINHA 1 — Rosa dos Ventos  |  Velocidade do Vento
# ════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Rosa dos Ventos")

    # 16 setores de 22.5 graus — frequencia e velocidade media por setor
    n_setores = 16
    largura   = 360 / n_setores          # 22.5 graus
    labels    = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                 "S","SSO","SO","OSO","O","ONO","NO","NNO"]

    df_r = df[["Dir.Vento_REG1", "Vel.Vento_REG1"]].dropna().copy()
    df_r["setor"] = (df_r["Dir.Vento_REG1"] / largura).astype(int) % n_setores
    df_r["label"] = df_r["setor"].map(dict(enumerate(labels)))

    por_setor = df_r.groupby("label", sort=False).agg(
        freq=("Vel.Vento_REG1", "count"),
        vel_media=("Vel.Vento_REG1", "mean"),
    ).reindex(labels).fillna(0)

    # frequencia relativa em %
    por_setor["freq_pct"] = por_setor["freq"] / por_setor["freq"].sum() * 100

    # barras coloridas por velocidade media
    fig_rosa = go.Figure()
    fig_rosa.add_trace(go.Barpolar(
        r        = por_setor["freq_pct"].values,
        theta    = labels,
        width    = [largura] * n_setores,
        marker   = dict(
            color      = por_setor["vel_media"].values,
            colorscale = "RdYlBu_r",
            showscale  = True,
            colorbar   = dict(
                title      = "m/s",
                thickness  = 12,
                len        = 0.6,
                tickformat = ".1f",
            ),
            line = dict(color="white", width=0.5),
        ),
        hovertemplate = (
            "<b>%{theta}</b><br>"
            "Frequencia: %{r:.1f}%<br>"
            "Vel. Media: %{marker.color:.2f} m/s"
            "<extra></extra>"
        ),
    ))

    fig_rosa.update_layout(
        height      = 420,
        margin      = dict(t=10, b=10, l=10, r=10),
        paper_bgcolor = "#ffffff",
        polar       = dict(
            bgcolor     = "#f8f9fb",
            angularaxis = dict(
                tickmode   = "array",
                tickvals   = labels,
                ticktext   = labels,
                direction  = "clockwise",
                rotation   = 90,
                gridcolor  = "#cccccc",
            ),
            radialaxis  = dict(
                ticksuffix = "%",
                gridcolor  = "#cccccc",
                tickfont   = dict(size=9),
            ),
        ),
    )
    st.plotly_chart(fig_rosa, use_container_width=True)

with col2:
    st.subheader("Velocidade e Direcao do Vento")

    vel = resample(df, "Vel.Vento_REG1", freq)
    dir_ = resample(df, "Dir.Vento_REG1", freq)

    fig_vel = make_subplots(specs=[[{"secondary_y": True}]])

    fig_vel.add_trace(go.Scatter(
        x=vel.index, y=vel,
        name="Vel. Vento (m/s)",
        fill="tozeroy",
        fillcolor="rgba(30,136,229,0.15)",
        line=dict(color="#1e88e5", width=1.8),
        hovertemplate="Vento: <b>%{y:.2f} m/s</b><extra></extra>",
    ), secondary_y=False)

    fig_vel.add_trace(go.Scatter(
        x=dir_.index, y=dir_,
        name="Direcao (graus)",
        mode="markers",
        marker=dict(size=3, color="#fb8c00", opacity=0.6),
        hovertemplate="Direcao: <b>%{y:.0f}°</b><extra></extra>",
    ), secondary_y=True)

    layout = {**LAYOUT_BASE,
              "title": "Velocidade e Direcao do Vento",
              "height": 420,
              "legend": dict(orientation="h", y=-0.28)}
    fig_vel.update_layout(**layout)
    fig_vel.update_yaxes(title_text="Velocidade (m/s)", secondary_y=False,
                         gridcolor="#e0e0e0")
    fig_vel.update_yaxes(title_text="Direcao (graus)", secondary_y=True,
                         range=[0, 360], showgrid=False)

    st.plotly_chart(fig_vel, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# LINHA 2 — Chuva: acumulada + incremental por dia
# ════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Precipitacao")

col3, col4 = st.columns(2)

with col3:
    # chuva acumulada (serie continua)
    chuva_acc = resample(df, "Chuva_Acumulada", freq)
    fig_acc = go.Figure(go.Scatter(
        x=chuva_acc.index, y=chuva_acc,
        name="Acumulada (mm)",
        line=dict(color="#1565c0", width=2),
        fill="tozeroy",
        fillcolor="rgba(21,101,192,0.12)",
        hovertemplate="Acumulada: <b>%{y:.1f} mm</b><extra></extra>",
    ))
    layout_acc = {**LAYOUT_BASE, "title": "Chuva Acumulada", "height": 340}
    fig_acc.update_layout(**layout_acc)
    fig_acc.update_yaxes(title_text="mm")
    st.plotly_chart(fig_acc, use_container_width=True)

with col4:
    # incremento diario (diferenca entre o maximo do dia e do dia anterior)
    df_diario = (
        df.set_index("DATETIME")["Chuva_Acumulada"]
        .resample("1D").max()
        .diff()
        .clip(lower=0)          # descarta resets do pluviometro
        .fillna(0)
    )
    fig_inc = go.Figure(go.Bar(
        x=df_diario.index, y=df_diario.values,
        name="Precipitacao diaria (mm)",
        marker_color="#00897b",
        hovertemplate="Data: %{x|%d/%m}<br>Chuva: <b>%{y:.1f} mm</b><extra></extra>",
    ))
    layout_inc = {**LAYOUT_BASE,
                  "title": "Precipitacao Diaria",
                  "height": 340}
    layout_inc["xaxis"] = dict(showgrid=True, gridcolor="#e0e0e0")
    fig_inc.update_layout(**layout_inc)
    fig_inc.update_yaxes(title_text="mm/dia")
    st.plotly_chart(fig_inc, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# LINHA 3 — Cards de metricas
# ════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Resumo do periodo")
cards_metricas(df[["DATETIME", "Temperatura.Ar", "Temp_solo", "Umidade.Relativa",
                    "Radiacao_Solar", "Pressao_Atm", "Vel.Vento_REG1",
                    "Dir.Vento_REG1", "Chuva_Acumulada"]])
