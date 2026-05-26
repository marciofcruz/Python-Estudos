import streamlit as st
from utils.data_loader import carregar_dados
from utils.header import cabecalho
from utils.metricas import cards_metricas

# ── Configuracao global ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="3F Solucoes Ambientais",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("assets/style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Dados ─────────────────────────────────────────────────────────────────────
df = carregar_dados("MemFlash.csv")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 3F Solucoes Ambientais")
    st.caption("Estacao Meteorologica")
    st.divider()

    data_min = df["DATETIME"].dt.date.min()
    data_max = df["DATETIME"].dt.date.max()

    intervalo = st.date_input(
        "Periodo",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max,
    )

    st.divider()
    st.caption(f"De:  {data_min.strftime('%d/%m/%Y')}")
    st.caption(f"Ate: {data_max.strftime('%d/%m/%Y')}")
    st.caption(f"{len(df):,} registros")

# ── Filtro ────────────────────────────────────────────────────────────────────
if len(intervalo) == 2:
    inicio, fim = intervalo
    df_filtrado = df[
        (df["DATETIME"].dt.date >= inicio) &
        (df["DATETIME"].dt.date <= fim)
    ]
else:
    df_filtrado = df

st.session_state["df"] = df_filtrado

# ── Conteudo ──────────────────────────────────────────────────────────────────
cabecalho("Estacao Meteorologica — Visao Geral")

st.markdown(
    f"Exibindo **{len(df_filtrado):,}** registros de "
    f"`{df_filtrado['DATETIME'].min().strftime('%d/%m/%Y %H:%M')}` "
    f"ate `{df_filtrado['DATETIME'].max().strftime('%d/%m/%Y %H:%M')}`"
)

st.subheader("Resumo do periodo")
cards_metricas(df_filtrado)

st.divider()
st.info("Use o menu lateral para navegar entre as paginas de analise.")
