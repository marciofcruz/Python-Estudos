import streamlit as st
from utils.header import cabecalho
from utils.graficos import fig_radiacao
from utils.metricas import cards_metricas

cabecalho("Radiacao Solar e Pressao Atmosferica")

df = st.session_state.get("df")
if df is None:
    st.warning("Volte a pagina inicial para carregar os dados.")
    st.stop()

freq = st.radio(
    "Resolucao temporal",
    options=["10min", "1h", "1D"],
    format_func=lambda x: {"10min": "10 minutos", "1h": "Horaria", "1D": "Diaria"}[x],
    horizontal=True,
)

st.plotly_chart(fig_radiacao(df, freq), use_container_width=True)

st.divider()
st.subheader("Resumo do periodo")
cards_metricas(df[["DATETIME", "Temperatura.Ar", "Temp_solo", "Umidade.Relativa",
                    "Radiacao_Solar", "Pressao_Atm", "Vel.Vento_REG1",
                    "Dir.Vento_REG1", "Chuva_Acumulada"]])
