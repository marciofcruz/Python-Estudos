import streamlit as st
from utils.header import cabecalho

cabecalho("Dados Brutos")

df = st.session_state.get("df")
if df is None:
    st.warning("Volte à página inicial para carregar os dados.")
    st.stop()

st.caption(f"{len(df):,} registros no período selecionado")

colunas = st.multiselect(
    "Colunas visíveis",
    options=[c for c in df.columns if c not in ("DATE", "TIME")],
    default=["DATETIME", "Temperatura.Ar", "Umidade.Relativa",
             "Radiacao_Solar", "Pressao_Atm", "Vel.Vento_REG1"],
)

st.dataframe(
    df[colunas].set_index("DATETIME"),
    use_container_width=True,
    height=500,
)

st.download_button(
    label="Baixar CSV filtrado",
    data=df[colunas].to_csv(index=False, sep=";", decimal=",").encode("utf-8"),
    file_name="meteo_filtrado.csv",
    mime="text/csv",
)
