"""
utils/header.py
Cabeçalho padrão de todas as páginas.
"""
import streamlit as st


def cabecalho(pagina: str):
    """
    Exibe o cabeçalho padrão:
        3F Soluções Ambientais
        <pagina>
    """
    st.markdown(
        """
        <div style="margin-bottom: 4px;">
            <span style="font-size:13px; font-weight:600;
                         color:#2e7d32; letter-spacing:.06em;
                         text-transform:uppercase;">
                3F Soluções Ambientais
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.title(f'3F Soluções Ambientais — {pagina}')
    
    st.divider()
