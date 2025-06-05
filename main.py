import streamlit as st
from src.demanda import AnalisisDemanda
from src.ml_xgboost import xgboost_prediction
#from ml_xgboost import xgboost_prediction  # el código de arriba, guárdalo en ml_xgboost.py

file_path = "HISTORIAL MENSUAL COMAS 2025.xlsx"
analisis = AnalisisDemanda(file_path)

st.title("Dashboard de Demanda y ML XGBoost")

proyecto = st.selectbox("Selecciona un proyecto:", analisis.proyectos)

tab1, tab2 = st.tabs(["Gráfico Demanda y Precio", "Predicción con XGBoost"])

with tab1:
    st.write(f"Gráfico para el proyecto: **{proyecto}**")
    fig = analisis.graficar_demanda(proyecto)
    st.pyplot(fig)

with tab2:
    xgboost_prediction(analisis.df, proyecto)
