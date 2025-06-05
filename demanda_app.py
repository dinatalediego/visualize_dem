import streamlit as st
from src.analisis_demanda import AnalisisDemanda  # asumiendo que guardas la clase en analisis_demanda.py

# Ruta al archivo (puedes dejarlo fijo o cargarlo con st.file_uploader)
file_path = "data/HISTORIAL MENSUAL COMAS 2025.xlsx"
analisis = AnalisisDemanda(file_path)

st.title("Dashboard de Demanda y Precio por Proyecto")
proyecto = st.selectbox("Selecciona un proyecto:", analisis.proyectos)

st.write(f"Gráfico para el proyecto: **{proyecto}**")
fig = analisis.graficar_demanda(proyecto)
st.pyplot(fig)
