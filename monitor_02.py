import streamlit as st
import io
import numpy as np
import pandas as pd
from src.demanda import AnalisisDemanda
from src.ml_xgboost import xgboost_prediction

file_path = "data/PROYECTOS RESULTADOS.xlsx"
analisis = AnalisisDemanda(file_path)

st.set_page_config(page_title="Dashboard Pricing & Revenue", layout="wide")
st.title("Dashboard de Demanda, Pricing Analytics & Revenue Management")

proyecto = st.selectbox("Selecciona un proyecto:", analisis.proyectos)

tab1, tab2, tab3 = st.tabs([
    "Gráfico Demanda y Precio", 
    "Predicción con XGBoost", 
    "Insights y Recomendaciones"
])

with tab1:
    st.header("Evolución de Demanda y Precio")
    fig = analisis.graficar_demanda(proyecto)
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="Descargar gráfico como PNG",
        data=buf.getvalue(),
        file_name=f"demanda_{proyecto}.png",
        mime="image/png"
    )

with tab2:
    st.header("Modelo ML y Descarga de Resultados")
    pred_df, feature_imp = xgboost_prediction(analisis.df, proyecto, return_data=True)
    if pred_df is not None and feature_imp is not None:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer) as writer:
            pred_df.to_excel(writer, sheet_name="Predicción", index=False)
            feature_imp.to_excel(writer, sheet_name="Importancia", index=False)
        st.download_button(
            label="Descargar resultados ML (Excel)",
            data=excel_buffer.getvalue(),
            file_name=f"resultados_ml_{proyecto}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with tab3:
    st.header("Insights Económicos y Sugerencias Pricing")
    df_proj = analisis.df[analisis.df['Proyecto'] == proyecto].copy()
    df_proj = df_proj.sort_values('Periodo')

    # Elasticidad-precio de la demanda (regresión log-log)
    beta = None
    if df_proj.shape[0] > 2:
        try:
            df_proj = df_proj[(df_proj['Precio Venta (Promedio)'] > 0) & (df_proj['Total en Ventas'] > 0)]
            log_p = np.log(df_proj['Precio Venta (Promedio)'])
            log_q = np.log(df_proj['Total en Ventas'])
            beta = np.polyfit(log_p, log_q, 1)[0]
            st.markdown(f"**Elasticidad-precio de la demanda estimada:** {beta:.2f}")
            if abs(beta) < 1:
                st.info("La demanda es inelástica al precio (variaciones de precio afectan poco a la cantidad).")
            else:
                st.info("La demanda es elástica al precio (subidas de precio reducen mucho la cantidad vendida).")
        except Exception as e:
            st.warning("No fue posible calcular elasticidad automáticamente.")

    st.markdown("#### Chequeo de outliers (demanda):")
    q1, q3 = df_proj['Total en Ventas'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df_proj[(df_proj['Total en Ventas'] < lower) | (df_proj['Total en Ventas'] > upper)]
    if not outliers.empty:
        st.warning(f"Hay posibles outliers en la demanda en los periodos: {', '.join(outliers['Periodo'].astype(str))}")
    else:
        st.success("No se detectan outliers fuertes en la demanda.")

    df_proj['variacion'] = df_proj['Total en Ventas'].pct_change()
    if df_proj['variacion'].abs().max() > 0.5:
        st.warning("Hay variaciones superiores al 50% en la demanda de un mes a otro.")

    avg_ventas = df_proj['Total en Ventas'].mean()
    total_ventas = None
    if "Total" in analisis.df['Periodo'].values:
        total_row = analisis.df[(analisis.df['Proyecto'] == proyecto) & (analisis.df['Periodo'] == "Total")]
        if not total_row.empty:
            total_ventas = total_row['Total en Ventas'].values[0]

    if df_proj['Total en Ventas'].iloc[-1] < avg_ventas * 0.7:
        st.warning("¡Atención! Última demanda muy baja vs. el promedio. Considera revisar precios, campañas, o mix de producto.")
    if df_proj['Precio Venta (Promedio)'].iloc[-1] > df_proj['Precio Venta (Promedio)'].mean() * 1.2:
        st.info("El precio promedio más reciente está notablemente por encima del promedio anual. Verifica si corresponde a una mejora de mix o reducción de unidades baratas.")

    st.markdown("**Sugerencia PhD:** Para forecasting avanzado, integra modelos híbridos (ARIMA + XGBoost) y testea price ladder por segmentos si tienes el dato. Mantén dashboards exportables y actualiza modelos con cada cierre mensual.")

    # Panel de descargas de todos los insights
    insight_lines = [
        f"Elasticidad estimada: {beta:.2f}" if beta is not None else "Elasticidad no calculada.",
        f"Promedio mensual ventas: {avg_ventas:.1f}",
    ]
    if total_ventas is not None:
        insight_lines.append(f"Total anual: {total_ventas}")
    insight_lines.append("Verifica outliers y shocks de demanda.")

    st.download_button(
        label="Descargar insights (texto)",
        data="\n".join(insight_lines),
        file_name=f"insights_{proyecto}.txt"
    )
