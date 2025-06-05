import streamlit as st
import io
import numpy as np
import pandas as pd
from src.demanda import AnalisisDemanda
from src.ml_xgboost import xgboost_prediction

class Dashboard:
    def __init__(self, file_path):
        self.analisis = AnalisisDemanda(file_path)
        st.set_page_config(page_title="Dashboard Pricing & Revenue", layout="wide")
        self.proyecto = st.selectbox("Selecciona un proyecto:", self.analisis.proyectos)

    def run(self):
        st.title("Dashboard de Demanda, Pricing Analytics & Revenue Management")
        tab1, tab2, tab3 = st.tabs([
            "Gráfico Demanda y Precio", 
            "Predicción con XGBoost", 
            "Insights y Recomendaciones"
        ])

        self.pagina_grafico(tab1)
        self.pagina_prediccion(tab2)
        self.pagina_insights(tab3)

    def pagina_grafico(self, tab):
        with tab:
            st.header("Evolución de Demanda y Precio")
            fig = self.analisis.graficar_demanda(self.proyecto)
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="Descargar gráfico como PNG",
                data=buf.getvalue(),
                file_name=f"demanda_{self.proyecto}.png",
                mime="image/png"
            )

    def pagina_prediccion(self, tab):
        with tab:
            st.header("Modelo ML y Descarga de Resultados")
            pred_df, feature_imp = xgboost_prediction(self.analisis.df, self.proyecto, return_data=True)
            if pred_df is not None and feature_imp is not None:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer) as writer:
                    pred_df.to_excel(writer, sheet_name="Predicción", index=False)
                    feature_imp.to_excel(writer, sheet_name="Importancia", index=False)
                st.download_button(
                    label="Descargar resultados ML (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name=f"resultados_ml_{self.proyecto}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    def pagina_insights(self, tab):
        with tab:
            st.header("Insights Económicos y Sugerencias Pricing")
            df_proj = self.analisis.df[self.analisis.df['Proyecto'] == self.proyecto].copy()
            df_proj = df_proj.sort_values('Periodo')

            insights = []

            beta = self.calcular_elasticidad(df_proj)
            if beta is not None:
                insights.append(f"Elasticidad estimada: {beta:.2f}")

            avg_ventas, total_ventas = self.analizar_ventas(df_proj)
            insights.append(f"Promedio mensual ventas: {avg_ventas:.1f}")
            if total_ventas is not None:
                insights.append(f"Total anual: {total_ventas}")

            self.chequear_outliers(df_proj)
            self.chequear_variacion(df_proj, avg_ventas)

            st.markdown("**Sugerencia PhD:** Para forecasting avanzado, integra modelos híbridos (ARIMA + XGBoost) y testea price ladder por segmentos si tienes el dato.")

            insights.append("Verifica outliers y shocks de demanda.")
            st.download_button(
                label="Descargar insights (texto)",
                data="\n".join(insights),
                file_name=f"insights_{self.proyecto}.txt"
            )

    def calcular_elasticidad(self, df):
        if df.shape[0] > 2:
            try:
                df = df[(df['Precio Venta (Promedio)'] > 0) & (df['Total en Ventas'] > 0)]
                log_p = np.log(df['Precio Venta (Promedio)'])
                log_q = np.log(df['Total en Ventas'])
                beta = np.polyfit(log_p, log_q, 1)[0]
                st.markdown(f"**Elasticidad-precio de la demanda estimada:** {beta:.2f}")
                elasticity_msg = "inelástica" if abs(beta) < 1 else "elástica"
                st.info(f"La demanda es {elasticity_msg} al precio.")
                return beta
            except Exception as e:
                st.warning("No fue posible calcular elasticidad automáticamente.")
        return None

    def analizar_ventas(self, df):
        avg_ventas = df['Total en Ventas'].mean()
        total_ventas = None
        if "Total" in self.analisis.df['Periodo'].values:
            total_row = self.analisis.df[(self.analisis.df['Proyecto'] == self.proyecto) & (self.analisis.df['Periodo'] == "Total")]
            if not total_row.empty:
                total_ventas = total_row['Total en Ventas'].values[0]
        return avg_ventas, total_ventas

    def chequear_outliers(self, df):
        q1, q3 = df['Total en Ventas'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df['Total en Ventas'] < lower) | (df['Total en Ventas'] > upper)]
        if not outliers.empty:
            st.warning(f"Hay posibles outliers en: {', '.join(outliers['Periodo'].astype(str))}")
        else:
            st.success("No se detectan outliers fuertes.")

    def chequear_variacion(self, df, avg_ventas):
        df['variacion'] = df['Total en Ventas'].pct_change()
        if df['variacion'].abs().max() > 0.5:
            st.warning("Variaciones superiores al 50% detectadas.")
        if df['Total en Ventas'].iloc[-1] < avg_ventas * 0.7:
            st.warning("Última demanda muy baja vs. promedio.")
        if df['Precio Venta (Promedio)'].iloc[-1] > df['Precio Venta (Promedio)'].mean() * 1.2:
            st.info("Precio promedio reciente notablemente alto.")

if __name__ == "__main__":
    dashboard = Dashboard(file_path="data/PROYECTOS RESULTADOS.xlsx")
    dashboard.run()
