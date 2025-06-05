import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from src.ml_xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap
import io

st.set_page_config(page_title="Dashboard Pricing & Revenue", layout="wide")
st.title("Dashboard de Demanda, Forecast & Interpretable ML")

uploaded_file = st.file_uploader("Carga tu archivo Excel de historial mensual:", type=["xlsx"])
if uploaded_file is None:
    st.info("Por favor sube un archivo Excel. El formato debe contener columnas como 'Proyecto', 'Periodo', 'Total en Ventas', 'Precio Venta (Promedio)', etc.")
    st.stop()

@st.cache_data
def cargar_datos(file):
    df = pd.read_excel(file, sheet_name="Export")
    return df

# 1. Lógica OOP para análisis
class AnalisisDemanda:
    def __init__(self, df):
        self.df = df[df['Periodo'] != "Total"]
        self.proyectos = sorted(self.df['Proyecto'].unique())

    def graficar_demanda(self, proyecto, cantidad_col='Total en Ventas', precio_col='Precio Venta (Promedio)'):
        df_proj = self.df[self.df['Proyecto'] == proyecto].copy()
        df_proj = df_proj.sort_values('Periodo')
        X = np.arange(len(df_proj)).reshape(-1,1)
        y = df_proj[cantidad_col].values
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        fig, ax1 = plt.subplots(figsize=(9,5))
        ax1.set_xlabel('Periodo')
        ax1.set_ylabel('Cantidad demandada', color='tab:blue')
        ax1.plot(df_proj['Periodo'], df_proj[cantidad_col], marker='o', color='tab:blue', label='Cantidad demandada')
        ax1.plot(df_proj['Periodo'], y_pred, '--', color='tab:cyan', label='Tendencia (Regresión)')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        plt.xticks(rotation=45)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Precio Promedio Venta', color='tab:red')
        ax2.plot(df_proj['Periodo'], df_proj[precio_col], marker='x', color='tab:red', label='Precio Promedio Venta')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        plt.title(f'Demanda y Precio - {proyecto}')
        fig.tight_layout()
        plt.grid(True)
        ax1.legend(loc='upper left')
        return fig, df_proj

df = cargar_datos(uploaded_file)
analisis = AnalisisDemanda(df)

proyecto = st.selectbox("Selecciona un proyecto:", analisis.proyectos)

tab1, tab2, tab3, tab4 = st.tabs([
    "Gráfico Demanda y Precio", 
    "Predicción ML & SHAP", 
    "Forecast a Futuro",
    "Insights y Recomendaciones"
])

with tab1:
    st.header("Evolución de Demanda y Precio")
    fig, df_proj = analisis.graficar_demanda(proyecto)
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
    st.header("Modelo XGBoost y Explicabilidad (SHAP)")
    features = ['Precio Venta (Promedio)', 'Precio Prom m2', 'Total SB Sin Firmar']
    dfp = df_proj.dropna(subset=features + ['Total en Ventas']).copy()
    dfp['Periodo_num'] = np.arange(len(dfp))
    X = dfp[features + ['Periodo_num']]
    y = dfp['Total en Ventas']

    if len(dfp) < 5:
        st.warning("No hay suficientes datos para entrenar el modelo ML.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = XGBRegressor(n_estimators=50, max_depth=2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = mean_squared_error(y_test, y_pred, squared=False)
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)

        st.write(f"RMSE en test: {score:.2f}")
        st.bar_chart(feature_importance.set_index('Feature'))

        pred_df = pd.DataFrame({'Real': y_test, 'Predicción': y_pred}, index=y_test.index)
        st.dataframe(pred_df)
        # Botón para descargar resultados ML
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer) as writer:
            pred_df.to_excel(writer, sheet_name="Predicción", index=False)
            feature_importance.to_excel(writer, sheet_name="Importancia", index=False)
        st.download_button(
            label="Descargar resultados ML (Excel)",
            data=excel_buffer.getvalue(),
            file_name=f"resultados_ml_{proyecto}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        # SHAP
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        st.subheader("SHAP - Importancia global")
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        # SHAP para una predicción específica
        st.subheader("SHAP - Explicación individual")
        idx = st.number_input("Selecciona fila para explicación individual (de 0 a N test)", min_value=0, max_value=len(X_test)-1, value=0)
        shap.plots.waterfall(shap_values[idx], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

with tab3:
    st.header("Forecast a futuro (XGBoost)")
    if len(dfp) < 6:
        st.warning("Se requieren al menos 6 periodos para forecast a futuro.")
    else:
        # Forecast usando XGBoost (solo para demo)
        # Entrena en todos los datos y predice los próximos 3 periodos
        model_full = XGBRegressor(n_estimators=50, max_depth=2)
        model_full.fit(X, y)
        last_idx = dfp['Periodo_num'].max()
        future_X = []
        for i in range(1,4):
            # Forecast usando el último valor conocido para features
            row = dfp.iloc[-1][features].values.tolist()
            row.append(last_idx+i)
            future_X.append(row)
        future_X = np.array(future_X)
        future_pred = model_full.predict(future_X)
        future_periods = [f"Futuro {i}" for i in range(1,4)]
        st.write("Pronóstico de demanda para los próximos 3 periodos:")
        forecast_df = pd.DataFrame({'Periodo': future_periods, 'Predicción demanda': future_pred})
        st.dataframe(forecast_df)
        # Gráfico extendido
        fig2, ax = plt.subplots(figsize=(9,5))
        ax.plot(dfp['Periodo'], dfp['Total en Ventas'], marker='o', label='Histórico')
        ax.plot(future_periods, future_pred, marker='x', color='red', label='Forecast')
        plt.xticks(rotation=45)
        plt.title("Demanda histórica y forecast a futuro")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig2)

with tab4:
    st.header("Insights Económicos y Sugerencias Pricing")
    avg_ventas = df_proj['Total en Ventas'].mean()
    # Elasticidad-precio de la demanda (regresión log-log)
    beta = None
    if df_proj.shape[0] > 2:
        try:
            df_proj_posit = df_proj[(df_proj['Precio Venta (Promedio)'] > 0) & (df_proj['Total en Ventas'] > 0)]
            log_p = np.log(df_proj_posit['Precio Venta (Promedio)'])
            log_q = np.log(df_proj_posit['Total en Ventas'])
            beta = np.polyfit(log_p, log_q, 1)[0]
            st.markdown(f"**Elasticidad-precio de la demanda estimada:** {beta:.2f}")
            if abs(beta) < 1:
                st.info("La demanda es inelástica al precio (variaciones de precio afectan poco a la cantidad).")
            else:
                st.info("La demanda es elástica al precio (subidas de precio reducen mucho la cantidad vendida).")
        except Exception as e:
            st.warning("No fue posible calcular elasticidad automáticamente.")
    # Chequeo de outliers
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
    total_ventas = None
    if "Total" in df['Periodo'].values:
        total_row = df[(df['Proyecto'] == proyecto) & (df['Periodo'] == "Total")]
        if not total_row.empty:
            total_ventas = total_row['Total en Ventas'].values[0]
    if df_proj['Total en Ventas'].iloc[-1] < avg_ventas * 0.7:
        st.warning("¡Atención! Última demanda muy baja vs. el promedio. Considera revisar precios, campañas, o mix de producto.")
    if df_proj['Precio Venta (Promedio)'].iloc[-1] > df_proj['Precio Venta (Promedio)'].mean() * 1.2:
        st.info("El precio promedio más reciente está notablemente por encima del promedio anual. Verifica si corresponde a una mejora de mix o reducción de unidades baratas.")
    st.markdown("**Sugerencia PhD:** Para forecasting avanzado, integra modelos híbridos (ARIMA + XGBoost) y testea price ladder por segmentos si tienes el dato. Mantén dashboards exportables y actualiza modelos con cada cierre mensual.")
    # Descarga de insights
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
