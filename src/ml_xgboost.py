import streamlit as st
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

""" def xgboost_prediction(df, proyecto):
    df_proj = df[df['Proyecto'] == proyecto].copy()
    df_proj = df_proj[df_proj['Periodo'] != "Total"].sort_values('Periodo')
    # Usamos solo filas con todos los datos numéricos completos
    features = ['Precio Venta (Promedio)', 'Precio Prom m2', 'Total SB Sin Firmar']
    df_proj = df_proj.dropna(subset=features + ['Total en Ventas'])
    # Convertir Periodo a int para ML
    df_proj['Periodo_num'] = np.arange(len(df_proj))
    X = df_proj[features + ['Periodo_num']]
    y = df_proj['Total en Ventas']
    
    if len(df_proj) < 5:
        st.warning("No hay suficientes datos para entrenar el modelo ML.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBRegressor(n_estimators=50, max_depth=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred, squared=False)
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)

    st.subheader("Predicción XGBoost")
    st.write(f"RMSE en test: {score:.2f}")
    st.bar_chart(feature_importance.set_index('Feature'))
    st.write("Predicción vs Real (test):")
    pred_df = pd.DataFrame({'Real': y_test, 'Predicción': y_pred}, index=y_test.index)
    st.dataframe(pred_df)
"""

def xgboost_prediction(df, proyecto, return_data=False):
    import pandas as pd
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import streamlit as st

    df_proj = df[df['Proyecto'] == proyecto].copy()
    df_proj = df_proj[df_proj['Periodo'] != "Total"].sort_values('Periodo')
    features = ['Precio Venta (Promedio)', 'Precio Prom m2', 'Total SB Sin Firmar']
    df_proj = df_proj.dropna(subset=features + ['Total en Ventas'])
    df_proj['Periodo_num'] = np.arange(len(df_proj))
    X = df_proj[features + ['Periodo_num']]
    y = df_proj['Total en Ventas']

    if len(df_proj) < 5:
        st.warning("No hay suficientes datos para entrenar el modelo ML.")
        if return_data:
            return None, None
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBRegressor(n_estimators=50, max_depth=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred, squared=False)
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)

    st.subheader("Predicción XGBoost")
    st.write(f"RMSE en test: {score:.2f}")
    st.bar_chart(feature_importance.set_index('Feature'))
    st.write("Predicción vs Real (test):")
    pred_df = pd.DataFrame({'Real': y_test, 'Predicción': y_pred}, index=y_test.index)
    st.dataframe(pred_df)

    # SOLO si se pide retornar los datos
    if return_data:
        return pred_df.reset_index(drop=True), feature_importance.reset_index(drop=True)
