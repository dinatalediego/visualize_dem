import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class AnalisisDemanda:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path, sheet_name="Export")
        # Limpia la fila "Total"
        self.df = self.df[self.df['Periodo'] != "Total"]
        self.proyectos = sorted(self.df['Proyecto'].unique())

    def graficar_demanda(self, proyecto, cantidad_col='Total en Ventas', precio_col='Precio Venta (Promedio)'):
        df_proj = self.df[self.df['Proyecto'] == proyecto].copy()
        df_proj = df_proj.sort_values('Periodo')
        # Para regresión, convertir Periodo a números consecutivos
        X = np.arange(len(df_proj)).reshape(-1,1)
        y = df_proj[cantidad_col].values

        # Regresión lineal sobre la cantidad
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
        return fig
