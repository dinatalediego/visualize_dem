import pandas as pd
import matplotlib.pyplot as plt

class AnalisisDemanda:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path)
        self.proyectos = sorted(self.df['Proyecto'].unique())
        #self.df = self.df.iloc[:-1]  # Skip the last row

    def graficar_demanda(self, proyecto, cantidad_col='Total en Ventas', precio_col='Precio Venta (Promedio)'):
        df_proj = self.df[self.df['Proyecto'] == proyecto].copy()
        df_proj = df_proj[df_proj['Periodo'] != "Total"]
        df_proj = df_proj.sort_values('Periodo')
        
        fig, ax1 = plt.subplots(figsize=(9,5))
        ax1.set_xlabel('Periodo')
        ax1.set_ylabel('Cantidad demandada', color='tab:blue')
        ax1.plot(df_proj['Periodo'], df_proj[cantidad_col], marker='o', color='tab:blue', label='Cantidad demandada')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        plt.xticks(rotation=45)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Precio Promedio Venta', color='tab:red')
        ax2.plot(df_proj['Periodo'], df_proj[precio_col], marker='x', color='tab:red', label='Precio Promedio Venta')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title(f'Demanda y Precio - {proyecto}')
        fig.tight_layout()
        plt.grid(True)
        return fig
