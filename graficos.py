# Importar las librerías
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix

# Cargar los datos
filename = 'dataset/metaverse_transactions_dataset.csv'
data = pd.read_csv(filename)

# Función matriz densidad
def matriz_densidad():
    # Código básico para gráficar
    figura = plt.figure(figsize=(6,6))
    
    # Código básico para gráficar una matriz de dispersión
    data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)

    # Agregar un título general a la figura
    figura.suptitle("MATRIZ DE DENSIDAD", fontsize=16)

    # Mostrar la figura
    plt.show()

# Función diagrama de caja
def diagrama_caja():
    # Código básico para gráficar
    figura = plt.figure(figsize=(8,8))

    # Código para gráfica una matriz de un diagrama de caja
    data.plot(kind='box', subplots=True, layout=(3,3), sharex=False)

    # Agregar un título general a la figura
    figura.suptitle("DIAGRAMA DE CAJA", fontsize=16)
    
    # Mostrar la figura
    plt.show()

# Función para generar la matriz de correlación
def matriz_correlacion():
    datos_numericos = data.select_dtypes(include=[np.number])
    correlacion = datos_numericos.corr(method='pearson')
    plt.figure(figsize=(6,6))
    plt.title('Matriz de Correlación')
    sns.heatmap(correlacion, vmax=1, square=True, annot=True, cmap='viridis')
    plt.show()    

# Función para matriz de dispersión
def matriz_dispersion():
    plt.rcParams['figure.figsize'] = (15,15)
    scatter_matrix(data)
    plt.show()

# Menú de opciones
def menu():
    while True:
        print("\n--- MENÚ DE OPCIONES ---")
        print("1. Matriz de Densidad")
        print("2. Diagrama de Caja")
        print("3. Matriz de Correlación")
        print("4. Matriz de Dispersión")
        print("5. Salir")
        
        opcion = input("Selecciona una opción (1-5): ")
        
        if opcion == '1':
            matriz_densidad()
        elif opcion == '2':
            diagrama_caja()
        elif opcion == '3':
            matriz_correlacion()
        elif opcion == '4':
            matriz_dispersion()
        elif opcion == '5':
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")

# Ejecutar el menú
menu()
