# Importar las librerías 
import numpy as np
import pandas as pd 
import seaborn as sns  
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Cargar la base 
filename = 'dataset/metaverse_transactions_dataset.csv'
data = pd.read_csv(filename)

# Función para calcular el sesgo en columnas numéricas
def sesgo(data):
    numeric_data = data.select_dtypes(include=['number'])
    result = numeric_data.skew()
    print("\nSesgo en columnas numéricas:")
    print(result)
    print("\nhour_of_day: -0.005089: El sesgo es muy cercano a cero, lo que sugiere que la distribución de esta columna es casi simétrica.")
    print("amount: 0.124223: Un sesgo positivo, aunque pequeño, lo que indica que la distribución está ligeramente sesgada hacia la derecha.")
    print("ip_prefix: -1.438522: Un sesgo negativo significativo, sugiriendo que la distribución está sesgada hacia la izquierda.")
    print("login_frequency: 0.174246: Un sesgo positivo leve, indicando que la distribución está ligeramente sesgada hacia la derecha.")
    print("session_duration: 0.660789: Un sesgo positivo notable, sugiriendo que la distribución está sesgada hacia la derecha.")
    print("risk_score: 1.047827: Un sesgo positivo significativo, lo que indica una distribución muy sesgada hacia la derecha.")

# Ejercicio de correlación
def matriz_correlacion(data):
    # Seleccionar solo las columnas numéricas
    numeric_data = data.select_dtypes(include=['number'])
    # Calcular la matriz de correlación
    correlacion = numeric_data.corr(method='pearson')
    # Mostrar la matriz de correlación en la consola
    print("\nMatriz de Correlación: ")
    print(correlacion)
    print("\nLa correlación más fuerte se observa entre login_frequency y session_duration (0.871915). Esto sugiere que los usuarios que inician sesión con más frecuencia tienden a tener sesiones más largas.")
    print("Existe una correlación moderada negativa entre hour_of_day y risk_score (-0.190985). Esto podría indicar que el riesgo tiende a ser ligeramente menor en horas más avanzadas del día.")
    print("La mayoría de las otras variables muestran correlaciones muy débiles entre sí (valores cercanos a 0), lo que indica que no hay una relación lineal fuerte entre ellas.")
    print("La fuerte correlación entre login_frequency y session_duration podría ser útil para entender el comportamiento del usuario, aunque no parece estar fuertemente relacionada con el riesgo.")

# Distribución entre clases 
def distribution_in_classes():
    class_counts = data.groupby('anomaly').size()
    print("\nDistribución entre clases:")
    print(class_counts)

# Resumen general del dataset
def summary():
    print("\nResumen general del dataset:")
    print(data.describe())

# Imprimir los tipos de datos
def print_data_types():
    print("\nTipos de datos:")
    print(data.dtypes)
    
# Imprimir la dimensión del dataframe
def print_dimensions():
    print("\nDimensión del dataframe:")
    print(data.shape)

# Menú interactivo
def menu():
    while True:
        print("\nMenú de opciones:")
        print("1. Lectura de datos")
        print("2. Cálculo del sesgo en columnas numéricas")
        print("3. Matriz de correlación")
        print("4. Distribución entre clases")
        print("5. Resumen general del dataset")
        print("6. Tipos de datos")
        print("7. Dimensión del dataframe")
        print("0. Salir")
        
        choice = input("Seleccione una opción: ")
        
        if choice == '1':
            print("\nLectura de datos:")
            print(data)
        elif choice == '2':
            sesgo(data)
        elif choice == '3':
            matriz_correlacion(data)
        elif choice == '4':
            distribution_in_classes()
        elif choice == '5':
            summary()
        elif choice == '6':
            print_data_types()
        elif choice == '7':
            print_dimensions()
        elif choice == '0':
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Por favor, seleccione una opción del menú.")

# Ejecutar el menú
menu()
