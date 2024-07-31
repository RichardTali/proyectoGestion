import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def cargar_data():
    # Cargar el dataset
    filename = 'dataset/metaverse_transactions_dataset.csv'
    df = pd.read_csv(filename)

    # Seleccionar las características y el objetivo
    X_reg = df[['hour_of_day', 'amount', 'ip_prefix', 'login_frequency', 'session_duration']]
    Y_reg = df['risk_score']
    
    return X_reg, Y_reg

# Error medio absoluto
def error_absoluto():
    test_size = 0.4
    seed = 8
    X_reg, Y_reg = cargar_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=test_size, random_state=seed)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # MAE: Error medio Absoluto
    MAE = mean_absolute_error(Y_test, predicted)
    print("Error Medio Absoluto: {}".format(MAE))

# Error Cuadrático Medio 
def error_cuadratico_medio():
    test_size = 0.4
    seed = 8
    X_reg, Y_reg = cargar_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=test_size, random_state=seed)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # MSE: Error Cuadrático Medio
    MSE = mean_squared_error(Y_test, predicted)
    print("Error Cuadrático Medio: {}".format(MSE))

# Coeficiente de Determinación conocido como R2
def coeficiente_determinacion():
    test_size = 0.4
    seed = 8
    X_reg, Y_reg = cargar_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=test_size, random_state=seed)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # Calculo del coeficiente de determinación
    R2 = r2_score(Y_test, predicted)
    print("Coeficiente de Determinación (R2): {}".format(R2))

# Función para mostrar el menú
def mostrar_menu():
    print("\n--- Menú de Evaluación ---")
    print("1. Error Medio Absoluto")
    print("2. Error Cuadrático Medio")
    print("3. Coeficiente de Determinación (R2)")
    print("4. Salir")
    seleccion = input("Selecciona una opción (1-4): ")
    return seleccion

# Función principal que ejecuta el menú
def ejecutar_menu():
    while True:
        seleccion = mostrar_menu()
        if seleccion == '1':
            error_absoluto()
        elif seleccion == '2':
            error_cuadratico_medio()
        elif seleccion == '3':
            coeficiente_determinacion()
        elif seleccion == '4':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, elige una opción del 1 al 4.")

# Llamar a la función principal
ejecutar_menu()
