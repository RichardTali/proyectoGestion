# Proyecto Final Gestión

Breve descripción de lo que hace tu proyecto.

## Requisitos

Lista de requerimientos necesarios para ejecutar tu proyecto.

- Python 3.x
- Pandas
- Otros paquetes necesarios

## Instituto Quito

![Instituto Quito](images/descarga.png)

## Análisis Dataset

### Sesgo

**SESGO**

![Sesgo 1](images/sesgo1.jpg)
![Sesgo 2](images/sesgo2.jpg)

### MATRIZ DE CORRELACIÓN

**CORRELACIÓN**

![Correlación 1](images/matrizCorrelacion1.jpg)
![Correlación 2](images/matrizCorrelacion2.jpg)

**GRÁFICA DE CORRELACIÓN**

![Gráfica de Correlación](images/Grafica_de_correlacion_codigo.jpg)
![Gráfica de Correlación 2](images/Grafica_de_correlacion.jpg)

**RESUMEN GENERAL DEL DATASET**

![Resumen General del dataset](images/Resumen_General_del_dataset_codigo.jpg)
![Resultado del Resumen](images/Resumen_General_del_dataset_resultado.jpg)

**TIPOS DE DATOS**

![Tipos de datos 1](images/Tipos_de_datos_1.jpg)
![Tipos de datos 2](images/Tipos_de_datos_2.jpg)

**DIMENSIÓN DEL DF**

![Dimensión del DF 1](images/Dimension_del_DF_1.jpg)
![Dimensión del DF 2](images/Dimension_del_DF_2.jpg)

## GRÁFICAS 


### Cargar los datos y función de matriz de densidad

```python
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos
filename = 'dataset/metaverse_transactions_dataset.csv'
data = pd.read_csv(filename)

# Función matriz densidad
def matriz_densidad():
    
    # Código básico para gráficar
    figura = plt.figure(figsize=(6,6))
    
    # Código básico para gráficar una matriz de dispersión
    data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)

    #Agregar un título general a la figura
    figura.suptitle("MATRIZ DE DENSIDAD", fontsize=16)

    # Mostrar la figura
    plt.show()

# Llamar a la función 
matriz_densidad()

**MATRIZ DE DENSIDAD**

![Matriz de Densidad](images/Matriz_densidad.jpg)


## Instalación

Pasos para instalar las dependencias y configurar el entorno.

bash
git clone https://github.com/tu-usuario/tu-proyecto.git
cd tu-proyecto
pip install -r requirements.txt
