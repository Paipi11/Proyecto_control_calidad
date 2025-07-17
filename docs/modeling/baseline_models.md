# Reporte del Modelo Baseline

Este documento contiene los resultados del modelo baseline.

## Descripción del modelo

El modelo base es un modelo que se entrena con una Red Neuronal Convolucional (cnn), el modelo pretende clasificar frutas (Manzanas, Babanos y Naranjas) en frutas frescas y frutas en mal estado para exportación.

Para este caso se crea un modelo secuencial en donde inicialmente, se aplican 64 filtros sobre las imagenes y se usa la función de activación 'relu', seguido por, la función ´MacPool2D' que permite reducir la cantidad de parámetros y que aprenda mayor detalle de las imagenes. En la segunda capa, se vuelve a aplicar la funcion ´relu´ y se propone una segunda ´MacPool2D´. Se propone un Dropout de 0.5 evitando así el overfitting, se añade la capa flatten que convierte los datos en un vector plano, para de esta manera, tener una capa densa con 128 neuronas que convine y aprenda relaciones entre las caracteristicas provenientes de las anteriores capas. Por ultimo se tiene la capa de salida con seis neuronas y activación 'softmax', esto debido que el modelo debe clasificar en seis clases diferentes y la funcion de activación para este caso, transforma la saluda en probabilidades que suman uno e indica la probabilidad de que la Imagen pertenezca a una clase en particular.

## Variables de entrada
Las variables de entrada seleccionadas para el modelo son:

0    freshapples: Esta clase contiene imagenes de Manzanas en buen estado (Apta para exportación)  
1    freshbanana: Esta clase contiene imagenes de Bananos en buen estado (Apta para exportación)  
2   freshoranges: Esta clase contiene imagenes de Naranjas en buen estado (Apta para exportación)  
3   rottenapples: Esta clase contiene imagenes de manzanas en mal estado (No apta para exportación)  
4   rottenbanana: Esta clase contiene imagenes de bananos en mal estado (No apta para exportación)  
5  rottenoranges: Esta clase contiene imagenes de naranjas en mal estado (No apta para exportación)  

## Variable objetivo
La variable objetivo a estimar, son dos variables categóricas que el modelo predecirá:

Nombre de la fruta (Banano, Manzana, Naranja)
Estado de la fruta (Apta para exportación, No apta para exportación)

Estas variables se predeciran a partir de la imagen que reciba el modelo

## Evaluación del modelo

### Métricas de evaluación

accuracy: Representa la proporción de predicciones correctas del modelo frente al total de predicciones.

loss (Función de pérdida): Mide el error del modelo. Un valor más bajo indica un mejor desempeño del modelo. 

### Resultados de evaluación

Tabla que muestra los resultados de evaluación del modelo baseline, incluyendo las métricas de evaluación.

## Análisis de los resultados

Descripción de los resultados del modelo baseline, incluyendo fortalezas y debilidades del modelo.

## Conclusiones

Conclusiones generales sobre el rendimiento del modelo baseline y posibles áreas de mejora.

## Referencias

Lista de referencias utilizadas para construir el modelo baseline y evaluar su rendimiento.

Espero que te sea útil esta plantilla. Recuerda que puedes adaptarla a las necesidades específicas de tu proyecto.
