# Reporte de Datos

Se presentan los resultados del análisis exploratorio de los datos.

## Resumen general de los datos

El conjunto de datos trae preestablecido la partición de los mismos en las carpetas `dataset`, `train` y `test`. Sin embargo, como los datos se repiten, se procede a eliminar estos datos repetidos, a demás, se eliminan los datos de una categoria que no esta contemplada dentro del alcance del proyecto.  

Se cuenta con imagenes de bananos, naranjas y manzanas repartidos y organizados en frutas en buen estado y frutas en mal estado, de esta manera se tienen las siguientes variables:

`0    freshapples`  
`1    freshbanana`  
`2   freshoranges`  
`3   rottenapples`  
`4   rottenbanana`  
`5  rottenoranges`  

Las imagenes se encuentran en formato `.png`, se verifica que todas las imagenes puedan ser abiertas, que no hayan imagenes vacias, que no hayan imagenes completamente negras o blancas, por ultimo se verifica la cantidad de imagenes con baja resolución. 

## Resumen de calidad de los datos

Se filtran las imagenes teniendo en cuenta la información minima que puede brindar la imagen al proyecto, segun su tamañano (Resolución 128X128) con un resultado de 27 imagenes que no cumplen esta condición, se procede a hacer la eliminación, ya que no representa un numero significativo respecto al total de imagenes con el que se cuenta.   
De esta manera despues de la limpieza de datos se tiene un total de 10879 imagenes de validacion, que seria un aproximado del 80% de los datos, y 2693 imagenes de prueba, que seria un aproximado del 20% de los datos, todas estas imagenes sin errores, no hay imagenes duplicadas.

## Variable objetivo
El producto final será utilizado por parte de los exportadores de fruta que podrán, a través del modelo, identificar si la fruta está en condiciones optimas o no, para ser exportada y evitar posibles contaminaciones, es decir, para este proyecto se tendra en cuenta si es una fruta fresca `fresh` o si es una fruta `rotten` además el modelo debe informar que tipo de fruta es `freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`.

En esta sección se describe la variable objetivo. Se muestra la distribución de la variable y se presentan gráficos que permiten entender mejor su comportamiento.

## Variables individuales

En esta sección se presenta un análisis detallado de cada variable individual. Se muestran estadísticas descriptivas, gráficos de distribución y de relación con la variable objetivo (si aplica). Además, se describen posibles transformaciones que se pueden aplicar a la variable.

## Ranking de variables

En esta sección se presenta un ranking de las variables más importantes para predecir la variable objetivo. Se utilizan técnicas como la correlación, el análisis de componentes principales (PCA) o la importancia de las variables en un modelo de aprendizaje automático.

## Relación entre variables explicativas y variable objetivo

En esta sección se presenta un análisis de la relación entre las variables explicativas y la variable objetivo. Se utilizan gráficos como la matriz de correlación y el diagrama de dispersión para entender mejor la relación entre las variables. Además, se pueden utilizar técnicas como la regresión lineal para modelar la relación entre las variables.
