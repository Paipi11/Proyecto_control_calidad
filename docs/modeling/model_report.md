# Reporte del Modelo Final

## Descripción del Problema

En lo que se lleva del 2025 las exportaciones del grupo de productos Agropecuarios, alimentos y bebidas ha tenido una contribución de 11 Millones de dólares habiendo crecido un 36.9% versus el inicio del año según cifras del DANE y su tendencia es positiva. Dado que la demanda de este tipo de productos crece, es crucial un buen control de calidad y contar con herramientas tecnológicas que permitan una mayor efectividad y precisión a la hora de escoger estos productos. Este rubro será vital para la economía colombiana y para todas las empresas que están en busca de nuevos clientes internacionales y genera confianza en la calidad de sus productos. En conclusión, el modelo brindará un apoyo a las empresas exportadoras de frutas colombianas que permitirá afianzar su posicionamiento dentro del mercado internacional y generar una mejor clasificación a la hora de escoger los productos perecederos.

**La solución:** un modelo que permita clasificar y reconocer el estado de las frutas, frutas en buen estado o en mal estado, y así poder escoger los mejores productos de exportación.

**El proyecto incluirá:**

•	La construcción de un modelo de Deep Learning utilizando redes neuronales y transfer learning.

•	El modelo se desarrollará en la API de Keras.

•	El modelo se evaluará por medio del F1-score y accuracy teniendo presente los pesos de cada clase, para no caer en conclusiones erróneas de la efectividad de nuestro modelo.

**Límites y fronteras del proyecto:**

El proyecto se centrará exclusivamente en reconocer e identificar el buen estado o el mal estado de las siguientes frutas:

•	Banano

•	Manzana

•	Naranja

**Excluido del alcance:** no se tendrá en cuenta el análisis del estado interno de la fruta, al igual que no se contemplan otras frutas, verduras o alimentos.
Uso del producto por parte del beneficiario:
El producto final será utilizado por parte de los exportadores de fruta que podrán, a través del modelo, identificar si la fruta está en condiciones óptimas o no, para ser exportada y evitar posibles contaminaciones, o su no conformidad, al pasar por las aduanas del país receptor.


## Descripción del Modelo

El modelo es una red neuronal convolucional utilizando transfer learning (específicamente la aplicación de MobileNetV2), capaz de clasificar entre las siguientes etiquetas:

•	Banano_buen_estado

•	Banano_podrido

•	Manzana_buen_estado

•	Manzana_podrida

•	Naranja_buen_estado

•	Naranja_podrida

## Evaluación del Modelo

Con base a los hallazgos obtenidos en la sección de baseline_models, seguimos explorando el 3er modelo de transfer learning (específicamente la aplicación de MobileNetV2) por su equilibrio y desempaño al evaluar al conjunto de test. 
<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/800744e5-5c9b-44da-9315-120f7316c9c5" />

Para hallar los mejores hiperparámetros del modelo utilizamos la herramienta *keras_tuner* para modificar las siguientes características:

1.	Número de neuronas en la primera capa densa. 
2.	La tasa de dropout en la tercera capa.
3.	La tasa de aprendizaje del optimizador Adam al momento de compilar.
   
El experimento se hizo para 20 testeos con 5 apocas cada uno. Se determino que los parámetros con un mayor despeño serían los siguientes:

1.	Número de neuronas: 64
2.	La tasa de dropout: 0.2
3.	La tasa de aprendizaje del optimizador Adam: 0.001
   
Finalmente se compilo el modelo con los mejores hiperparámetros hallados con las mismas variables de el modelo base con la diferencia que para este se corrieron 30 épocas para poder validar el overfitting en la data de validación. 

## Resultados modelo final
<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/c90f436b-cce7-4a06-88a0-85265e169712" />
<img width="794" height="692" alt="image" src="https://github.com/user-attachments/assets/9376b3a7-5195-452c-9e09-fd9aef98bb9c" />
<img width="485" height="246" alt="image" src="https://github.com/user-attachments/assets/114c7d3b-2e99-415e-acf2-1e4ee44a9312" />

## Resumen Ejecutivo

El modelo ya no presenta overfitting al haber bajado la tasa de Dropout y con los nuevos parámetros presenta una precisión general del 91% y un F1-score mayor al 80% para todas las clases, al ser la manzana en buen estado la que aún presenta más problemas para identificar y discernir con precisión sobre otras clases, confundiéndola principalmente con la naranja en buen estado. En general el modelo puede clasificar de forma eficiente todas las clases, mas se le dificulta el identificar la manzana en buen estado. 
Finalmente, al evaluar el modelo final en el conjunto de test, presenta una precisión del 91% aproximadamente, lo cual muestra una mejoría del 6% respecto al modelo inicial propuesto en el módulo 2 de Deep learning.


## Conclusiones y Recomendaciones

En conclusión, el modelo tiene una buena base, más sin embargo aún tiene oportunidad de mejora ya que la precisión a un nivel de producción tan alto debe ser muy cercana al 100% y un 1% pude representar millones de productos con clasificación errónea. En cuanto al alcance se podrían incluir otros productos diferentes al de este proyecto y aplicarles un tratamiento similar para su entrenamiento y testeo. Finalmente, hay oportunidad para mejorar la clasificación de la clase manzana en buen estado al ser la que tiene menor f1-score dentro todas las clases y lograr que el modelo discierna mejor.

## Referencias

Datos de exportación del DANE: [Datos de exportación del DANE 2025]( https://www.dane.gov.co/index.php/estadisticas-por-tema/comercio-internacional/exportaciones#:~:text=De%20acuerdo%20con%20la%20informaci%C3%B3n,en%20las%20ventas%20externas%20del)

