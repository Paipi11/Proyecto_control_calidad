# Reporte del Modelo Final

## Descripción del Problema

En lo que se lleva del 2025 las exportaciones del grupo de productos Agropecuarios, alimentos y bebidas ha tenido una contribución de 11 Millones de dólares habiendo crecido un 36.9% versus el inicio del año según cifras del DANE y su tendencia es positiva. Dado que la demanda de este tipo de productos crece, es crucial un buen control de calidad y contar con herramientas tecnológicas que permitan una mayor efectividad y precisión a la hora de escoger estos productos. Este rubro será vital para la economía colombiana y para todas las empresas que están en busca de nuevos clientes internacionales y genera confianza en la calidad de sus productos.
En conclusión, el modelo brindará un apoyo a las empresas exportadoras de frutas colombianas que permitirá afianzar su posicionamiento dentro del mercado internacional y generar una mejor clasificación a la hora de escoger los productos perecederos.

**La solución:** un modelo que permita clasificar y reconocer el estado de las frutas, frutas en buen estado o en mal estado, y así poder escoger los mejores productos de exportación.

**El proyecto incluirá:**

- La construcción de un modelo de Deep Learning utilizando redes neuronales y transfer learning.
- El modelo se desarrollará en la API de Keras.
- El modelo se evaluará por medio del F1-score y accuracy teniendo presente los pesos de cada clase, para no caer en conclusiones erróneas de la efectividad de nuestro modelo.
  
**Límites y fronteras del proyecto:**

El proyecto se centrará exclusivamente en reconocer e identificar el buen estado o el mal estado de las siguientes frutas:
- Banano
- Manzana
- Naranja

Excluido del alcance: No se abordaran problemas de predicción a largo plazo ni seguimiento, no se tendrá en cuenta el análisis del estado interno de la fruta, al igual que no se contemplan otras frutas, verduras o alimentos.

Uso del producto por parte del beneficiario:

El producto final será utilizado por parte de los exportadores de fruta que podrán, a través del modelo, identificar si la fruta está en condiciones óptimas o no, para ser exportada y evitar posibles contaminaciones, o su no conformidad, al pasar por las aduanas del país receptor.

## Descripción del Modelo

El modelo es una red neuronal convolucional utilizando transfer learning (específicamente la aplicación de MobileNetV2), capaz de clasificar entre las siguientes etiquetas:
- Banano_buen_estado
- Banano_podrido
- Manzana_buen_estado
- Manzana_podrida
- Naranja_buen_estado
- Naranja_podrida

## Evaluación del Modelo

En esta sección se presentará una evaluación detallada del modelo final. Se deben incluir las métricas de evaluación que se utilizaron y una interpretación detallada de los resultados.
Se realizaron dos modelos inicialmente:
1. Una red convolucional de 8 capas en las cuales se utilizaron CONV2D, MaxPool2D, Dropout, Flattern y Dense.
2. Una red convolucional con transfer learning utilizando la aplicación de keras MobilNetV2 y agregándole 4 capas de GlobaLAveragePooling2D, Dropout y Dense.

Entre los dos modelos el que obtuvo un mejor desempaño en las primeras 10 épocas fue el 2do al obtener más de un 80% VS el 70% del 1ro.

Adicionalmente, se utilizó Keras tuner para determinar los mejores hiperparámetros del segundo modelo siendo estas 192 neuronas en la capa densa, una tasa de dropout de 0.3 y un learning rate de 0.001 para una precisión del 82% con 10 épocas.

## Resumen Ejecutivo

El modelo presente una precisión del 99.8% en el conjunto de entrenamiento para 20 épocas y se visualiza como la pérdida del conjunto de validación se va minimizando al igual que el conjunto de entrenamiento va mejorando su precisión.

<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/0bb4c2a4-e7ba-44dd-b9ff-1010a26edaa4" />

Adicionalmente, se realizó un gráfico T-SNE para ver como el conjunto de entrenamiento separa y diferencia las características de cada clase. Como se puede ver en el gráfico hay una separación gradual de cada clase que le permite al modelo discernir con una precisión mayor al 90% una clase de otra. 

<img width="853" height="624" alt="image" src="https://github.com/user-attachments/assets/eba974b2-b37d-44c3-8c5b-3a8030481ae6" />

Finalmente, el modelo final muestra una precisión del 91% aproximadamente en el conjunto de prueba, lo cual muestra una mejoría del 5% respecto al modelo inicial propuesto en el módulo 2 de Deep learning.

## Conclusiones y Recomendaciones

En conclusión, el modelo tiene una buena base, más sin embargo aún tiene oportunidad de mejora ya que la precisión a un nivel de producción tan alto debe ser muy cercana al 100% y un 1% pude representar millones de productos con clasificación errónea. En cuanto al alcance se podrían incluir otros productos diferentes al de este proyecto. Finalmente, hay oportunidad para mejorar la clasificación de características que el modelo discierne y hacer más pruebas para consolidar su uso. 

## Referencias

Datos de exportación del DANE: Datos de exportación del DANE 2025

