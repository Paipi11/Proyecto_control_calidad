# Reporte del Modelo Baseline

Este documento contiene los resultados del modelo baseline.

## Descripción del modelo

El modelo base es un modelo que se entrena con una Red Neuronal Convolucional (cnn), el modelo pretende clasificar frutas (Manzanas, Babanos y Naranjas) en frutas frescas y frutas en mal estado para exportación.

El modelo base es un modelo que se entrena con una Red Neuronal Convolucional (cnn), el modelo pretende clasificar frutas (Manzanas, Babanos y Naranjas) en frutas frescas y frutas en mal estado para exportación.
Para este caso se crearon 3 modelos iniciales:

1.	Un primer modelo secuencial, que tuvo como entrada las imágenes del conjunto training en gama de colores **RGB**. El modelo tiene primeramente 64 filtros sobre las imágenes y se usa la función de activación 'relu', seguido por, la función ´MaxPool2D' que permite reducir la cantidad de parámetros y que aprenda mayor detalle de las imágenes. En la segunda capa, se vuelve a aplicar la función ´relu´ y se propone una segunda ´MaxPool2D´. Se propone un Dropout de 0.5 evitando así el overfitting, se añade la capa Flatten que convierte los datos en un vector plano, para de esta manera, tener una capa densa con 128 neuronas que convine y aprenda relaciones entre las características provenientes de las anteriores capas. Por último, se tiene la capa de salida con seis neuronas y activación 'softmax', esto debido que el modelo debe clasificar en seis clases diferentes y la función de activación para este caso, transforma la salida en probabilidades que suman uno e indica la probabilidad de que la Imagen pertenezca a una clase en particular.
   
2.	Un segundo modelo secuencial, que tuvo como entrada las imágenes del conjunto training en **escala de grises** y con las mismas capas que el modelo anterior.
   
3.	Un tercer modelo secuencial, pero utilizando **transfer learning** en específico la aplicación de MobileNetV2 donde se agregó una capa de GlobalAveragePooling2D para reducir drásticamente la dimensionalidad de las imágenes y detectar mejor sus características, una de Dropout de 0.5 para evitar el overfitting, una capa densa de 128 neuronas para que interactúe con los resultados de la capa anterior  con activación 'relu' y finalmente la última capa densa donde el modelo clasifica en las 6 clases requeridas y activación 'softmax'.


## Variables de entrada
Las variables de entrada seleccionadas para el modelo son:

0 freshapples: Esta clase contiene imágenes de Manzanas en buen estado (Apta para exportación)

1 freshbanana: Esta clase contiene imágenes de Bananos en buen estado (Apta para exportación)

2 freshoranges: Esta clase contiene imágenes de Naranjas en buen estado (Apta para exportación)

3 rottenapples: Esta clase contiene imágenes de manzanas en mal estado (No apta para exportación)

4 rottenbanana: Esta clase contiene imágenes de bananos en mal estado (No apta para exportación)

5 rottenoranges: Esta clase contiene imágenes de naranjas en mal estado (No apta para exportación)

Para todos los modelos se utilizó tf.keras.utils.image_dataset_from_directory para crear el vector de entrada para todos los modelos propuestos. La única diferencia es que en el 1er y 3er modelo se utilizo la gama de colores RGB, mientras que en el 2do se utilizo la escala de grises. 


## Variable objetivo
La variable objetivo a estimar son las mismas categorías propuestas al inicio del proyecto y que el modelo pueda diferenciar entre una clase y otra de forma discriminada y precisa. Como se muestra en la imagen. Este es un resultado preliminar de lo que se quiere lograr para las 6 clases y que el modelo pueda agrupar características de cada clase para una mejor predicción. 

<img width="853" height="624" alt="image" src="https://github.com/user-attachments/assets/ab4dc5e8-8395-4e65-8b91-7af1d0f9fa14" /> 

## Evaluación del modelo

### Métricas de evaluación

Para la evaluación de los 3 modelos se realizaron graficas de Accuracy y Loss para la data de entrenamiento y de validación para observar el comportamiento y poder determinar si había un overfitting. Adicionalmente, se realizo la matriz de confusión para cada uno de ellos, reporte de clasificación con el conjunto de validación y se evaluó el modelo en el conjunto test para ver su precisión. Finalmente, para los tres modelos se utilizó el optimizador Adam al ser el más equilibrado para el entrenamiento, 'categorical_crossentropy', métrica de 'accuracy' y un total de 20 épocas. 

### Resultados de evaluación
**1. Primer modelo: RGB CNN Model**
   <img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/1469768e-c81b-48fe-87c4-09e532b17303" />
   <img width="794" height="692" alt="image" src="https://github.com/user-attachments/assets/f489b796-df6a-475b-bd28-374c9cece056" />
   <img width="483" height="254" alt="image" src="https://github.com/user-attachments/assets/39b40a9b-87bc-4296-946a-48f277e634bb" />

**2. Segundo modelo: Grayscale CNN Model**
    <img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/d0e9b6dd-4ad7-441b-a22b-17ed20e7c140" />
    <img width="794" height="692" alt="image" src="https://github.com/user-attachments/assets/e2b46a93-1055-4376-b09f-ca705fb22f54" />
    <img width="483" height="240" alt="image" src="https://github.com/user-attachments/assets/42fd0f1a-e1fc-46cd-9543-12eb895f6c8c" />
    
**3. Tercer modelo: Transfer Learning Model**
    <img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/64a042c9-282a-472a-8b85-d4525c30d5cb" />
    <img width="794" height="692" alt="image" src="https://github.com/user-attachments/assets/3bd175a1-bf8a-44c6-a037-f83de30f0c2a" />
    <img width="499" height="251" alt="image" src="https://github.com/user-attachments/assets/ad20e888-9e31-4ecd-b6d7-a7f85ba530b2" />

## Análisis de los resultados
| Gráficas y métricas de evaluación | RGB CNN Model | Grayscale CNN Model | Transfer Learning Model | Análisis |
|---------|---------|---------|---------|---------|
| Accuracy and Loss training set VS Validation set|Acc final training 0.48 y Acc final validation 0.55 |Acc final training 0.84 y Acc final validation 0.86 |Acc final training 0.79 y Acc final validation 0.80|Podemos observar que en los tres modelos se presenta overfitting al estar el valor de accuracy del validation set por encima del conjunto de entrenamiento lo que os indica que nuestra capa de Dropout puede tener un ajuste para mejorar este resultado.|
| Confusion Matrix|  | || **1er modelo** la naranja en buen estado es la que más dispersión tiene y el banano podrido es la clase con mejor reconocimiento. **2do modelo** ya discrimina mejor todas las clases siendo la que tiene mayor dispersión la manzana en buen estado y el banano podrido el que menos tiene.**3er modelo** clasifica con una dispersión media las clases de naranja y de manzana y las de banano si presenta una dispersión muy baja.|
|Classification report|	0.55|	0.86|	0.81| **1er modelo** tiene un accuracy general de 0.55 con dificultades para detectar la naranja en buen estado y solo el banano podrido presenta un acc cercano al 0.7.**2do modelo** ya tiene tanto un mejor acc, recall y f1-score teniendo un 0.8 de acc en todas las clases.**3er modelo** detecta muy bien las imágenes de las clases de banao mas sin embargo tiene oportunidad de mejora para las demás 4 clases al estar por debajo de 0.8 de acc.|
|Evaluación en el conjunto test|	0.29|	0.52|	0.81| Conforme a este resultado nos basamos para escoger con que modelo continuaríamos, ya que puede determinar mejor las características del conjunto test. En conclusión, continuamos explorando el modelo de transfer learning para buscar los mejores hiperparámetros.|

## Conclusiones

El 3er modelo ofrece una primera aproximación útil para desplegar un camino y poder cumplir con los objetivos propuestos para el proyecto. Aunque su rendimiento es aceptable, se continuará explorando posibilidades de mejora como la búsqueda de mejores hiperparámetros que permitan discernir mejor las características de cada clase e incrementar sus parámetros de evaluación (acc, recall, f1-score).  

## Referencias

[Fruit Ripeness: Unripe, Ripe, and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten)  
[Scikit-learn documentation](https://scikit-learn.org/stable/)  
[tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/all_symbols)  
[keras documentation](https://keras.io/)  


