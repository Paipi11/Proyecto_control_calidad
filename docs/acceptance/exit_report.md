# Informe de salida

## Resumen Ejecutivo

Este informe describe los resultados del proyecto de machine learning y presenta los principales logros y lecciones aprendidas durante el proceso.

## Resultados del proyecto

### Resumen de los entregables y logros alcanzados en cada etapa del proyecto.
**1.	Entendimiento del negocio.**
Este proyecto busca ayudar a mejorar el proceso de control de calidad de ciertos productos perecederos, en específico, las frutas banano manzana y naranja prediciendo si están en un estado óptimo de exportación o no. El proyecto se trabajará con base a la metodología Cross Industry Standard Process for Data Mining (CRISP - DM), gracias a su flexibilidad y facilidad para personalizar. El cronograma de actividades se repartió en 5 semanas en las cuales se hará la entrega final y se estima un costo estimado de entre 22’000.000 de pesos a 49´000.000 de pesos según los requerimientos de mano de obra y los stakeholders. 

**2.	Data y preprocesamiento.**
Los datos utilizados en este proyecto provienen del dataset [Fruit Ripeness: Unripe, Ripe, and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten), disponible en la plataforma Kaggle. Con base en los objetivos del proyecto eliminamos la carpeta Unripe, ya que son frutas biches y en su totalidad con una resolución menor a 128x128 pixeles, que es el mínimo requerido para obtener buenos resultados en el entrenamiento. Adicionalmente al aplicar el filtro de 128x128 pixeles eliminamos 27 imágenes adicionales del corpus inicial quedando al final con una cantidad de 13572 imágenes distribuidas en:

- Imágenes test: 2693

- Imágenes train: 10879
  
Se presenta el diccionario de los datos finales:

| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
| --- | --- | --- | --- | --- |
| Frutas frescas | Imagenes de frutas en buen estado (`fresh`), representadas como: `freshapples`,  `freshbanana`,  `freshoranges` | categórico  (Imagen) | `freshapples`,  `freshbanana`,  `freshoranges` | [Fruit Ripeness: Unripe, Ripe, and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten) |
  | Frutas en descomposición | Imagenes de frutas en mal estado (`rotten`), representadas como: `rottenapples`,  `rottenbanana`,  `rottenoranges`| categórico  (Imagen) | `rottenapples`,  `rottenbanana`,  `rottenoranges` | [Fruit Ripeness: Unripe, Ripe, and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten) |

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/b84d6041-8f27-4696-a199-c6d80d2d934f" />

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/da53a855-b798-40f9-a243-70fa07536ebb" />

Adicionalmente, se genera un histograma para conocer la distribución de la resolución total de las imágenes de cada clase de fruta sin importar si esta en buen o mal estado, dándonos un panorama de como se distribuyen estos valores en cada fruta.

<img width="795" height="494" alt="image" src="https://github.com/user-attachments/assets/56709635-66cc-4d40-90d1-aaf96e6f656a" />


**3.	Modelamiento y evaluación.**

Se realizaron 3 modelos iniciales

3.1 Modelo secuencial, de 13 capas que tuvo como entrada las imágenes del conjunto training en gama de colores **RGB**.

3.2 Modelo secuencial, de 13 capas que tuvo como entrada las imágenes del conjunto training en **escala de grises**.

3.3 Modelo secuencial, utilizando **transfer learning**, en específico la aplicación de MobileNetV2 que tuvo como entrada las imágenes del conjunto training en gama de colores RGB.
    
Variables de entrada:

0 freshapples: Esta clase contiene imágenes de Manzanas en buen estado (Apta para exportación)

1 freshbanana: Esta clase contiene imágenes de Bananos en buen estado (Apta para exportación)

2 freshoranges: Esta clase contiene imágenes de Naranjas en buen estado (Apta para exportación)

3 rottenapples: Esta clase contiene imágenes de manzanas en mal estado (No apta para exportación)

4 rottenbanana: Esta clase contiene imágenes de bananos en mal estado (No apta para exportación)

5 rottenoranges: Esta clase contiene imágenes de naranjas en mal estado (No apta para exportación)

| Gráficas y métricas de evaluación | RGB CNN Model | Grayscale CNN Model | Transfer Learning Model | Análisis |
|---------|---------|---------|---------|---------|
| Accuracy and Loss training set VS Validation set|Acc final training 0.48 y Acc final validation 0.55 |Acc final training 0.84 y Acc final validation 0.86 |Acc final training 0.79 y Acc final validation 0.80|Podemos observar que en los tres modelos se presenta overfitting al estar el valor de accuracy del validation set por encima del conjunto de entrenamiento lo que os indica que nuestra capa de Dropout puede tener un ajuste para mejorar este resultado.|
| Confusion Matrix|  | || **1er modelo** la naranja en buen estado es la que más dispersión tiene y el banano podrido es la clase con mejor reconocimiento. **2do modelo** ya discrimina mejor todas las clases siendo la que tiene mayor dispersión la manzana en buen estado y el banano podrido el que menos tiene.**3er modelo** clasifica con una dispersión media las clases de naranja y de manzana y las de banano si presenta una dispersión muy baja.|
|Classification report|	0.55|	0.86|	0.81| **1er modelo** tiene un accuracy general de 0.55 con dificultades para detectar la naranja en buen estado y solo el banano podrido presenta un acc cercano al 0.7.**2do modelo** ya tiene tanto un mejor acc, recall y f1-score teniendo un 0.8 de acc en todas las clases.**3er modelo** detecta muy bien las imágenes de las clases de banao mas sin embargo tiene oportunidad de mejora para las demás 4 clases al estar por debajo de 0.8 de acc.|
|Evaluación en el conjunto test|	0.29|	0.52|	0.81| Conforme a este resultado nos basamos para escoger con que modelo continuaríamos, ya que puede determinar mejor las características del conjunto test. En conclusión, continuamos explorando el modelo de transfer learning para buscar los mejores hiperparámetros.|

Con base en los anteriores resultados se determino que el 3er modelo sería el más preciso para generar el modelo. Con la herramienta de *keras-tuner* se determino que los mejores hiperparámetros serían los siguientes:
- Número de neuronas: 64
- La tasa de dropout: 0.2
- La tasa de aprendizaje del optimizador Adam: 0.001

<img width="794" height="692" alt="image" src="https://github.com/user-attachments/assets/1032d127-0afa-4442-a3ee-cbfe40f554bd" />

<img width="430" height="231" alt="image" src="https://github.com/user-attachments/assets/99e6e769-74f3-406d-8fc4-b71ec81fcdeb" />

Finalmente, el modelo puede clasificar de forma eficiente todas las clases, más se le dificulta el identificar la manzana en buen estado. Al evaluar el modelo final en el conjunto de test, presenta una precisión del 91% aproximadamente.

**4.	Despliegue.**
Por medio de la herramienta Render y la librería Streamlit se desplego la aplicación [Identificación del estado de productos de exportación perecederos para mejorar el proceso de control de calidad: Bananos, Manzanas y Naranjas](https://proyecto-control-calidad-1.onrender.com) donde es posible cargar cualquier imagen de un banano, manzana y naranja para determinar su estado. Para ello se utilizo la contenerización por medio de Docker para evitar cualquier tipo de inviabilidad entre las versiones de las librerías y Python. Finalmente, la aplicación no requiere de ningún tipo de autenticación y es de carácter público.

### Evaluación del modelo final y comparación con el modelo base.
Con base en el modelo inicial que se planteo se tiene lo siguiente:
|Métricas	|Modelo Base |	Modelo Final |	Diferencia |
|-----------|-----------|-----------|-----------|
|Accuracy	|0.81	|0.94	|13%|
|F1-Score	|0.81	|0.91	|10%|
|Evaluación en el conjunto test	|0.81	|0.91	|10%|

En conclusión, el modelo final supera en 10% y más en todas las 3 métricas analizadas y comparadas con el modelo inicial. 

### Descripción de los resultados y su relevancia para el negocio.
El modelo tiene una buena base, más sin embargo aún tiene oportunidad de mejora ya que la precisión a un nivel de producción tan alto debe ser muy cercana al 100% y un 1% pude representar millones de productos con clasificación errónea. En cuanto al alcance se podrían incluir otros productos diferentes al de este proyecto y aplicarles un tratamiento similar para su entrenamiento y testeo. Finalmente, hay oportunidad para mejorar la clasificación de la clase manzana en buen estado al ser la que tiene menor f1-score dentro todas las clases y lograr que el modelo discierna mejor.
## Lecciones aprendidas

- **Identificación de los principales desafíos y obstáculos encontrados durante el proyecto:** El principal desafío fue el entrenamiento y el despliegue, al notar que durante las épocas de entrenamiento de los tres modelos iniciales la data de validación presentaba valores de acc mayores a la de entrenamiento lo cual indicaba un overfitting por parte de la data. Adicionalmente, al contar con nula experiencia en el despliegue el buscar una herramienta o varias que pudiera plasmar y que fuera de fácil uso para que el modelo pudiera cumplir su función fue uno de los mayores retos.
  
- **Lecciones aprendidas en relación al manejo de los datos, el modelamiento y la implementación del modelo:** Se aprendieron varias estrategias al tener un conjunto de datos tan diverso y desequilibrio al igual que el poder manipular la escala de colores de las imágenes de entrada para obtener mejores resultados. Adicionalmente, la librería de keras-tuner nos ayudo bastante a encontrar los mejores hiperparámetros dentro del rango de testeos realizados y con ello poder mejorar cada vez mas nuestro modelo. Finalmente, el poder integrar de forma rápida el modelo guardado del código de despliegue para realizar varios testeos.
  
- **Recomendaciones para futuros proyectos de machine learning:**  Conocer muy bien los diferentes modelos que mejor se pueden implementar para el problema planteado. En nuestro caso omitimos el transfer learning en un foco inicial, pero en este modulo pudimos apreciar su poder y precisión. Adicionalmente, el versionamiento habilitado en las herramientas del despliegue ya que a pesar de que muchas librerías modernas han avanzado mucho, su soporte en aplicaciones de despliegue aún es muy bajo y es recomendable analizar esto primero antes de escoger alguna técnica en especial. Finalmente, el nunca dejar de aprender sobre el tema ya que esta creciendo aniveles exponenciales y cada día los enfoques son más precisos y rápidos.

## Impacto del proyecto  

Las nuevas tecnologías emergentes permiten tener un control de la información en tiempo real sobre los productos ofertados en las empresas, más aún, la calidad con que ellos llegan o podrían llegar al consumidor final. Por un lado, los tiempos de identificación de frutas en buen estado o mal estado, se reducen en gran medida al hacer la transición de manera manual a manera virtual, resultando una mejora significativa en los tiempos de entrega y de esta manera un crecimiento económico. Por otro lado, al notar que la aplicación de estas tecnologías emergentes  a los procesos productivos aumentan ganancias, reducen tiempo, la empresa inicia a destinar un rubro para capacitar el personal de su planta y de esta manera hacer una actualización de su cadena de producción, esto para hacer una transición progresiva que promueva el bienestar social, productivo y económico.  

Se tiene como plan de mejoramiento ampliar la variedad de frutas que se pueden identificar, también se pueden añadir vegetales, además se abren un sinfín de oportunidades como la capacidad de predecir características como: fruta verde, fruta con alguna enfermedad, tiempo aproximado para no ser apta para exportación, etc.   



## Conclusiones  

Se desarrollo un algoritmo útil para la identificación de frutas para exportación que da respuesta a los objetivos propuestos inicialmente, aunque el proyecto está en una fase temprana, la empresa planea continuar explorando posibilidades de mejora como la búsqueda de mejores hiperparámetros que permitan discernir mejor las características de cada clase e incrementar sus parámetros de evaluación.  
  
Uno de los logros más importantes del proyecto fue la reducción significativa del tiempo requerido para la identificación de los productos, representando un avance importante en los procesos logísticos y de calidad.  
  
A pesar de los buenos resultados obtenidos, cabe destacar que, como se ha mencionado antes, el algoritmo aún tiene oportunidad de mejora ya que la precisión a un nivel de producción alto debe ser muy cercana al 100% y un 1% pude representar millones de productos con clasificación errónea.  

## Agradecimientos

Queremos expresar nuestro más sincero agradecimiento al equipo de trabajo que hizo posible el desarrollo de este proyecto, en especial a Andrés Paipilla el líder del proyecto y a Daniel Pedreros, por su compromiso, dedicación y colaboración, fueron fundamentales para alcanzar los objetivos propuestos, además de su esfuerzo, creatividad y disposición para enfrentar los retos que surgieron en el camino.  
  
Les agradecemos especialmente a nuestros patrocinadores y entidades financiadoras, cuya confianza y respaldo económico permitieron llevar a cabo esta iniciativa. Sin su apoyo, este proyecto no habría sido posible.  


