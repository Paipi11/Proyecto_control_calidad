# Informe de salida

## Resumen Ejecutivo

Este informe describe los resultados del proyecto de machine learning y presenta los principales logros y lecciones aprendidas durante el proceso.

## Resultados del proyecto

### Resumen de los entregables y logros alcanzados en cada etapa del proyecto.
**1.	Entendimiento del negocio.**
Este proyecto busca ayudar a mejorar el proceso de control de calidad de ciertos productos perecederos, en específico, las frutas banano manzana y naranja prediciendo si están en un estado óptimo de exportación o no. El proyecto se trabajará con base a la metodología Cross Industry Standard Process for Data Mining (CRISP - DM), gracias a su flexibilidad y facilidad para personalizar. El cronograma de actividades se repartió en 5 semanas en las cuales se hará la entrega final y se estima un costo estimado de entre 22’000.000 de pesos a 49´000.000 de pesos según los requerimientos de mano de obra y los stakeholders. 

**2.	Data y preprocesamiento.**
Los datos utilizados en este proyecto provienen del dataset "Fruit Ripeness: Unripe, Ripe, and Rotten", disponible en la plataforma Kaggle. Con base en los objetivos del proyecto eliminamos la carpeta Unripe, ya que son frutas biches y en su totalidad con una resolución menor a 128x128 pixeles, que es el mínimo requerido para obtener buenos resultados en el entrenamiento. Adicionalmente al aplicar el filtro de 128x128 pixeles eliminamos 27 imágenes adicionales del corpus inicial quedando al final con una cantidad de 13572 imagenes distribuidas en:
imágenes test: 2693
imágenes train: 10879
Se presenta el diccionario de los datos finales:


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


Con base en los anteriores resultados se determino que el 3er modelo sería el más preciso para generar el modelo. Con la herramienta de keras-tuner se determino que los mejores hiperparámetros serían los siguientes:
- Número de neuronas: 64
- La tasa de dropout: 0.2
- La tasa de aprendizaje del optimizador Adam: 0.001

<img width="794" height="692" alt="image" src="https://github.com/user-attachments/assets/1032d127-0afa-4442-a3ee-cbfe40f554bd" />


<img width="430" height="231" alt="image" src="https://github.com/user-attachments/assets/99e6e769-74f3-406d-8fc4-b71ec81fcdeb" />

Finalmente, el modelo puede clasificar de forma eficiente todas las clases, más se le dificulta el identificar la manzana en buen estado. Al evaluar el modelo final en el conjunto de test, presenta una precisión del 91% aproximadamente.

**4.	Despliegue.**
Por medio de la herramienta Render y la librería Streamlit se desplego la aplicación (Link) donde es posible cargar cualquier imagen de un banano, manzana y naranja para determinar su estado. Para ello se utilizo la contenerización por medio de Docker para evitar cualquier tipo de inviabilidad entre las versiones de las librerías y Python. Finalmente, la aplicación no requiere de ningún tipo de autenticación y es de carácter público.

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

- Descripción del impacto del modelo en el negocio o en la industria.
- Identificación de las áreas de mejora y oportunidades de desarrollo futuras.

## Conclusiones

- Resumen de los resultados y principales logros del proyecto.
- Conclusiones finales y recomendaciones para futuros proyectos.

## Agradecimientos

- Agradecimientos al equipo de trabajo y a los colaboradores que hicieron posible este proyecto.
- Agradecimientos especiales a los patrocinadores y financiadores del proyecto.
