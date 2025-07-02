# Definición de los datos

## Origen de los datos

Los datos utilizados en este proyecto provienen del dataset "Fruit Ripeness: Unripe, Ripe, and Rotten", disponible en la plataforma Kaggle. Este conjunto de datos incluye imágenes de frutas tropicales en diferentes etapas de maduración, se encontraran imagenes de plátanos, manzanas y naranjas en diferentes estados como:

- Frutas frescas
- Frutas verdes
- Frutas en descomposición o podridas

Estos datos se encuentran en: [Fruit Ripeness: Unripe, Ripe, and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten)

Los datos se descargan desde la página oficial, utilizando sus funcionalidades de descarga directa, pusto que Kaggle ofrece la opción de acceder a los datos usando su API para integrarlos automaticamente a entornos de trabajo, facilitando la integración de los datos para su análisis en tiempo real. 

## Especificación de los scripts para la carga de datos
1. Se hace instalación de la biblioteca `kagglehub`
   
`!pip install kagglehub`  

2. Se importan los módulos necesarios para el manejo de los archivos:  

`import os`  
`import matplotlib.pyplot as plt`  
`import matplotlib.image as mpimg`  
`import kagglehub`  

3. Se descarga el dataset de Kaggle mediante la función `dataset_download()`, Se define el conjunto de datos en una variable de ruta local, se comprueba que no hallan versiones anteriores, por ultimo, se copia el dataset para poder trabajar en un entorno local.  

`ruta = kagglehub.dataset_download("leftin/fruit-ripeness-unripe-ripe-and-rotten")`  
`path = "/kaggle/working/fruit_dataset_editable"`  
`if os.path.exists(path):`  
    `shutil.rmtree(path)`  
`shutil.copytree(ruta, path)`  

## Referencias a rutas o bases de datos origen y destino
- Ruta de origen:  
El dataset se obtiene desde la plataforma Kaggle, desde el repositorio `"leftin/fruit-ripeness-unripe-ripe-and-rotten"`, este se descarga utilizando la función `kagglehub.dataset_download()` y se almacena temporalmente en una ruta local.
- Ruta de destino:  
  Allí los datos son copiados con la función `shutil.copytree()` que permitira el posterior procesamiento y analisis de datos.

### Rutas de origen de datos
Respecto al entendimiento de los datos se tiene:  

El dataset está organizado en una carpeta principal `dataset` que contiene tres sub carpetas  `'test'`,  `'train'`,  `'dataset'`:  
`'test'`: contiene imágenes destinadas a la evaluación del modelo.   
`'train'`:contiene imágenes destinadas al entrenamiento del modelo.   
`'dataset'`: contiene dos subcarpetas `'test'`,  `'train'`.  

El total de imagenes con las que se cuenta es de 41936 en formatos jpg y png.  

Luego de ver la organización del Dataset, se realizan las siguientes filtraciones para su etapa de limpieza:  

1. Eliminación de datos repetidos: En este proceso se identifica que las subcarpetas en `'dataset'` contienen los mismos datos que las carpetas   `'test'`,  `'train'`, antes nombradas, por ende, se proceden a eliminar estas carpetas sueltas, quedando solamente `'dataset'` con sus respectivas sub carpetas.  

2. Eliminación de información no relevante: La información dentro de las subcarpetas se filtra teniendo en cuenta que se debe eliminar una categoria que no esta contemplada dentro del alcance del proyecto, `unripe` correspondiente a frutas verdes. Además, tambien se eliminan las imagenes que no cumplen con un estandar basico de información minima que pueda brindar la imagen al proyecto, segun su tamañano (Resolución 128X128).

3. Codificación de etiquetas:
   Se hace una codificación de etiquetas de la siguiente manera:  
   `0    freshapples`  
   `1    freshbanana`  
   `2   freshoranges`  
   `3   rottenapples`  
   `4   rottenbanana`  
   `5  rottenoranges`  
De esta manera el proyecto queda con una cantidad de 13572 imagenes distribuidas en:  
imágenes `test`: 2693  
imágenes `train`: 10879

Con un peso total de 1.82 GB

### Base de datos de destino
Como base de datos de destino se tendra al dataset limpio, en la ruta de la copia local, se hara uso de la herramienta de TensorFlow `tf.keras.utils.image_dataset_from_directory()` que permite cargar imágenes desde una estructura de carpetas y convertirlas directamente en un dataset listo para entrenar el modelo. 

Se carga la ruta del dataset con el debido preprocesamiento:
`training_set_rgb = tf.keras.utils.image_dataset_from_directory(
    '/kaggle/working/fruit_dataset_editable/archive (1)/dataset/dataset/train')`  
Allí se configuran los parámetros para el modelamiento:  
`labels='inferred'`,  
 `label_mode='categorical'`,  
 `class_names=None`,  
 `color_mode='rgb'`,  
 `batch_size=32`,  
 `image_size=(128,128)`,  
 `shuffle=True`,  
 `seed=None`, 
 `validation_split=None`,  
 `subset=None`,  
 `interpolation='bilinear'`,  
 `follow_links=False`,  
 `crop_to_aspect_ratio=False`  
De esta manera se obtiene imagenes en formato RGB, redimensionadas a 128x128 pixeles, en lotes de 32 y las etiquetas inferidas por el mismo sistema y codificadas de forma categorica en formato one-hot.
