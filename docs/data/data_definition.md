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
El dataset está organizado en una carpeta principal `dataset` que contiene tres sub carpetas  `'test'`,  `'train'`,  `'dataset'`:  
`'test'`: contiene imágenes destinadas a la evaluación del modelo.   
`'train'`:contiene imágenes destinadas al entrenamiento del modelo.   
`'dataset'`: contiene dos subcarpetas `'test'`,  `'train'`.  

El total de imagenes con las que se cuenta es de 41936 en formatos jpg y png.  

- [ ] Describir los procedimientos de transformación y limpieza de los datos.

### Base de datos de destino

- [ ] Especificar la base de datos de destino para los datos.
- [ ] Especificar la estructura de la base de datos de destino.
- [ ] Describir los procedimientos de carga y transformación de los datos en la base de datos de destino.
