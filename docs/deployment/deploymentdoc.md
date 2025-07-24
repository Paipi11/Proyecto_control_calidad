# Despliegue del modelo - link: [Identificación del estado de productos de exportación perecederos para mejorar el proceso de control de calidad: Bananos, Manzanas y Naranjas](https://proyecto-control-calidad-1.onrender.com)  

## Infraestructura

**Nombre del modelo:** trained_best_model2.h5  
  
**Plataforma de despliegue:** Render es una plataforma que permite desplegar aplicaciones web, APIs, bases de datos y demás servicios relacionados, sin necesidad de gestionar infrastructura compleja como servidores físicos o maquinas virtuales.  Algo importante para mensionar es que se integra directamente con GitHub y cada vez que se hace un commit, Render actualiza automaticamente la aplicación.  
  
  - Brinda multiples servicios como: Web Services, Static Sites, Background Workers, Cron Jobs y Databases. 
  
**Requisitos técnicos:**  
Respecto a la configuración del entorno en Render se tiene:  
  
Render es compatible con versiones modernas de Python y se puede escoger la version a conveniencia, para esto se crea un archivo `runtime.txt` especificando la version con la cual se desea trabajar. Render utiliza linux como sistema operativo base, para el presente caso, se utilizo un Doker dando respuesta al requerimeinto de una version especifica de python para trabajar de la mano con tensorflow y lograr el despliegue del modelo propuesto. Respecto a las dependencias, se requiere un archivo `requirements.txt` el cual debe contener las librerias necesarias para el funcionamiento del proyecto, en este caso: 
  
`streamlit==1.35.0`  
`tensorflow>=2.12.0`  
`numpy==1.26.4`  
`gdown`  

La estructura correspondiente al desplique en Render del presente proyecto contiene: 

`main.py` - Que es el archivo principal de la aplicacion a desplegar  
  
`requirements.txt` - Son las librerias necesarias para el funcionamiento del proyecto  
  
`Dockerfile` - Es un archivo que contiene un conjunto de instrucciones para preparar el entorno en el que la aplicación se ejecutara, se incluye el sistema operativo base, las dependencias, el codigo fuente y el comando de incio.  
 
**Requisitos de seguridad:**  
La aplicación será de libre acceso y no manejará datos sensibles ni documentos que requieran confidencialidad. No obstante, se considerarán las siguientes medidas mínimas:
Mantener las dependencias actualizadas para evitar vulnerabilidades conocidas.

Asegurar que no se expongan credenciales o tokens en el código fuente, utilizando variables de entorno si fuera necesario.

Configurar el servidor para escuchar únicamente en la dirección y puerto especificados, delegando la seguridad de acceso al proveedor que para el presente proyecto es Render.  

  
**Diagrama de arquitectura:** Se presenta el diagrama de arquitectura correspondiente al desplieque del modelo.
```
                    ┌─────────────────────────────┐
                    │         Usuario             │
                    │ (Navegador / Cliente)       │
                    └───────────┬─────────────────┘
                                │
                    ┌───────────▼─────────────────┐
                    │       Servidor Web          │
                    │     (Render/Streamlit)      │
                    │ - Autenticación             │
                    │ - Validación de datos       │
                    └───────────┬─────────────────┘
                                │
                    ┌───────────▼─────────────────┐
                    │       Contenedor            │
                    │ - main.py                   │
                    │ - requirements.txt          |
                    │ - Dockerfile                |
                    └───────────┬─────────────────┘
                                │
                    ┌───────────▼─────────────────┐
                    │       Modelo ML             │
                    │     (TensorFlow)            │
                    └─────────────────────────────┘
```

## Código de despliegue

- **Archivo principal:** `main.py`
  
- **Rutas de acceso a los archivos:**  
`Proyecto_control_calidad/main.py`  
`Proyecto_control_calidad/labels.txt`  
`Proyecto_control_calidad/requirements.txt`  
`Proyecto_control_calidad/Dockerfile`


- **Variables de entorno:**  
`PORT = 8501` - Puerto asignado por render  
`RENDER` - Indica que el servicio se ejecuta en Render  
`HOSTNAME` - Nombre interno del contenedor  
`SERVICE_NAME`  Nombre del servicio desplegado en Render  
`MODEL_PATH = model/trained_best_model2.h5`- Ruta del archivo  
`ENV` - Modo (Producción o desarrollo)  
`SECRET_KEY`- Clave de seguridad  
`DEBUG = False`- Indica si se deben activar los logs de depuración

  
## Documentación del despliegue

- **Instrucciones de instalación:**
  - Se prepara el repositorio con los archivos: `main.py`, `requirements.txt`, `Dockerfile` y las imagenes que contendra la pagina web.  
  - Se configuran las variables de entorno: `PORT = 8080`, `MODEL_PATH = model/trained_best_model2.h5`, `SECRET_KEY`, `DEBUG = False`  
  - Se configura el Dockerfile para inciar la aplicación (script de inicio)  
  - Desde Rendel:  
    - Se crea un nuevo Web Service  
    - Se conecta al repositorio en Github  
    - Se especifica la rama y la configuracion de arranque  
    - Se inicia a generar el despliegue  
      
- **Instrucciones de configuración:**
Se define el comando de inicio con Dockerfile
  
`FROM python:3.11.4-slim` Esta es la linea es el punto de partida ya que brinda la base sobre la cual se construira el contenedor del despliegue, es una versión no tan resiente de Python para que tenga conexion con tensorflow.
  
`WORKDIR /app` Se define la ruta del directorio de trabajo  

`COPY . .` Se copian todos los archivos y carpetas del entorno local al entorno del contenedor `/app`  

`RUN pip install --upgrade pip`se actualiza `pip` a la versión más reciente  

`RUN pip install -r requirements.txt` se iistalan las librerias listadas en `requirements.txt`  

`EXPOSE 8501` Se declara el contenedor  

`CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]`Se indica el comando que se ejecutará cuando se inicie el contenedor. 

`streamlit run main.py` Se ejecuta para desplegar
`--server.port=8501` Se define el puerto  
`--server.address=0.0.0.0` Se despliega en la web  

Nota: Cabe añadir que el modelo esta guardado en la nube y el archivo `main.py` lo llama para su despliegue
  
Instrucciones de uso:  
El modelo estara disponible en el siguiente link:  
[Identificación del estado de productos de exportación perecederos para mejorar el proceso de control de calidad: Bananos, Manzanas y Naranjas](https://proyecto-control-calidad-1.onrender.com)  

Cuando se esta en la pagina web, esta tendra tres apartados, dando clic en la viñeta superior izquierda, en el cual estara el panel de control, allí de desplegara una lista con el contenido:
Inicio: Estara el nombre con el cual se identifica eel algoritmo construido y una imagen de presentación.  
  
Información general del producto: Allí se logra observar el objetivo general, el Stakeholders y las expectativas del algoritmo construido.  

Identificación del producto: En este apartado el cliente podra subir desde el entorno local una imagen de las frutas que abarcan el presente proyecto (Manzana, Naranja o bananos), mediante el boton `Browse files`, allí con la opción `Visualizar la imagen`se desplegara la imagen en cuestión y con el boton `predecir` la pagina arrojara la predicción (Fruta en buen estado o en mal estado) para la exportación. 

Este proceso podra realizarse cuantas veces el usuario lo requiera. 

**Instrucciones de mantenimiento:**
Respecto al mantenimiento, se deben considerar los siguientes apartados:

- Actualización de dependencias:
Editar periódicamente el archivo requirements.txt para actualizar las versiones de las librerías.

- Verificar la compatibilidad de las nuevas versiones con el código antes de desplegar. En el futuro se podrán agregar nuevas librerías al ampliar el modelo para identificar más productos.

- Revisión de funcionamiento: Supervisar regularmente los logs de la aplicación (en la plataforma de despliegue) para identificar errores o advertencias.

- Tomar acciones correctivas si se detectan fallos.

- Copias de seguridad: Crear copias del modelo en la nube, etiquetadas con la fecha de la actualización.

- Mantener al menos una versión estable lista para restauración en caso de error.

- Pruebas regulares: Ejecutar pruebas locales del modelo con streamlit run main.py antes de cada despliegue.



