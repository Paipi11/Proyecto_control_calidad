# Despliegue de modelos

## Infraestructura

**Nombre del modelo:** trained_best_model2.h5  
  
**Plataforma de despliegue:** Render es una plataforma que permite desplegar aplicaciones web, APIs, bases de datos y demás servicios relacionados, sin necesidad de gestionar infrastructura compleja como servidores físicos o maquinas virtuales.  Algo importante para mensionar es que se integra directamente con GitHub y cada vez que se hace un commit, Render actualiza automaticamente la aplicación.  
  - Brinda multiples servicios como: Web Services, Static Sites, Background Workers, Cron Jobs y Databases. 
  
**Requisitos técnicos:** Respecto a la configuración del entorno en Render se tiene:  
Render es compatible con versiones modernas de Python y se puede escoger la version a conveniencia, para esto se crea un archivo `runtime.txt` especificando la version con la cual se desea trabajar. Render utiliza linux como sistema operativo base, para el presente caso, se utilizo un Doker dando respuesta al requerimeinto de una version especifica de python para trabajar de la mano con tensorflow y lograr el despliegue del modelo propuesto.  
Respecto a las dependencias, se requiere un archivo `requirements.txt` el cual debe contener las librerias necesarias para el funcionamiento del proyecto, en este caso: 
  
`streamlit==1.35.0`  
`tensorflow>=2.12.0`  
`numpy==1.26.4`  
`gdown`  

En este mismo sentido, la estructura para el desplique en Render del presente proyecto contiene: 

`main.py` - Que es el archivo principal de la aplicacion a desplegar  
  
`requirements.txt` - Son las librerias necesarias para el funcionamiento del proyecto  
  
`Dockerfile` - Es un archivo que contiene un conjunto de instrucciones para preparar el entorno en el que la aplicación se ejecutara, se incluye el sistema operativo base, las dependencias, el codigo fuente y el comando de incio.  
 
- **Requisitos de seguridad:** (lista de requisitos de seguridad necesarios para el despliegue, como autenticación, encriptación de datos, etc.)
  
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

- (lista de rutas de acceso a los archivos necesarios para el despliegue)
- **Variables de entorno:** (lista de variables de entorno necesarias para el despliegue)

## Documentación del despliegue

- **Instrucciones de instalación:** (instrucciones detalladas para instalar el modelo en la plataforma de despliegue)
- **Instrucciones de configuración:** (instrucciones detalladas para configurar el modelo en la plataforma de despliegue)
- **Instrucciones de uso:** (instrucciones detalladas para utilizar el modelo en la plataforma de despliegue)
- **Instrucciones de mantenimiento:** (instrucciones detalladas para mantener el modelo en la plataforma de despliegue)
