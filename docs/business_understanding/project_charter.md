# Project Charter - Entendimiento del Negocio

## Integrantes del equipo de trabajo
Grupo N° 13

- Andrés David Paipilla Salgado - CC 1015477880 - apaipillas@unal.edu.co
- Daniel Alejandro Pedreros Cifuentes- CC 1233493224 - d-cifuentes@hotmail.com

## Nombre del Proyecto

Identificación del estado de productos de exportación perecederos para mejorar el proceso de control de calidad

## Objetivo del Proyecto

Este proyecto busca ayudar a mejorar el proceso de control de calidad de ciertos productos perecederos, en específico, frutas por medio de un modelo que identifique si el producto está en buen o mal estado. Este tipo de productos tienen que contar con una rigurosa inspección antes de ser exportados, por lo cual este modelo ayudaría a separar y filtrar este tipo de productos y garantizar la efectividad del control de calidad. En otras palabras, el producto debe estar en óptimas condiciones para el consumo y para prevenir contaminar al resto del producto durante todo su proceso de exportación.

## Alcance del Proyecto

### Incluye:

- La construcción de un modelo de Deep Learning utilizando redes neuronales que permita clasificar y reconocer el estado de las frutas, y así poder escoger los mejores productos de exportación. El modelo se desarrollará en la API de Keras y se evaluará por medio del accuracy teniendo presente los pesos de cada clase, para no caer en conclusiones erróneas de la efectividad de nuestro modelo.
  
- El proyecto se centrará exclusivamente las siguientes frutas:
  - Banano
  - Manzana
  - Naranja
    
*Nota:* Conforme vayamos avanzando en el proyecto se irán añadiendo otros tipos de frutas o verduras para ampliar la variedad a identificar.


- Identificar desde varios ángulos las imágenes de frutas con una precisión de al menos 90%. 

### Excluye:

- Por el momento algún otro tipo de fruta que no sea banano, naranja o manzana.

## Metodología

La metodología utilizada para este proyecto será Cross Industry Standard Process for Data Mining (CRISP - DM), gracias a su flexibilidad y facilidad para personalizar.

## Cronograma

| Etapa | Duración Estimada | Fechas | Encargado |
|------|---------|-------|-------|
| Entendimiento del negocio y carga de datos | 1 semana | del 25 de Junio al 3 de Julio | Daniel Pedreros y Andrés Paipilla |
| Preprocesamiento, análisis exploratorio | 1 semanas | del 4 de Julio al 10 de Julio | Daniel Pedreros y Andrés Paipilla |
| Modelamiento y extracción de características | 1 semana | del 11 de JuLio al 17 de Julio | Daniel Pedreros y Andrés Paipilla |
| Despliegue | 1 semana | del 18 de julio al 24 de julio | Daniel Pedreros y Andrés Paipilla |
| Evaluación y entrega final | 1 semana | del 25 de Julio al 28 de Julio | Daniel Pedreros y Andrés Paipilla |


## Equipo del Proyecto

- Andrés David Paipilla Salgado -> Líder del proyecto
- Daniel Alejandro Pedreros Cifuentes -> Miembro de equipo

## Presupuesto

| Categoría | Descripción | Rango estimado | Observaciones |
|------|---------|-------|-------|
| Recolección de datos | Fuente de los datos y etiquetado para el aprendizaje | 0 - 4'000.000 COP | La fuente de datos es Kaggle por lo que no implicaría un costo. Sin embargo, dado el caso que la empresa quisiera hacerlo con imágenes propias de sus productos se debería contratar un técnico para tomar las fotos y en diferentes condiciones (luz, fondo, estado de madurez) |
| Infraestructura | Opciones donde se realizará el desarrollo y entrenamiento del modelo y si será un computador local o en la nube | 1'000.000 - 5'000.000 COP | La opción más asequible sería Google Colab y AWS como una opción escalable a futuro. En caso de que se quiera hacer de forma local si implicara un costo más alto, pero con completa independencia |
| Desarrollo del modelo | Comprende la limpieza de datos, entrenamiento, validación y pruebas | 12'000.000 - 20'000.000 COP | El costo dependerá del número de Data Scientist que se requieran (entre 1 y 2) para realizar todos los pasos de la metodología CRISP - DM |
| Despliegue y mantenimiento | Aplicación donde se hará uso del modelo y se realizarán los diferentes ajustes necesarios para el usuario final | 4'000.000 - 12'000.000 COP | Dependerá de factores como el servidor (local o en la nube) y si es necesario un desarrollador Backend, Frontend o Fullstack. Adicionalmente, el integrar las cámaras de la planta de producción con la aplicación |
| Otros | aquí se incluyen licencias de software, capacitación del personal de la planta y las pruebas piloto | 4'000.000 - 8'000.000 COP | Este costo dependerá de las licencias que se utilicen y del tiempo que tome las capacitaciones del personal |
| | Total | Entre 22'000.000 - 49'000.000 COP  | |

## Stakeholders

- Carlos Suarez QMS Coordinator de Ocati S.A.
- Relación con los stakeholders: En lo que se lleva del 2025 las exportaciones del grupo de productos Agropecuarios, alimentos y bebidas ha tenido una contribución de 11 Millones de dólares habiendo crecido un 36.9% versus el inicio del año según cifras del DANE y su tendencia es positiva. Dado que la demanda de este tipo de productos crece, es crucial un buen control de calidad y contar con herramientas tecnológicas que permitan una mayor efectividad y precisión a la hora de escoger estos productos. Este rubro será vital para la economía colombiana y para todas las empresas que están en busca de nuevos clientes internacionales y genera confianza en la calidad de sus productos.
- Expectativas: mejorar la efectividad del proceso de selección de frutas, minimizando el riesgo de reportar frutas en mal estado en pedidos de exportación y evitar pérdidas monetarias y de reputación a la compañía.

## Aprobaciones

- Andrés David Paipilla Salgado -> Líder del proyecto
- **Andrés Paipilla**
- Julio 3 de 2025
