# Informe de salida

## Resumen Ejecutivo

Este informe describe los resultados del proyecto de machine learning y presenta los principales logros y lecciones aprendidas durante el proceso.

## Resultados del proyecto

- Resumen de los entregables y logros alcanzados en cada etapa del proyecto.
- Evaluación del modelo final y comparación con el modelo base.
- Descripción de los resultados y su relevancia para el negocio.

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
