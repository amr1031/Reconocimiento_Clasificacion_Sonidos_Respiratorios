# Reconocimiento_Clasificacion_Sonidos_Respiratorios
<img src='INPUT/IMAGENES/escudoUBU.jpeg' align="right" height="120" />

> Repositorio del Trabajo Fin de Grado para el grado de Ingeniería de la Salud, Universidad de Burgos (UBU), 2024-2025;

> **ALVARO MARTIN** ([amr1031\@alu.ubu.es](mailto:amr1031@alu.ubu.es))[📩](https://emojipedia.org/shortcodes)
> - 4º INGENERIA DE LA SALUD / [TRABAJO FIN DE GRADO.](https://ubuvirtual.ubu.es/course/view.php?id=15233)[🎓](https://emojipedia.org/shortcodes) 


## Índice

1.  [Sección 1: Resumen / Abstract](#sección-1-resumen--abstract)  
2.  [Sección 2: Palabras clave / Keywords](#sección-2-palabras-clave--keywords)  
3.  [Sección 3: Objetivo general](#sección-3-objetivo-general)  
4.  [Sección 4: Subobjetivos del proyecto](#sección-4-subobjetivos-del-proyecto)  
    4.1. [Preprocesamiento y extracción de características](#subsección-41-preprocesamiento-y-extracción-de-características)  
    4.2. [Detección acústica](#subsección-42-detección-acústica)  
    4.3. [Clasificación clínica híbrida](#subsección-43-clasificación-clínica-híbrida)  
    4.4. [Interfaz de usuario e informes](#subsección-44-interfaz-de-usuario-e-informes)  

5.  [Sección 5: Obtención del dataset](#sección-5-obtención-del-dataset)  
6.  [Sección 6: Despliegue de la aplicación](#sección-6-despliegue-de-la-aplicación)  
7.  [Sección 7: Bibliografía](#sección-7-bibliografía)  
8.  [Sección 8: Conclusión](#sección-8-conclusión)  
 


## Sección 1: Resumen / Abstract

El presente Trabajo de Fin de Grado aborda el desarrollo de una aplicación web interactiva de auscultación digital para el análisis y diagnóstico de patologías respiratorias a partir de grabaciones de sonido torácico. Aprovechando técnicas de aprendizaje automático, el sistema procesa señales de audio en ventanas de cinco segundos, extrae características mediante transformaciones espectrales y genera predicciones sobre la presencia de crepitaciones y sibilancias, así como un diagnóstico clínico en tres categorías (Sano, Infecciosa y Obstructiva). La interfaz, implementada con Streamlit, permite grabar o cargar archivos WAV, visualizar mel-espectrogramas y obtener resultados en tiempo real.

Para la clasificación acústica se integró un modelo de redes neuronales convolucionales previamente entrenado, adaptándolo al entorno local y optimizando su rendimiento mediante ajustes de muestreo y serialización. A nivel de paciente, las probabilidades promedio de la CNN se emplearon como vectores de entrada para un clasificador Random Forest afinado mediante ponderación interna de clases y búsqueda de hiperparámetros con validación cruzada estratificada. Este enfoque híbrido demostró una mejora notable en métricas como F1–macro y tiempos de inferencia, manteniendo robustez frente al desequilibrio entre categorías.


La solución resultante, abierta y reproducible, incluye generación automática de informes clínicos en PDF y almacenamiento de historiales en SQLite. Sus hallazgos ponen de manifiesto la viabilidad de las herramientas de inteligencia artificial para apoyar la auscultación tradicional, ofreciendo a profesionales de la salud y a usuarios sin conocimientos técnicos una experiencia intuitiva y fiable para la detección precoz de anomalías respiratorias.

## Sección 2: Palabras clave / Keywords

**Palabras clave:**  
Auscultación digital, sonidos respiratorios, mel-espectrograma, Streamlit, clasificación acústica, diagnóstico clínico, aprendizaje automático, balanceo de clases, SMOTE, Random Forest, redes neuronales convolucionales, PDF clínico, SQLite.

**Keywords:**  
Digital auscultation, respiratory sounds, mel-spectrogram, Streamlit, acoustic classification, clinical diagnosis, machine learning, class balancing, SMOTE, Random Forest, convolutional neural networks, clinical PDF, SQLite.

## Sección 3: Objetivo general

Desarrollar una aplicación web interactiva que, a partir de grabaciones de sonido torácico, detecte automáticamente crepitaciones y sibilancias mediante una CNN acústica y, seguidamente, asigne un diagnóstico clínico (Sano, Infecciosa u Obstructiva) a cada paciente mediante un clasificador Random Forest, garantizando un entorno reproducible y de bajo coste.


## Sección 4: Subobjetivos del proyecto

### 4.1 Preprocesamiento y extracción de características  
Segmentar las señales en ventanas de cinco segundos, generar mel-espectrogramas normalizados y estructurar los datos para su análisis.

### 4.2 Detección acústica  
Adaptar y optimizar un modelo de CNN preentrenado para discriminar las cuatro clases de sonido (none, crackles, wheezes, both) con altas métricas de validación.

### 4.3 Clasificación clínica híbrida  
Entrenar y afinar un Random Forest sobre las probabilidades promedio de la CNN para obtener un diagnóstico interpretable y equilibrado frente al desequilibrio de clases.

### 4.4 Interfaz de usuario e informes  
Implementar una herramienta en Streamlit que permita cargar o grabar audios, visualizar resultados en tiempo real y descargar informes clínicos en PDF.  

## Sección 5: Obtención del Dataset

Debido al tamaño y a las restricciones de la licencia, el dataset no se incluye directamente en este repositorio. Para obtenerlo, sigue estos pasos:

1. Regístrate o inicia sesión en [Kaggle](https://www.kaggle.com).
2. Accede a la [Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) y haz clic en "Download" para descargar el archivo.
3. Descomprime el archivo descargado.


## Sección 6: Despliegue de la aplicación

### Streamlit Cloud

1. Abre tu navegador y visita  
   (https://aplicacionsonidostfg-u7tde8xl7eo65sxusmcqjv.streamlit.app/)  
2. Espera a que cargue; no es necesario instalar nada.  
3. Para probar sin necesidad de tener tus propios audios, descarga los archivos WAV de ejemplo que encontrarás en la carpeta **Audios Muestra** de este repositorio. A continuación, en la aplicación haz clic en **Subir archivo**, selecciona uno de los WAV descargado y pulsa **Analizar**.  


## Sección 8: Conclusión

Este repositorio recopila de forma organizada todo el código fuente de la aplicación web desarrollada en el TFG. Está diseñado para que cualquier persona pueda clonar el proyecto, instalarlo y ejecutarlo sin complicaciones, garantizando la transparencia y reproducibilidad de cada componente. De este modo, se facilita el acceso y la comprensión de la solución propuesta y sus resultados.  
