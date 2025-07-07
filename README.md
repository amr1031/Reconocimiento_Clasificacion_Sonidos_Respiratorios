# Reconocimiento_Clasificacion_Sonidos_Respiratorios
<img src='INPUT/IMAGENES/escudoUBU.jpeg' align="right" height="120" />

> Repositorio del Trabajo Fin de Grado para el grado de Ingeniería de la Salud, Universidad de Burgos (UBU), 2024-2025;

> **ALVARO MARTIN** ([amr1031\@alu.ubu.es](mailto:amr1031@alu.ubu.es))[📩](https://emojipedia.org/shortcodes)
> - 4º INGENERIA DE LA SALUD / [TRABAJO FIN DE GRADO.](https://ubuvirtual.ubu.es/course/view.php?id=15233)[🎓](https://emojipedia.org/shortcodes) 




## Índice

1.  [Sección 1: Resumen / Abstract](#sección-1-resumen--abstract)  
2.  [Sección 2: Palabras clave / Keywords](#sección-2-palabras-clave--keywords)  
3.  [Sección 3: Objetivo principal](#sección-3-objetivo-principal)  
    3.1. [Subsección 3.1: Objetivo 1](#subsección-31-objetivo-1)  
    3.2. [Subsección 3.2: Objetivo 2](#subsección-32-objetivo-2)  
    3.3. [Subsección 3.3: Objetivo 3](#subsección-33-objetivo-3)  
    3.4. [Subsección 3.4: Objetivo 4](#subsección-34-objetivo-4)  

4.  [Sección 4: Obtención del dataset](#sección-4-obtención-del-dataset)  
5.  [Sección 5: Despliegue de la aplicación](#sección-5-despliegue-de-la-aplicación)  
6.  [Sección 6: Explicación de los objetivos](#sección-6-explicación-de-los-objetivos)  
    6.1. [Subsección 6.1: Explicación Objetivo 1](#subsección-61-explicación-objetivo-1)  
    6.2. [Subsección 6.2: Explicación Objetivo 2](#subsección-62-explicación-objetivo-2)  
    6.3. [Subsección 6.3: Explicación Objetivo 3](#subsección-63-explicación-objetivo-3)  
    6.4. [Subsección 6.4: Explicación Objetivo 4](#subsección-64-explicación-objetivo-4)  

7.  [Sección 7: Bibliografía](#sección-7-bibliografía)  
8.  [Sección 8: Conclusión](#sección-8-conclusión)  

## Sección 1: Resumen / Abstract

El presente trabajo desarrolla una aplicación web interactiva de auscultación digital que, mediante aprendizaje automático, procesa grabaciones de sonido torácico en ventanas de cinco segundos, extrae características espectrales y predice la presencia de crepitaciones y sibilancias. La interfaz, implementada con Streamlit, permite grabar o cargar archivos WAV, visualizar mel-espectrogramas y obtener resultados en tiempo real, clasificando cada caso en tres categorías clínicas: Sano, Infecciosa u Obstructiva.

Para la clasificación acústica se emplea un modelo de redes neuronales convolucionales optimizado localmente, cuyas probabilidades promedio por paciente alimentan un clasificador Random Forest ajustado con validación cruzada estratificada. Este enfoque híbrido mejora métricas como la F1–macro y reduce los tiempos de inferencia, además de generar informes clínicos automáticos en PDF y almacenar historiales en SQLite, garantizando una herramienta abierta, reproducible y fácil de usar para profesionales y usuarios sin experiencia técnica.  



## Obtención del Dataset

Debido al tamaño y a las restricciones de la licencia, el dataset no se incluye directamente en este repositorio. Para obtenerlo, sigue estos pasos:

1. Regístrate o inicia sesión en [Kaggle](https://www.kaggle.com).
2. Accede a la [Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) y haz clic en "Download" para descargar el archivo.
3. Descomprime el archivo descargado.


