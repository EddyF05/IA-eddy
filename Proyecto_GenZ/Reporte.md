# Proyecto 3: Sistema RAG (Retrieval-Augmented Generation)

## Introducción
En este proyecto se desarrolló un sistema de tipo RAG (Retrieval-Augmented Generation), cuyo objetivo es mejorar la calidad de las respuestas de un modelo de lenguaje mediante la recuperación de información relevante desde un corpus de documentos previamente construido.

El sistema permite responder preguntas considerando información externa, evitando respuestas genéricas y aumentando la precisión contextual.

---

## Construcción del corpus
Se construyó un corpus de información a partir de documentos relacionados con el tema del proyecto.  
Los datos fueron recolectados y posteriormente limpiados para eliminar ruido, caracteres innecesarios y contenido irrelevante.

Posteriormente, la información fue convertida a un formato estructurado que permite su procesamiento por el sistema RAG.

---

## Procesamiento de datos
El procesamiento de los datos se realizó en distintas etapas:

1. Limpieza de los datos originales.
2. Conversión de los datos a formato JSON/JSONL.
3. Almacenamiento del corpus para su posterior consulta.

Este proceso permitió preparar la información para ser utilizada en la recuperación de contexto.

---

## Uso del sistema RAG
El sistema RAG funciona recuperando fragmentos de información relevantes del corpus con base en una pregunta del usuario.  
Dichos fragmentos se utilizan como contexto adicional para que el modelo de lenguaje genere una respuesta más precisa y fundamentada.

El sistema fue probado mediante distintas preguntas relacionadas con el contenido del corpus, obteniendo respuestas coherentes y alineadas con la información recuperada.

---

## Resultados
Los resultados obtenidos demuestran que el uso del enfoque RAG mejora la calidad de las respuestas en comparación con un modelo sin acceso a información externa.

El sistema es capaz de:
- Identificar información relevante.
- Generar respuestas más contextualizadas.
- Reducir respuestas ambiguas o incorrectas.

---

## Conclusiones
El desarrollo del sistema RAG permitió comprender la importancia de la recuperación de información como complemento de los modelos de lenguaje.

Este enfoque resulta útil para aplicaciones donde se requiere precisión, contexto y respaldo en información específica, demostrando ser una solución efectiva para el problema planteado.

---

## Herramientas utilizadas
- Python
- Modelos de lenguaje
- Procesamiento de texto
- Sistema de recuperación de información
