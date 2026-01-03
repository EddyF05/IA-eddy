# Proyecto 3: Análisis de Datos mediante RAG

## Introducción
En la actualidad, la Generación Z se desarrolla en un contexto marcado por la hiperconectividad, el uso intensivo de redes sociales y la presencia constante de algoritmos que median la información, el entretenimiento y la interacción social. Este entorno ha generado cuestionamientos relacionados con el sentido de vida, la identidad y la autonomía individual.

El objetivo de este proyecto es analizar si existe una crisis de sentido en la Generación Z y cómo los algoritmos y la inteligencia artificial influyen en su percepción de identidad y libertad. Para ello, se implementó un enfoque de **Retrieval-Augmented Generation (RAG)**, que permite combinar información teórica con discursos juveniles obtenidos de entornos digitales.

---

## Metodología
Se construyó un sistema RAG utilizando **AnythingLLM** junto con un modelo de lenguaje local (**Llama 3 a través de Ollama**). Este sistema permite recuperar información desde un corpus previamente cargado y generar respuestas fundamentadas en dicho contenido.

El proceso se desarrolló en las siguientes etapas:

1. Definición del problema de investigación, centrado en la crisis de sentido, la identidad juvenil y la autonomía frente a los algoritmos.
2. Construcción del corpus a partir de fuentes teóricas y empíricas.
3. Carga de documentos en AnythingLLM para su procesamiento.
4. Generación de embeddings para permitir la recuperación semántica.
5. Planteamiento de preguntas analíticas para identificar patrones emocionales, identitarios y sociales en los discursos juveniles.

---

## Construcción del Corpus
El corpus fue diseñado con un enfoque mixto, integrando dos tipos de información:

### Corpus teórico
Se incluyeron textos filosóficos y académicos que abordan temas como el sentido de la vida, la identidad, la técnica, la vigilancia y la cultura del rendimiento. Entre los autores considerados se encuentran:

- Jean-Paul Sartre  
- Albert Camus  
- Zygmunt Bauman  
- Byung-Chul Han  
- Michel Foucault  
- Martin Heidegger  
- Jürgen Habermas  

### Corpus empírico
Se integraron discursos juveniles representativos obtenidos a partir de comentarios simulados de plataformas digitales como **TikTok, YouTube y X**, relacionados con ansiedad, identidad, presión social y el impacto de los algoritmos.

Esta combinación permitió contrastar la teoría filosófica con experiencias reales del entorno digital contemporáneo.

---

## Resultados del Análisis RAG

### Vacío existencial en los discursos juveniles
El sistema identificó expresiones recurrentes de incertidumbre, desorientación y falta de propósito. Frases relacionadas con no saber qué hacer con la vida, miedo al futuro y sensación de estar perdido reflejan un vacío existencial similar al planteado por el existencialismo de Sartre y Camus.

### Identidad líquida en entornos digitales
Los discursos analizados muestran una identidad cambiante y flexible. Los jóvenes describen cómo su personalidad, gustos y opiniones varían según el contenido consumido, lo que coincide con el concepto de **identidad líquida** propuesto por Zygmunt Bauman.

### Presión, rendimiento y burnout
Las emociones predominantes en los discursos sobre el rendimiento fueron ansiedad, estrés, frustración e inseguridad. Estos patrones respaldan la crítica de Byung-Chul Han sobre la **sociedad del rendimiento**, donde el individuo se autoexplota y se responsabiliza de su propio cansancio.

### Algoritmos y percepción de autonomía
Los resultados muestran una relación ambivalente con los algoritmos. Mientras algunos usuarios se sienten beneficiados por la personalización del contenido, otros perciben dependencia y pérdida de control, lo que indica que la autonomía juvenil se encuentra condicionada por los sistemas de recomendación.

---

## Interpretación Filosófica
Los datos recuperados permiten establecer relaciones claras entre la experiencia juvenil contemporánea y distintos marcos filosóficos:

- **Sartre y Camus**: el vacío existencial y el absurdo se manifiestan en la falta de sentido y propósito.
- **Bauman**: la identidad se presenta como fluida, inestable y fragmentada.
- **Byung-Chul Han**: el cansancio, la presión constante y la autoexplotación aparecen como elementos recurrentes.
- **Foucault**: los algoritmos pueden interpretarse como nuevas formas de vigilancia.
- **Heidegger**: la tecnología no solo actúa como herramienta, sino como configuradora de la experiencia humana.
- **Habermas**: el espacio público digital se muestra fragmentado y dominado por dinámicas algorítmicas.

---

## Conclusiones
El análisis realizado mediante el sistema RAG sugiere la existencia de indicadores claros de una crisis de sentido en la Generación Z, estrechamente vinculada a la hiperconectividad, la presión social y la mediación algorítmica.

Asimismo, se observa que la autonomía juvenil no desaparece por completo, pero se ve condicionada por los sistemas de recomendación que influyen en hábitos, gustos y percepciones. El uso del enfoque RAG permitió fundamentar estas conclusiones tanto en datos empíricos como en teoría filosófica.

Este proyecto demuestra que la inteligencia artificial puede emplearse como una herramienta de análisis crítico, integrando filosofía, análisis de datos y tecnologías de IA para comprender problemáticas sociales contemporáneas.

---

## Herramientas utilizadas
- Python  
- AnythingLLM  
- Ollama  
- Llama 3  
- Procesamiento de lenguaje natural  
- Sistema RAG
