
# Reporte del Proyecto: Tutor Académico con Fine-Tuning usando LoRA

## Autor
Edgar Posadas Fraga

## Descripción General del Proyecto
Este proyecto consiste en el diseño y desarrollo de un **tutor académico basado en un modelo de lenguaje**, ajustado mediante la técnica de **fine-tuning con LoRA (Low-Rank Adaptation)**.  
El objetivo principal es que el modelo pueda **responder preguntas académicas relacionadas con la materia de Inteligencia Artificial y temas derivados**, de forma clara, estructurada y con un enfoque educativo.

El proyecto **no trabaja con imágenes**, sino con **texto**, específicamente con un dataset de **preguntas y respuestas académicas**.

---

## Modelo Utilizado
Se utilizó un modelo de lenguaje tipo **Transformer** como base, el cual fue adaptado mediante fine-tuning ligero.

- Arquitectura: Transformer
- Tipo: Modelo de lenguaje generativo
- Ajuste: Fine-tuning con **LoRA**
- Framework principal: `transformers` de Hugging Face

El modelo base no se reentrena completamente; solo se entrenan adaptadores LoRA, lo que reduce costos computacionales.

---

## ¿Por qué se usó Transformers?
La librería **Transformers** se utilizó porque:
- Permite cargar modelos de lenguaje preentrenados.
- Facilita el entrenamiento y ajuste fino de modelos NLP.
- Proporciona compatibilidad directa con LoRA y PEFT.
- Simplifica la tokenización y generación de texto.

---

## ¿Qué es LoRA y por qué se utilizó?
**LoRA (Low-Rank Adaptation)** es una técnica de fine-tuning eficiente que:
- Congela los pesos originales del modelo.
- Entrena solo matrices pequeñas adicionales (adaptadores).
- Reduce drásticamente el uso de memoria y tiempo de entrenamiento.

Esto permite entrenar el tutor académico incluso en hardware limitado.

---

## PEFT (Parameter-Efficient Fine-Tuning)
Se utilizó la librería **PEFT** para implementar LoRA.

- Tamaño del PEFT: pequeño (solo adaptadores)
- Ventaja: no se guarda el modelo completo, solo los pesos LoRA
- Resultado: checkpoints ligeros y fáciles de versionar

---

## Dataset Utilizado
El dataset está compuesto por **preguntas y respuestas académicas**, enfocadas en:
- Inteligencia Artificial
- Conceptos básicos y derivados
- Explicaciones teóricas tipo tutor

Formato del dataset:
- Archivos `.jsonl`
- Campos principales: `instruction`, `input`, `output`

Ejemplo:
```json
{
  "instruction": "Explica qué es el aprendizaje supervisado",
  "input": "",
  "output": "El aprendizaje supervisado es un tipo de aprendizaje automático..."
}
```

El dataset fue dividido en:
- Entrenamiento (`train.jsonl`)
- Validación (`val.jsonl`)
- Pruebas (`test.jsonl`)

---

## Procesamiento de Datos
Se realizó un preprocesamiento que incluyó:
- Limpieza de texto
- Normalización de preguntas
- Conversión al formato esperado por el modelo
- Tokenización usando el tokenizer del modelo base

Script utilizado:
- `procesado_datos.py`

---

## Entrenamiento del Modelo
El entrenamiento se realizó mediante:
- Script: `train_lora.py`
- Técnica: Fine-tuning con LoRA
- Optimización: solo adaptadores entrenables
- Checkpoints generados automáticamente

Se generó al menos un checkpoint funcional (`checkpoint-66`).

---

## Estructura del Proyecto
```
Proyecto_Tutor/
│── data/
│   ├── raw/
│   └── processed/
│── models/
│   └── tutor-lora/
│── src/
│   ├── train_lora.py
│   ├── chatt_tutor.py
│   └── procesado_datos.py
│── requirements.txt
│── README.md
```

---

## Uso del Tutor Académico
El tutor se ejecuta mediante:
- Script: `chatt_tutor.py`
- Permite realizar preguntas académicas
- El modelo responde con explicaciones educativas

---

## Control de Versiones
- Repositorio gestionado con Git y GitHub
- Se ignoró el entorno virtual (`venv`) mediante `.gitignore`
- Solo se versionaron:
  - Código
  - Dataset procesado
  - Adaptadores LoRA
  - Documentación

---

## Conclusión
Este proyecto demuestra la aplicación práctica de:
- Modelos Transformer
- Fine-tuning eficiente con LoRA
- Uso de datasets académicos
- Desarrollo de un tutor inteligente enfocado en educación

El resultado es un **tutor académico funcional**, ligero y especializado en temas de Inteligencia Artificial.
