import numpy as np
from pathlib import Path
from PIL import Image
import gradio as gr
import tensorflow as tf

# Orden de clases del modelo
CLASS_NAMES = ['ants', 'cats', 'dogs', 'ladybugs', 'turtles']
IMG_SIZE = (128, 128)
MODEL_PATH = Path(__file__).resolve().parent / "animal_classifier_optimized.h5"



def load_model():
    if not hasattr(load_model, "_model"):
        load_model._model = tf.keras.models.load_model(MODEL_PATH)
    return load_model._model


def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr



def predict(image):
    """Clasifica la imagen y retorna las probabilidades por clase."""
    if image is None:
        return None
    
    try:
        # Convertir a PIL Image si es necesario
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        arr = preprocess_image(image)
        model = load_model()
        preds = model.predict(arr, verbose=0)[0]
        
        # Crear diccionario de resultados para Gradio
        results = {name: float(prob) for name, prob in zip(CLASS_NAMES, preds)}
        return results
    
    except Exception as exc:
        raise gr.Error(f"Error al procesar la imagen: {exc}")


# Crear la interfaz
with gr.Blocks(title="Clasificador de Animales", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐾 Clasificador de Animales
        
        Sube una imagen de un animal para clasificarlo. El modelo puede identificar:
        **hormigas, gatos, perros, mariquitas y tortugas**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Imagen de entrada",
                type="pil",
                height=320
            )
            predict_btn = gr.Button("🔍 Predecir", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_label = gr.Label(
                label="Resultados de clasificación",
                num_top_classes=5
            )
    
    gr.Markdown(
        """
        ---
        ### Instrucciones:
        1. Arrastra una imagen o haz clic para seleccionar
        2. Presiona el botón **Predecir**
        3. Observa las probabilidades de cada clase
        """
    )
    
    # Conectar eventos
    predict_btn.click(fn=predict, inputs=image_input, outputs=output_label)
    image_input.change(fn=predict, inputs=image_input, outputs=output_label)


if __name__ == "__main__":
    demo.launch()
