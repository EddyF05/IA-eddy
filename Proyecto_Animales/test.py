import argparse
from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Usa el mismo modelo y clases que flet_app por defecto
DEFAULT_MODEL = "animal_classifier_optimized"
CLASS_NAMES = ["ants", "cats", "dogs", "ladybugs", "turtles"]
TARGET_SIZE = (128, 128)


def cargar_modelo(model_path: Path) -> object:
    if not model_path.exists():
        alternativa = model_path.with_suffix(".h5")
        if alternativa.exists():
            model_path = alternativa
        else:
            raise FileNotFoundError(f"No se encontró el modelo en {model_path} ni {alternativa}")
    return load_model(model_path)


def preprocess_image(img_path: Path, target_size: tuple[int, int], normalizar: bool) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    arr = img.astype("float32")
    if normalizar:
        arr /= 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predecir(imagen: Path, modelo, labels: list[str], target_size=(128, 128), normalizar=True):
    tensor = preprocess_image(imagen, target_size, normalizar)
    pred = modelo.predict(tensor)
    idx = int(np.argmax(pred))
    confianza = float(pred[0][idx] * 100)
    return labels[idx], confianza, pred[0]


def main():
    parser = argparse.ArgumentParser(description="Test de clasificador de animales")
    parser.add_argument(
        "imagen",
        nargs="?",
        default="test-animals/hormiga2.jpg",
        help="Ruta de la imagen a clasificar",
    )
    parser.add_argument(
        "--modelo",
        default=DEFAULT_MODEL,
        help="Ruta del modelo (.keras o .h5). Por defecto coincide con flet_app (animals_cnn.keras)",
    )
    parser.add_argument(
        "--sin-normalizar",
        action="store_true",
        help="No dividir por 255. Úsalo si tu modelo se entrenó sin rescale (como flet_app actual)",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=CLASS_NAMES,
        help="Lista de clases en el orden del modelo",
    )
    args = parser.parse_args()

    raiz = Path(__file__).resolve().parent
    modelo = cargar_modelo((raiz / args.modelo).resolve())
    labels = args.labels

    imagen_path = Path(args.imagen)
    if not imagen_path.is_absolute():
        imagen_path = raiz / imagen_path

    clase, confianza, vector = predecir(
        imagen_path,
        modelo,
        labels,
        target_size=TARGET_SIZE,
        normalizar=not args.sin_normalizar,
    )

    print(f"Imagen: {imagen_path}")
    print(f"Clase predicha: {clase}")
    print(f"Confianza: {confianza:.2f}%")
    print(f"Vector de probabilidades: {vector}")

    img_display = cv2.cvtColor(cv2.imread(str(imagen_path)), cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    plt.text(
        img_display.shape[1] - 10,
        30,
        f"{clase} ({confianza:.1f}%)",
        fontsize=14,
        color="white",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
