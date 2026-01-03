import argparse
from pathlib import Path
import tensorflow as tf

# Clases en el mismo orden que se usó al entrenar
CLASSES = ['ants', 'cats', 'dogs', 'ladybugs', 'turtles']
IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def load_dataset(data_dir: Path):
    # Forzar class_names para que coincidan con el orden de entrenamiento
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    # Saltar archivos corruptos/no soportados para evitar fallos de decodificación
    ds = ds.ignore_errors()
    return ds.prefetch(tf.data.AUTOTUNE)


def evaluate(model_path: Path, data_dir: Path):
    model = tf.keras.models.load_model(model_path)
    ds = load_dataset(data_dir)
    loss, acc = model.evaluate(ds, verbose=1)
    print(f"\nAccuracy: {acc:.4f}\nLoss: {loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evalúa accuracy en un folder de prueba")
    parser.add_argument(
        "--model",
        default="animals_cnn.keras",
        help="Ruta al modelo Keras guardado",
    )
    parser.add_argument(
        "--data",
        default="test-animals",
        help="Directorio con subcarpetas por clase (ants/cats/dogs/ladybugs/turtles)",
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    data_dir = Path(args.data).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de datos en {data_dir}")

    evaluate(model_path, data_dir)


if __name__ == "__main__":
    main()
