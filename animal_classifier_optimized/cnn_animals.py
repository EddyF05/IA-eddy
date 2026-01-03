import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "CNN_ejemplo" / "animals-dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 60
VAL_SPLIT = 0.15
SEED = 42

# Cargar dataset
train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Obtener nombres de clases 
class_names = train_ds_raw.class_names
num_classes = len(class_names)
print("Clases detectadas:", class_names)

# Saltar elementos que fallen al decodificar 
train_ds = train_ds_raw.ignore_errors()
val_ds = val_ds_raw.ignore_errors()

# Prefetch para rendimiento
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Modelo CNN
def build_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_model()
model.summary()

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
)

# Evaluación en el split de validación (si necesitas test, separa un folder aparte o usa validation_split menor y un test_ds)
val_loss, val_acc = model.evaluate(val_ds)
print(f"Val accuracy: {val_acc:.3f}")

# Guardar modelo
model.save("animals_cnn.keras")
print("Modelo guardado en animals_cnn.keras")