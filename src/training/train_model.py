import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.model_building.load_mobilenet import load_model
from src.training.compile_model import compile_model
from src.training.callbacks import get_callbacks
from src.training.training_visualization import plot_training_history


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
N_LAST_LAYERS = 10
SEED = 1337


# Dataset directories
train_dir = "data/processed/train"
val_dir = "data/processed/val"


# Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)


# Load dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)


# Number of classes
num_classes = train_generator.num_classes


# Load MobileNetV2 model
model = load_model(num_classes)


# Compile model
model = compile_model(model)


# Get callbacks
callbacks = get_callbacks()


# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)


# Save final model
model.save("models/plant_disease_model.h5")

print("Training completed. Model saved in models folder.")


# Plot training graphs
plot_training_history(history)