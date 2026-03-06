import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224,224)

# Load base model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.35)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.25)(x)

# Output layer (38 plant disease classes)
outputs = Dense(38, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=outputs)

model.summary()