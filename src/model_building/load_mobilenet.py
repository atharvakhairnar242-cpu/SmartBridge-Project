import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2


IMG_SIZE = (224, 224)


base_model = MobileNetV2(
    weights='imagenet',        
    include_top=False,         
    input_shape=(224, 224, 3) 
)


for layer in base_model.layers:
    layer.trainable = False

print("MobileNetV2 base model loaded successfully")