import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator


model_path = "models/plant_disease_model.h5"
valid_dir = "data/raw/plant_disease_dataset/valid"


model = tf.keras.models.load_model(model_path)


datagen = ImageDataGenerator(rescale=1./255)

valid_generator = datagen.flow_from_directory(
    valid_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)


loss, accuracy = model.evaluate(valid_generator)

print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)


predictions = model.predict(valid_generator)

y_pred = np.argmax(predictions, axis=1)

y_true = valid_generator.classes


print("\nClassification Report\n")
print(classification_report(y_true, y_pred, target_names=valid_generator.class_indices.keys()))


cm = confusion_matrix(y_true, y_pred)


plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=valid_generator.class_indices.keys(),
    yticklabels=valid_generator.class_indices.keys()
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()