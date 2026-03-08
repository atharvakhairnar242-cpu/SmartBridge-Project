from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, valid_dir):

    # Training generator with minimal augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # normalize pixels
        rotation_range=15,           # small rotations
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Validation generator (no augmentation)
    valid_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),      # MobileNetV2 input size
        batch_size=32,
        class_mode="categorical"
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical"
    )

    return train_generator, valid_generator