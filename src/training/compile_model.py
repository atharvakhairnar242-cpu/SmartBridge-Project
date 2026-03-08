from tensorflow.keras.optimizers import Adam

def compile_model(model):

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model