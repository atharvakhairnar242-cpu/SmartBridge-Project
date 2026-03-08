from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def get_callbacks():

    # Save best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        "/content/mobilenetv2_best.keras",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    # Reduce learning rate when validation loss plateaus
    reducelr = ReduceLROnPlateau(
        monitor='val loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # Stop training if no improvement
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
       
    return [early_stop, checkpoint, reducelr]