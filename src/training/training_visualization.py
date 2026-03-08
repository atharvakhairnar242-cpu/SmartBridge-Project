import matplotlib.pyplot as plt


def plot_training_history(history):

    # Accuracy Graph
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get('accuracy',[]), label='train_accuracy')
    plt.plot(history.history.get('val_accuracy',[]), label='val_accuracy')
    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(history.history.get('loss',[]), label='train_loss')
    plt.plot(history.history.get('val_loss',[]), label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title('Loss')
    plt.show()