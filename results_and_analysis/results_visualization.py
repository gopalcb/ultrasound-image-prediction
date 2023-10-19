from matplotlib import pyplot as plt


def visualize_training_loss_accuracy(train_loss, train_accuracy):
    fig, ax = plt.subplots(2, 1, figsize=(12,12))
    ax[0].plot(train_loss)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')

    ax[1].plot(train_accuracy)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training Accuracy')

    plt.tight_layout()
    plt.show()