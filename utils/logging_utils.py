
# logging_utils.py
import matplotlib.pyplot as plt
import os

def save_plots(history, fold=None):
    # Define the folder for saving plots
    plot_dir = "logs/plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    if fold:
        plt.savefig(os.path.join(plot_dir, f'accuracy_fold{fold}.png'))
    else:
        plt.savefig(os.path.join(plot_dir, 'accuracy.png'))
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if fold:
        plt.savefig(os.path.join(plot_dir, f'loss_fold{fold}.png'))
    else:
        plt.savefig(os.path.join(plot_dir, 'loss.png'))
    plt.close()































# import matplotlib.pyplot as plt
# import os

# def save_plots(history, fold=None):
#     if not os.path.exists("logs/plots"):
#         os.makedirs("logs/plots")

#     # Generate plot filenames based on the fold number
#     fold_str = f"_fold{fold}" if fold else ""

#     # Plot accuracy
#     plt.figure()
#     plt.plot(history.history['accuracy'], label='train accuracy')
#     plt.plot(history.history['val_accuracy'], label='validation accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.savefig(f'logs/plots/accuracy{fold_str}.png')
#     plt.close()

#     # Plot loss
#     plt.figure()
#     plt.plot(history.history['loss'], label='train loss')
#     plt.plot(history.history['val_loss'], label='validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(f'logs/plots/loss{fold_str}.png')
#     plt.close()












# import matplotlib.pyplot as plt
# import os

# def save_plots(history):
#     if not os.path.exists("logs/plots"):
#         os.makedirs("logs/plots")
    
#     # Plot accuracy
#     plt.figure()
#     plt.plot(history.history['accuracy'], label='train accuracy')
#     plt.plot(history.history['val_accuracy'], label='validation accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.savefig('logs/plots/accuracy.png')
#     plt.close()

#     # Plot loss
#     plt.figure()
#     plt.plot(history.history['loss'], label='train loss')
#     plt.plot(history.history['val_loss'], label='validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig('logs/plots/loss.png')
#     plt.close()
