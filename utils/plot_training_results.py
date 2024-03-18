import numpy as np
import matplotlib.pyplot as plt

from configurations_grid import classes

def plot_acc_loss(acc_train, acc_valid, loss_train, loss_valid, log_dir_path = './', date_time = '2000_01_01_00_00_00', fold_idx = 0):
    # Save image of plotting the accuracy and loss in a single figure
    fig, axs = plt.subplots(2, 1, figsize=(15, 6))
    axs[0].plot(acc_train, label = 'train')
    axs[0].plot(acc_valid, label = 'valid')
    axs[0].set_title('Accuracy')
    axs[0].legend()
    axs[1].plot(loss_train, label = 'train')
    axs[1].plot(loss_valid, label = 'valid')
    axs[1].set_title('Loss')
    axs[1].legend()
    plt.savefig(f'./{log_dir_path}/{date_time}_acc_loss_{fold_idx}.png')

def plot_confusion_matrix(best_confusion_matrix, classes, log_dir_path='./', date_time='2000_01_01_00_00_00', fold_idx=0, set='valid'):
    # Define a single font size variable
    print('plot_confusion_matrix')
    
    font_size = 12

    # Create the plot for confusion matrix
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.matshow(best_confusion_matrix, cmap=plt.cm.Blues)
    # plt.colorbar(cax)

    # Add values to the confusion matrix
    for (i, j), val in np.ndenumerate(best_confusion_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', fontsize=font_size)

    # Offset the tick marks and add class labels
    # ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    # ax.set_xticklabels(classes, rotation=90, fontsize=font_size)
    ax.set_yticklabels(classes, fontsize=font_size)
    ax.set_xticks([])  # This line turns off the x-axis values


    # Set the labels for the axes
    ax.set_xlabel('Predicted Labels', fontsize=font_size)
    ax.set_ylabel('True Labels', fontsize=font_size)
    # plt.title(f'Confusion Matrix', fontsize=font_size)

    # Save the figure
    # plt.savefig(f'{log_dir_path}/{date_time}_confusion_matrix_fold_{fold_idx}_{set}.png', dpi=150)
    plt.savefig(f'{log_dir_path}/{date_time}_confusion_matrix_fold_{fold_idx}_{set}.png', dpi=150, bbox_inches= 'tight')


# def create_plot_input_gradcam(model, dataloader, )
