import matplotlib.pyplot as plt
from configurations_grid import classes

def plot_examples(dataloader, input_features, log_dir_path='./', date_time='2000_01_01_00_00_00', name_title='samples_train', figsize=(15, 10)):
    # Save x random samples of the dataset
    for j, feature in enumerate(input_features):
        fig, axs = plt.subplots(3, 5, figsize=figsize)  # Adjust figure size
        axs = axs.flatten()
        for i, batch in enumerate(dataloader[0]):
            if i == 15:
                break
            x, y, label, _, _, _, _ = batch
            axs[i].imshow(x[0, j, :, :].numpy())
            # Give title of the same name as the y value, adjust y position if needed
            axs[i].set_title(f"{classes[y[0]]}", fontsize=14)  # Adjust font size
            axs[i].axis('off')
        # Automatically adjust subplot parameters to give specified padding
        plt.tight_layout()
        # Save the figure
        plt.savefig(f'./{log_dir_path}/{date_time}_{name_title}_{feature}.png', dpi=100, bbox_inches='tight')