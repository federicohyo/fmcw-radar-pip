import cv2
import numpy as np

import torch

import matplotlib.pyplot as plt

from configurations_grid import classes, classes_nice_text

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):    # x is data of a batch
        self.gradients = []   # clear
        self.activations = []  # clear
        return self.model(x)  # trigger forward_hook and backward_hook

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
             model,
             target_layers,
             reshape_transform=None,
             use_cuda=True):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True) # mean in height and width

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):  # number of imgs in the current batch 
            loss = loss + output[i, target_category[i]] # sum of y_c
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        
#         print("weight.shape:", weights.shape)
#         print("_______________________")
#         print("acti.shape:", activations.shape)
        
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                      for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                  for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            
#             print("Original_______________________")
#             print(cam.shape)
#             print(cam)
#             print("Later_______________________")
            
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # Forward to get output logits without softmax    
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            #print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad() # clear historial grad
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

def show_cam_on_image(img: np.ndarray,
               mask: np.ndarray,
               use_rgb: bool = False,     
               colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    #print(mask)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    
    #cam = heatmap
    #print(cam)
    
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def create_figure_grad_cam(model, file_name_best_model, model_name, data_loader, log_dir_path = './', date_time = '2000_00_00_00_00_00' ,fold_idx = 0):
    
    model.eval()
    model.load_state_dict(torch.load(f'./models/{file_name_best_model}'))
    # x.x test if model.backbone works
    if model_name == 'efficientnetb1':
        target_layers = [model.base_model.conv_stem]
    elif model_name == 'levit':
        target_layers = [model.conv_embedding]
    elif model_name == 'levit_128s':
        target_layers = [model.base_model.stem[0].linear]
    elif model_name == 'moganet':
        target_layers = [model.patch_embed1.projection[0]]
    else:
        ValueError("model_name not supported, define it in create_figure_grad_cam")

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    for batch in data_loader:
        x, y, labels, _, _, _, _ = batch

        number_of_images = 16

        fig, ax = plt.subplots(number_of_images , 2, figsize=(12, 5*number_of_images))
        img_count = 0

        for image, label in zip(x,y):
            
            image = image.cuda()
            image = image.unsqueeze(dim=0)

            grayscale_cam = cam(input_tensor=image, target_category=None)
            grayscale_cam = grayscale_cam[0, :]

            output = model(image)
            pred = torch.argmax(output,dim = 1).item()
            image = image.squeeze().squeeze().cpu().numpy()

            ax[img_count,0].imshow(image)
            ax[img_count,0].set_title(f"Label: {classes[int(label.item())]}")
            ax[img_count,0].axis('off')

            ax[img_count,1].imshow(grayscale_cam, cmap = 'hot')
            ax[img_count,0].set_title(f"Label: {classes[int(label.item())]}")
            ax[img_count,1].axis('off')
            img_count += 1

            if img_count == number_of_images:
                break

        fig.savefig(f'./{log_dir_path}/{date_time}_gradcam_fold_{fold_idx}.png', dpi = 100, bbox_inches = 'tight')

        break


def create_plot_input_gradcam(model ,dataloader, log_dir_path = './', date_time = '2000_01_01_00_00_00', fold_idx = 0, name_title = 'input_with_gradcam_per_label'):

    # Get pandas data frame with meta data
    # print(datasetmodule.data_as_df)
    # print(datasetmodule.data_as_df.size)

    # Assuming df_full is your DataFrame
    unique_labels = dataloader.df['labels'].unique()
    # print(unique_labels)
    # print(dataloader.df)

    model.eval()

    # Only applicable for Efficientnet see create_figure_grad_cam for other defining the first layer
    target_layers = [model.base_model.conv_stem]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    fig, axs = plt.subplots(nrows=2, ncols=11, figsize=(22,4))
    for i, rad_idx in enumerate(dataloader.df.groupby('labels').agg({'rad_chunk_idx': 'first'}).reset_index()['rad_chunk_idx'].to_numpy()):
        matching_rows = dataloader.df.loc[dataloader.df['rad_chunk_idx'] == rad_idx]
        x, y, label, _, _, _, _ = dataloader.__getitem__(int(matching_rows.index.item()))
        # print(label)
        axs[0, i].imshow(x[0].numpy())
        axs[0, i].axis('off')

        image = x.cuda()
        image = image.unsqueeze(dim = 0)
        # image = image.unsqueeze(dim = 0)
        # image.unsqueeze(dim=0)

        grayscale_cam = cam(input_tensor = image, target_category = None)
        grayscale_cam = grayscale_cam[0,:]

        output = model(image)

        axs[1,i].imshow(grayscale_cam, cmap = 'hot')
        axs[1,i].axis('off')
    plt.savefig(f'./{log_dir_path}/{date_time}_{name_title}.png', dpi=100, bbox_inches='tight')

def fill_check_and_sort_predictions(correct, incorrect, total_labels=11):
    expected_labels = set(range(total_labels))
    
    # Extracting labels from both lists
    existing_labels_correct = {label for label, _, _ in correct}
    existing_labels_incorrect = {label for label, _, _ in incorrect}
    
    # Identifying missing labels
    missing_labels_correct = expected_labels - existing_labels_correct
    missing_labels_incorrect = expected_labels - existing_labels_incorrect
    
    # Filling missing labels with placeholders
    for label in missing_labels_correct:
        correct.append((label, None, None))
    for label in missing_labels_incorrect:
        incorrect.append((label, None, None))
    
    # Sorting both lists by their labels
    correct_sorted = sorted(correct, key=lambda x: x[0])
    incorrect_sorted = sorted(incorrect, key=lambda x: x[0])
    
    return correct_sorted, incorrect_sorted

def create_plot_input_gradcam_correct_incorrect(model ,dataloader, train_logs ,log_dir_path = './', date_time = '2000_01_01_00_00_00', fold_idx = 0, name_title = 'input_with_gradcam_per_label'):

    # Only applicable for Efficientnet see create_figure_grad_cam for other defining the first layer
    target_layers = [model.base_model.conv_stem]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    correct_predictions = []
    incorrect_predictions = []
    correct_labels_added = set()
    incorrect_labels_added = set()

    expected_labels = set(range(len(classes)))

    for true_label, pred_label, loader_idx in zip(
        train_logs['best_valid_model_predictions']['true_labels'],
        train_logs['best_valid_model_predictions']['pred_labels'],
        train_logs['best_valid_model_predictions']['loader_idx'],
    ):
        if true_label == pred_label:
            if true_label not in correct_labels_added:
                correct_predictions.append((true_label, pred_label ,loader_idx))
                correct_labels_added.add(true_label)
        else:
            if true_label not in incorrect_labels_added:
                incorrect_predictions.append((true_label, pred_label, loader_idx))
                incorrect_labels_added.add(true_label)

    # correct_predictions.sort(key=lambda x: x[0])
    # incorrect_predictions.sort(key=lambda x: x[0])

    correct_predictions, incorrect_predictions = fill_check_and_sort_predictions(correct_predictions,incorrect_predictions)

    # print(correct_predictions)
    # print(incorrect_predictions)

    fig, axs = plt.subplots(nrows=2, ncols=11, figsize=(22,4))
    for i, (label, pred, rad_idx) in enumerate(correct_predictions):
        # matching_rows = dataloader.df.loc[dataloader.df['rad_chunk_idx'] == rad_idx]
        if pred == None:
            x = torch.zeros((1,240,240))
            y = None
        else:
            x, y, label, _, _, _, _ = dataloader.__getitem__(rad_idx)
    
        axs[0, i].set_title(f"{classes_nice_text[int(label)]}")
        axs[0, i].imshow(x[0].numpy())
        axs[0, i].axis('off')

        image = x.cuda()
        image = image.unsqueeze(dim = 0)
        # image = image.unsqueeze(dim = 0)
        # image.unsqueeze(dim=0)

        grayscale_cam = cam(input_tensor = image, target_category = None)
        grayscale_cam = grayscale_cam[0,:]

        output = model(image)
        if pred == None:
            output == np.zeros((240,240))

        axs[1,i].set_title(f"{classes_nice_text[int(pred)]}")
        axs[1,i].imshow(grayscale_cam, cmap = 'hot')
        axs[1,i].axis('off')
    
    plt.savefig(f'./{log_dir_path}/{date_time}_grad_cam_correct.png', dpi=100, bbox_inches='tight')


    fig, axs = plt.subplots(nrows=2, ncols=11, figsize=(22,4))
    for i, (label, pred ,rad_idx) in enumerate(incorrect_predictions):
        # matching_rows = dataloader.df.loc[dataloader.df['rad_chunk_idx'] == rad_idx]
        if pred == None:
            x = torch.zeros((1,240,240))
            pred = None
            y = None
        else:
            x, y, label, _, _, _, _ = dataloader.__getitem__(rad_idx)
        # print(label)

        axs[0, i].set_title(f"{classes_nice_text[int(label)]}")
        axs[0, i].imshow(x[0].numpy())
        axs[0, i].axis('off')

        image = x.cuda()
        image = image.unsqueeze(dim = 0)
        # image = image.unsqueeze(dim = 0)
        # image.unsqueeze(dim=0)

        grayscale_cam = cam(input_tensor = image, target_category = None)
        grayscale_cam = grayscale_cam[0,:]

        output = model(image)
        if pred == None:
            output = np.zeros((240,240))
            axs[1,i].set_title(f"Absent")
        else:
            axs[1,i].set_title(f"{classes_nice_text[int(pred)]}")
        
        axs[1,i].imshow(grayscale_cam, cmap = 'hot')
        axs[1,i].axis('off')
    
    plt.savefig(f'./{log_dir_path}/{date_time}_grad_cam_incorrect.png', dpi=100, bbox_inches='tight')

    # exit()

    # fig, axs = plt.subplots(nrows=2, ncols=11, figszie = (22,8))
