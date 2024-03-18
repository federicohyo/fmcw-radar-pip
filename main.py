import torch
import numpy as np
from scipy import constants
import argparse
import pickle

import json

import h5py

import sys
sys.path.append('../')
# from data_module.MarshalSigns import *
# from data_module.MarshalSignsJIT import *

from data_module.MarshalSimple import DataModuleRadarDynamicPreprocessing, DataSetRadarDynamicPreprocessing, DataModuleRadarDynamicPreprocessing_test
from models_architecture.LeViTJim import LeViTJim
from models_architecture.MobileNetV2_grey import MobileNetV2_grey
from models_architecture.efficientnetb1 import EfficientNetB1, EfficientNetB1_custom
from models_architecture.levit_timm import LeViT_128s, LeViT_128
from models_architecture.moganet import *
from models_architecture.moganet import custom_moganet_pretrained
from models_architecture.two_channels_two_models import TwoChannelsTwoModels

import torch.nn.utils.prune as prune
from model_compression.pruning import prune_model, prune_model_iterative, human_readable_count, count_model_parameters, count_model_parameters_nonzero, get_prunable_parameters
from model_compression.quantization import QuantizationMarshalling, print_model_size

from sklearn.metrics import confusion_matrix

from tqdm import tqdm, trange
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import models

from torchinfo import summary

from processing.parallelized_preprocessing import *

from configurations_grid import *

from datetime import datetime

## Grad-CAM for Compact LeViT
from PIL import Image
from utils.gradcam_lib import create_figure_grad_cam, create_plot_input_gradcam, create_plot_input_gradcam_correct_incorrect
from utils.plot_training_results import plot_acc_loss, plot_confusion_matrix
from utils.train_validation_testing import train_model, validate_model, test_model
from utils.show_examples import plot_examples
#importlib.reload(gradcam)  # reload the latest gradcam file

def seed_everything(seed: int = 0):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description='Grid search for radar preprocessing')
    # add argument for job number, it is so multiple jobs can run at the same dataset
    parser.add_argument('-j','--job_number', type=int, default=0, dest = 'job_number' , help='job number')
    # add argument for only displaying the processing results and not train, so debug mode
    parser.add_argument('-d', '--debug', action='store_true', dest='debugging_mode', help='debug mode')
    parser.add_argument('-d2', '--debug_2', action='store_true', dest='debugging_mode_2', help='debug mode 2 only does one batch')
    parser.add_argument('-nsp', '--not_show_progress', action='store_true', dest='not_show_progress', help='show progress bar')
    parser.add_argument('-pdm', '--prepared_data_module', action='store_true', dest='prepared_data_module', help='use prepared data module')
    parser.add_argument('-ppm', '--prepared_preprocessing_module', action='store_true', dest='prepared_preprocessing_module', help='use prepared preprocessing module')
    parser.add_argument('-uts', '--use_test_set', action='store_true', dest='use_test_set', help='use test set')
    parser.add_argument('-nt',  '--no_training',  action='store_false', default=True , dest='training', help='no training')

    return parser.parse_args()

def default_converter(o):
    if isinstance(o, torch.dtype):
        return str(o)
    elif isinstance(o, np.int64):
        return int(o)  # Convert numpy int64 to a native Python int

    raise TypeError(f"Object of type '{o.__class__.__name__}' is not JSON serializable")

if __name__ == "__main__":

    # How to execute program, example:
    # python grid_search_marshalling.py -j 0 -d

    # Parse arguments
    args = parse_args()
    # print if debugging mode is on
    print('job_number', args.job_number)
    print('debugging_mode', args.debugging_mode)

    precision = 16
    learning_rate = 1e-3

    # Create datasets and dataloaders
    # data_dir = '/local/efficient_radar_pipeline/'
    # data_dir = './radar8Ghz-DVS-marshaling_signals_20220421_rad/'
    
    if args.prepared_data_module:
        with open('./data_module/prepared_datamodule.pkl', 'rb') as f:
            datasetmodule = pickle.load(f)
        with open('./data_module/prepared_datamodule_test.pkl', 'rb') as f:
            datasetmodule_test = pickle.load(f)
    else:
        datasetmodule = DataModuleRadarDynamicPreprocessing(data_dir,frame_limit, sample_limit)
        with open('./data_module/prepared_datamodule.pkl', 'wb') as f:
            pickle.dump(datasetmodule, f)
        datasetmodule_test = DataModuleRadarDynamicPreprocessing_test(data_dir_test, frame_limit)
        with open('./data_module/prepared_datamodule_test.pkl', 'wb') as f:
            pickle.dump(datasetmodule_test, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prefix = './models/'

    grid = []

    # For debugging
    # grid.append(debugging_config)

    # grid.append(baseline)
    # grid.append(la_grande_finale_v4)
    
    # grid.append(la_grande_finale_float16)

    # grid.extend(testing_preprocessing_window)
    # grid.append(hamming_mean_removal)
    # grid.extend(no_delta)
    # grid.extend(testing_data_type)

    # grid.append(baseline_moganet)
    # grid.extend(combinations_features_only_one)
    # grid.extend(baseline_combination_data_aug)
    # grid.append(baseline_levit_official_128s)

    # grid.extend(combinations_of_features)

    # grid.append(range_doppler_mu_doppler)

    # grid.append(baseline_mobilenetv2)
    # grid.append(mobilenetv2_pruning)
    # grid.append(moganet_pruning) 
    # grid.append(la_grande_finale_pruning_unstructured_03)
    # grid.append(la_grande_finale_pruning_unstructured_array[2])
    # grid.extend(la_grande_finale_pruning_unstructured_array)
    # grid.append(la_grande_finale_pruning_unstructured_iterative_array[2])
    # grid.append(la_grande_finale_pruning_unstructured_iterative_array[6])
    # grid.extend(la_grande_finale_pruning_unstructured_iterative_array[len(la_grande_finale_pruning_unstructured_iterative_array)-1])
    # grid.append(la_grande_finale_pruning_unstructured_iterative_array[len(la_grande_finale_pruning_unstructured_iterative_array)-1])
    # grid.extend(la_grande_finale_pruning_unstructured_iterative_array_even_further)
    # grid.append(la_grande_finale_pruning_unstructured_iterative_array_even_further[2])
    # grid.append(la_grande_finale_best_pruning)

    grid.append(efficientnetb1_ptq_x86)
    # grid.append(efficientnetb1_qat_x86_epochs_2)
    # grid.append(baseline_levit_official_128)

    # grid.append(efficientnetb1_ptq_x86_optimized_pipeline)
    # grid.extend(array_increasing_in_epochs_qat_x86_optmized_pipeline)

    # grid.extend([efficientnetb1_ptq_x86_optimized_pipeline,efficientnetb1_qat_x86_epochs_2_optimzied_pipeline])

    # grid.append(baseline_data_aug_random_horizontal_flip)
    # grid.append(baseline_mean_removal_no_delta)
    # grid.append(baseline_float8_no_filtering)

    grid.append(efficientnetb1_ptq_x86_optimized_pipeline)

    file_name_best_model = []
    best_confusion_matrix = None

    for square in grid:
        # seed_everything(seed)
        seed_everything(seed)

        date_now = datetime.now()
        date_time = date_now.strftime("%Y_%m_%d_%H_%M_%S")
        if args.debugging_mode:
            log_dir_path = './trashcan/'
        else:
            log_dir_path = square['log_dir_path']
        print('log_dir_path', log_dir_path + date_time)
        filename_save_stats = f'./{log_dir_path}/{date_time}_stats.txt'
        filename_save_model = f'./models/{date_time}_model.pt'

        with open(filename_save_stats, 'w') as file_stats:
            file_stats.write(f'####################### \n')
        file_stats = open(filename_save_stats, 'a')
        file_stats.write(f'Grid search with {square} \n \n')
        file_stats.write(f'configuration_name:{square["config_name"]} \n')
        print(f'configuration_name:{square["config_name"]} \n')
        file_stats.write(f'input_features: {square["input_features"]} \n')
        file_stats.write(f'apply_delta: {square["apply_delta"]} \n')
        file_stats.write(f'change_data_type_to: {square["change_data_type_to"]} \n')
        file_stats.write(f'preprocessing_range_chirp: {square["preprocessing_range_chirp"]} \n')
        file_stats.write(f'postprocessing_range_chirp: {square["postprocesssing_range_chirp"]} \n')
        file_stats.write(f'preprocessing_range_doppler: {square["preprocessing_range_doppler"]} \n')
        file_stats.write(f'postprocessing_range_doppler: {square["postprocessing_range_doppler"]} \n')
        file_stats.write(f'rad_transform: {square["rad_transform"]} \n')
        file_stats.write(f'data_aug: {square["data_augmentation"]} \n')
        file_stats.write(f'image_size_save_hdf5: {square["image_size_save_hdf5"]} \n')
        file_stats.write(f'image_size_model: {square["image_size_model"]} \n')
        file_stats.write(f'####################### \n')        

        # Model initialization and definint at the moment static parameters
        input_features = square['input_features']
        apply_delta = square['apply_delta']
        change_data_type_to = square['change_data_type_to']
        preprocessing_range_chirp = square['preprocessing_range_chirp']
        postprocessing_range_chirp = square['postprocesssing_range_chirp']
        preprocessing_range_doppler = square['preprocessing_range_doppler']
        postprocessing_range_doppler = square['postprocessing_range_doppler']
        rad_transform = square['rad_transform']
        data_aug = square['data_augmentation']
        image_size_model = square['image_size_model']       # Needed for augmentaition of range velocity
        image_size_hdf5 = square['image_size_save_hdf5']    # Needed for augmentation of range velocity
        if 'epochs' in square:
            N_EPOCHS = square['epochs']
        pretrained = True
        if 'pretrained' in square:
            pretrained = square['pretrained']
        
        load_model_weights = False
        if 'load_model_path' in square:
            # print("I come here")
            load_model_weights = True
            load_model_path = square['load_model_path']
            print('load_model_path', load_model_path)

        pruning_the_model = False
        if 'pruning_configuration' in square:
            pruning_the_model = True
            pruning_configuration = square['pruning_configuration']

        model_name = 'efficientnetb1'
        if 'model_name' in square:
            model_name = square['model_name']

        quantize_model = False
        if "quantization" in square:
            quantize_model = True
            quantization = square['quantization']
            print(square['quantization'])
            file_stats.write(f'quantization: {square["quantization"]} \n')

        use_test_set = False
        if "use_test_set" in square:
            use_test_set = square['use_test_set']

        input_channels = len(input_features)
        image_size = image_size_model[0]
        num_classes = 11
        file_stats.write(f'Configuration ML model \n')
        file_stats.write(f'input_features: {input_features} \n')
        file_stats.write(f'input_channels: {input_channels} \n')
        file_stats.write(f'image_size: {image_size_model[0]} \n')
        file_stats.write(f'num_classes: {num_classes} \n')

        # Create report values
        dummy_signal = torch.zeros((1, fft_y, fft_x))
        report_values = create_report_preprocessing(dummy_signal, fft_x, fft_y, input_features, apply_delta ,preprocessing_range_chirp, postprocessing_range_chirp, preprocessing_range_doppler, postprocessing_range_doppler, change_data_type_to, rad_transform)
        print(visualize_nested_dict(report_values))
        file_stats.write(f'report_values: {json.dumps((report_values), indent=2, default=default_converter)} \n')
        total_ops, total_memory = sum_ops_memory(report_values)
        file_stats.write(f'total_operations_preprocessing {human_readable_ops(total_ops)} \n')
        file_stats.write(f'total_forward_memory_preprocessing {human_readable_size(total_memory)} \n')
        print('total operations:', human_readable_ops(total_ops))
        print('total memory:', human_readable_size(total_memory))

        # Do preprocessing here of the stated settings
        print('fftx',fft_x)
        print('ffty',fft_y)

        cache_file_train = f'{cache_file_dir}/radar_data_train_{args.job_number}.hdf5'
        cache_file_test = f'{cache_file_dir}/radar_data_test_{args.job_number}.hdf5'
        if not args.prepared_preprocessing_module:
            datasetmodule.preprocess_data_and_store( data_dir , cache_file_train, fft_Nx = fft_x, fft_Ny = fft_y, features = input_features, apply_delta = apply_delta ,preprocessing_range_chirp = preprocessing_range_chirp, postprocessing_range_chirp = postprocessing_range_chirp, preprocessing_range_doppler = preprocessing_range_doppler, postprocessing_range_doppler = postprocessing_range_doppler, change_data_type_to = change_data_type_to, transform = rad_transform, image_size_x = image_size_hdf5[0], image_size_y = image_size_hdf5[1])
            datasetmodule_test.preprocess_data_and_store( data_dir_test , cache_file_test, fft_Nx = fft_x, fft_Ny = fft_y, features = input_features, apply_delta = apply_delta ,preprocessing_range_chirp = preprocessing_range_chirp, postprocessing_range_chirp = postprocessing_range_chirp, preprocessing_range_doppler = preprocessing_range_doppler, postprocessing_range_doppler = postprocessing_range_doppler, change_data_type_to = change_data_type_to, transform = rad_transform, image_size_x = image_size_hdf5[0], image_size_y = image_size_hdf5[1])

        # if args.use_test_set and use_test_set:
        #     datasetmodule_test.preprocess_data_and_store( data_dir_test , cache_file_test, fft_Nx = fft_x, fft_Ny = fft_y, features = input_features, apply_delta = apply_delta ,preprocessing_range_chirp = preprocessing_range_chirp, postprocessing_range_chirp = postprocessing_range_chirp, preprocessing_range_doppler = preprocessing_range_doppler, postprocessing_range_doppler = postprocessing_range_doppler, change_data_type_to = change_data_type_to, transform = rad_transform, image_size_x = image_size_hdf5[0], image_size_y = image_size_hdf5[1])
        # Get 

        # Create dataloaders
        dataloader_train = []
        dataloader_valid = []
        dataloader_python_train = []
        dataloader_python_test = []
        if use_kfold_leave_one_out:
            mean, std = None, None
            for fold in datasetmodule.data_leave_one_out_two_fold_train:
                dataloader_python = (DataSetRadarDynamicPreprocessing(fold, cache_file_train, transform = data_aug))
                if input_channels > 1:
                    mean, std = dataloader_python.compute_mean_std_off_train()
                    # print('mean', mean)
                    # print('std', std)
                dataloader_train.append(DataLoader(dataloader_python, batch_size = batch_size, shuffle = True, num_workers = 1)) 
            for fold in datasetmodule.data_leave_one_out_two_fold_valid:              
                dataloader_python = (DataSetRadarDynamicPreprocessing(fold, cache_file_train, transform = transforms.Compose([transforms.CenterCrop([image_size_model[0], image_size_model[1]])])  ))
                
                if input_channels > 1:
                    dataloader_python.set_mean_std_for_test_or_validation(mean,std)
                total_samples = len(dataloader_python)  # Assuming `dataset` is your dataset object
                print('total_samples', total_samples)
                dataloader_valid.append(DataLoader(dataloader_python, batch_size = batch_size, shuffle = True, num_workers = 1))

            if args.use_test_set and use_test_set:
                dataloader_python_test = (DataSetRadarDynamicPreprocessing(datasetmodule_test.data_test, cache_file_test, transform = transforms.Compose([transforms.CenterCrop([image_size_model[0], image_size_model[1]])])  ))
                dataloader_test = DataLoader(dataloader_python_test, batch_size = 4, shuffle = True, num_workers = 1)            
        else:
            dataloader_python_train = (DataSetRadarDynamicPreprocessing(datasetmodule.data_full, cache_file_train, transform = data_aug))  
            dataloader_python_valid = (DataSetRadarDynamicPreprocessing(datasetmodule_test.data_test, cache_file_test, transform = transforms.Compose([transforms.CenterCrop([image_size_model[0], image_size_model[1]])])  ))
            
            print("length datalaoder python train",len(dataloader_python_train))
            if not(args.debugging_mode) and False:
                mean, std = dataloader_python_train.compute_mean_std_off_train()
                dataloader_python_train.set_mean_std_for_test_or_validation(mean,std)
                dataloader_python_valid.set_mean_std_for_test_or_validation(mean,std)
            dataloader_train.append(DataLoader(dataloader_python_train, batch_size = batch_size, shuffle = True, num_workers = 1))

            # if input_channels > 1 or model_name == 'moganet':
            # if not(args.debugging_mode):
            dataloader_valid.append(DataLoader(dataloader_python_valid, batch_size = batch_size, shuffle = True, num_workers = 1))

        
        plot_examples(dataloader_train, input_features, log_dir_path = log_dir_path, date_time = date_time, name_title = 'samples_train')
        plot_examples(dataloader_valid, input_features, log_dir_path = log_dir_path, date_time = date_time, name_title = 'samples_valid')


        for fold_idx, (fold_dataloader_train, fold_dataloader_valid) in enumerate(zip(dataloader_train, dataloader_valid)):

            print("Fold:", fold_idx + 1)

            train_logs = {
                'best_acc_valid': 0,
                'best_acc_test': 0,
                'acc_train': [],
                'acc_valid': [],
                'acc_test': [],
                'loss_train': [],
                'loss_valid': [],
                'loss_test': [],
                'best_confusion_matrix_valid': None,
                'best_confusion_matrix_test': None,
                'file_name_best_model_valid': None,
                'file_name_best_model_test': None,
                'best_valid_model_predictions': {
                    'true_labels': [],
                    'pred_labels': [],
                    'loader_idx': [],
                } 
            }

            if model_name == 'mobilenetv2':
                model = MobileNetV2_grey(num_classes = num_classes, input_channels = input_channels)
            elif model_name == 'efficientnetb1':
                # model = EfficientNetB1(num_classes = num_classes, input_channels = input_channels ,pretrained = pretrained) 
                model = EfficientNetB1_custom(num_classes = num_classes, input_channels = input_channels ,pretrained = pretrained) 
            elif model_name == 'vgg':
                model = SimpleVGG(input_channels = input_channels, num_classes = num_classes)
            elif model_name == 'levit':
                model = LeViTJim(input_channels = input_channels, image_size = image_size, num_classes = num_classes, n_convs = 4, stages = 2, dim = (8,16), depth = 4, heads = (2,4), mlp_mult = 2, dropout = 0.1)
            elif model_name == 'levit_128s':
                model = LeViT_128s(num_classes = num_classes, input_channels = input_channels)
            elif model_name == 'levit_128':
                model = LeViT_128(num_classes = num_classes, input_channels = input_channels)
            elif model_name == 'moganet':
                model = custom_moganet_pretrained()
            elif model_name == 'efficientnetb1_two_channels_two_models':
                sub_model = EfficientNetB1(num_classes = num_classes, input_channels = 1 ,pretrained = pretrained)
                model = TwoChannelsTwoModels(sub_model, num_classes = num_classes)
            else:
                ValueError('Model name not found')
            
            if load_model_weights:
                # Check if path exist and otherwise throw Value error
                if not os.path.exists(load_model_path[fold_idx]):
                    raise ValueError(f'Path does not exist: {load_model_path[fold_idx]}')
                print('load_model_path', load_model_path[fold_idx])
                if load_model_path[fold_idx].endswith('.pth'):
                    model = torch.load(load_model_path[fold_idx])
                elif load_model_path[fold_idx].endswith('.pt'):
                    model.load_state_dict(torch.load(load_model_path[fold_idx]))
                else:
                    raise ValueError(f'File format not recognized: {load_model_path[fold_idx]}')

            # file_stats.write(f'{str(summary(model, [(1, 3, image_size_model[0], image_size_model[1])]))}\n')
            file_stats.write(f'{str(summary(model, [(1, input_channels, image_size_model[0], image_size_model[1])]))}\n')
            # exit()
            model = model.to(device)
            file_stats.write(f'model: {model.__class__.__name__} \n')
            # Model initialization and definint at the moment static parameters
            if args.debugging_mode:
                N_EPOCHS = 1

            LR = 0.0005
            if model_name == 'moganet':
                LR = 0.0005

            # optimizer = Adam(model.parameters(), lr = LR)
            optimizer = Adam(model.parameters(), lr = LR, weight_decay = 1E-3)

            # Best learning rate for transformers
            # optimzer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
            # schedular = torch.optim.lr_scheduler.StepLR(step_size = 5, optimizer = optimizer, gamma = 0.3)
            
            file_stats.write(f'Optimizer: {optimizer} \n')
            file_stats.write(f'LR: {LR} \n')

            file_stats.write(f'N_EPOCHS: {N_EPOCHS} \n')
            file_stats.write(f'LR: {LR} \n')
            file_stats.write(f'optimizer: {optimizer} \n')
            file_stats.write(f'criterion: {criterion} \n')
            file_stats.write(f'batch size: {batch_size} \n') 
            file_stats.write(f'pretrained: {pretrained} \n')           

            for epoch in trange(N_EPOCHS, desc = 'training and validating'):
                # print("train model")
                if args.training:
                    train_model(model, fold_dataloader_train, optimizer, args, file_stats, train_logs, epoch)
                
                # print("valid model")
                validate_model(model, fold_dataloader_valid, args, file_stats, train_logs, epoch, date_time = date_time)

            if train_logs['best_acc_valid'] > 1:
                file_stats.write(f"final_best_accuracy: {train_logs['best_acc_valid'].item()}\n")
                print('final_best_accuracy', train_logs['best_acc_valid'].item())

            if pruning_the_model:
                print('pruning_configuration', pruning_configuration)

                if pruning_configuration['iterative_pruning'] == True:

                    file_stats.write(f"Number of parameters before pruning: {human_readable_count(count_model_parameters(model))}\n")
                    print(f"{human_readable_count(count_model_parameters_nonzero(model))} parameters before pruning.")
                    file_stats.write(f"Model size before pruning {print_model_size(model)}")
                    print(f"Model size before pruning {print_model_size(model)}")


                    LR = 0.00005
                    optimizer = Adam(model.parameters(), lr = LR)
                    for i in range(pruning_configuration['itaretive_pruning_steps']):
                        model = prune_model_iterative(pruning_configuration, model, model_name = model_name, iteration = i)
                        for epoch in trange(pruning_configuration['itarative_pruning_epochs'], desc = 'training'):
                            train_model(model, fold_dataloader_train, optimizer, args, file_stats, train_logs, i)
                            validate_model(model, fold_dataloader_valid, args, file_stats, train_logs, i, date_time = date_time)

                    # prune_model(pruning_configuration, model, model_name = model_name)
                    print(f"{human_readable_count(count_model_parameters_nonzero(model))} parameters after pruning.")
                    file_stats.write(f"{human_readable_count(count_model_parameters_nonzero(model))} parameters after pruning.")

                    # Remove the pruned parameters
                    parameters_to_prune = get_prunable_parameters(model)
                    for module, param in parameters_to_prune:
                        prune.remove(module, param)
                    vall_acc, _ = validate_model(model, fold_dataloader_valid, args, file_stats, train_logs, 0, date_time = date_time)

                    # Save the quantized model
                    torch.save(model.state_dict(), f'./models_pruned/{date_time}_model_pruned_{vall_acc}.pt')

                    print(f"Model saved as: ./models_pruned/{date_time}_model_pruned_{vall_acc}.pt")
                    file_stats.write(f"final_best_accuracy_after_pruning: {train_logs['best_acc_valid'].item()}\n")
                    print(f"Model size after pruning {print_model_size(model)}")
                    file_stats.write(f"Model size after pruning {print_model_size(model)}")

                    print(f"{human_readable_count(count_model_parameters_nonzero(model))} parameters after pruning.")
                    file_stats.write(f"{human_readable_count(count_model_parameters_nonzero(model))} parameters after pruning.")
                else:
                    vall_acc, _ = validate_model(model, fold_dataloader_valid, args, file_stats, train_logs, 0 , date_time = date_time)
                    print(f"validation accuracy before pruning {vall_acc}")
                    print(f"{human_readable_count(count_model_parameters_nonzero(model))} parameters after pruning.")
                    file_stats.write(f"{human_readable_count(count_model_parameters_nonzero(model))} parameters after pruning.")
                    file_stats.write(f"Model size before pruning {print_model_size(model)}")
                    print(f"Model size before pruning {print_model_size(model)}")

                    model = prune_model(pruning_configuration, model, model_name = model_name)
                    vall_acc, _ = validate_model(model, fold_dataloader_valid, args, file_stats, train_logs, 0, date_time = date_time)

                    torch.save(model.state_dict(), f'./models_pruned/{date_time}_model_pruned_{vall_acc}.pt')
                    print(f"Model saved as: ./models_pruned/{date_time}_model_pruned_{vall_acc}.pt")

                    print(f"{human_readable_count(count_model_parameters_nonzero(model))} parameters after pruning.")
                    file_stats.write(f"{human_readable_count(count_model_parameters_nonzero(model))} parameters after pruning.")
                    print(f"Model size before pruning {print_model_size(model)}")
                    file_stats.write(f"Model size before pruning {print_model_size(model)}")

                    # model = custom_moganet_pretrained()
                    # model.to(device)
                    # model.load_state_dict(torch.load(f'./models_pruned/{date_time}_model_pruned_{vall_acc}.pt'))
                    # vall_acc, _ = validate_model(model, fold_dataloader_valid, args, file_stats, train_logs, 0, date_time = date_time)
                    # exit()

            else:
                print('no pruning_configuration')

            if quantize_model:

                # print(f"{human_readable_count(count_model_parameters_nonzero(model))} parameters after pruning.")
                print(f"Model size before quantization: {human_readable_count(count_model_parameters(model))} parameters")

                print("Validation before quantization")
                vall_acc, _ = validate_model(model, fold_dataloader_valid, args, file_stats, train_logs, 0, date_time = date_time)
                # continue # Skip the quantization for now 

                quantization_marshalling = QuantizationMarshalling(args, file_stats, train_logs, date_time)
                if quantization['quantization_type'] == 'ptq':
                    model = quantization_marshalling.ptq(model, quantization, fold_dataloader_train, args)
                elif quantization['quantization_type'] == 'qat':
                    model = quantization_marshalling.qat(model, quantization, fold_dataloader_train, optimizer, args)
                else:
                    raise ValueError('Quantization method not found')

                print("Validation after quantization")
                vall_acc, _ = validate_model(model, fold_dataloader_valid, args, file_stats, train_logs, 0, date_time = date_time)
                
                print(f"Model saved as: ./models_quantized/{date_time}_model_quantized_{vall_acc}.pt")
                torch.save(model.state_dict(), f'./models_quantized/{date_time}_model_quantized_{vall_acc}.pt') 
                # file_stats.write(f"final_best_accuracy_after_quantization: {train_logs['best_acc'].item()}\n")
                # print('final_best_accuracy_after_quantization', train_logs['best_acc'].item())

                print(f"Model size after quantization: {human_readable_count(count_model_parameters(model))} parameters")

            # Plotting the accuracy and loss in a single figure
            plot_acc_loss(train_logs['acc_train'], train_logs['acc_valid'], train_logs['loss_train'], train_logs['loss_valid'], log_dir_path = log_dir_path, date_time = date_time ,fold_idx = fold_idx)
            plot_confusion_matrix(train_logs['best_confusion_matrix_valid'], classes = classes_nice_text , log_dir_path = log_dir_path, date_time = date_time ,fold_idx = fold_idx)   

            # create_plot_input_gradcam(model ,dataloader_python_train, log_dir_path = log_dir_path, date_time = date_time, fold_idx = fold_idx )
            create_plot_input_gradcam_correct_incorrect(model, dataloader_python_valid, train_logs, log_dir_path = log_dir_path, date_time = date_time, fold_idx = fold_idx)
            
            if args.use_test_set and use_test_set:
                plot_acc_loss(train_logs['acc_train'], train_logs['acc_test'], train_logs['loss_train'], train_logs['loss_test'], log_dir_path = log_dir_path, date_time = date_time ,fold_idx = fold_idx)
                plot_confusion_matrix(train_logs['best_confusion_matrix_test'], log_dir_path = log_dir_path, date_time = date_time ,fold_idx = fold_idx, set = 'test')

            # Grad-CAM
            if use_gradcam and (input_channels == 1) and (not quantize_model) and (not pruning_the_model):
                create_figure_grad_cam(model, train_logs['file_name_best_model_valid'], model_name, fold_dataloader_valid, log_dir_path = log_dir_path, date_time = date_time ,fold_idx = fold_idx)

            # print(train_logs["best_valid_model_predictions"])