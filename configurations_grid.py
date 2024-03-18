from torchvision import transforms
import torch
import numpy as np
import math

# Confiugre these to your own directories:
# data_dir = '/local/efficient_radar_pipeline/'
# cache_file_dir = '/local/efficient_radar_pipeline/preprocessed_data/'
for_parallelized_preprocessing_example_rad_file = './data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v1/train/foyer/2-move_back_v1-300/ofxRadar8Ghz_2022-04-21_13-54-09.rad'

# data_dir = './data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v2/train'
data_dir = './data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v1/train'
# data_dir_test = './data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v2/test'
data_dir_test = './data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v1/test'
# cache_file_dir = './data/preprocessed_data'
cache_file_dir = '/local/marshalling_efficient/'

n_chirps = 192
samples_per_chirp = 512
f_sampling = 12.5*(10**6) #??? sampling frequency of the ADC

encoding = 'doppler-mag' # None, range-mag, range-phase, doppler-mag, doppler-phase, mu-doppler
# x.x og 5 and 960
frame_limit = 5 # 1 frame has 192 chirps with spans 1.3ms
sample_limit = 960 # 1 chirp has 512 samples with spans 40.96us

# x.x
super_x = 3
super_y = 1

center_percent = 0.5

fft_y = 192*frame_limit*super_y
fft_x = 512*super_x

f_c = 7.3E9 # center frequency
bandwidth = 750E6 # bandwidth
chirp_time = 40.96E-6 # time of one chirp
n_chirps = 192 # number of chirps
n_samples = 512 # number of samples per chirp
PRT = 1.3E-3 # Pulse Repetition Time
c = 3E8 # speed of light
m_w = bandwidth/chirp_time   # (frequency shift per unit of time) 

resolution_per_fft_bin = c/((2*bandwidth * fft_x) / samples_per_chirp)
range_maximum =  resolution_per_fft_bin * fft_x
frequencies_per_bin_range = np.arange(0, fft_x//2) * (f_sampling/fft_x)

def freq_to_range(f):
  return f*c/(2*m_w)

range_per_bin = np.concatenate([ freq_to_range(frequencies_per_bin_range)[::-1], -freq_to_range(frequencies_per_bin_range)])

# print(range_per_bin)
# print(range_per_bin.shape)


def angle_freq_to_velocity(w):
  return w*c/(4*np.pi*f_c)

f_chirp = 1/PRT
omega_second = 2*np.pi*np.concatenate((np.arange(0, fft_y//2)[::-1], np.arange(-fft_y//2, 0)[::-1]))*(f_chirp/fft_y)
velocities_per_bin = angle_freq_to_velocity(omega_second)

# training settings
N_EPOCHS = 30
batch_size = 16

use_gradcam = True
use_kfold_leave_one_out = False

seed = 0

# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.functional.cross_entropy

# Define the classes
classes = ['none', 'emergency_stop', 'move_ahead', 'move_back_v1', 'move_back_v2', 
        'slow_down', 'start_engines', 'stop_engines', 'straight_ahead', 
        'turn_left', 'turn_right']

classes_nice_text = ['None', 'Emergency', 'Move ahead', 'Back v1', 'Back v2', 
        'Slow down', 'Start engine', 'Stop engine', 'Straight ahead', 
        'Turn left', 'Turn right']        

baseline = {
    "config_name": "baseline",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_baseline",
    # "epochs": 10,
    'pretrained': True,
}

la_grande_finale = {
    "config_name": "la_grande_finale",
    "input_features": ['range_doppler'],
    "apply_delta": False,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ["hamming_window", 'mean_removal'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ["hamming_window"],
    "postprocessing_range_doppler": ["magnitude", "transform_db"],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ]),
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_baseline",
    # "epochs": 50,
    "model_name": "moganet",
}

la_grande_finale_v2 = {
    "config_name": "la_grande_finale_v2",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float8',
    "preprocessing_range_chirp": ["hanning_window"],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ["hanning_window"],
    "postprocessing_range_doppler": ["magnitude"],
    "rad_transform": transforms.Compose([transforms.CenterCrop([250, 250]),]),
    "data_augmentation": transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
        transforms.CenterCrop([240,240])
    ]),
    "image_size_save_hdf5": [250,250],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_baseline",
    # "epochs": 50,
    # "model_name": "effi",
}

# la_grande_finale_v3 = {
#     "config_name": "la_grande_finale_v3",
#     "input_features": ['range_doppler'],
#     "apply_delta": True,
#     "change_data_type_to": 'float8',
#     "preprocessing_range_chirp": ["hanning_window"],
#     "postprocesssing_range_chirp": [],
#     "preprocessing_range_doppler": ["hanning_window"],
#     "postprocessing_range_doppler": ["magnitude"],
#     "rad_transform": transforms.Compose([transforms.CenterCrop([270, 270]),]),
#     "data_augmentation": transforms.Compose([
#         transforms.RandomVerticalFlip(p=0.5),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
#         transforms.CenterCrop([240,240])
#     ]),
#     "image_size_save_hdf5": [270,270],
#     "image_size_model": [240,240],
#     "log_dir_path": "log_dir_baseline",
#     # "epochs": 50,
#     "model_name": "moganet",
# }



baseline_without_filtering = {
    "config_name": "baseline_without_filtering",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_windowing",
}

baseline_without_filtering_and_normalization = {
    "config_name": "baseline_without_filtering",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ['magnitude'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_windowing",
}

baseline_hamming = {
    "config_name": "baseline_hamming",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hamming_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hamming_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_windowing",
}

baseline_chebychev =  {
    "config_name": "baseline_chebychev",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['cheby_chev_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['cheby_chev_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_windowing",
}

baseline_blackman_harris =  {
    "config_name": "baseline_blackman_harris",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['blackman_harris_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['blackman_harris_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_windowing",
}

baseline_mean_removal =  {
    "config_name": "baseline_mean_removal",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['mean_removal', 'hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['mean_removal', 'hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_windowing",
}


baseline_mean_removal_no_delta =  {
    "config_name": "baseline_mean_removal_no_delta",
    "input_features": ['range_doppler'],
    "apply_delta": False,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['mean_removal', 'hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['mean_removal', 'hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_windowing",
}

testing_preprocessing_window = [baseline_without_filtering, baseline_chebychev, baseline_hamming , baseline_blackman_harris, baseline_mean_removal, baseline_mean_removal_no_delta]

hamming_mean_removal =  {
    "config_name": "baseline_mean_removal",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['mean_removal', 'hamming_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['mean_removal', 'hamming_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_windowing",
}

# change_data_type exploration

# this resulted in bullish results, because there is no good representation of the data doing this
baseline_std_data_type = {
    "config_name": "baseline_std_data_type",
    "input_features": ['range_doppler'],
    "apply_delta": False,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_delta",
}

baseline_std_data_type_2 = {
    "config_name": "baseline_std_data_type_2",
    "input_features": ['range_doppler'],
    "apply_delta": False,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'transform_db'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_delta",
}

baseline_std_data_type_3 = {
    "config_name": "baseline_std_data_type_3",
    "input_features": ['range_doppler'],
    "apply_delta": False,
    "change_data_type_to": '',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ['magnitude', 'transform_db'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_delta",
}

no_delta = [baseline_std_data_type, baseline_std_data_type_2 ]

baseline_binary = {
    "config_name": "baseline_binary",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'binary',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": ['noramlize'],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ['magnitude'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_dtype",
}

baseline_float8 = {
    "config_name": "baseline_float8",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float8_virtually',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_dtype",
}

baseline_float8_no_filtering = {
    "config_name": "baseline_float8_no_filtering",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float8_virtually',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_dtype",
}

baseline_float16 = {
    "config_name": "baseline_float16",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float16',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude','normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_dtype",
}

baseline_float32 = {
    "config_name": "baseline_float32",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude','normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_dtype",
}

baseline_float64 = {
    "config_name": "baseline_float64",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float64',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_dtype",
}

baseline_int16 = {
    "config_name": "baseline_int16",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'int16',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['mangitude'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_dtype",
}

baseline_int32 = {
    "config_name": "baseline_int32",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'int32',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ['magnitude'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_dtype",
}

# testing_data_type = [baseline_float8 ,baseline_float16, baseline_float32, baseline_int16, baseline_int32]
testing_data_type = [baseline_binary, baseline_float8 ,baseline_float16, baseline_float32, baseline_float64, baseline_int32]
# testing_data_type = [baseline_std_data_type, baseline_std_data_type_2, baseline_float16, baseline_float32, baseline_delta_float16, baseline_delta_float32, baseline_delta_int32]

# Feature exploration
raw = {
    "config_name": "raw",
    "input_features": ['raw'],
    "apply_delta": False,
    "change_data_type_to": '',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": [],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_features",
}

baseline_raw_delta = {
    "config_name": "baseline_raw_delta",
    "input_features": ['raw'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": [],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_features",
}

baseline_raw_delta_img_size_320 = {
    "config_name": "raw_delta_img_size_320",
    "input_features": ['raw'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": [],
    "rad_transform": transforms.Compose([transforms.CenterCrop([320, 320])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [320,320],
    "image_size_model": [320,320],
    "log_dir_path": "log_dir_features",
}


baseline_range = {
    "config_name": "baseline_range",
    "input_features": ['range_chirp'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": ['magnitude'],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": [],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_features",
}

baseline_mu_doppler = {
    "config_name": "baseline_mu_doppler",
    "input_features": ['mu_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window', 'mean_removal'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": [],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_features",
}

baseline_raw_range_doppler = {
    "config_name": "baseline_raw_range_doppler",
    "input_features": ['raw','range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_features",
}

baseline_range_and_range_doppler = {
    "config_name": "baseline_range_and_range_doppler",
    "input_features": ['range_chirp','range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_features",
}

baseline_range_and_mu_doppler = {
    "config_name": "baseline_range_and_mu_doppler",
    "input_features": ['range_chirp','mu_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_features",
}

range_doppler_mu_doppler = {
    "config_name": "range_doppler_mu_doppler",
    "input_features": ['range_doppler','mu_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_features",
}

range_doppler_mu_doppler_two_models = {
    "config_name": "range_doppler_mu_doppler",
    "input_features": ['range_doppler','mu_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_features",
    "model_name": "efficientnetb1_two_channels_two_models",
}

all_features = {
    "config_name": "all_features", 
    "input_features": ['raw','range_doppler','mu_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": None,
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_features",
}

combinations_features_only_one = [raw, baseline_raw_delta, baseline_range, baseline_mu_doppler]
combinations_of_features = [baseline_raw_range_doppler, range_doppler_mu_doppler, all_features]
# all_combinations_features = [baseline_raw_range_doppler, baseline_range_and_range_doppler, baseline_range_and_mu_doppler, range_doppler_mu_doppler, all_features]
# all_combinations_features = [raw, baseline_raw_delta, baseline_range, baseline_mu_doppler, baseline_raw_range_doppler

baseline_data_aug_range_velocity = {
    "config_name": "baseline_data_aug_range_velocity",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([180, 180])
    ]),
    "data_augmentation": transforms.Compose([   
        transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
        transforms.CenterCrop([240, 240]),
        ]),
    "image_size_save_hdf5": [180,180],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_data_aug",
}

# Define your noise-adding function as a lambda
add_gaussian_noise = lambda x: x + torch.randn(x.size()) * 0.01
baseline_data_aug_noise_01 = {
    "config_name": "baseline_data_aug_noise_01",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([180, 180])
    ]),
    "data_augmentation": transforms.Compose([   
        add_gaussian_noise,
        transforms.CenterCrop([240, 240]),
        ]),
    "image_size_save_hdf5": [180,180],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_data_aug",
}

add_gaussian_noise = lambda x: x + torch.randn(x.size()) * 0.005
baseline_data_aug_noise_005 = {
    "config_name": "baseline_data_aug_noise_005",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([180, 180])
    ]),
    "data_augmentation": transforms.Compose([   
        add_gaussian_noise,
        transforms.CenterCrop([240, 240]),
        ]),
    "image_size_save_hdf5": [180,180],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_data_aug",
}

add_gaussian_noise = lambda x: x + torch.randn(x.size()) * 0.001
baseline_data_aug_noise_001 = {
    "config_name": "baseline_data_aug_noise_001",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([180, 180])
    ]),
    "data_augmentation": transforms.Compose([   
        add_gaussian_noise,
        transforms.CenterCrop([240, 240]),
        ]),
    "image_size_save_hdf5": [180,180],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_data_aug",
}

add_gaussian_noise = lambda x: x + torch.randn(x.size()) * 0.0005
baseline_data_aug_noise_0005 = {
    "config_name": "baseline_data_aug_noise_0005",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([180, 180])
    ]),
    "data_augmentation": transforms.Compose([   
        add_gaussian_noise,
        transforms.CenterCrop([240, 240]),
        ]),
    "image_size_save_hdf5": [180,180],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_data_aug",
}

baseline_data_aug_noise_range_velocity = {
    "config_name": "baseline_data_aug_noise_range_velocity",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([180, 180])
    ]),
    "data_augmentation": transforms.Compose([   
        add_gaussian_noise,                                        
        transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
        transforms.CenterCrop([240, 240]),
        ]),
    "image_size_save_hdf5": [180,180],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_data_aug",
}

baseline_data_aug_random_horizontal_flip = {
    "config_name": "baseline_data_aug_horizontal_random_flip",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([240, 240])
    ]),
    "data_augmentation": transforms.Compose([   
        transforms.RandomHorizontalFlip(p=0.5),
    ]),
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_data_aug",
}

baseline_data_aug_random_vertical_flip = {
    "config_name": "baseline_data_aug_random_vertical_flip",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([240, 240])
    ]),
    "data_augmentation": transforms.Compose([   
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]),
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_data_aug",
}

baseline_combination_data_aug_flip = [baseline_data_aug_random_horizontal_flip, baseline_data_aug_random_vertical_flip]
baseline_combinations_data_aug_noise = [baseline_data_aug_noise_01, baseline_data_aug_noise_005, baseline_data_aug_noise_001, baseline_data_aug_noise_0005]
# basline_combinations_data_aug = [baseline_data_aug_range_velocity ,baseline_data_aug_noise_range_velocity]

baseline_combination_data_aug = [baseline_data_aug_range_velocity]
baseline_combination_data_aug.extend(baseline_combination_data_aug_flip)
baseline_combination_data_aug.extend(baseline_combinations_data_aug_noise)

# quantization
"""
Posibilities for quantization
- PTQ (post training quantization) 
- PTQ + FT (fine tuning)
- QAT (quantization aware training)

- Qscheme
    - per tensor
        - affine (assymetric)
            - scale
            - zero_point
        - symmetric
    - per channel
        - affine
            - scale
            - zero_point
        - symmetric

dtype:
    - quint8
    - qint8
    - qint32
    - float16

"""

baseline_mobilenetv2 = {
    "config_name": "baseline_mobilenetv2",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([224, 224])
    ]),
    "data_augmentation": None,
    "image_size_model": [224,224],
    "image_size_save_hdf5": [224,224],
    "log_dir_path": "log_dir_model",
    "model_name": "mobilenetv2",
}

no_windowing_mobilenetv2 = {
    "config_name": "no_windowing_mobilenetv2",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([224, 224])
    ]),
    "data_augmentation": None,
    "image_size_model": [224,224],
    "image_size_save_hdf5": [224,224],
    "log_dir_path": "log_dir_model",
    "quantization": "ptq_mobilenetv2",
    "model_name": "mobilenetv2",
}

baseline_efficientnetb1 = {
    "config_name": "baseline_mobilenetv2",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([240, 240])
    ]),
    "data_augmentation": None,
    "image_size_model": [240,240],
    "image_size_save_hdf5": [240,240],
    "log_dir_path": "log_dir_model",
    "model_name": "efficientnetb1",
    "epochs": 50,
    # "pretrained": False,
}

baseline_efficientnetb1_pretrained = {
    "config_name": "baseline_mobilenetv2_pretrained",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([240, 240])
    ]),
    "data_augmentation": None,
    "image_size_model": [240,240],
    "image_size_save_hdf5": [240,240],
    "log_dir_path": "log_dir_model",
    "model_name": "efficientnetb1",
    "epochs": 50,
    "pretrained": True,
}

no_windowing_efficientnetb1 = {
    "config_name": "no_windowing_efficientnetb1",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([240, 240])
    ]),
    "data_augmentation": None,
    "image_size_model": [240,240],
    "image_size_save_hdf5": [240,240],
    "log_dir_path": "log_dir_model",
    "quantization": "ptq_mobilenetv2",
    "model_name": "efficientnetb1",
}

baseline_levit_240 = {
    "config_name": "baseline_levit",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([240, 240])
    ]),
    "data_augmentation": None,
    "image_size_model": [240,240],
    "image_size_save_hdf5": [240,240],
    "log_dir_path": "log_dir_model",
    "model_name": "levit",
}

baseline_levit_official_128s = {
    "config_name": "baseline_levit_official_128s",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([224, 224])
    ]),
    "data_augmentation": None,
    "image_size_model": [224,224],
    "image_size_save_hdf5": [224,224],
    "log_dir_path": "log_dir_model",
    "model_name": "levit_128s",
}

baseline_levit_official_128 = {
    "config_name": "baseline_levit_official_128",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([224, 224])
    ]),
    "data_augmentation": None,
    "image_size_model": [224,224],
    "image_size_save_hdf5": [224,224],
    "log_dir_path": "log_dir_model",
    "model_name": "levit_128s",
}


baseline_moganet = {
    "config_name": "baseline_moganet",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([224, 224])
    ]),
    "data_augmentation": None,
    "image_size_model": [224,224],
    "image_size_save_hdf5": [224,224],
    "log_dir_path": "log_dir_model",
    "model_name": "moganet",
}

no_windowing_vgg = {
    "config_name": "no_windowing_vgg",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float32',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([128, 128])
    ]),
    "data_augmentation": None,
    "image_size_model": [128,128],
    "image_size_save_hdf5": [128,128],
    "log_dir_path": "log_dir_model",
    "quantization": "ptq_mobilenetv2",
    "model_name": "vgg",
}


#### Pruning ####
"""
Post-training pruning
Iterative pruning 

Pruning techniques in Pytorch:
- Unstructured pruning: It does not considerd the structure i.e. the channels or filters in convs
- Structured pruning: It considers the structure i.e. the channels or filters in convs
- Global Pruning: It prunes the entire model based on the percentage rather then a single layer.
- L1: Based on the L1 norm based on the absolute value of the weights
- LN: Based on the power of the L norm, L1 is the absolute L2 is Euclidean distance and Linf is the maximum value
- Global Unstructured pruning

Pytorch pruning objects:
- torch.nn.utils.prune.RandomUnstructured
- torch.nn.utils.prune.L1Unstructured
- torch.nn.utils.prune.LnStructured 
- torch.nn.utils.prune.random_unstructured
- torch.nn.utils.prune.l1_unstructured
- torch.nn.utils.prune.random_structured
- torch.nn.utils.prune.ln_structured
- torch.nn.utils.prune.global_unstructured

- The goal is not to assume to being implemented on specific hardware therefor the focus of unstructured pruning.

"""

# efficientnetb1_pruning = {
#     "config_name": "efficientnetb1_pruning",
#     "input_features": ['range_doppler'],
#     "apply_delta": True,
#     "change_data_type_to": '',
#     "preprocessing_range_chirp": ['hanning_window'],
#     "postprocesssing_range_chirp": [],
#     "preprocessing_range_doppler": ['hanning_window' ],
#     "postprocessing_range_doppler": ['magnitude', 'normalize'],
#     "rad_transform": transforms.Compose([
#         transforms.CenterCrop([240, 240])
#     ]),
#     "data_augmentation": None,
#     "image_size_model": [240,240],
#     "image_size_save_hdf5": [240,240],
#     "log_dir_path": "log_dir_pruning",
#     "model_name": "efficientnetb1",
#     "pruning_configuration": {
#         "pruning_type": "random",
#     }
# }

la_grande_finale_pruning_unstructured_array = []
pruning_amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for pruning_amount in pruning_amounts:
    la_grande_finale_pruning_unstructured = {
        "config_name": f"la_grande_finale_pruning_unstructured_{pruning_amount}",
        "input_features": ['range_doppler'],
        "apply_delta": True,
        "change_data_type_to": 'float16',
        "preprocessing_range_chirp": [],
        "postprocesssing_range_chirp": [],
        "preprocessing_range_doppler": [],
        "postprocessing_range_doppler": ["magnitude"],
        "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240]),]),
        "data_augmentation": transforms.Compose([
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
            transforms.CenterCrop([240,240])
        ]),
        "image_size_save_hdf5": [240,240],
        "image_size_model": [240,240],
        "log_dir_path": "log_dir_pruning",
        "epochs": 0,
        # Must be size of folds because of cross validation, define it as array
        "load_model_path": ["./models/2024_03_04_20_53_11_acc_0.7141model_valid.pt"],
        # "load_model_path": "./models/2024_07_02_14_57_23_acc_0.6477model.pth",
        # "model_name": "moganet",
        "pruning_configuration": {
            "global_pruning": True,
            "pruning_type": "l1_unstructured",
            "pruning_amount" : pruning_amount,
            "iterative_pruning": False,
        }
    }
    la_grande_finale_pruning_unstructured_array.append(la_grande_finale_pruning_unstructured)

la_grande_finale_pruning_unstructured_iterative_array = []
iterative_steps = 5
pruning_amounts = [0.9**(1/iterative_steps),0.8**(1/iterative_steps),0.7**(1/iterative_steps),0.6**(1/iterative_steps),0.5**(1/iterative_steps),0.4**(1/iterative_steps),0.3**(1/iterative_steps)]
for pruning_amount in pruning_amounts:
    la_grande_finale_pruning_unstructured_iterative = {
        "config_name": f"la_grande_finale_pruning_unstructured_itereative{pruning_amount}",
        "input_features": ['range_doppler'],
        "apply_delta": True,
        "change_data_type_to": 'float16',
        "preprocessing_range_chirp": [],
        "postprocesssing_range_chirp": [],
        "preprocessing_range_doppler": [],
        "postprocessing_range_doppler": ["magnitude"],
        "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240]),]),
        "data_augmentation": transforms.Compose([
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
            transforms.CenterCrop([240,240])
        ]),
        "image_size_save_hdf5": [240,240],
        "image_size_model": [240,240],
        "log_dir_path": "log_dir_pruning",
        "epochs": 0,
        # Must be size of folds because of cross validation, define it as array
        "load_model_path": ["./models/2024_03_04_20_53_11_acc_0.7141model_valid.pt"],
        # "load_model_path": "./models/2024_07_02_14_57_23_acc_0.6477model.pth",
        # "model_name": "moganet",
        "pruning_configuration": {
            "global_pruning": True,
            "pruning_type": "l1_unstructured",
            "pruning_amount" : pruning_amount,
            "iterative_pruning": True,
            "itaretive_pruning_steps": iterative_steps,
            "itarative_pruning_epochs": 1,
        }
    }
    la_grande_finale_pruning_unstructured_iterative_array.append(la_grande_finale_pruning_unstructured_iterative)

la_grande_finale_pruning_unstructured_iterative_array_even_further = []
pruning_amounts = [0.2**(1/iterative_steps), 0.1**(1/iterative_steps), 0.05**(1/iterative_steps), 0.025**(1/iterative_steps), 0.01**(1/iterative_steps)]
for pruning_amount in pruning_amounts:
    la_grande_finale_pruning_unstructured_iterative = {
        "config_name": f"la_grande_finale_pruning_unstructured_itereative{pruning_amount}",
        "input_features": ['range_doppler'],
        "apply_delta": True,
        "change_data_type_to": 'float16',
        "preprocessing_range_chirp": [],
        "postprocesssing_range_chirp": [],
        "preprocessing_range_doppler": [],
        "postprocessing_range_doppler": ["magnitude"],
        "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240]),]),
        "data_augmentation": transforms.Compose([
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
            transforms.CenterCrop([240,240])
        ]),
        "image_size_save_hdf5": [240,240],
        "image_size_model": [240,240],
        "log_dir_path": "log_dir_pruning",
        "epochs": 0,
        # Must be size of folds because of cross validation, define it as array
        "load_model_path": ["./models/2024_03_04_20_53_11_acc_0.7141model_valid.pt"],
        # "load_model_path": "./models/2024_07_02_14_57_23_acc_0.6477model.pth",
        # "model_name": "moganet",
        "pruning_configuration": {
            "global_pruning": True,
            "pruning_type": "l1_unstructured",
            "pruning_amount" : pruning_amount,
            "iterative_pruning": True,
            "itaretive_pruning_steps": iterative_steps,
            "itarative_pruning_epochs": 1,
        }
    }
    la_grande_finale_pruning_unstructured_iterative_array_even_further.append(la_grande_finale_pruning_unstructured_iterative)

# iterative_steps = 5
# la_grande_finale_best_pruning = {
#     "config_name": f"la_grande_finale_pruning_unstructured_itereative{pruning_amount}",
#     "input_features": ['range_doppler'],
#     "apply_delta": False,
#     "change_data_type_to": 'float16',
#     "preprocessing_range_chirp": ['hamming_window','mean_removal'],
#     "postprocesssing_range_chirp": [],
#     "preprocessing_range_doppler": ['hamming_window'],
#     "postprocessing_range_doppler": ["magnitude", "transform_db"],
#     "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
#     "data_augmentation": transforms.Compose([
#         transforms.RandomVerticalFlip(p=0.5),
#         transforms.RandomHorizontalFlip(p=0.5),
#     ]),
#     "image_size_save_hdf5": [240,240],
#     "image_size_model": [240,240],
#     "log_dir_path": "log_dir_pruning",
#     "epochs": 0,
#     # Must be size of folds because of cross validation, define it as array
#     # "load_model_path": ["./models/la_grande_finale_fold1.pt","./models/la_grande_finale_fold2.pt"],
#     # "load_model_path": ["./models/77ghzradarpipeline/marshalling/models_pruned/2024_18_02_15_29_24_model_pruned_0.8635518550872803.pth","./models/2024_18_02_15_29_24_model_pruned_0.7901856899261475.pth"],
#     # "load_model_path": "./models/2024_07_02_14_57_23_acc_0.6477model.pth",
#     "load_model_path": ["./models_pruned/2024_18_02_16_47_33_model_pruned_0.8622245788574219.pt","./models_pruned/2024_18_02_16_47_33_model_pruned_0.8281167149543762.pt"],
#     "model_name": "moganet",
#     "pruning_configuration": {
#         "global_pruning": True,
#         "pruning_type": "l1_unstructured",
#         "pruning_amount" : 0.2**(1/iterative_steps),
#         "iterative_pruning": True,
#         "itaretive_pruning_steps": iterative_steps,
#         "itarative_pruning_epochs": 1,
#     }
# }



### Quantization ###
shell_efficientnet_pruned_for_quantization = {
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float16',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ["magnitude"],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240]),]),
    "data_augmentation": transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
        transforms.CenterCrop([240,240])
    ]),
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_quantization",
    "epochs": 0,
    # "load_model_path": ['./models_pruned/2024_03_05_11_56_53_model_pruned_0.6985446810722351.pt']
    "load_modal_path": ["./models/2024_03_11_10_12_39_acc_0.6954model_valid.pt"]
}

efficientnetb1_ptq_x86 = {
    "config_name": "efficientnetb1_ptq_x86",
    "quantization": {
        "quantization_type": "ptq",
        "native_backend": "x86",
    },
}
efficientnetb1_ptq_x86.update(shell_efficientnet_pruned_for_quantization)

efficientnetb1_qat_x86_epochs_2 = {
    "config_name": "efficientnetb1_qat_x86_epochs_2",
    "quantization": {
        "quantization_type": "qat",
        "native_backend": "x86",
        "number_of_epochs_qat": 2,  
    },
}
efficientnetb1_qat_x86_epochs_2.update(shell_efficientnet_pruned_for_quantization)

shell_efficientnet_for_quantization = {
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float16',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ["magnitude"],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240]),]),
    "data_augmentation": transforms.Compose([
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
        transforms.CenterCrop([240,240])
    ]),
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_quantization",
    "epochs": 0,
    # "load_model_path": ["./models/2024_03_04_20_53_11_acc_0.7141model_valid.pt"],
    "load_model_path": ["./models/2024_03_14_12_07_19_acc_0.6299model_valid.pt"]
    # "load_model_path": ["./models_pruned/2024_03_05_11_40_49_model_pruned_0.6995841860771179.pt"],
}

efficientnetb1_ptq_x86_optimized_pipeline = {
    "config_name": "efficientnetb1_ptq_x86",
    "quantization": {
        "quantization_type": "ptq",
        "native_backend": "x86",
    },
}
efficientnetb1_ptq_x86_optimized_pipeline.update(shell_efficientnet_for_quantization)

array_increasing_in_epochs_qat_x86_optmized_pipeline = []
for total_epochs in range(1,6):
    efficientnetb1_qat_x86_epochs_optimzied_pipeline = {
        "config_name": f"efficientnetb1_qat_x86_epochs_{total_epochs}",
        "quantization": {
            "quantization_type": "qat",
            "native_backend": "x86",
            "number_of_epochs_qat": total_epochs,  
        },
    }
    efficientnetb1_qat_x86_epochs_optimzied_pipeline.update(shell_efficientnet_for_quantization)
    array_increasing_in_epochs_qat_x86_optmized_pipeline.append(efficientnetb1_qat_x86_epochs_optimzied_pipeline)


# efficientnetb1_qat_x86_epochs_optimzied_pipeline = {
#     "config_name": "efficientnetb1_qat_x86_epochs_2",
#     "quantization": {
#         "quantization_type": "qat",
#         "native_backend": "x86",
#         "number_of_epochs_qat": 1,  
#     },
# }
# efficientnetb1_qat_x86_epochs_2_optimzied_pipeline.update(shell_efficientnet_for_quantization)

moganet_ptq_x86 = {
    "config_name": "moganet_ptq_x86",
    "input_features": ['range_doppler'],
    "apply_delta": False,
    "change_data_type_to": 'float16',
    "preprocessing_range_chirp": ['hamming_window','mean_removal'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hamming_window'],
    "postprocessing_range_doppler": ["magnitude", "transform_db"],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ]),
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "model_name": "moganet",
    "log_dir_path": "log_dir_quantization",
    "epochs": 0,
    # Must be size of folds because of cross validation, define it as array
    # "load_model_path": ["./models_pruned/2024_02_18_18_10_22_model_pruned_0.8439075946807861.pt","./models_pruned/2024_02_18_18_10_22_model_pruned_0.8159151077270508.pt"],
    # "load_model_path": ["./models_pruned/2024_18_02_16_47_33_model_pruned_0.8622245788574219.pt","./models_pruned/2024_18_02_16_47_33_model_pruned_0.8281167149543762.pt"],
    "quantization": {
        "quantization_type": "ptq",
        "native_backend": "x86",
    },
}

moganet_qat_x86_2 = {
    "config_name": "moganet_qat_x86",
    "input_features": ['range_doppler'],
    "apply_delta": False,
    "change_data_type_to": 'float16',
    "preprocessing_range_chirp": ['hamming_window','mean_removal'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hamming_window'],
    "postprocessing_range_doppler": ["magnitude", "transform_db"],
    "rad_transform": transforms.Compose([transforms.CenterCrop([240, 240])]),
    "data_augmentation": transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ]),
    "image_size_save_hdf5": [240,240],
    "image_size_model": [240,240],
    "model_name": "moganet",
    "log_dir_path": "log_dir_quantization",
    "epochs": 0,
    # Must be size of folds because of cross validation, define it as array
    # "load_model_path": ["./models_pruned/2024_18_02_16_47_33_model_pruned_0.8622245788574219.pt","./models_pruned/2024_18_02_16_47_33_model_pruned_0.8281167149543762.pt"],
    "load_model_path": ["./models_pruned/2024_02_18_18_10_22_model_pruned_0.8439075946807861.pt","./models_pruned/2024_02_18_18_10_22_model_pruned_0.8159151077270508.pt"],
    "quantization": {
        "quantization_type": "qat",
        "native_backend": "x86",
        "number_of_epochs_qat": 2,  
    },
}

moganet_qat_custom = {
    "config_name": "moganet_qat_custom",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([224, 224])
    ]),
    "data_augmentation": None,
    "image_size_model": [224,224],
    "image_size_save_hdf5": [224,224],
    "log_dir_path": "log_dir_quantization",
    "quantization": {
        "quantization_type": "qat",
        "native_backend": "custom",
        "number_of_epochs_qat": 2,
        "custom_qconfig": {
            "bits": 8,
            "scheme": "per_tensor",
            "observer": "minmax",
            "ch_axis": None,
        }      
    },
}

moganet_ptq_custom = {
    "config_name": "moganet_ptq_custom",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": '',
    "preprocessing_range_chirp": ['hanning_window'],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": ['hanning_window'],
    "postprocessing_range_doppler": ['magnitude', 'normalize'],
    "rad_transform": transforms.Compose([
        transforms.CenterCrop([224, 224])
    ]),
    "data_augmentation": None,
    "image_size_model": [224,224],
    "image_size_save_hdf5": [224,224],
    "log_dir_path": "log_dir_quantization",
    "quantization": {
        "quantization_type": "ptq",
        "native_backend": "custom",
        # "number_of_epochs_qat": 2,
        "custom_qconfig": {
            "bits": 8,
            "scheme": "per_tensor",
            "observer": "minmax",
            "ch_axis": None,
        }      
    },
}

# efficientnetb1_ptq_x86 = {
#     "config_name": "efficientnetb1_ptq_x86",
#     "input_features": ['range_doppler'],
#     "apply_delta": True,
#     "change_data_type_to": '',
#     "preprocessing_range_chirp": ['hanning_window'],
#     "postprocesssing_range_chirp": [],
#     "preprocessing_range_doppler": ['hanning_window'],
#     "postprocessing_range_doppler": ['magnitude', 'normalize'],
#     "rad_transform": transforms.Compose([
#         transforms.CenterCrop([240, 240])
#     ]),
#     "data_augmentation": None,
#     "image_size_model": [240,240],
#     "image_size_save_hdf5": [240,240],
#     "log_dir_path": "log_dir_quantization",
#     "quantization": {
#         "quantization_type": "ptq",
#         "native_backend": "x86"       
#     },
# }

# efficientnetb1_qat_x86 = {
#     "config_name": "efficientnetb1_qat_x86",
#     "input_features": ['range_doppler'],
#     "apply_delta": True,
#     "change_data_type_to": '',
#     "preprocessing_range_chirp": ['hanning_window'],
#     "postprocesssing_range_chirp": [],
#     "preprocessing_range_doppler": ['hanning_window'],
#     "postprocessing_range_doppler": ['magnitude', 'normalize'],
#     "rad_transform": transforms.Compose([
#         transforms.CenterCrop([240, 240])
#     ]),
#     "data_augmentation": None,
#     "image_size_model": [240,240],
#     "image_size_save_hdf5": [240,240],
#     "log_dir_path": "log_dir_quantization",
#     "quantization": {
#         "quantization_type": "qat",
#         "native_backend": "x86",
#         "number_of_epochs_qat": 5,      
#     },
# }

# ptq_mobilenetv2 = {
#     "config_name": "ptq_mobilenetv2",
#     "input_features": ['range_doppler'],
#     "apply_delta": True,
#     "change_data_type_to": 'float32',
#     "preprocessing_range_chirp": ['hanning_window'],
#     "postprocesssing_range_chirp": [],
#     "preprocessing_range_doppler": ['hanning_window'],
#     "postprocessing_range_doppler": ['magnitude', 'normalize'],
#     "rad_transform": transforms.Compose([
#         transforms.CenterCrop([224, 224])
#     ]),
#     "data_augmentation": None,
#     "image_size_model": [224,224],
#     "image_size_save_hdf5": [224,224],
#     "log_dir_path": "log_dir_quantization",
#     "quantization": "ptq_mobilenetv2",
# }

# ptq_qnnpack = {
#     "config_name": "ptq_qnnpack",
#     "input_features": ['range_doppler'],
#     "apply_delta": True,
#     "change_data_type_to": '',
#     "preprocessing_range_chirp": [],
#     "postprocesssing_range_chirp": [],
#     "preprocessing_range_doppler": [],
#     "postprocessing_range_doppler": ['magnitude'],
#     "rad_transform": transforms.Compose([
#         transforms.CenterCrop([240, 240])
#     ]),
#     "data_augmentation": None,
#     "image_size_model": [240,240],
#     "image_size_save_hdf5": [240,240],
#     "log_dir_path": "log_dir_quantization",
#     "quantization": "ptq_qnnpack",
# }

# qat_qnnpack = {
#     "config_name": "qat_qnnpack",
#     "input_features": ['range_doppler'],
#     "apply_delta": True,
#     "change_data_type_to": '',
#     "preprocessing_range_chirp": [],
#     "postprocesssing_range_chirp": [],
#     "preprocessing_range_doppler": [],
#     "postprocessing_range_doppler": ['magnitude'],
#     "rad_transform": transforms.Compose([
#         transforms.CenterCrop([240, 240])
#     ]),
#     "data_augmentation": None,
#     "image_size_model": [240,240],
#     "image_size_save_hdf5": [240,240],
#     "log_dir_path": "log_dir_quantization",
#     "quantization": "qat_qnnpack",
# }

# qat_per_tensor_symmetric_min_max_8_bits = {
#     "config_name": "new_baseline",
#     "input_features": ['range_doppler'],
#     "apply_delta": True,
#     "change_data_type_to": '',
#     "preprocessing_range_chirp": [],
#     "postprocesssing_range_chirp": [],
#     "preprocessing_range_doppler": [],
#     "postprocessing_range_doppler": ['magnitude', 'normalize'],
#     "rad_transform": transforms.Compose([
#         transforms.CenterCrop([240, 240])
#     ]),
#     "data_augmentation": None,
#     "image_size_save_hdf5": [240,240],
#     "image_size_model": [240,240],
#     "log_dir_path": "log_dir_baseline",
#     "quantization": {
#         "quantization_type": "qat",
#         "quantization_scheme": "per_tensor_symmetric",
#         "observer": "min_max",
#         "ch_axis": None,
#         "bits": 8,
#     }
# }

add_gaussian_noise = lambda x: x + torch.randn(x.size()) * 0.0005
la_grande_finale_v4 = {
    "config_name": "la_grande_finale_v4",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float16',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ["magnitude"],
    "rad_transform": transforms.Compose([transforms.CenterCrop([270, 270]),]),
    "data_augmentation": transforms.Compose([
        add_gaussian_noise,
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
        transforms.CenterCrop([240,240])
    ]),
    "image_size_save_hdf5": [270,270],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_baseline",
    # "epochs": 50,
    # "model_name": "moganet",
}

la_grande_finale_v4_with_weight_decay = {
    "config_name": "la_grande_finale_v4",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float16',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ["magnitude"],
    "rad_transform": transforms.Compose([transforms.CenterCrop([270, 270]),]),
    "data_augmentation": transforms.Compose([
        add_gaussian_noise,
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
        transforms.CenterCrop([240,240])
    ]),
    "image_size_save_hdf5": [270,270],
    "image_size_model": [240,240],
    "log_dir_path": "log_dir_baseline",
    # "epochs": 50,
    # "weight_decay": 
    # "model_name": "moganet",
}


debugging_config = {
    "config_name": "debugging_config",
    "input_features": ['range_doppler'],
    "apply_delta": True,
    "change_data_type_to": 'float16',
    "preprocessing_range_chirp": [],
    "postprocesssing_range_chirp": [],
    "preprocessing_range_doppler": [],
    "postprocessing_range_doppler": ["magnitude"],
    "rad_transform": transforms.Compose([transforms.CenterCrop([270, 270]),]),
    "data_augmentation": transforms.Compose([
        add_gaussian_noise,
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees = 0, scale = (0.9, 1.1)),
        transforms.CenterCrop([240,240])
    ]),
    "image_size_save_hdf5": [270,270],
    "image_size_model": [240,240],
    "log_dir_path": "trashcan",
    "epochs": 1,
    # "load_model_path": ["./models/2024_03_04_20_53_11_acc_0.7141model_valid.pt"],
    # "load_model_path": ["./models_pruned/2024_03_05_11_40_49_model_pruned_0.6995841860771179.pt"],
    "load_model_path": ["./models/2024_03_13_14_26_46_acc_0.7089model_valid.pt"]
}
