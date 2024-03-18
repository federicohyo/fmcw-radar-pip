import sys
sys.path.append('./')
print(sys.path)
from processing.radar import *
from torchvision import transforms
import torchvision.transforms as transforms
import torch.distributions as dist

from configurations_grid import *

from tqdm import tqdm
import glob

import os
import shutil

from torchvision.transforms import InterpolationMode

from scipy.signal import chebwin
from scipy.signal.windows import blackmanharris
from scipy.signal.windows import hamming
from scipy import stats

from utils.radar_file_parser import RadarFileParser 
import matplotlib.pyplot as plt

import numpy as np
import torch

from einops import rearrange, reduce, repeat

visualization_of_processing = False

dir_save_images_of_preprocessing = ''

def compute_ops_windowing(data, axis = None):
    if type(data) is np.ndarray:
        return data.size
    elif type(data) is torch.Tensor:
        return data.numel()
    else:
        print("Unknown type")
        return -1

def calculate_sparsity(tensor):
    # Step 1: Count the number of zero elements
    num_zero_elements = torch.sum(tensor == 0).item()
    
    # Step 2: Calculate the total number of elements
    total_elements = tensor.numel()
    
    # Step 3: Compute the sparsity
    sparsity = num_zero_elements / total_elements
    
    return sparsity


def remove_files_in_folder(folder_path):
    # Check if the given path is an actual directory
    if not os.path.isdir(folder_path):
        print("The provided path is not a directory.")
        return

    # List all file paths in the given directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file or a directory
        if os.path.isfile(file_path) or os.path.islink(file_path):
            # It's a file or a link, so remove it
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            # It's a directory, remove it and all its contents
            shutil.rmtree(file_path)

def perform_ks_test(histogram):
    """Perform the Kolmogorov-Smirnov test on the histogram data."""
    histogram_values = [index for index, count in enumerate(histogram) for _ in range(int(count))]
    return stats.kstest(histogram_values, 'norm', args=(np.mean(histogram_values), np.std(histogram_values)))

def ks_test_normal(data):
    # Ensure the data is a PyTorch tensor
    data = torch.tensor(data, dtype=torch.float32)

    # Move computations to the same device as the data
    device = data.device

    # Standardize the data
    mean = torch.mean(data)
    std = torch.std(data)
    data_standardized = (data - mean) / std

    # Generate a standard normal distribution
    standard_normal = dist.Normal(0, 1)
    standard_normal_sample = standard_normal.sample(data_standardized.shape).to(device)

    # Sort the data
    data_sorted = torch.sort(data_standardized).values
    normal_sorted = torch.sort(standard_normal_sample).values

    # Compute EDFs
    n = data_sorted.size(0)
    edf_data = torch.arange(1, n + 1, device=device) / n
    edf_normal = torch.arange(1, n + 1, device=device) / n

    # Calculate KS statistic: maximum difference between EDFs
    data_all = torch.cat([data_sorted, normal_sorted])
    data_unique = torch.unique(data_all)
    edf_data_all = torch.searchsorted(data_sorted, data_unique, right=True) / n
    edf_normal_all = torch.searchsorted(normal_sorted, data_unique, right=True) / n
    ks_statistic = torch.max(torch.abs(edf_data_all - edf_normal_all)).item()

    return ks_statistic

def ks_test_normal_histogram(histogram, bin_edges):

    histogram_values = torch.tensor([
        bin_edges[i] for i, count in enumerate(histogram) 
        for _ in range(int(count))
    ], dtype=torch.float32)


    data = histogram_values
    device = data.device

    mean = torch.mean(data)
    std = torch.std(data)
    data_standardized = (data - mean) / std

    standard_normal = dist.Normal(0, 1)
    standard_normal_sample = standard_normal.sample(data_standardized.shape).to(device)

    data_sorted = torch.sort(data_standardized).values
    normal_sorted = torch.sort(standard_normal_sample).values

    # Compute EDFs
    n = data_sorted.size(0)
    edf_data = torch.arange(1, n + 1, device=device) / n
    edf_normal = torch.arange(1, n + 1, device=device) / n

    # Calculate KS statistic
    data_all = torch.cat([data_sorted, normal_sorted])
    data_unique = torch.unique(data_all)
    edf_data_all = torch.searchsorted(data_sorted, data_unique, right=True) / n
    edf_normal_all = torch.searchsorted(normal_sorted, data_unique, right=True) / n
    ks_statistic = torch.max(torch.abs(edf_data_all - edf_normal_all)).item()

    return ks_statistic

def compute_ops_fft(matrix, nbin=None, axis=None):
    """
    Calculate the operation count for computing the FFT along a given axis in a multi-dimensional PyTorch tensor.
    
    The FFT computational complexity is generally O(N log N) for N data points.
    
    Parameters:
    - matrix: input data (PyTorch tensor)
    - nbin: FFT bin size. If None, it defaults to the size of the specified axis.
    - axis: Axis along which the FFT should be computed. If None, it defaults to the last axis.
    """

    # Ensure matrix is a PyTorch tensor
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)
    
    # If no axis is specified, use the last axis
    if axis is None:
        axis = -1

    # If nbin is not specified, use the size of the axis
    if nbin is None:
        nbin = matrix.shape[axis]
    
    # print(torch.tensor(matrix.shape))
    # print(matrix.shape[axis])
    
    # Number of FFTs to compute is the size of the matrix divided by the size of the axis
    num_ffts = torch.div( torch.prod(torch.tensor(matrix.shape)), torch.tensor(matrix.shape[axis]), rounding_mode='floor')
    
    # For each FFT, the operation count is O( (log(N)*N)/2 * mul + log(N)N * add)
    operations_per_fft = int((torch.log(torch.tensor(nbin, dtype=torch.float))*nbin)/2 + torch.log(torch.tensor(nbin, dtype=torch.float))*nbin)
    
    return int(num_ffts * operations_per_fft)

def compute_ops_absolute(data, axis = None):
    """
    Calculate the operation count for computing the absolute value of each element in a multi-dimensional numpy matrix.
    For each complex number, we have:
    - 2 squaring operations (for real and imaginary parts)
    - 1 addition
    - 1 square root
    Total = 4 operations per complex number.
    """
    total_elements = np.prod(data.shape)
    operations_per_element = 4  # As described above
    return total_elements * operations_per_element

def compute_ops_sum(matrix, axis=None):
    """
    Calculate the operation count for computing the sum of elements along a given axis in a multi-dimensional numpy matrix.
    
    If summing along an axis:
    - (size of that axis - 1) additions for each position along the other axes.
    If summing the entire matrix:
    - (total elements - 1) additions.
    """
    
    # If no axis is specified, summing the entire matrix
    if axis is None:
        total_elements = np.prod(matrix.shape)
        return total_elements - 1
    
    # If an axis is specified
    axis_size = matrix.shape[axis]
    other_axes_size = np.prod(matrix.shape) // axis_size
    return (axis_size - 1) * other_axes_size

def compute_mean_removal_ops(tensor, axis = None):
    """
    Calculate the number of operations to remove the mean from a PyTorch tensor.

    Parameters:
    - tensor: input data (PyTorch tensor)
    """

    return tensor.numel() * 2

def compute_size_memory(data):
    if isinstance(data, np.ndarray):
        # It's a NumPy array
        dtype_bytes = data.itemsize
        return dtype_bytes * data.size
    elif isinstance(data, torch.Tensor):
        # It's a PyTorch tensor
        dtype_bytes = data.element_size()
        return dtype_bytes * data.numel()
    else:
        raise TypeError("Unsupported data type")

special_cases_dtypes = ['binary', 'float8_virtually', 'complex16_virtually', 'complex32_virtually']
def compute_size_memory_special_case(data, dtype):
    
    if dtype == 'binary':
        return data.numel() / 4

    if dtype == 'float8_virtually':
        return data.numel() * 1

    if dtype == 'complex16_virtually':
        return data.numel() * 2

    if dtype == 'complex32_virtually':
        return data.numel() * 4
    
def add_resources_info(pipeline, key, compute_flops_func, data, dtype = None):
    pipeline[key] = {}
    pipeline[key]['ops'] = compute_flops_func(data)
    pipeline[key]['memory'] = compute_size_memory(data)
    pipeline[key]['data_type'] = data.dtype
    pipeline[key]['shape'] = data.shape
    if dtype in special_cases_dtypes:
        pipeline[key]['data_type'] = dtype
        pipeline[key]['memory'] = compute_size_memory_special_case(data, dtype)
    
def add_resources_info_fft(pipeline, key, compute_flops_func, data, nbin = None, axis = None, dtype = None):
    assert nbin != None, 'define nbin'
    assert axis != None, 'define axis'
    pipeline[key] = {}
    pipeline[key]['ops'] = compute_flops_func(data, nbin, axis)
    pipeline[key]['memory'] = compute_size_memory(data)
    pipeline[key]['data_type'] = data.dtype
    pipeline[key]['shape'] = data.shape
    if dtype in special_cases_dtypes:
        pipeline[key]['data_type'] = dtype
        pipeline[key]['memory'] = compute_size_memory_special_case(data, dtype)

def human_readable_size(size, decimal_places=2):
    if not isinstance(size, (int, float)):
        return size
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024.0
    return f"{size:.{decimal_places}f} PB"

def human_readable_ops(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # floating point precision formatting can be adjusted in the format string
    return f'{num:.1f}{["", "K", "M", "G", "T", "P"][magnitude]}'

def apply_human_readable(dct):
    for key, value in dct.items():
        if isinstance(value, dict):
            apply_human_readable(value)
        else:
            if key == 'memory':
                dct[key] = human_readable_size(value)

def visualize_nested_dict(dct, depth=0, parent_key=None):
    indent = '  ' * depth
    if parent_key is not None:
        print(indent + parent_key + ':')
    for key, value in dct.items():
        if isinstance(value, dict):
            visualize_nested_dict(value, depth + 1, key)
        else:
            if key == 'memory':
                value = human_readable_size(value)
            if key == 'ops':
                value = human_readable_ops(value)
            print(indent + '  ' + key + ': ' + str(value))

def sum_nested_dict_values(dct):
    total = 0
    for key, value in dct.items():
        if isinstance(value, dict):
            total += sum_nested_dict_values(value)
        elif isinstance(value, (int, float)):  # Check if the value is a number
            total += value
    return total  

def sum_ops_memory(dct):
    total_ops = 0
    total_memory = 0
    for key, value in dct.items():
        if isinstance(value, dict):
            ops, memory = sum_ops_memory(value)
            total_ops += ops
            total_memory += memory
        else:
            if key == 'ops':
                total_ops += value
            elif key == 'memory':
                total_memory += value
    return total_ops, total_memory

plotted_figures = 0

def visualize_radar_data(tensor, save_dir_image ,title_plot ,x_axis_name = '?', y_axis_name = '?', x_ticks = None, y_ticks = None, skip_x_ticks = 50, skip_y_ticks = 50, font_size=12, add_axis = True ):
    # Convert the PyTorch tensor to a NumPy array.
    # Assumes the tensor is real-valued (not complex).
    
    global plotted_figures

    plt.clf()

    if torch.is_complex(tensor):
        tensor = torch.abs(tensor)  # You could also use tensor.real or tensor.imag
    
    tensor = tensor.float()    
    np_data = tensor.cpu().detach().numpy()

    # If the tensor has more than 2 dimensions, select the first 2D slice for visualization.
    if np_data.ndim > 2:
        np_data = np_data[0]

    plt.imshow(np_data)

    if add_axis:
        # Add title if provided
        if title_plot:
            plt.title(title_plot, fontsize = font_size)

        # Add axis labels if provided
        if x_axis_name:
            plt.xlabel(x_axis_name, fontsize = font_size)
        if y_axis_name:
            plt.ylabel(y_axis_name, fontsize = font_size)

        # Check if x_ticks is not None and has elements
        if x_ticks is not None and len(x_ticks) > 0:
            plt.xticks(range(len(x_ticks))[::skip_x_ticks], np.round(x_ticks[::skip_x_ticks], 1), rotation=45, fontsize = font_size)

        # Check if y_ticks is not None and has elements
        if y_ticks is not None and len(y_ticks) > 0:
            print("len y_ticks",len(y_ticks))
            print(np_data.shape)
            plt.yticks(range(len(y_ticks))[::skip_y_ticks], np.round(y_ticks[::skip_y_ticks], 1), rotation=0, fontsize= font_size)
    else:
        # Completely remove axes if not needed
        plt.axis('off')

    # Adjust layout to make room for the x-axis label
    plt.tight_layout()

    # save image
    print('save image to: ', save_dir_image + str(plotted_figures) + '_' + title_plot + '.png')
    # plt.colorbar()  # Adds a color bar to interpret values
    plt.savefig(  save_dir_image + str(plotted_figures) + '_' + title_plot + '.png', dpi = 200, bbox_inches= 'tight')
    # plt.savefig(  save_dir_image + str(plotted_figures) + '_' + title_plot + '.pdf', bbox_inches= 'tight')
    # plt.savefig(save_dir_image + str(plotted_figures) + '_' + title_plot +'.eps', format='eps', bbox_inches='tight')

    plotted_figures += 1


def to_db(vals):
    '''
    Replace zero values with a small epsilon value and then convert to dB scale.
    @param vals: Tensor of values to convert
    @return: vals in dB
    '''
    epsilon = 1e-10  # Small epsilon value
    # Ensure epsilon is of the same type as vals
    epsilon_tensor = torch.tensor(epsilon, dtype=vals.dtype, device=vals.device)
    vals_stable = torch.where(vals == 0, epsilon_tensor, vals)
    return 10 * torch.log10(vals_stable)

def delta_frames_radar(signal, rsc_report = None, dtype = None):
    # signal dimension: batch, chirp, sample
    # print('signal shape: ', signal.shape)
    # flattened_batched_tensors = signal.view(signal.shape[0],-1)\\

    if rsc_report:
        add_resources_info(rsc_report, 'delta_frames_radar', compute_ops_windowing, signal, dtype = dtype)

    flattened_batched_tensors = signal.transpose(1,2).reshape(signal.shape[0],-1)
    # print('flattened_batched_tensors shape: ', flattened_batched_tensors.shape)

    delta_tensor_flatt = flattened_batched_tensors[:,1:] - flattened_batched_tensors[:,:-1]
    zero_column = signal.new_zeros(signal.shape[0], 1)
    delta_tensor_flatt = torch.cat((zero_column, delta_tensor_flatt), 1)

    delta_tensor = delta_tensor_flatt.reshape(signal.shape[0], signal.shape[2], signal.shape[1]).transpose(1,2)
    # delta_tensor = delta_tensor.permute(0,2,1)
        
    # print("Sparsity tensor",calculate_sparsity(delta_tensor))

    return delta_tensor

def float_to_float8(input_tensor):
    # Define the range and precision for the custom float8 format
    exponent_bits = 4
    mantissa_bits = 3
    bias = (2 ** (exponent_bits - 1) - 1)  # Subtracting 1 for the bias

    # Define the int8 range
    int8_min = -128
    int8_max = 127

    # Step 1: Handle special cases (zero, negative, inf, nan)
    # Replace NaN values with zero (or some other default)
    input_tensor = torch.where(torch.isnan(input_tensor), torch.zeros_like(input_tensor), input_tensor)

    # Handle zero, inf, and -inf by clamping the values within a representable range
    input_tensor = torch.clamp(input_tensor, min=float(int8_min), max=float(int8_max))

    sign_mask = torch.sign(input_tensor)  # Extract sign
    input_tensor = torch.abs(input_tensor)  # Work with positive values for simplicity

    # Calculate the min and max for clamping based on the exponent range
    clamp_min = max(2**(-bias), int8_min)
    clamp_max = min(2**bias - 1, int8_max)  # Ensure the max is within int8 range

    # Step 2: Scale values to the float8 range and ensure they are within the representable range to avoid overflow
    scaled = torch.clamp(input_tensor, min=clamp_min, max=clamp_max)

    # Step 3: Normalize and extract exponent and mantissa
    exponent = torch.floor(torch.log2(scaled)).to(torch.float16) + bias  # Biasing the exponent
    mantissa = scaled / (2. ** exponent) * (2 ** mantissa_bits)

    # Step 4: Quantize mantissa
    quantized_mantissa = torch.round(mantissa).to(torch.float16)

    # Step 5: Combine sign, exponent, and mantissa
    # Ensure that the combination does not result in overflow
    float8_virtual = sign_mask*(2**(exponent - bias)) * (1 + mantissa / float(2** mantissa_bits))

    return float8_virtual

# architecture design
def change_data_type(signal, transform_to = None):
    '''
    signal: just the maps in 3D tensor
    transform_to arguments: (ints, floats, binary, 'delta')
    transform_to: (float16, float32, binary, )
    
    'pytorch does only support the following for fft: float32, float64'
    '''
    
    if transform_to == 'complex64':
        return signal.to(torch.complex64)

    if 'float8_virtually' == transform_to:
        return float_to_float8(signal)

    if 'float16' == transform_to:
        return signal.to(torch.float16)
    
    if 'float32' == transform_to:
        return signal.to(torch.float32)

    if 'float64' == transform_to:
        return signal.to(torch.float64)
    
    if 'int32' == transform_to:
        return signal.to(torch.int32)

    if  'int16' == transform_to:
        return signal.to(torch.int16)

    if 'complex16_virtually' == transform_to:
        real_part = signal.real
        imag_part = signal.imag
        real_part = float_to_float8(real_part)
        imag_part = float_to_float8(imag_part)
        real_part = real_part.to(torch.float32)
        imag_part = imag_part.to(torch.float32)
        complex_tensor = torch.complex(real_part, imag_part)
        return complex_tensor

    if transform_to == 'complex32_virtually':
        real_part = signal.real
        imag_part = signal.imag
        real_part = real_part.to(torch.float16)
        imag_part = imag_part.to(torch.float16)
        real_part = real_part.to(torch.float32)
        imag_part = imag_part.to(torch.float32)
        complex_tensor = torch.complex(real_part, imag_part)
        return complex_tensor

    if transform_to == 'binary':
        # signal = delta_frames_radar(signal)
        # transform signal to int
        # singal = signal.to(torch.float16)
        signal = signal - torch.mean(signal, dim=(1,2), keepdim=True)
        signal = torch.where(signal> 0.0,  torch.tensor(1.0, dtype=signal.dtype, device=signal.device), signal)
        signal = torch.where(signal< 0.0, torch.tensor(-1.0, dtype=signal.dtype, device=signal.device), signal)
        
        return signal

    return signal

def preprocessing_radar_range_chirp(signal, bins_fft_range, change_data_type_to = '' ,preprocessing = [] , postprocessing = [], tracking_resources = False, rsc_report = None):
    """
    Preprocesses a radar signal for range-chirp processing with optional preprocessing and postprocessing steps.

    This function is designed to prepare a radar signal for further analysis, particularly for range-chirp processing. 
    It applies various preprocessing techniques like mean removal and Hanning windowing, followed by FFT operations 
    and optional postprocessing. The function is also capable of handling batch processing by working with 3D tensors 
    where each tensor in the batch represents an individual signal.

    Parameters:
    - signal (torch.Tensor): The input radar signal data, expected to be a 2D or 3D tensor.
    - bins_fft_range (int): The number of FFT bins for range processing.
    - preprocessing (list of str): A list of preprocessing operations to apply. Supported operations are 
      'mean_removal' and 'hanning_window'.
    - postprocessing (list of str): A list of optional postprocessing operations to apply.

    Steps:
    1. Input Validation: Checks if the input is a PyTorch tensor and reshapes 2D tensors to 3D for batch processing.
    2. Mean Removal: If 'mean_removal' is specified, the mean along the chirp axis is removed from the signal.
    3. Hanning Window: If 'hanning_window' is specified, a Hanning window is applied to the signal.
    4. FFT Processing: Applies FFT along the specified axis with the given number of bins.
    5. FFT Shift: The FFT result is shifted for better visualization and analysis.

    The function supports visualization at each step if `visualization_of_processing` is set to True.

    Returns:
    - signal (torch.Tensor): The processed signal, suitable for range-chirp analysis or further processing.
    """
    
    # print('singal dtype: ', signal.dtype)
    # print('signal change dtype: ', change_data_type_to)

    # check if pytorch tensor
    if not isinstance(signal, torch.Tensor):
        raise TypeError("Input data must be a pytorch tensor")

    # check if 3D for batch processing even if its 2D
    if signal.ndim == 2:
        signal = signal.unsqueeze(0)

    change_data_type_to_complex = ''
    if change_data_type_to == 'binary': change_data_type_to_complex =  'complex16_virtually'
    if change_data_type_to == 'float8_virtually': change_data_type_to_complex =  'complex16_virtually'
    if change_data_type_to == 'float16_virtually': change_data_type_to_complex =  'complex16_virtually'
    if change_data_type_to == 'float16': change_data_type_to_complex = 'complex32_virtually'
    if change_data_type_to == 'float32': change_data_type_to_complex = 'complex64'
    if change_data_type_to == 'float64': change_data_type_to_complex = 'complex128'
    if change_data_type_to == 'int32': change_data_type_to_complex = 'complex64'


    if change_data_type_to != '': 
        signal = change_data_type(signal, transform_to = change_data_type_to)
        if visualization_of_processing:
            print('change data type')
            visualize_radar_data(rearrange(signal, 'b s c ->  b c s' ), './processing/parallelized_images/', 'range change data type', 'chirp', 'sample')

    # Apply processing before fft
    if 'mean_removal' in preprocessing:
        if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to)
        if rsc_report:
            add_resources_info(rsc_report, 'mean_removal', compute_mean_removal_ops, signal, change_data_type_to)
        signal = signal - torch.mean(signal, dim=(2), keepdim=True)
        if visualization_of_processing:
            print("mean removal")
            # visualize_radar_data(signal)
            visualize_radar_data(rearrange(signal, 'b s c ->  b c s' ), './processing/parallelized_images/', 'mean removal', 'chirp', 'sample')
        if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to)

        # visualize_radar_data(signal, './processing/parallelized_images/', 'x_mean_removal', 'chirp', 'sample')

    hann_window = []
    # optimization define the hann_window once
    if 'hanning_window' in preprocessing:
        # Track resources
        # Apply operation
        # if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to)

        if rsc_report:
            add_resources_info(rsc_report, 'hanning_window', compute_ops_windowing, signal, dtype = change_data_type_to)
        hann_window = np.hanning(signal.shape[2])
        hann_window = torch.from_numpy(hann_window).to(device=signal.device)
        hann_window = change_data_type(hann_window, transform_to = change_data_type_to)
        # print('signal dtype: ', signal.dtype)
        # print('hann_window dtype: ', hann_window.dtype)
        signal = signal * hann_window
        if visualization_of_processing:
            print('hanning window')
            visualize_radar_data(rearrange(signal, 'b s c ->  b c s' ), './processing/parallelized_images/', 'Hanning window', 'chirp', 'sample')
        # visualize_radar_data(signal, './processing/parallelized_images/', 'xx_hanning_window', 'chirp', 'hanning')
        if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to)


    if 'hamming_window' in preprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'hamming_window', compute_ops_windowing, signal, dtype = change_data_type_to)
        hamming_window = hamming(signal.shape[2])
        hamming_window = torch.from_numpy(hamming_window).to(device=signal.device)
        signal = signal * hamming_window
        if visualization_of_processing:
            print('hamming window')
            visualize_radar_data(rearrange(signal, 'b s c ->  b c s' ), './processing/parallelized_images/', 'range Hamming window', 'chirp', 'sample')

    if 'blackman_harris_window' in preprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'blackmanharris_window', compute_ops_windowing, signal, dtype = change_data_type_to)
        blackmanharris_window = blackmanharris(signal.shape[2])
        blackmanharris_window = torch.from_numpy(blackmanharris_window).to(device=signal.device)
        signal = signal*blackmanharris_window
        if visualization_of_processing:
            print('blackmanharris window')
            visualize_radar_data(rearrange(signal, 'b s c ->  b c s' ), './processing/parallelized_images/', 'range Blackman Harris window', 'chirp', 'sample')
    
    if 'cheby_chev_window' in preprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'chebychev_window', compute_ops_windowing, signal, dtype = change_data_type_to)
        # at is 45 dB
        cheby_chev_window = chebwin(signal.shape[2], at = 60)
        cheby_chev_window = torch.from_numpy(cheby_chev_window).to(device=signal.device)
        signal = signal * cheby_chev_window
        if visualization_of_processing:
            print('chebychev window')
            visualize_radar_data(rearrange(signal, 'b s c ->  b c s' ), './processing/parallelized_images/', 'range Chebychev window', 'chirp', 'sample')

    # apply FFT on sample per chirp axis
    if rsc_report:
        add_resources_info_fft(rsc_report, 'fft', compute_ops_fft, signal, nbin=bins_fft_range, axis=2, dtype = change_data_type_to_complex)
    
    if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to)
    signal = signal.double()
    signal = torch.fft.fft(signal, bins_fft_range, dim=2, norm = 'ortho')
    if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to_complex)

    if visualization_of_processing:
        print('range_chirps')
        visualize_radar_data(rearrange(signal, 'b r c ->  b c r' ), './processing/parallelized_images/', 'range FFT', 'chirp', 'range')
    # signal = 
    # visualize_radar_data(signal, './processing/parallelized_images/', 'xxx_fft', 'chirp', 'fft')

    # apply FFT shift
    if rsc_report:
        add_resources_info(rsc_report, 'fft_shift', compute_ops_windowing, signal, dtype = change_data_type_to_complex)

    signal = torch.fft.fftshift(signal, dim=2)

    # if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to_complex) 
    if visualization_of_processing:
        print('fft shift')
        # visualize_radar_data(rearrange(signal, 'b r c ->  b c r' ), './processing/parallelized_images/', 'range FFT shift', 'chirp', 'range')
        visualize_radar_data(rearrange(signal, 'b r c ->  b c r' ), './processing/parallelized_images/', 'range FFT shift', 'chirp', 'range [m]', y_ticks = range_per_bin, skip_y_ticks = 100)
        visualize_radar_data(rearrange(signal, 'b r c ->  b r c' ), './processing/parallelized_images/', 'range FFT', 'range [m]', 'chirp', x_ticks = range_per_bin, skip_x_ticks = 100)
        visualize_radar_data(rearrange(to_db(signal), 'b r c ->  b c r' ), './processing/parallelized_images/', 'range FFT shift (dB)', 'chirp', 'range [m]', y_ticks = range_per_bin, skip_y_ticks = 100)

    # visualize_radar_data(signal, './processing/parallelized_images/', 'xxxx_fft_shift', 'chirp', 'fft_shift')

    if 'normalize' in postprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'normalize', compute_ops_sum, signal, dtype = change_data_type_to_complex)
        
        signal = signal / torch.max(torch.abs(signal))
        # signal = signal / torch.linalg.norm(torch.abs(signal), dim = (1,2), keepdims=True)

        if visualization_of_processing:
            print('normalize')
            visualize_radar_data(rearrange(signal, 'b r c ->  b c r' ), './processing/parallelized_images/', 'range normalize', 'chirp', 'range')
        if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to_complex)

    if 'magnitude' in postprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'magnitude', compute_ops_absolute, signal)
        signal = torch.abs(signal)
        if visualization_of_processing:
            print('magnitude')
            visualize_radar_data(rearrange(signal, 'b r c ->  b c r' ), './processing/parallelized_images/', 'magnitude', 'chirp', 'magnitude')
        if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to)

    return signal

    
def preprocessing_radar_range_doppler(signal, bins_fft_doppler, change_data_type_to = '', preprocessing = ['hanning_window'] ,postprocessing = ['transform_db','magnitude'], rsc_report = None):
    """
    Preprocess a radar signal for range-Doppler map generation with optional preprocessing and postprocessing steps.

    This function applies a series of preprocessing and optional postprocessing steps to a radar signal, 
    typically used to generate a range-Doppler map. The function allows for mean removal, application of a 

    Parameters:
    - signal (torch.Tensor): The input radar signal data.
    - bins_fft_doppler (int): The number of FFT bins for the Doppler processing.
    - preprocessing (list of str): A list of preprocessing operations to apply. Supported operations are
      'mean_removal' and 'hanning_window'.
    - postprocessing (list of str): A list of postprocessing operations to apply. Supported operations are
      'transform_db' and 'magnitude_range_doppler'.

    Processing Steps:
    1. Mean Removal: If 'mean_removal' is in the preprocessing list, the mean of the signal is removed.
    2. Hanning Window: If 'hanning_window' is in the preprocessing list, a Hanning window is applied to the signal.
    3. FFT Processing: An FFT is applied along the specified axis with the given number of bins.
    4. FFT Shift: The FFT result is shifted for better visualization.
    5. Postprocessing: Magnitude computation and dB transformation are applied if specified.

    The function supports visualization at each step if `visualization_of_processing` is set to True.

    Returns:
    - signal (torch.Tensor): The processed signal, which can be directly used for generating range-Doppler maps.
    """

    change_data_type_to_complex = ''
    if change_data_type_to == 'binary': 
        change_data_type_to_complex = 'complex16_virtually'
        change_data_type_to = 'binary'
    elif change_data_type_to == 'float8_virtually': change_data_type_to_complex = 'complex16_virtually'
    elif change_data_type_to == 'float16': change_data_type_to_complex = 'complex32_virtually'
    elif change_data_type_to == 'float32': change_data_type_to_complex = 'complex64'
    elif change_data_type_to == 'float64': change_data_type_to_complex = 'complex128'
    if change_data_type_to == 'int32': change_data_type_to_complex = 'complex64'


    # Apply preprocessing
    if 'mean_removal' in preprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'mean_removal_range_doppler', compute_mean_removal_ops, signal, dtype = change_data_type_to_complex)
        signal = signal - torch.mean(signal, dim=(1), keepdim=True)
        if visualization_of_processing:
            print('mean removal')
            visualize_radar_data(rearrange(signal, 'b r c ->  b c r' ), './processing/parallelized_images/', 'Doppler mean removal', 'chirp', 'range')

    '''
    Possible to add other filtering functions here
    '''

    if 'hanning_window' in preprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'hanning_window_range_doppler', compute_ops_windowing, signal, dtype = change_data_type_to_complex)
        hann_window = np.hanning(signal.shape[1])
        hann_window = torch.from_numpy(hann_window).to(device=signal.device)
        hann_window = hann_window.unsqueeze(1)
        hann_window = change_data_type(hann_window, transform_to = change_data_type_to)
        # hann_window = hann_window.unsqueeze(1).repeat(1,range_chirp_maps.shape[2])
        
        # # Calculate the size of 'signal' in bytes
        # print(signal.shape)

        # signal_size_bytes = signal.element_size() * signal.numel()

        # # Calculate the size of 'hann_window' in bytes
        # hann_window_size_bytes = hann_window.element_size() * hann_window.numel()

        # # Print the size in bytes
        # print(f"Size of 'signal' in GPU memory: {signal_size_bytes} bytes")
        # print(f"Size of 'hann_window' in GPU memory: {hann_window_size_bytes} bytes")

        # # Optionally, convert bytes to MB for easier interpretation
        # bytes_to_mb_factor = 1 / (1024 * 1024)
        # print(f"Size of 'signal' in GPU memory: {signal_size_bytes * bytes_to_mb_factor} MB")
        # print(f"Size of 'hann_window' in GPU memory: {hann_window_size_bytes * bytes_to_mb_factor} MB")

        signal = signal*hann_window      
        if visualization_of_processing:
            print('hanning window')
            visualize_radar_data(rearrange(signal, 'b r c ->  b c r' ), './processing/parallelized_images/', 'Doppler hanning window', 'chirp', 'range')
        # visualize_radar_data(signal, './processing/parallelized_images/', 'zz_hanning_window', 'chirp', 'hanning')
        
    if 'hamming_window' in preprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'hamming_window_range_doppler', compute_ops_windowing, signal, dtype = change_data_type_to)
        hamming_window = hamming(signal.shape[1])
        hamming_window = torch.from_numpy(hamming_window).to(device=signal.device)
        hamming_window = hamming_window.unsqueeze(1)
        signal = signal * hamming_window
        if visualization_of_processing:
            print('hamming window')
            visualize_radar_data(rearrange(signal, 'b r c ->  b c r' ), './processing/parallelized_images/', 'Doppler Hamming window', 'chirp', 'range')

    if 'blackman_harris_window' in preprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'blackmanharris_window_range_doppler', compute_ops_windowing, signal)
        blackmanharris_window = blackmanharris(signal.shape[1])
        blackmanharris_window = torch.from_numpy(blackmanharris_window).to(device=signal.device)
        blackmanharris_window = blackmanharris_window.unsqueeze(1)
        signal = signal*blackmanharris_window
        if visualization_of_processing:
            print('blackmanharris window')
            visualize_radar_data(rearrange(signal, 'b r c ->  b c r' ), './processing/parallelized_images/', 'Doppler Blackman Harris window', 'chirp', 'range')

    
    if 'cheby_chev_window' in preprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'chebychev_window_range_doppler', compute_ops_windowing, signal)
        cheby_chev_window = chebwin(signal.shape[1], at = 60)
        cheby_chev_window = torch.from_numpy(cheby_chev_window).to(device=signal.device)
        cheby_chev_window = cheby_chev_window.unsqueeze(1)
        signal = signal * cheby_chev_window
        if visualization_of_processing:
            print('chebychev window')
            visualize_radar_data(rearrange(signal, 'b r c ->  b c r' ), './processing/parallelized_images/', 'Doppler Chebychev window', 'chirp', 'range')
        
    # implement statement that it is possible to do 
    
    # apply FFT on sample per chirp axis
    if rsc_report:
        add_resources_info_fft(rsc_report, 'fft_range_doppler', compute_ops_fft, signal, nbin=bins_fft_doppler, axis=1, dtype = change_data_type_to_complex)
    if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to_complex)
    signal = signal.to(torch.complex128)
    signal = torch.fft.fft(signal, bins_fft_doppler, dim=1)
    if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to_complex)
    if visualization_of_processing:
        print('range_doppler map:')
        visualize_radar_data(rearrange((signal), 'b r d ->  b d r' ), './processing/parallelized_images/', 'Doppler FFT', 'Doppler', 'range')
        visualize_radar_data(rearrange(to_db(signal), 'b r d ->  b d r' ), './processing/parallelized_images/', 'Doppler FFT to dB', 'Doppler', 'range')
    # visualize_radar_data(signal, './processing/parallelized_images/', 'zzz_fft', 'chirp', 'fft')

    # apply FFT shift
    
    if rsc_report:
        add_resources_info(rsc_report, 'fft_shift_range_doppler', compute_ops_windowing, signal, dtype = change_data_type_to_complex)
    signal = torch.fft.fftshift(signal, dim=1)
    if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to_complex)
    if visualization_of_processing:
        visualize_radar_data( rearrange((signal), 'b r d ->  b d r'), './processing/parallelized_images/', 'Doppler shift FFT', 'velocity [m/s]', 'range [m]', x_ticks = velocities_per_bin, y_ticks = range_per_bin, skip_x_ticks = 100, skip_y_ticks = 100)
        visualize_radar_data( rearrange(to_db(signal), 'b r d ->  b d r'), './processing/parallelized_images/', 'Doppler shift FFT to dB', 'velocity [m/s]', 'range [m]')

    # - Normalize
    # - Threshold to zero
    
    # apply processing after fft This was a false assumption that this was neccesary
    if 'magnitude' in postprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'magnitude_range_doppler', compute_ops_absolute, signal, change_data_type_to_complex)

        signal = torch.abs(signal)

        change_data_type_to_complex = change_data_type_to 
        if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to_complex)

        if visualization_of_processing:
            visualize_radar_data(rearrange(signal, 'b r d ->  b d r'), './processing/parallelized_images/', 'Doppler magnitude', 'Doppler', 'range')

    if 'normalize' in postprocessing:
        if 'hanning_window' in preprocessing:
            if rsc_report:
                add_resources_info(rsc_report, 'normalize_by_hanning_range_doppler', compute_ops_windowing, signal, change_data_type_to_complex)
            hann_window = np.hanning(signal.shape[1])
            hann_window = torch.from_numpy(hann_window).to(device=signal.device)
            hann_window = hann_window.unsqueeze(1)
            # print(1/torch.sum(hann_window))
            signal = (1/torch.sum(hann_window)) * signal
            if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to_complex)
            if visualization_of_processing:
                print('normalize by hanning window')
                visualize_radar_data(rearrange(signal, 'b r d ->  b d r'), './processing/parallelized_images/', 'Doppler normalize', 'Doppler', 'range')
        else:
            if rsc_report:
                add_resources_info(rsc_report, 'normalize_range_doppler', compute_ops_sum, signal, change_data_type_to_complex)
            signal = (1/signal.shape[1]) * signal
            if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to_complex)
            if visualization_of_processing:
                print('normalize')
                visualize_radar_data(rearrange(signal, 'b r d ->  b d r'), './processing/parallelized_images/', 'Doppler normalize', 'Doppler', 'range')

    if 'transform_db' in postprocessing:
        if rsc_report:
            add_resources_info(rsc_report, 'transform_db_range_doppler', compute_mean_removal_ops, signal, change_data_type_to_complex)
        signal = signal.double()
        signal = to_db(signal)
        if change_data_type_to != '': signal = change_data_type(signal, transform_to = change_data_type_to_complex)
        if visualization_of_processing:
            print('to_db')
            visualize_radar_data(rearrange(signal, 'b r d ->  b d r'), './processing/parallelized_images/', 'Doppler transform dB', 'Doppler', 'range')
    return signal


def preprocessing_radar_mu_doppler_loop(signal, fft_N_mdopp, wnd_stride, wnd_lenght, preprocessing = ['hanning_window'] ,postprocessing = [], rsc_report = None):
    # n_fft is the size the fourier transform to use, determining the size of frequency bins larger n_fft finer frequency resolution
    # hop_lenngth is the jump size of the sliding window
    # win_lenght is the size of the winodw function of each sement of the signal is applied on
    # window a filtering technique which can be applied before each fft

    list_of_max_frames = []

    # print("frame config", signal.shape)

    for frame in signal:

        iterations_of_bins = 5

        frame = rearrange(frame, 'c s -> s c')
        # print('before mean',frame.shape)
        mean = torch.mean(torch.abs(frame), dim = 1)
        value, indices = torch.topk(mean, iterations_of_bins)

        # select the middle frame its almost everytime the largest amplitude
        # this save a huge amount of computation time
        range_bin = indices[0].item()
        if indices[0].item() == (frame.shape[0]//2):
            range_bin = indices[1].item()

        for indic , val in zip(indices, value):
            print("range:", indic.item(), "value:", val.item())

        # print('range_bin',range_bin)
        print(range_bin)
        print(frame.shape)
        frame = frame[range_bin]
        frame = frame.unsqueeze(0)

        # frame = torch.stft(frame, n_fft = fft_N_mdopp, hop_length = wnd_stride, win_length = wnd_lenght, window = torch.hann_window(wnd_lenght))
        # frame = torch.stft(frame, n_fft = fft_N_mdopp, hop_length = wnd_stride, win_length = wnd_lenght, return_complex = True)
        frame = torch.stft(frame, n_fft = 192, hop_length = 6, win_length = 96, window = torch.hann_window(96).to(device=signal.device) ,return_complex = True)
        frame = torch.abs(frame)

        # fftshift
        frame = torch.cat((frame[:,frame.shape[1]//2:,:], frame[:,:frame.shape[1]//2,:]), dim=1)

        if visualization_of_processing:
            visualize_radar_data(frame, './processing/parallelized_images/', 'mu_doppler', 'time', 'frequency')

        list_of_max_frames.append((frame[0,:,:]))
        print(len(list_of_max_frames))

    # if True:
    #     visualize_radar_data(list_of_max_frames[0], './processing/parallelized_images/', 'mu_doppler', 'time', 'frequency')
    
    if visualization_of_processing:
        visualize_radar_data(list_of_max_frames[0], './processing/parallelized_images/', 'mu_doppler', 'time', 'frequency')

    # print("list_of_max_frames", len(list_of_max_frames))
    # print("shape frames", list_of_max_frames[0].shape)

    return torch.stack(list_of_max_frames, dim=0)

def preprocessing_radar_mu_doppler_loop_best_bin_finding(signal, fft_N_mdopp, wnd_stride, wnd_lenght, preprocessing = ['hanning_window'] ,postprocessing = [], nr_chunks = 30 ,rsc_report = None):
    # n_fft is the size the fourier transform to use, determining the size of frequency bins larger n_fft finer frequency resolution
    # hop_lenngth is the jump size of the sliding window
    # win_lenght is the size of the winodw function of each sement of the signal is applied on
    # window a filtering technique which can be applied before each fft

    list_of_max_frames = []

    signal = rearrange(signal, 'f c s -> (f c) s')
    full_frame = rearrange(signal, 'c s -> s c')

    ks = 20

    mean = torch.mean(torch.abs(full_frame), dim = 1)
    value, indices = torch.topk(mean, ks)

    list_of_mu_doppler = []

    # Create mu_doppler maps
    for indic in indices:
        # select the middle frame its almost everytime the largest amplitude
        # this save a huge amount of computation time
        range_bin = indic.item()

        sub_frame = full_frame[range_bin]
        sub_frame = sub_frame.unsqueeze(0)

        # sub_frame 

        # What is the time axis what is the frequency axis
        # 
        # print("sub frame shape",sub_frame.shape)

        sub_frame = torch.stft(sub_frame, n_fft = 192, hop_length = 6, win_length = 96, window = torch.hann_window(96).to(device=signal.device) ,return_complex = True)
        # print(sub_frame.shape)/
        
        sub_frame = torch.abs(sub_frame)

        # fftshift
        sub_frame = torch.cat((sub_frame[:,sub_frame.shape[1]//2:,:], sub_frame[:,:sub_frame.shape[1]//2,:]), dim=1)
        if visualization_of_processing:
            visualize_radar_data(sub_frame, './processing/parallelized_images/', 'mu_doppler', 'time', 'frequency')
        sub_frame = sub_frame[:,:,1:]
        sub_frame = reduce(sub_frame, '1 s c -> s c', 'mean')

        # frame = rearrange(frame, 's (f c) -> f s c ', f = nr_chunks)
        list_of_mu_doppler.append(sub_frame)

    ks_statistic_best = float('inf')  # Adjust based on your requirements
    best_frame = list_of_mu_doppler[0]

    for index, frame in enumerate(list_of_mu_doppler):
        image = frame.clone()  # Make a copy of the frame
        
        histogram = torch.mean(image, dim=1)

        # Perform kolgomorov-smirnov test
        ks_statistic = ks_test_normal_histogram(histogram, torch.arange(0, histogram.shape[0]+1, 1))
        
        if ks_statistic < ks_statistic_best:
            ks_statistic_best = ks_statistic
            best_frame = frame.clone()  # Make sure to clone the frame
        
    best_frame = rearrange(best_frame, 's (f c) -> f s c ', f = nr_chunks)

    return best_frame

def preprocessing_radar_mu_doppler_straight(signal, fft_N_mdopp, wnd_stride, wnd_lenght, preprocessing = ['hanning_window'] ,postprocessing = [], nr_chunks = 30 ,rsc_report = None):
    # n_fft is the size the fourier transform to use, determining the size of frequency bins larger n_fft finer frequency resolution
    # hop_lenngth is the jump size of the sliding window
    # win_lenght is the size of the winodw function of each sement of the signal is applied on
    # window a filtering technique which can be applied before each fft

    list_of_max_frames = []

    # print("frame config", signal.shape)
    # print("signal shape input mu-doppler",signal.shape)
    signal = rearrange(signal, 'f c s -> (f c) s')
    # print("shape after rearrange",signal.shape)
    frame = rearrange(signal, 'c s -> s c')
    
    # print('before mean',frame.shape)
    mean = torch.mean(torch.abs(frame), dim = 1)
    value, indices = torch.topk(mean, 5)

    # select the middle frame its almost everytime the largest amplitude
    # this save a huge amount of computation time
    range_bin = indices[0].item()
    if indices[0].item() == (frame.shape[0]//2):
        range_bin = indices[1].item()

    frame = frame[range_bin]
    frame = frame.unsqueeze(0)

    # frame = torch.stft(frame, n_fft = fft_N_mdopp, hop_length = wnd_stride, win_length = wnd_lenght, window = torch.hann_window(wnd_lenght))
    # frame = torch.stft(frame, n_fft = fft_N_mdopp, hop_length = wnd_stride, win_length = wnd_lenght, return_complex = True)
    
    print("frame shape uh",frame.shape)


    frame = torch.stft(frame, n_fft = 192, hop_length = 6, win_length = 96, window = torch.hann_window(96).to(device=signal.device) ,return_complex = True)
    frame = torch.abs(frame)

    # fftshift
    frame = torch.cat((frame[:,frame.shape[1]//2:,:], frame[:,:frame.shape[1]//2,:]), dim=1)
        
    if visualization_of_processing:
        visualize_radar_data(frame, './processing/parallelized_images/', 'mu_doppler', 'time', 'frequency')
    
    frame = frame[:,:,1:]
    frame = reduce(frame, '1 s c -> s c', 'mean')
    frame = rearrange(frame, 's (f c) -> f s c ', f = nr_chunks) 

    return frame


def preprocessing_radar_mu_doppler_loop_whole_recording(signal, fft_N_mdopp, wnd_stride, wnd_lenght, preprocessing = ['hanning_window'] ,postprocessing = [], rsc_report = None):
    # n_fft is the size the fourier transform to use, determining the size of frequency bins larger n_fft finer frequency resolution
    # hop_lenngth is the jump size of the sliding window
    # win_lenght is the size of the winodw function of each sement of the signal is applied on
    # window a filtering technique which can be applied before each fft

    signal = torch.from_numpy(signal).to('cuda')

    print(signal.shape)

    signal = rearrange(signal, 'f c s -> (f c) s')
    print(signal.shape)
    signal = signal.unsqueeze(0)
    signal = signal - torch.mean(signal, dim=(1,2), keepdim=True)
    signal = delta_frames_radar(signal)
    print('shape after unsquueeze',signal.shape)
    # signal = signal - torch.mean(signal, dim=(1,2), keepdim=True)
    
    print(signal.shape)
    
    # Apply hanning window
    hann_window = np.hanning(signal.shape[2])
    hann_window = torch.from_numpy(hann_window).to(device=signal.device)
    signal = signal * hann_window

    signal = preprocessing_radar_range_chirp(signal, bins_fft_range = signal.shape[2], change_data_type_to = '',  preprocessing = [''], postprocessing = [], rsc_report = None)
    print(signal.shape)
    # 1 chirp is 1.3 milliseconds, so 13 seconds for 10,000 chirps.
    visualize_radar_data(rearrange(signal[:,:], 'b c s -> b s c'), './processing/parallelized_images/', 'before_mu-Doppler', 'chirp', 'range')

    # signal = signal[:,0:10000,:]

    # Max amplitude for the ranges

    list_of_max_frames = []

    for frame in signal:
        # print("frame shape",frame.shape)
        # print("frame shape",frame.shape)
        frame = rearrange(frame, 'c s -> s c')
        # print("signal after rearrange",signal.shape)
        # print("fft_N_mdopp", fft_N_mdopp)
        # print("wnd_stride", wnd_stride)
        # print("wnd_lenght", wnd_lenght)
        if visualization_of_processing:
            visualize_radar_data(frame[:,:1000], './processing/parallelized_images/', 'range-chirp', 'chirp', 'range')

        print('before mean',frame.shape)

        mean = torch.mean(torch.abs(frame), dim = 1)
        value, indices = torch.topk(mean, 5)
        for indic , val in zip(indices, value):
            print("range:", indic.item(), "value:", val.item())

        print(frame.shape)

        range_bin = indices[0].item()
        if indices[0].item() == (frame.shape[0]//2):
            range_bin = indices[1].item()

        print("range_bin", range_bin)


        frame = frame[range_bin]

        print(frame.shape)

        frame = frame.unsqueeze(0)

        print(frame.shape)

        # time between chirps is 1.3 milliseconds
        # single time stamp
        hop_length = 12
        time_window_stft = 1.3E-3 * hop_length
        f_chirp = 1/PRT
        win_length = 96
        n_fft = 192
        # need to fill in the following, fft_y(win_length), 
        velocities_per_bin_stft = 2*np.pi*np.concatenate((np.arange(0, n_fft//2)[::-1], np.arange(-n_fft//2, 0)[::-1]))*(f_chirp/n_fft)

        frame = torch.stft(frame, n_fft = n_fft, hop_length = hop_length, win_length = win_length, window= torch.hann_window(96).to(device=frame.device) ,return_complex = True) # this give something different then normal
        time_per_bin_stft = np.arange(0, frame.shape[2]) * time_window_stft


        print(frame.shape)
        frame = torch.abs(frame)
        # frame = to_db(frame)
        if visualization_of_processing:
            visualize_radar_data(frame[:,:,:], './processing/parallelized_images/', 'mu_doppler', 'time', 'frequency')

        frame = torch.cat((frame[:,int(frame.shape[1]/2):,:], frame[:,:int(frame.shape[1]/2),:]), dim=1)

        if visualization_of_processing:
            untill_time = 1000
            visualize_radar_data(frame[0,:,:untill_time], './processing/parallelized_images/', 'mu doppler shifted', 'time [s]', 'frequency [hz]', x_ticks = time_per_bin_stft[0:untill_time], y_ticks = velocities_per_bin_stft)

        print("frame shape",frame.shape)
        print("velocities_per_bin_stft",velocities_per_bin_stft.shape)
        print("time_per_bin_stft",time_per_bin_stft.shape)

        # exit()

        # Used for computing things not anymore
        # max_amplitude_index = torch.argmax(torch.sum(torch.abs(frame), dim = (1,2)))
        # print("max amplitude", max_amplitude_index)
        frame_diff = frame.shape[1] - frame.shape[2]

        if frame_diff > 0:
            frame = frame[:,frame_diff//2:-frame_diff//2,:]
        elif frame_diff < 0:
            frame = frame[:,:,:frame.shape[1]]

        # print("frame cropped shape",frame.shape)

        list_of_max_frames.append((frame[0,:,:]))

    if visualization_of_processing:
        visualize_radar_data(list_of_max_frames[0], './processing/parallelized_images/', 'mu_doppler', 'time', 'frequency')

    # print("list_of_max_frames", len(list_of_max_frames))
    # print("shape frames", list_of_max_frames[0].shape)

    return torch.stack(list_of_max_frames, dim=0)

def create_report_preprocessing(signal, fft_Nx, fft_Ny, features, apply_delta ,preprocessing_range_chirp, postprocessing_range_chirp, preprocessing_range_doppler, postprocessing_range_doppler, change_dtype, rad_transform):
    '''
    Create a report of the preprocessing steps
    '''

    rsc_report = {'initial_signal': {}}

    # print('signal chagne dtype:', change_dtype)

    if change_dtype != '' and change_dtype != 'float8_virtually' and change_dtype != 'binary': 
        print('signal dtype:', signal.dtype)
        signal = change_data_type(signal, transform_to = change_dtype)

    if visualization_of_processing:
        visualize_radar_data(rearrange(signal, 'b s c -> b c s'), './processing/parallelized_images/', 'raw data', 'chirp', 'sample')
        visualize_radar_data(rearrange(signal, 'b s c -> b c s'), './processing/parallelized_images/', 'raw data', 'chirp', 'sample', add_axis = False)

    if apply_delta:
        print("applying delta")
        signal = delta_frames_radar(signal, rsc_report = rsc_report, dtype = change_dtype)
        if visualization_of_processing:
            print('delta')
            visualize_radar_data(rearrange(signal, 'b s c -> b c s'), './processing/parallelized_images/', 'delta', 'chirp', 'sample')

    if 'range_chirp' in features or 'range_doppler' in features or 'mu_doppler' in features:
        signal = preprocessing_radar_range_chirp(signal, fft_Nx, change_data_type_to = change_dtype,  preprocessing = preprocessing_range_chirp, postprocessing = postprocessing_range_chirp, rsc_report = rsc_report)
    
    if 'range_doppler' in features:
        print("signal shape before mu_doppler")
        signal = preprocessing_radar_range_doppler(signal, fft_Ny ,change_data_type_to = change_dtype, preprocessing = preprocessing_range_doppler, postprocessing =  postprocessing_range_doppler, rsc_report = rsc_report) 
    # preprocessing range chirp

    if 'mu_doppler' in features:
        print("applying mu doppler")
        # singal = torch.concatneate((signal, signal), dim = 0)
        # signal = preprocessing_radar_range_chirp()
        signal = preprocessing_radar_mu_doppler_loop(signal, fft_N_mdopp = 256, wnd_lenght = 256, wnd_stride = 2, preprocessing = [''], postprocessing =  [], rsc_report = rsc_report)
    
        # if 'mu-doppler' in features:
        # mu_doppler_maps = preprocessing_radar_mu_doppler(range_chrip_maps, fft_Ny , preprocessing = preprocessing_range_doppler, postprocessing =  postprocessing_range_doppler)

    print('signal shape: ', signal.shape)
    signal = rad_transform(signal)

    if visualization_of_processing:
        # visualize_radar_data(rearrange(signal, 'b r d -> b d r'), './processing/parallelized_images/', 'Doppler-Range', 'velocity [m/s]', 'range [m]')

        print(rearrange(signal, 'b r d -> b d r').shape)

        # print(signal.shape[1]-120,signal.shape[1]+120)
        # print(signal.shape[2]-120,signal.shape[2]+120)
        # print(range_per_bin.shape[0]/2)
        # print(range_per_bin.shape[0]//2)
        # print(velocities_per_bin[ velocities_per_bin.shape[0]//2 -120:velocities_per_bin.shape[0]//2+120])
        range_ticks = range_per_bin[range_per_bin.shape[0]//2-120: range_per_bin.shape[0]//2+120]
        velocity_ticks = velocities_per_bin[velocities_per_bin.shape[0]//2-120: velocities_per_bin.shape[0]//2+120]
        visualize_radar_data(rearrange(signal, 'b r d -> b d r'), './processing/parallelized_images/', 'Doppler-Range', 'velocity [m/s]', 'range [m]', x_ticks = velocity_ticks, y_ticks = range_ticks)

    if visualization_of_processing:
        visualize_radar_data(signal, './processing/parallelized_images/', 'out', '?', '?', add_axis = False)

    if visualization_of_processing:
        total_ops, total_memory = sum_ops_memory(rsc_report)
        print(human_readable_ops(total_ops))
        print(human_readable_size(total_memory))

    return rsc_report

if __name__ == "__main__":

    remove_files_in_folder('./processing/parallelized_images/')

    visualization_of_processing = False
    
    # Load radar file 
    # file = '/local/radar8ghz-efficient/holst_foyer/capture-emergency_stop-leon/foyer-emergency_stop-350-cm-ofxRadar8Ghz_2022-04-21_20-33-10.rad'
    # file = '/local/radar8ghz-efficient/holst_foyer/capture-start_engines-leon/foyer-start_engines-150-cm-ofxRadar8Ghz_2022-04-21_15-41-11.rad'
    # file = '/local/efficient_radar_pipeline/holst_foyer/capture-straight_ahead-federico/foyer-straight_ahead-250-cm-ofxRadar8Ghz_2021-10-19-03_04_01.rad'
    file = for_parallelized_preprocessing_example_rad_file

    chunk_idx = 6
    rad_file = RadarFileParser(file, read_dvs=False, read_radar= True)

    print('signal: ', 'emergency_stop')

    frame_limit = 5
    nr_chunks = int(len(rad_file.radar_frames) / frame_limit)
    print("number of chirps per frame", rad_file.num_chirps_per_frame)
    print("number of samples per chirp", rad_file.num_samples_per_chirp)
    print("sign", rad_file)
    chunk_size = frame_limit * rad_file.num_chirps_per_frame

    all_radar_frames = []
    for frame in rad_file.radar_frames:
        all_radar_frames.append(frame.td_matrix)
    all_radar_frames = np.array(all_radar_frames, dtype=float)

    # Transform to other frame sizes
    chunked_radar_frames = np.reshape(all_radar_frames, [-1, chunk_size, rad_file.num_samples_per_chirp ])
    radar_sample = chunked_radar_frames[chunk_idx]

    visualization_of_processing = True
    # Testing for mu-doppler
    # mu_doppler = preprocessing_radar_mu_doppler_testing(chunked_radar_frames, fft_N_mdopp = 512, wnd_lenght = 512//32, wnd_stride = 64, preprocessing = ['hanning_window'], postprocessing =  [], rsc_report = None)
    # chunked_radar_frames = torch.from_numpy(chunked_radar_frames)
    # chunked_radar_frames = change_data_type(chunked_radar_frames, 'delta')
    # mu_doppler = preprocessing_radar_mu_doppler_loop(chunked_radar_frames, fft_N_mdopp = 512, wnd_lenght = 512, wnd_stride = 2, preprocessing = ['hanning_window'], postprocessing =  [], rsc_report = None)

    super_x = 3
    super_y = 1

    image_size = 240

    # mu_doppler_width = 512
    # mu_doppler_height = 481
    # aspect_ratio = mu_doppler_width/mu_doppler_height

    radar_crop = [image_size, image_size] #960*0.5/2 = 480, 192 (64x3)
    radar_shape = [image_size, image_size] #960*0.5/2 (center_crop + halfed), 32*2*3*1.25

    # Additional Transforms
    rad_transform = transforms.Compose([
        transforms.CenterCrop(radar_crop),
    ])

    signal = torch.from_numpy(radar_sample)
    signal = signal.unsqueeze(0)

    fft_Ny = 192*frame_limit*super_y 
    fft_Nx = 512*super_x

    print('fft_Nx: ', fft_Nx)
    print('fft_Ny: ', fft_Ny)
    
    preprocessing_radar_mu_doppler_loop_whole_recording(all_radar_frames, fft_N_mdopp = 256, wnd_lenght = 256, wnd_stride = 2, preprocessing = ['hanning_window'], postprocessing =  [], rsc_report = None)

    # supported: ['float16', 'float32', 'binary']
    # change_dtype = 'binary'

    # supported: [ 'mean_removal', 'blackman_harris_window', 'cheby_chev_window', 'hanning_window', 'hamming_window']
    # preprocessing_range_chirp = ['']
    # supported arguments for postprocessing: ['magnitude', 'normalize', 'transform_db']
    # postprocessing_range_chirp = ['normalize']
    # signal = preprocessing_radar_range_chirp(tensor_radar_play_snipper, fft_Nx , preprocessing = preprocessing_range_chirp, postprocessing = postprocessing_range_chirp)

    # preprocessing_range_doppler = ['']
    # supported arguments for postprocessing: ['transform_db', 'magnitude', 'normalize']
    # postprocessing_range_doppler = ['magnitude', '']
    # signal = preprocessing_radar_range_doppler(range_chrip_maps, fft_Ny , preprocessing = [ 'hanning_window'], postprocessing =  postprocessing_range_chirp)

    # square = la_grande_finale
    square = baseline_mean_removal
    input_features = square['input_features']
    apply_delta = square['apply_delta']
    change_dtype = square['change_data_type_to']
    preprocessing_range_chirp = square['preprocessing_range_chirp']
    postprocessing_range_chirp = square['postprocesssing_range_chirp']
    preprocessing_range_doppler = square['preprocessing_range_doppler']
    postprocessing_range_doppler = square['postprocessing_range_doppler']

    tracking_resources = True
    print('Input signal of processing pipeline: ', signal.shape)
    report_values = create_report_preprocessing(signal, fft_Nx, fft_Ny, input_features, apply_delta,  preprocessing_range_chirp, postprocessing_range_chirp, preprocessing_range_doppler, postprocessing_range_doppler, change_dtype, rad_transform)

    print('Report of processing pipeline: ')
    visualize_nested_dict(report_values)