import copy
import numpy as np
import torch
from .util import to_db
from .traditional import *

import matplotlib.pyplot as plt

def visualize_radar_data(tensor, save_dir_image ,title_plot ,x_axis_name, y_axis_name ):
    # Convert the PyTorch tensor to a NumPy array.
    # Assumes the tensor is real-valued (not complex).
    
    plt.clf()

    if isinstance(tensor, torch.Tensor):
        tensor = torch.abs(tensor)  # You could also use tensor.real or tensor.imag
        np_data = tensor.cpu().detach().numpy()

    if np.iscomplexobj(tensor):
        tensor = np.abs(tensor)    # You could also use tensor.real or tensor.imag
    np_data = tensor

    # Check if the data type is object
    if np_data.dtype == object:
        # Try to convert to float
        try:
            np_data = np.array(np_data, dtype=float)
        except ValueError:
            raise ValueError("Data cannot be converted to float. Check the data format.")


    # If the tensor has more than 2 dimensions, select the first 2D slice for visualization.
    if np_data.ndim > 2:
        np_data = np_data[0]

    print('np_data shape', np_data.shape)
    print('np_data type', np_data.dtype)

    plt.imshow(np_data)

    # Add title if provided
    if title_plot:
        plt.title(title_plot)

    # Add axis labels if provided
    if x_axis_name:
        plt.xlabel(x_axis_name)
    if y_axis_name:
        plt.ylabel(y_axis_name)

    # save image
    print('save image to: ', save_dir_image + title_plot + '.png')
    plt.colorbar()  # Adds a color bar to interpret values
    plt.savefig(save_dir_image + title_plot + '.png')

def process_radar(radar_adc, delta, stride, clipping, processing, fft_Nx, fft_Ny, shift_fft, transform, crop_fft = False, delta_to_db = True):
    
    radar_adc = np.reshape(radar_adc, [1, radar_adc.shape[0], radar_adc.shape[1]])
    print('radar adc shape', radar_adc.shape)
    print('radar adc type', radar_adc.dtype)
    print('radar adc[0]', radar_adc[0][0][0])
    visualize_radar_data(radar_adc, save_dir_image = './processing/unparallelized_images/', title_plot = 'radar_adc', x_axis_name = 'chirp', y_axis_name = 'sample')

    # print(radar_adc.shape)

    # apply delta
    if delta is not None:
        # Delta is applied to the radar_adc data
        print('delta is applied to the radar_adc data')
        encoded_sample = perform_delta(radar_adc, delta, stride, clipping)
        encoded_sample = np.array(encoded_sample)
    else:
        print('no delta is applied to the radar_adc data')
        encoded_sample = radar_adc

    visualize_radar_data(encoded_sample, save_dir_image = './processing/unparallelized_images/', title_plot = 'encoded_sample', x_axis_name = 'chirp', y_axis_name = 'sample')

    # how to pre-process the radar adc data
    print('processing', processing)
    if processing == None:
        sample = encoded_sample.astype(int)[0]

    elif(processing == 'range-mag'):
        processed_sample = range_mag(encoded_sample, fft_Nx)[0]

    elif(processing == 'range-phase'):
        processed_sample = range_phase(encoded_sample, fft_Nx)[0]

    elif(processing == 'doppler-mag'):
        processed_sample = doppler_mag(encoded_sample, fft_Nx, fft_Ny)[0]

    elif(processing == 'doppler-phase'):
        processed_sample = doppler_phase(encoded_sample, fft_Nx, fft_Ny)[0]

    elif(processing == 'mu-doppler'):
        print('mu-doppler is executed')
        processed_sample = mu_doppler(encoded_sample)

    else:
        print(f'radar processing {processing} not yet supported.')

    # visualize_radar_data(processed_sample, save_dir_image = './processing/unparallelized_images/', title_plot = 'processed_sample', x_axis_name = 'range', y_axis_name = 'doppler')

    # apply additional steps like cropping and to_db if needed.
    
    if processing is not None and ('mag' in processing or 'phase' in processing):
        print('processing is not none and mag or phase is in processing')
        if crop_fft:
            print('crop_fft')
            range_bins = int(32 * (fft_Nx / 512))

            if shift_fft:
                cropped_sample = processed_sample[:, fft_Nx//2-range_bins:fft_Nx//2+range_bins]
            else:
                cropped_sample = processed_sample[:, fft_Nx//2:fft_Nx//2+range_bins]

        else:
            cropped_sample = processed_sample # no cropping plz

        if (delta is None or delta_to_db) and not 'phase' in processing:  
            print('to db is executed')     
            sample = to_db(cropped_sample)
        else:
            print('not to db is executed')
            sample = cropped_sample #without to_db

    elif processing == 'mu-doppler':
        np_sample = np.array(processed_sample)
        sample = np_sample #without to_db
        print('sample shape mu doppler', sample.shape)
        
    
    # convert to tensor and apply transform
    rad_tensor = torch.tensor(sample, dtype=torch.float32)

    if processing != 'mu-doppler':
        rad_tensor = rad_tensor.reshape(1, rad_tensor.shape[0], rad_tensor.shape[1])
        visualize_radar_data(rad_tensor, save_dir_image = './processing/unparallelized_images/', title_plot = 'mu-doppler', x_axis_name = 'range', y_axis_name = 'time')


    if (transform is not None):
        rad_tensor = transform(rad_tensor)

    visualize_radar_data(rad_tensor, save_dir_image = './processing/unparallelized_images/', title_plot = 'zz_end_result', x_axis_name = 'doppler', y_axis_name = 'range')

    print("Exit is put in radar processing because of debugging what is acutally happening in the processing.")


    exit()

    return rad_tensor


def perform_delta(sample, delta, stride, clipping):

    delta_radar = DeltaRadar(sample)

    # Print apply delta_radar deltas
    delta_all_chirps, delta_frames = delta_radar.deltas(delta_step = delta, stencil_stride = stride, pad_start = 'same', frame_size = 'same', base_chirp=False)

    print('delta_all_chirps shape', delta_all_chirps.shape)
    print('delta_frames shape', len(delta_frames) , delta_frames[0].shape)

    visualize_radar_data(delta_all_chirps, save_dir_image = './processing/unparallelized_images/', title_plot = 'delta_all_chirps', x_axis_name = 'chirp', y_axis_name = 'sample')
    # visualize_radar_data(delta_frames[0], save_dir_image = './processing/unparallelized_images/', title_plot = 'delta_frames', x_axis_name = 'chirp', y_axis_name = 'sample')

    # apply clipping / tresholding
    print('clipping', clipping)
    if clipping is not None:
        for frame in range(0, len(delta_frames)):

            if(clipping == "binary"):
                delta_frames[frame] = np.where(delta_frames[frame] > 0, 1, delta_frames[frame])
                delta_frames[frame] = np.where(delta_frames[frame] < 0, -1, delta_frames[frame])

            elif (clipping == "positive"):
                delta_frames[frame] = np.where(delta_frames[frame] > 0, delta_frames[frame], 0)

            elif (clipping == "negative"):
                delta_frames[frame] = np.where(delta_frames[frame] < 0, delta_frames[frame], 0)

            elif (clipping == "positive-binary"):
                delta_frames[frame] = np.where(delta_frames[frame] > 0, 1, 0)
            
            elif (clipping == "negative-binary"):
                delta_frames[frame] = np.where(delta_frames[frame] < 0, -1, 0)

            else:
                clipping_params = clipping.split('b')
                if len(clipping_params) > 1:
                    number = float(clipping_params[0])
                    binary = True
                else:
                    number = float(clipping)
                    binary = False

                delta_stddev = delta_all_chirps[2:].std()

                # if above 0, lvl-x with specified treshhold
                if (number > 0):
                    delta_frames[frame] = np.where(abs(delta_frames[frame]) >= number, delta_frames[frame], 0)

                # if below 0, multiple of std on all chirps (on a per-capture basis)
                elif (number < 0):
                    delta_frames[frame] = np.where(abs(delta_frames[frame]) >= delta_stddev*abs(number), delta_frames[frame], 0)

                if binary:
                    delta_frames[frame] = np.where(delta_frames[frame] > 0, 1, delta_frames[frame])
                    delta_frames[frame] = np.where(delta_frames[frame] < 0, -1, delta_frames[frame])

    return delta_frames

## -------------------------------------------------------------------------------------------------
##
## Class to generate deltas out of a sequence of radar frames
##
## NOTE: we make a shallow copy of the orig ADC data, we assume that they are readonly data!!!!
## (for limited mem reasons)
##

class DeltaRadar:
    def __init__(self, radar_frames):

        # This initialization is potato potato

        print("init delta radar object")
        #self.frames = copy.deepcopy( radar_frames )
        self.frames = copy.copy( radar_frames )       ## save mem
        self.frame_shape = radar_frames[0].shape

        self.all_chirps = np.copy( self.frames[0] )
        for frame in self.frames[1:]:
            self.all_chirps = np.concatenate( (self.all_chirps, frame), axis=0 )

        print("Made deep copy of radar frames")
        print("self.frames shape", len(self.frames))
        print("self.frames sahpe", self.frames[0].shape)
        print("self.all_chirps shape", self.all_chirps.shape)

        return

    def delta_stencil (self, radar_chirps, lookback_n=1, stride=1 ):
        ''' Stencil that scans a radar frame and computes a line at a time of deltas
            between chirps at a specified step distance.
            @param lookback_n: time-step/distance between chirps to compute delta
            @param stride: positions to shift stensil for next delta
        '''
        for step in range(lookback_n, len(radar_chirps), stride):
            deltas_one_step = radar_chirps[step] - radar_chirps[step - lookback_n]
            yield deltas_one_step


    def delta_frames (self, delta_vecs, frame_size ):
        ''' Slice a matrix of delta vectors in frames
            @param frame_size: num of delta vectors in one frame
            @param delta_vecs
        '''

        assert len(delta_vecs) % frame_size == 0, "ERROR: delta_vecs not an integer multiple of frame_size! Consider padding delta_ves or changing the frame_size."

        num_frames = len(delta_vecs) // frame_size

        delta_frames = []
        for f in range(num_frames):
            delta_frames.append( delta_vecs[f * frame_size : (f+1) * frame_size, : ] )

        return delta_frames


    def deltas (self, delta_step = 1, stencil_stride = 1, pad_start = 'same', frame_size = None, base_chirp = False):
        ''' Generate deltas for a given delta-step size and stride. For the computation of deltas
            all radar frames are first concatenated together in one big chirp matrix before applying
            the delta stensil. Padding (if any) is prepended to that all-in-1 matrix. In other words
            deltas are not computed per frame independently.
            @param delta_step
            @param stencil_stride
            @param pad_start: To generate an equal num of delta vectors as the num of chirps we need to prepend the
                              chirp data with some padding. Options are 'same' as the first chirp, a 'zero' valued chirp,
                              or None for no use of padding (in this case the delta vectors will be less than the num of
                              chirps.
            @param frame_size: If true, return the deltas also as a list of frames (works only with pad_start != None)
            @base_chirp: If a base chirp  with initial state should be appended
        '''
        print("Apply deltas to radar data")

        # This code is probably only for padding purposes
        if pad_start == 'same':
            print("Apply deltas to radar the same")
            pad = np.copy( self.all_chirps[0,:] )
            pad_vecs = pad * np.ones([delta_step, len(pad)], dtype=int)
            adc_all_chirps = np.concatenate( (pad_vecs, self.all_chirps), axis=0)
            print("pad shape", pad.shape)
            print("pad_vecs shape", pad_vecs.shape)
            print("adc_all_chirps shape", adc_all_chirps.shape)
        elif pad_start == 'zero':
            print("Apply deltas to radar the zero")
            pad_vecs = np.zeros( (delta_step, self.all_chirps.shape[1]), dtype=int )
            adc_all_chirps = np.concatenate( (pad_vecs, self.all_chirps), axis=0)
        elif pad_start == None:
            print("Apply deltas to radar the None")
            adc_all_chirps = self.all_chirps
        else:
            raise ValueError("'pad_start' can be None|'same'|'zero' only")


        delta_all_chirps = np.array( [ delta_row for delta_row in self.delta_stencil( adc_all_chirps, lookback_n=delta_step, stride=stencil_stride ) ] )
        delta_mats = []

        # adds a full base chirp at the front, but removes last - could be a good tradeoff.
        if base_chirp:
            delta_all_chirps = np.concatenate((self.all_chirps[0,:].reshape((1,512)), delta_all_chirps), axis=0)[:-1]

        if frame_size == 'same':
            assert pad_start != None, "ERROR: frame_size 'same' requires specification of pad_start!"
            frame_size = self.frame_shape[0]
            delta_mats = self.delta_frames (delta_all_chirps, frame_size)
        elif isinstance(frame_size, int) and frame_size > 0:
            delta_mats = self.delta_frames (delta_all_chirps, frame_size)
        elif frame_size == None:
            pass
        else:
            assert False, "ERROR: frame_size incorrectly set"


        return delta_all_chirps, delta_mats