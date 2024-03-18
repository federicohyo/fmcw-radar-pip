import glob
import os
from tqdm import tqdm
import re
import sys
sys.path.append('../')

import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange, reduce, repeat

import pandas as pd

import h5py

from utils.radar_file_parser import RadarFileParser 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

from imblearn.under_sampling import NearMiss, RandomUnderSampler
from collections import Counter

from processing.parallelized_preprocessing import change_data_type, preprocessing_radar_range_chirp, preprocessing_radar_range_doppler, preprocessing_radar_mu_doppler_straight, preprocessing_radar_mu_doppler_loop_best_bin_finding
from processing.parallelized_preprocessing import visualize_radar_data
from processing.parallelized_preprocessing import delta_frames_radar, preprocessing_radar_mu_doppler_loop

class DataModuleRadarDynamicPreprocessing():

    def __init__(self, data_dir, frame_limit, sample_limit = 960, under_sample = True):
        
        self.frame_limit = frame_limit
        
        self.sample_limit = sample_limit
        self.signals = ['none', 'emergency_stop', 'move_ahead', 'move_back_v1', 'move_back_v2', 'slow_down', 'start_engines', 'stop_engines', 'straight_ahead', 'turn_left', 'turn_right']
        self.seed = 42
        
        self.data = {'indices': [], 'labels': [], 'full_labels': [], 'groups': [], 'rad_chunk_idx': [], 'person_id': [], 'room': []}
        # Goal is to fill these up, following ML dataset rules
        self.data_train = {'indices': [], 'labels': [], 'full_labels': [], 'groups': [], 'rad_chunk_idx': [], 'person_id': []}
        self.data_val = {'indices': [],'labels': [], 'full_labels': [], 'groups': [], 'rad_chunk_idx': [], 'person_id': []}
        self.data_test = {'indices': [], 'labels': [], 'full_labels': [], 'groups': [], 'rad_chunk_idx': [], 'person_id': []}

        # Get path of radar data
        if os.path.exists(data_dir):
            paths_of_radar_data = glob.glob(data_dir + '/**/*.rad', recursive=True)
            print('Found radar sequences:', len(paths_of_radar_data))
            self.paths_of_radar_data = paths_of_radar_data
        else:
            print("data dir not found:",data_dir)
            raise ValueError("Path does not exist")

        self.signals_to_ignore = ["mixed", "noise"]
        self.distances_to_ignore = ["100", "450"]
        
        frame_counter = 0
        # Get all the labels
        for  rad_file in (tqdm(self.paths_of_radar_data, desc='Parsing Files')):

            # example of rad file:
            # foyer/2-move_back_v2-350/ofxRadar8Ghz_2022-04-21_15-26-08.rad

            room = rad_file.split("/")[-3]
            person = rad_file.split("/")[-2].split("-")[0]
            distance = rad_file.split("/")[-2].split("-")[2]
            signal = rad_file.split("/")[-2].split("-")[1]

            date = rad_file.split("/")[-1].split("_")[1]
            time = rad_file.split("/")[-1].split("_")[2]
            
            if signal == "start_engine":
                signal = "start_engines"
            if signal == "stop_engine":
                signal = "stop_engines"   


            description = '-'.join([room, signal, distance, date, time, person]) 
            group = '-'.join([signal, distance])

            rad = RadarFileParser(rad_file, read_radar=True, read_dvs=False) # radar data is a lot quicker to read and all we really need
            chirps_per_frame = rad.num_chirps_per_frame
            samples_per_chirp = rad.num_samples_per_chirp
            radar_frames = rad.radar_frames
            frame_count = len(radar_frames)

            indices = []
            full_labels = []
            labels = []
            groups = []

            nr_chunks = 0
            chunk_size = 0

            if(self.frame_limit != None):
                nr_chunks = int(frame_count / frame_limit)
                # print(nr_chunks)
                indices = [ (rad_file, i) for i in range(0, nr_chunks)]
                full_labels = [f"{description}_{i}" for i in range(0, nr_chunks)]
                groups = [group for i in range(0, nr_chunks)]
                person = [person for i in range(0, nr_chunks)]
                rad_chunk_idx = [ frame_counter + i for i in range(0, nr_chunks)]
                rooms = [room for i in range(0, nr_chunks)]
                frame_counter += nr_chunks
            else:
                raise Exception("Input samples with potetntially different sizes are not supported")

            # Keep this before chunk_counter, otherwise reference to preprocessed is lost
            # ignore unused signals / classes
            if signal in self.signals_to_ignore:
                continue

            # ignore dropped distances
            if distance in self.distances_to_ignore:
                continue

            if(self.frame_limit != None):
                labels = [self.signals.index(signal) for i in range(0, nr_chunks)]
            
            self.data['indices'].extend(indices)
            self.data['full_labels'].extend(full_labels)
            self.data['labels'].extend(labels)
            self.data['groups'].extend(groups)
            self.data['rad_chunk_idx'].extend(rad_chunk_idx)
            self.data['person_id'].extend(person)
            self.data['room'].extend(rooms)
        
        self.data['indices'] = [ f'rad_file:{rad_file} rad_idx:{i}' for rad_file, i in self.data['indices']]

        # convert everything to NP arrays so we can undersample and do the train/test/val split
        self.data['indices'] = np.array(self.data['indices'])
        self.data['labels'] = np.array(self.data['labels'])
        self.data['full_labels'] = np.array(self.data['full_labels'])
        self.data['groups'] = np.array(self.data['groups'])   
        self.data['rad_chunk_idx'] = np.array(self.data['rad_chunk_idx'])
        self.data['person_id'] = np.array(self.data['person_id']) 
        self.data['room'] = np.array(self.data['room'])

        print(f'Undersampling dataset to balance...')
        print('Original dataset shape %s' % Counter(self.data['labels']))
        sampling_strategy = dict(zip([i for i in range(0, len(self.signals))], [self.sample_limit] * len(self.signals)))
        print(f'Sampling strategy: {sampling_strategy}')
        
        # print([self.sample_limit] * len(self.signals))
        if (under_sample == True):
            resampler = RandomUnderSampler(random_state=self.seed, sampling_strategy=sampling_strategy)
            dummy_x = np.zeros([len(self.data['labels']), 1])
            _, y_res = resampler.fit_resample(dummy_x, self.data['labels'])
            groups_res = self.data['groups'][resampler.sample_indices_]
            full_labels_res = self.data['full_labels'][resampler.sample_indices_]
            rad_chunk_idx_res = self.data['rad_chunk_idx'][resampler.sample_indices_]
            person_res = self.data['person_id'][resampler.sample_indices_]
            X_res = self.data['indices'][resampler.sample_indices_]
            room_res = self.data['room'][resampler.sample_indices_]

            print('Resampled dataset shape %s' % Counter(y_res))
            
            self.data['indices'] = X_res 
            self.data['labels'] = y_res
            self.data['full_labels'] = full_labels_res
            self.data['groups'] = groups_res
            self.data['rad_chunk_idx'] = rad_chunk_idx_res
            self.data['person_id'] = person_res
            self.data['room'] = room_res

        ### LEAVE ONE OUT TWO FOLD CROSS VALIDATION ###
        # Code for leave one out two fold cross validation
        df_full = pd.DataFrame({
            'indices': self.data['indices'],
            'labels': self.data['labels'],
            'full_labels': self.data['full_labels'],
            'groups': self.data['groups'],
            'rad_chunk_idx': self.data['rad_chunk_idx'],
            'person_ids': self.data['person_id'],
            'room': self.data['room'], 
        })

        self.data_as_df = df_full
        self.data_full = df_full.to_dict('list')

        df_without_3 = df_full[~df_full['person_ids'].str.contains('3')].to_dict('list')
        df_3 = df_full[df_full['person_ids'].str.contains('3')].to_dict('list')
        df_without_2 = df_full[~df_full['person_ids'].str.contains('2')].to_dict('list')
        df_2 = df_full[df_full['person_ids'].str.contains('2')].to_dict('list')

        self.data_leave_one_out_two_fold_train = [df_without_3,df_without_2] 
        self.data_leave_one_out_two_fold_valid = [df_3,df_2]

        # print(f"Number of samples lenght leave one out two fold train: {len(df_without_3['indices'])}, {len(df_without_2['indices'])}")
        # print(f"Number of samples lenght leave one out two fold valid: {len(df_3['indices'])}, {len(df_2['indices'])}")


    def preprocess_data_and_store(self , data_dir, save_dir, fft_Nx, fft_Ny, features = [] ,preprocessing_range_chirp = [], postprocessing_range_chirp = [], preprocessing_range_doppler = [], postprocessing_range_doppler = [], change_data_type_to = '', apply_delta = False ,transform = None, image_size_x = 160, image_size_y = 160):

        if data_dir == None:
            ValueError("data_dir cannot be None")

        print("Preprocessing data and storing it in hdf5 format")
        print("Number of files to process:", len(self.data['indices']))

        print("Number of features to process:", len(features))

        # Create hdf5 file
        with h5py.File(save_dir, 'w') as f:
            f.create_dataset('array_160x160', (0, len(features), image_size_x, image_size_y), maxshape=(None, None ,image_size_x, image_size_y), dtype='float32')
            f.create_dataset('signs', (0,), maxshape=(None,), dtype='int64')

        # print what is going to be processed
        print("Preprocessing data and storing it in hdf5 format")
        print("fft_Nx:", fft_Nx)
        print("fft_Ny:", fft_Ny)
        print("change_data_type_to:", change_data_type_to)
        print("preprocessing_range_chirp:", preprocessing_range_chirp)
        print("postprocessing_range_chirp:", postprocessing_range_chirp)
        print("preprocessing_range_doppler:", preprocessing_range_doppler)
        print("postprocessing_range_doppler:", postprocessing_range_doppler)
        print("transform:", transform)

        for file in tqdm(self.paths_of_radar_data, desc='Processing files'):
            # file, chunk_idx = self.data['indices'][idx]

            # print("Processing file:", file)
            rad_file = RadarFileParser(file, read_radar=True, read_dvs=False) # radar data is a lot quicker to read and all we really need
            all_radar_frames = []
            for frame in rad_file.radar_frames:
                all_radar_frames.append(frame.td_matrix)
            all_radar_frames = np.array(all_radar_frames, dtype=float)
            chunk_size =  self.frame_limit * rad_file.num_chirps_per_frame
            chunked_radar_frames = np.reshape(all_radar_frames, [-1, chunk_size, int(rad_file.num_samples_per_chirp)])
            # visualize_radar_data(torch.from_numpy(chunked_radar_frames[0]), './', 'raw_adc_chunked', 'chirps', 'samples')

            # room_dir, person_dir, filename = file.split(os.sep)[-3:]
            # print('filename:', filename)
            # room, sign, distance, distance_unit, *rest = filename.split('-')
            # print('file information:', room, signal, distance, distance_unit)

            # print("rad_file:", file)

            room = file.split("/")[-3]
            person = file.split("/")[-2].split("-")[0]
            distance = file.split("/")[-2].split("-")[2]
            sign = file.split("/")[-2].split("-")[1]

            date = file.split("/")[-1].split("_")[1]
            time = file.split("/")[-1].split("_")[2]

            if sign == "start_engine":
                sign = "start_engines"
            if sign == "stop_engine":
                sign = "stop_engines"   
            

            frame_count = len(rad_file.radar_frames)
            nr_chunks = int(frame_count / self.frame_limit)
            if sign in self.signals_to_ignore:
                with h5py.File(save_dir, 'a') as f:
                    original_size_160x160 = f['array_160x160'].shape[0]
                    # print("original_size_160x160:", original_size_160x160)
                    # print("nr_chunks:", nr_chunks)
                    f['array_160x160'].resize((original_size_160x160 + nr_chunks ), axis=0)
                    original_size_1 = f['signs'].shape[0]
                    f['signs'].resize((original_size_160x160 + nr_chunks), axis=0)
                continue
            if distance in self.distances_to_ignore:
                with h5py.File(save_dir, 'a') as f:
                    original_size_160x160 = f['array_160x160'].shape[0]
                    # print('original_size_160x160:', original_size_160x160)
                    # print('nr_chunks:', nr_chunks)
                    f['array_160x160'].resize((original_size_160x160 + nr_chunks), axis=0)
                    original_size_1 = f['signs'].shape[0]
                    f['signs'].resize((original_size_160x160 + nr_chunks), axis=0)
                continue

            signal_raw = None
            signal_range_chirp = None
            signal_range_doppler = None
            signal_mu_doppler = None


            signs = [self.signals.index(sign) for i in range(0, nr_chunks)]
            # print("signs:", signs)
            signs = np.array(signs)

            signal = torch.from_numpy(chunked_radar_frames).cuda()

            if 'raw' in features:
                signal_raw = torch.abs(torch.from_numpy(chunked_radar_frames).cuda())

            if change_data_type_to != '' and change_data_type_to != 'float8_virtually' and change_data_type_to != 'binary': 
                signal = change_data_type(signal, change_data_type_to)

            if apply_delta == True:
                signal = delta_frames_radar(signal)
                if 'raw' in features:
                    signal_raw = signal.clone()

            if 'range_chirp' in features or 'range_doppler' in features or 'mu_doppler' in features:
                signal = preprocessing_radar_range_chirp(
                    signal = signal,
                    bins_fft_range = fft_Nx,
                    preprocessing = preprocessing_range_chirp,
                    postprocessing = postprocessing_range_chirp,
                    change_data_type_to = change_data_type_to,
                )
                signal_range_chirp = signal.clone()
            
            if ('range_doppler' in features):
                signal = preprocessing_radar_range_doppler(
                    signal = signal,
                    bins_fft_doppler = fft_Ny,
                    preprocessing = preprocessing_range_doppler,
                    postprocessing = postprocessing_range_doppler,
                    change_data_type_to = change_data_type_to,
                )
                signal_range_doppler = signal.clone()

            if ('mu_doppler' in features):
                # chunked_radar_frames = torch.from_numpy(chunked_radar_frames).cuda()
                # print("chunked_radar_frames.shape:", signal.shape)
                
                signal_mu = torch.from_numpy(chunked_radar_frames).cuda()
                
                signal_mu = signal_mu - torch.mean(signal_mu, dim=(1,2), keepdim=True)

                signal_mu = rearrange(signal_mu, 'f c s -> (f c) s')
                
                signal_mu = signal_mu.unsqueeze(0)
                signal_mu = delta_frames_radar(signal_mu)

                signal_mu = preprocessing_radar_range_chirp(
                    signal = signal_mu,
                    bins_fft_range = 256,
                    preprocessing = ['mean_removal', 'hanning_window'],
                    postprocessing = [],
                    )

                signal = preprocessing_radar_mu_doppler_loop_best_bin_finding(
                # signal = preprocessing_radar_mu_doppler_straight(
                # signal = preprocessing_radar_mu_doppler_loop(
                    signal = signal_mu,
                    fft_N_mdopp = 192,
                    wnd_lenght = 96,
                    wnd_stride = 2,
                    preprocessing = [],
                    postprocessing = [],
                    nr_chunks = nr_chunks,
                )
                signal_mu_doppler = signal.clone()

                # visualize_radar_data(rearrange(signal_mu_doppler, 'f s c -> s (f c)'), './processing/parallelized_images/', distance + '_' + person_dir, 'time', 'frequency')

                # print('doppler shape',signal_mu_doppler.shape)
                # exit()
                # input("Press Enter to continue...")

            X = []
            if signal_raw != None and 'raw' in features:
                # print('raw')
                if transform:
                    X.append(transform(signal_raw))

            if signal_range_chirp != None and 'range_chirp' in features:
                # print("range_chirp")
                if transform:
                    X.append(transform(signal_range_chirp))
            
            if signal_range_doppler != None and 'range_doppler' in features:
                # print("dtype signal range doppler",signal_range_doppler.dtype)
                if transform:
                    X.append(transform(signal_range_doppler))

            if signal_mu_doppler != None and 'mu_doppler' in features:
                # torch.abs(signal_mu_doppler)
                # print("signal_mu_doppler",signal_mu_doppler.dtype)
                if transform:
                    X.append(transform(signal_mu_doppler))

            # Transform list to pytorch matrix
            X = torch.stack(X, dim=1).cuda()

            # Save signal to hdf5 file
            with h5py.File(save_dir, 'a') as f:
                original_size_160x160 = f['array_160x160'].shape[0]
                new_size_160x160 = original_size_160x160 + signal.shape[0]  # Increase by the number of new samples
                f['array_160x160'].resize((new_size_160x160, len(features), image_size_x, image_size_y))
                f['array_160x160'][original_size_160x160:new_size_160x160] = X.cpu().numpy()
                original_size_1 = f['signs'].shape[0]
                new_size_1 = original_size_1 + signs.shape[0]
                f['signs'].resize((new_size_1,))
                f['signs'][original_size_1:new_size_1] = signs

                # print(f"shape of signs: {signal.shape}")
                # print(f"shape of array_160x160: {f['array_160x160'].shape}")


class DataSetRadarDynamicPreprocessing(Dataset):
    def __init__(self, dataset, cache_file, transform = None):
        self.data = dataset
        # Converting each string back to a tuple
        indices = []
        # pattern = r"'(.+\.rad)', (\d+)"
        pattern = r"rad_file:(.+\.rad) rad_idx:(\d+)"
        for s in self.data['indices']:
            match = re.match(pattern, s)
            if match:
                rad_file = match.group(1)
                chunk_idx = int(match.group(2))
                indices.append((rad_file,chunk_idx))
            else:
                print(f"String:{s} does not match pattern '{pattern}'")
        
        self.indices_loader = indices
        self.labels_loader = self.data['labels']
        self.full_labels_loader = self.data['full_labels']
        self.groups_loader = self.data['groups']
        self.rad_chunk_idx_loader = self.data['rad_chunk_idx']
        self.cache_file = cache_file
        self.transform = transform

        self.mean = None
        self.std = None

        self.df = pd.DataFrame({
            'indices': self.indices_loader,
            'labels': self.labels_loader,
            'full_labels': self.full_labels_loader,
            'groups': self.groups_loader,
            'rad_chunk_idx': self.rad_chunk_idx_loader,
            # Include other attributes here if needed
        })

        # print(self.indices_loader)

    def compute_mean_std_off_train(self):
        
        mean = 0.
        std = 0.
        for idx in tqdm(range(len(self)), desc='Computing mean and std'):
            X, _, _, _, _, _, _ = self.__getitem__(idx)
            # print(X.shape)
            mean += X.mean(dim=(1,2), keepdim=True)
            std += X.std(dim=(1,2), keepdim=True)

        mean /= len(self)
        std /= len(self)
        self.mean = mean
        self.std = std

        return mean, std
    
    def set_mean_std_for_test_or_validation(self, mean, std):
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.rad_chunk_idx_loader)

    def __getitem__(self,idx):        
        file, chunk_idx = self.indices_loader[idx]
        label = self.labels_loader[idx]
        full_label = self.full_labels_loader[idx]
        group = self.groups_loader[idx]
        rad_chunk_idx = self.rad_chunk_idx_loader[idx]

        with h5py.File(self.cache_file, 'r') as f:
            # print(f['array_160x160'].shape)
            X = f['array_160x160'][rad_chunk_idx] # because of preprocessing purposes this is a pointer to preprocessed data, precalculated
            y = f['signs'][rad_chunk_idx]

        X = torch.from_numpy(X)

        if self.mean != None and self.std != None:
            X = (X - self.mean)/self.std

        if self.transform:
            X = self.transform(X)

        # return X, y ,label, file, chunk_idx, full_label, group
        return X, y ,label, file, chunk_idx, idx, group



class DataModuleRadarDynamicPreprocessing_test():

    def __init__(self, data_dir, frame_limit):
        
        self.frame_limit = frame_limit
        
        self.signals = ['none', 'emergency_stop', 'move_ahead', 'move_back_v1', 'move_back_v2', 'slow_down', 'start_engines', 'stop_engines', 'straight_ahead', 'turn_left', 'turn_right']
        self.seed = 42
        
        self.data = {'indices': [], 'labels': [], 'full_labels': [], 'groups': [], 'rad_chunk_idx': [], 'person_id': []}
        # Goal is to fill these up, following ML dataset rules

        # Get path of radar data
        if os.path.exists(data_dir):
            paths_of_radar_data = glob.glob(data_dir + '/**/*.rad', recursive=True)
            print('Found radar sequences:', len(paths_of_radar_data))
            self.paths_of_radar_data = paths_of_radar_data
        else:
            print("data dir not found:",data_dir)
            raise ValueError("Path does not exist")

        self.signals_to_ignore = ["mixed", "noise"]
        self.distances_to_ignore = ["100", "450"]
        
        frame_counter = 0
        # Get all the labels
        for  rad_file in (tqdm(self.paths_of_radar_data, desc='Parsing Files')):

            # example of rad file:
            # foyer/2-move_back_v2-350/ofxRadar8Ghz_2022-04-21_15-26-08.rad

            room = rad_file.split("/")[-3]
            person = rad_file.split("/")[-2].split("-")[0]
            distance = rad_file.split("/")[-2].split("-")[2]
            signal = rad_file.split("/")[-2].split("-")[1]

            date = rad_file.split("/")[-1].split("_")[1]
            time = rad_file.split("/")[-1].split("_")[2]
            
            if signal == "start_engine":
                signal = "start_engines"
            if signal == "stop_engine":
                signal = "stop_engines"   


            description = '-'.join([room, signal, distance, date, time, person]) 
            group = '-'.join([signal, distance])
            # print('person', person)
            # print('group',group)

            rad = RadarFileParser(rad_file, read_radar=True, read_dvs=False) # radar data is a lot quicker to read and all we really need
            chirps_per_frame = rad.num_chirps_per_frame
            samples_per_chirp = rad.num_samples_per_chirp
            radar_frames = rad.radar_frames
            # self.chunk_size = frame_limit * chirps_per_frame
            # self.nr_chunks = int(len(radar_frames) / frame_limit)
            frame_count = len(radar_frames)

            indices = []
            full_labels = []
            labels = []
            groups = []

            nr_chunks = 0
            chunk_size = 0

            if(self.frame_limit != None):
                nr_chunks = int(frame_count / frame_limit)
                # print(nr_chunks)
                indices = [ (rad_file, i) for i in range(0, nr_chunks)]
                full_labels = [f"{description}_{i}" for i in range(0, nr_chunks)]
                groups = [group for i in range(0, nr_chunks)]
                person = [person for i in range(0, nr_chunks)]
                rad_chunk_idx = [ frame_counter + i for i in range(0, nr_chunks)]
                frame_counter += nr_chunks
            else:
                raise Exception("Input samples with potetntially different sizes are not supported")

            # Keep this before chunk_counter, otherwise reference to preprocessed is lost
            # ignore unused signals / classes
            if signal in self.signals_to_ignore:
                # print(f"ignoring signal {signal} in file {rad_file}")
                continue

            # ignore dropped distances
            if distance in self.distances_to_ignore:
                # print(f"ignoring distance {distance} in file {rad_file}")
                continue

            if(self.frame_limit != None):
                labels = [self.signals.index(signal) for i in range(0, nr_chunks)]
            
            self.data['indices'].extend(indices)
            self.data['full_labels'].extend(full_labels)
            self.data['labels'].extend(labels)
            self.data['groups'].extend(groups)
            self.data['rad_chunk_idx'].extend(rad_chunk_idx)
            self.data['person_id'].extend(person)
        
        self.data['indices'] = [ f'rad_file:{rad_file} rad_idx:{i}' for rad_file, i in self.data['indices']]

        # convert everything to NP arrays so we can undersample and do the train/test/val split
        self.data['indices'] = np.array(self.data['indices'])
        self.data['labels'] = np.array(self.data['labels'])
        self.data['full_labels'] = np.array(self.data['full_labels'])
        self.data['groups'] = np.array(self.data['groups'])   
        self.data['rad_chunk_idx'] = np.array(self.data['rad_chunk_idx'])
        self.data['person_id'] = np.array(self.data['person_id']) 

        df_full = pd.DataFrame({
            'indices': self.data['indices'],
            'labels': self.data['labels'],
            'full_labels': self.data['full_labels'],
            'groups': self.data['groups'],
            'rad_chunk_idx': self.data['rad_chunk_idx'],
            'person_ids': self.data['person_id'],
        })

        self.data_as_df = df_full
        self.data_test = df_full.to_dict('list')
        

    def preprocess_data_and_store(self , data_dir, save_dir, fft_Nx, fft_Ny, features = [] ,preprocessing_range_chirp = [], postprocessing_range_chirp = [], preprocessing_range_doppler = [], postprocessing_range_doppler = [], change_data_type_to = '', apply_delta = False ,transform = None, image_size_x = 160, image_size_y = 160):

        if data_dir == None:
            ValueError("data_dir cannot be None")

        print("Preprocessing data and storing it in hdf5 format")
        print("Number of files to process:", len(self.data['indices']))

        print("Number of features to process:", len(features))

        # Create hdf5 file
        with h5py.File(save_dir, 'w') as f:
            f.create_dataset('array_160x160', (0, len(features), image_size_x, image_size_y), maxshape=(None, None ,image_size_x, image_size_y), dtype='float32')
            f.create_dataset('signs', (0,), maxshape=(None,), dtype='int64')

        # print what is going to be processed
        print("Preprocessing data and storing it in hdf5 format")
        print("fft_Nx:", fft_Nx)
        print("fft_Ny:", fft_Ny)
        print("change_data_type_to:", change_data_type_to)
        print("preprocessing_range_chirp:", preprocessing_range_chirp)
        print("postprocessing_range_chirp:", postprocessing_range_chirp)
        print("preprocessing_range_doppler:", preprocessing_range_doppler)
        print("postprocessing_range_doppler:", postprocessing_range_doppler)
        print("transform:", transform)

        for file in tqdm(self.paths_of_radar_data, desc='Processing files'):
            # file, chunk_idx = self.data['indices'][idx]

            # print("Processing file:", file)
            rad_file = RadarFileParser(file, read_radar=True, read_dvs=False) # radar data is a lot quicker to read and all we really need
            all_radar_frames = []
            for frame in rad_file.radar_frames:
                all_radar_frames.append(frame.td_matrix)
            all_radar_frames = np.array(all_radar_frames, dtype=float)
            chunk_size =  self.frame_limit * rad_file.num_chirps_per_frame
            chunked_radar_frames = np.reshape(all_radar_frames, [-1, chunk_size, int(rad_file.num_samples_per_chirp)])
            # visualize_radar_data(torch.from_numpy(chunked_radar_frames[0]), './', 'raw_adc_chunked', 'chirps', 'samples')

            # room_dir, person_dir, filename = file.split(os.sep)[-3:]
            # print('filename:', filename)
            # room, sign, distance, distance_unit, *rest = filename.split('-')
            # print('file information:', room, signal, distance, distance_unit)

            # print("rad_file:", file)

            room = file.split("/")[-3]
            person = file.split("/")[-2].split("-")[0]
            distance = file.split("/")[-2].split("-")[2]
            sign = file.split("/")[-2].split("-")[1]

            date = file.split("/")[-1].split("_")[1]
            time = file.split("/")[-1].split("_")[2]

            if sign == "start_engine":
                sign = "start_engines"
            if sign == "stop_engine":
                sign = "stop_engines"   
            

            frame_count = len(rad_file.radar_frames)
            nr_chunks = int(frame_count / self.frame_limit)
            if sign in self.signals_to_ignore:
                with h5py.File(save_dir, 'a') as f:
                    original_size_160x160 = f['array_160x160'].shape[0]
                    # print("original_size_160x160:", original_size_160x160)
                    # print("nr_chunks:", nr_chunks)
                    f['array_160x160'].resize((original_size_160x160 + nr_chunks ), axis=0)
                    original_size_1 = f['signs'].shape[0]
                    f['signs'].resize((original_size_160x160 + nr_chunks), axis=0)
                continue
            if distance in self.distances_to_ignore:
                with h5py.File(save_dir, 'a') as f:
                    original_size_160x160 = f['array_160x160'].shape[0]
                    # print('original_size_160x160:', original_size_160x160)
                    # print('nr_chunks:', nr_chunks)
                    f['array_160x160'].resize((original_size_160x160 + nr_chunks), axis=0)
                    original_size_1 = f['signs'].shape[0]
                    f['signs'].resize((original_size_160x160 + nr_chunks), axis=0)
                continue

            signal_raw = None
            signal_range_chirp = None
            signal_range_doppler = None
            signal_mu_doppler = None


            signs = [self.signals.index(sign) for i in range(0, nr_chunks)]
            # print("signs:", signs)
            signs = np.array(signs)

            signal = torch.from_numpy(chunked_radar_frames).cuda()
            
            if 'raw' in features:
                signal_raw = torch.abs(torch.from_numpy(chunked_radar_frames).cuda())

            if change_data_type_to != '' and change_data_type_to != 'float8_virtually' and change_data_type_to != 'binary': 
                signal = change_data_type(signal, change_data_type_to)

            if apply_delta == True:
                signal = delta_frames_radar(signal)
                if 'raw' in features:
                    signal_raw = signal.clone()

            if 'range_chirp' in features or 'range_doppler' in features or 'mu_doppler' in features:
                signal = preprocessing_radar_range_chirp(
                    signal = signal,
                    bins_fft_range = fft_Nx,
                    preprocessing = preprocessing_range_chirp,
                    postprocessing = postprocessing_range_chirp,
                    change_data_type_to = change_data_type_to,
                )
                signal_range_chirp = signal.clone()
            
            if ('range_doppler' in features):
                signal = preprocessing_radar_range_doppler(
                    signal = signal,
                    bins_fft_doppler = fft_Ny,
                    preprocessing = preprocessing_range_doppler,
                    postprocessing = postprocessing_range_doppler,
                    change_data_type_to = change_data_type_to,
                )
                signal_range_doppler = signal.clone()

            if ('mu_doppler' in features):
                # chunked_radar_frames = torch.from_numpy(chunked_radar_frames).cuda()
                # print("chunked_radar_frames.shape:", signal.shape)
                
                signal_mu = torch.from_numpy(chunked_radar_frames).cuda()
                
                signal_mu = signal_mu - torch.mean(signal_mu, dim=(1,2), keepdim=True)

                signal_mu = rearrange(signal_mu, 'f c s -> (f c) s')
                
                signal_mu = signal_mu.unsqueeze(0)
                signal_mu = delta_frames_radar(signal_mu)


                signal_mu = preprocessing_radar_range_chirp(
                    signal = signal_mu,
                    bins_fft_range = 256,
                    preprocessing = ['mean_removal', 'hanning_window'],
                    postprocessing = [],
                    )

                signal = preprocessing_radar_mu_doppler_loop_best_bin_finding(
                # signal = preprocessing_radar_mu_doppler_straight(
                # signal = preprocessing_radar_mu_doppler_loop(
                    signal = signal_mu,
                    fft_N_mdopp = 192,
                    wnd_lenght = 96,
                    wnd_stride = 2,
                    preprocessing = [],
                    postprocessing = [],
                    nr_chunks = nr_chunks,
                )
                signal_mu_doppler = signal.clone()

                # visualize_radar_data(rearrange(signal_mu_doppler, 'f s c -> s (f c)'), './processing/parallelized_images/', distance + '_' + person_dir, 'time', 'frequency')

                # print('doppler shape',signal_mu_doppler.shape)
                # exit()
                # input("Press Enter to continue...")

            X = []
            if signal_raw != None and 'raw' in features:
                if transform:
                    X.append(transform(signal_raw))

            if signal_range_chirp != None and 'range_chirp' in features:
                if transform:
                    X.append(transform(signal_range_chirp))
            
            if signal_range_doppler != None and 'range_doppler' in features:
                if transform:
                    X.append(transform(signal_range_doppler))

            if signal_mu_doppler != None and 'mu_doppler' in features:
                if transform:
                    X.append(transform(signal_mu_doppler))

            # Transform list to pytorch matrix
            X = torch.stack(X, dim=1).cuda()
            # print("X.shape:", X.shape)

            # Save signal to hdf5 file
            with h5py.File(save_dir, 'a') as f:
                original_size_160x160 = f['array_160x160'].shape[0]
                new_size_160x160 = original_size_160x160 + signal.shape[0]  # Increase by the number of new samples
                f['array_160x160'].resize((new_size_160x160, len(features), image_size_x, image_size_y))
                f['array_160x160'][original_size_160x160:new_size_160x160] = X.cpu().numpy()
                original_size_1 = f['signs'].shape[0]
                new_size_1 = original_size_1 + signs.shape[0]
                f['signs'].resize((new_size_1,))
                f['signs'][original_size_1:new_size_1] = signs

                # print(f"shape of signs: {signal.shape}")
                # print(f"shape of array_160x160: {f['array_160x160'].shape}")
