import glob

import h5py

data_dir = '/local/radar8ghz-efficient'
files_to_be_processed = glob.glob(data_dir + '/**/*.rad', recursive = True)
# print(files_to_be_processed)
# print('maximally available samples',len(files_to_be_processed)*30)

cache_file = '/local/efficient_pipeline/preprocessed_data/radar_data.hdf5'

with h5py.File(cache_file, 'r') as f:
    print(f['array_160x160'])

# for i in range(8000):
with h5py.File(cache_file, 'r') as f:
    print(f['signs'])
    for i in range(f['signs'].shape[0]):
        print(f['signs'][i])
    # labels = f['signs']
    # print(labels[0])
    # print(labels[1013])
    print(f['signs'][9064]) # should be 7
