
# Requirements
python version >= 3.10.8.4
CUDA version >= 11.8.0 

# Setting up the Python Environment and Getting the data automatically

Creates python environment 'scratch', downloads the dataset and puts it into correct folder and uzips it.
~~~
. setup.sh
~~~

# Setting up the Python environment


~~~
python -m venv scratch
~~~

activate python environment
~~~
. scratch/bin/activate
~~~

When on high performance cluster load modules
~~~
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.8.0
~~~

Install packages
~~~
pip install --upgrade pip
pip install -r requirements.txt
~~~

# Getting the data

Getting the dataset
~~~
wget https://zenodo.org/records/10359770/files/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized.7z?download=1
mv 'radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized.7z?download=1' ./data/radar8GHz-DVS-marshaling_signals_v1.7z
python unzip7zip.py
mv ./data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized ./data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v1
~~~


alteranative for unpacking 7z file is:
~~~
7z x radar8GHz-DVS-marshaling_signals_v1.7z
~~~

# Running the code

Edit configuration file for defining folders for data, so in 'configuration_grid.py' change:
~~~
# data_dir = '/local/efficient_radar_pipeline/'
for_parallelized_preprocessing_example_rad_file = './data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v1/train/foyer/2-move_back_v1-300/ofxRadar8Ghz_2022-04-21_13-54-09.rad'

data_dir = './data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v1/train'
data_dir_test = './data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v1/test'
# cache_file_dir = './data/preprocessed_data'
cache_file_dir = '/local/marshalling_efficient/'
~~~



simple run
~~~
python main.py
~~~

arguments:
-j 'integer' # makes copy of processing dataset so multiple python scripts can read from md5 file
-d # Debugging mode only runs 2 epochs (usefull for debugging)
-pdm # (preprocessed datamodule) skips creating datamodule (saves 40sec, usefull for debuggin)
-ppm # (preprocessed preprocessing module) skips preprocessing (saves 2min on GTX2080, usefull for debugging)
