# Setting up the Python Environment and Getting the data automatically
. setup.sh

# Setting up the Python environment

python version >= 3.10.8.4
CUDA version >= 11.8.0 

~~~
python -m venv scratch
~~~

activate
~~~
. scratch/bin/activate
pip install --upgrade pip
~~~

~~~
pip install -r requirements.txt
~~~

# Getting the data

v1 data
~~~
wget https://zenodo.org/records/7656911/files/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized.7z?download=1
mv 'radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized.7z?download=1' ./data/radar8GHz-DVS-marshaling_signals_v1.7z
python unzip7zip.py
mv ./data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized ./data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v1
~~~

v2 data
~~~
wget https://zenodo.org/records/10359770/files/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized.7z?download=1
mv 'radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized.7z?download=1' ./data/radar8GHz-DVS-marshaling_signals_v2.7z
python unzip7zip.py
mv ./data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized ./data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v2
~~~

alteranative for unpacking 7z file is:
~~~
7z x radar8GHz-DVS-marshaling_signals_v1.7z
~~~

# Running the code

Edit configuration file for defining folders for data


simple run
~~~
python main.py

arguments:
-j 'integer' # makes copy of processing dataset so multiple python scripts can read from md5 file
-d # Debugging mode only runs 2 epochs (usefull for debugging)
-pdm # (preprocessed datamodule) skips creating datamodule (saves 40sec, usefull for debuggin)
-ppm # (preprocessed preprocessing module) skips preprocessing (saves 2min on GTX2080, usefull for debugging)
