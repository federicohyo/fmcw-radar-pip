python -m venv scratch

. scratch/bin/activate
pip install --upgrade pip

pip install -r requirements.txt

wget https://zenodo.org/records/7656911/files/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized.7z?download=1
mv 'radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized.7z?download=1' ./data/radar8GHz-DVS-marshaling_signals_v1.7z
python unzip7zip.py
mv ./data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized ./data/radar8Ghz-DVS-marshaling_signals_20220901_publication_anonymized_v1
