import py7zr
import tqdm

# path to 7z file
file_path = './data/radar8GHz-DVS-marshaling_signals_v1.7z'

# path to the directory to unpack
extract_path = './data/'

print("Can take a few minutes to unpack 30GB or so.")

with py7zr.SevenZipFile(file_path, mode = 'r') as z:
 z.extractall(path=extract_path)

print("Extraction completed.")