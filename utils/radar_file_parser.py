import struct
import numpy as np
import io
import os
import sys
import time
from collections import namedtuple


Frame = namedtuple("Frame", "td_matrix timestamps hw_timestamp_pri hw_timestamp_dma ant_config_from_hw")
Polarity = namedtuple("Polarity", "timestamp x y polarity")


class RadarFileParser:

    def __init__(self, filepath, read_radar=True, read_dvs=False):
        with open(filepath, 'rb') as file:
            tag, length, version, radar_type = self._read_and_parse(file, "<4sLHH")
            if tag != b'radr':
                raise Exception("No header found")

            # Found Radar file header
            self._num_tx_antennas, \
                self._num_rx_antennas, \
                self._num_chirps_per_frame, \
                self._num_samples_per_chirp, \
                flags = self._read_and_parse(file, "<LLLLL")
            self._with_complex_samples         = ((flags >> 0) & 1) == 1
            self._with_timestamps_per_chirp    = ((flags >> 1) & 1) == 1
            self._with_hw_timestamps_per_chirp = ((flags >> 2) & 1) == 1
            self._with_ant_config_per_chirp    = ((flags >> 3) & 1) == 1

            radar_frames = []
            dvs_tbins = []
            while True:
                tag, length = self._read_tlv_tag_and_length(file)
                if tag is None:
                    break
                if tag == b"rafd" and read_radar:
                    value = self._read_tlv_value(file, length)
                    f = self._parse_radar_frame(value)
                    radar_frames.append(f)
                elif tag == b"polb" and read_dvs:
                    value = self._read_tlv_value(file, length)
                    t = self._parse_polarities_bin(value)
                    dvs_tbins.append(t)
                else:
                    self._skip_tlv_value(file, length)
            self._radar_frames = radar_frames
            self._dvs_polarity_tbins = dvs_tbins

    @property
    def num_tx_antenns(self):
        return self._num_tx_antennas
        
    @property
    def num_rx_antennas(self):
        return self._num_rx_antennas
        
    @property
    def num_chirps_per_frame(self):
        return self._num_chirps_per_frame
        
    @property
    def num_samples_per_chirp(self):
        return self._num_samples_per_chirp

    @property
    def with_complex_samples(self):
        return self._with_complex_samples

    @property
    def radar_frames(self):
        return self._radar_frames

    @property
    def dvs_polarity_tbins(self):
        return self._dvs_polarity_tbins

    @staticmethod
    def _read_and_parse(file, format):
        n = struct.calcsize(format)
        bytes = file.read(n)
        return struct.unpack(format, bytes)

    @staticmethod
    def _read_tlv_tag_and_length(file):
        bytes = file.read(8)
        if len(bytes) < 8:
            return None, "EOF" if len(bytes) == 0 else "Incomplete header"
        tag, length = struct.unpack("<4sL", bytes)
        return tag, length

    @staticmethod
    def _read_tlv_value(file, length):
        value = file.read(length)
        if len(value) < length:
            return None, "Incomplete value"
        return value

    @staticmethod
    def _skip_tlv_value(file, length):
        file.seek(length, os.SEEK_CUR)

    @staticmethod
    def _extract_data(stream, dtype, count):
        buffer = stream.read(count * np.dtype(dtype).itemsize)
        return np.frombuffer(buffer, dtype=dtype, count=count)

    @staticmethod
    def _extract_uint32(stream, count):
        return RadarFileParser._extract_data(stream, np.uint32, count)

    @staticmethod
    def _extract_uint16(stream, count):
        return RadarFileParser._extract_data(stream, np.uint16, count)

    def _parse_radar_frame(self, value):
        stream = io.BytesIO(value)
        # Read timestamps
        n = self._num_chirps_per_frame if self._with_timestamps_per_chirp else 1
        ts_sec = self._extract_uint32(stream, n).astype(dtype=np.double)
        ts_nsec = self._extract_uint32(stream, n).astype(dtype=np.double)
        timestamps = ts_sec + ts_nsec / 1000000000
        # Read td_matrix
        m = 2 if self._with_complex_samples else 1 
        n = self._num_chirps_per_frame * self._num_tx_antennas * self._num_rx_antennas * self._num_samples_per_chirp
        td_matrix = self._extract_uint16(stream, n*m)
        if self._with_complex_samples:
            td_matrix = td_matrix[0::2] + 1j * td_matrix[1::2]
        if self._num_tx_antennas * self._num_rx_antennas == 1:
            td_matrix = td_matrix.reshape((self._num_chirps_per_frame, self._num_samples_per_chirp))
        else:
            td_matrix = td_matrix.reshape((self._num_chirps_per_frame, self._num_tx_antennas, self._num_rx_antennas, self._num_samples_per_chirp))
        # Read hw_timestamp_pri and hw_timestamp_dma
        if self._with_hw_timestamps_per_chirp:
            hw_timestamp_pri = self._extract_uint32(stream, self._num_chirps_per_frame)
            hw_timestamp_dma = self._extract_uint32(stream, self._num_chirps_per_frame)
        else:
            hw_timestamp_pri = np.empty((0,))
            hw_timestamp_dma = np.empty((0,))
        # Read ant_config_from_hw
        if self._with_ant_config_per_chirp:
            ant_config_from_hw = self._extract_uint16(stream, self._num_chirps_per_frame)
        else:
            ant_config_from_hw = np.empty((0,))
        # All together
        return Frame(td_matrix, timestamps, hw_timestamp_pri, hw_timestamp_dma, ant_config_from_hw)
        
    def _parse_polarities_bin(self, value):
        polarity_t = "<qHH?3x"  # int64_t, uint16_t, uint16_t, bool, 3 padding bytes
        m = struct.calcsize(polarity_t)
        tbin = [Polarity(*e) for e in struct.iter_unpack(polarity_t, value)]
        return tbin


if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "addons/ofxRadar/example_radarGui/recording.rad"

    start_time = time.time()
    for i in range(10):
        rf = RadarFileParser(input_file, read_dvs=False)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(len(rf.radar_frames))


