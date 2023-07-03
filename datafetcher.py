import os
import sys
import glob
import random

import h5py

import librosa
from librosa.core import audio

import numpy as np

import torch
from torch import FloatTensor
from torch.utils.data import Dataset


def tensor_size(x):
    return list(x.size())

class DataFetcher:
    def __init__(self, nofx_path, fx_path):
        self.nofx_path = nofx_path
        self.fx_path = fx_path

        self.wavelist_dict = {}
        self.train_dict = {}
        self.test_dict = {}
        self.num_samples = 0

        self.train_tensor_list = []
        self.test_tensor_list = []

        self.tensor_train_path = os.path.join(os.getcwd(), 'processed_train.hdf5')
        self.tensor_test_path = os.path.join(os.getcwd(), 'processed_test.hdf5')
        self.device = 'cuda'

    def extract_data(self):
        wavelist_original = glob.glob(
            os.path.join(self.nofx_path, '*.wav'))
        wavelist_distortion = glob.glob(
            os.path.join(self.fx_path, '*.wav'))
        
        #wavelist_distortion.sort()
        #wavelist_original.sort()

        for index in range(len(wavelist_original)):
            prefix = wavelist_original[index].split('/')[-1][:9] + '-4411'

            j_index = -1
            for i, wavfile in enumerate(wavelist_distortion):
                if prefix in wavfile:
                    j_index = i

            self.wavelist_dict[index] = {
                'x': wavelist_original[index],
                'y': wavelist_distortion[j_index]
            }

        self.num_samples = len(self.wavelist_dict)

    def train_test_split(self):
        split_index = int(0.85 * self.num_samples)
        kv_pairs = [v for k, v in self.wavelist_dict.items()]
        #random.shuffle(kv_pairs)

        self.train_list = kv_pairs[:split_index]
        self.test_list = kv_pairs[split_index:]

    def write_to_hdf5(self, data, subset_type):
        data_path = self.tensor_train_path if subset_type == "train" else self.tensor_test_path

        sr = 44100
        num_samples = len(data)

        #if not os.path.exists(data_path):
        if os.path.exists(data_path):
            h5_file = h5py.File(data_path, 'w')
            h5_data = h5_file.create_dataset(subset_type, shape=(
                2*num_samples, 1 + 2*sr, 1), dtype=np.float32)

            for index, audio_pair in enumerate(data):
                y11 = librosa.load(
                    audio_pair['x'], sr=sr)[0]
                h5_data[2*index] = torch.FloatTensor(librosa.load(
                    audio_pair['x'], sr=sr)[0]).reshape(1 + 2*sr, 1)
                h5_data[1 + 2*index] = torch.FloatTensor(librosa.load(
                    audio_pair['y'], sr=sr)[0]).reshape(1 + 2*sr, 1)

            h5_file.close()
        else:
            print("HDF5 file with path", data_path, "exists.")

    def get_train_data(self):
        hf = h5py.File(self.tensor_train_path, 'r')
        return hf['train']

    def get_test_data(self):
        hf = h5py.File(self.tensor_test_path, 'r')
        return hf['test']

    def write_hdf5_files(self):
        self.write_to_hdf5(self.train_list, "train")
        self.write_to_hdf5(self.test_list, "test")

    def get_train_test_data(self):
        train_data = self.get_train_data()
        test_data = self.get_test_data()

        X_train = FloatTensor([train_data[i]
                              for i in range(0, len(train_data), 2)])
        y_train = FloatTensor([train_data[i]
                              for i in range(1, len(train_data), 2)])

        X_test = FloatTensor([test_data[i]
                             for i in range(0, len(test_data), 2)])
        y_test = FloatTensor([test_data[i]
                             for i in range(1, len(test_data), 2)])
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    base_path = os.path.join(os.getcwd(), 'data', 'IDMT-SMT-AUDIO-EFFECTS', 'IDMT-SMT-AUDIO-EFFECTS', 'Gitarre_monophon', 'Samples')
    x = DataFetcher(nofx_path=os.path.join(base_path, 'NoFX'),
                fx_path=os.path.join(base_path, 'Distortion'))
    x.extract_data()
    x.train_test_split()
    x.write_hdf5_files()
    #x.get_train_test_data()
