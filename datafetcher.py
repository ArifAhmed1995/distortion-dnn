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
    def __init__(self, folder='Gitarre monophon'):
        self.folder = folder

        self.wavelist_dict = {}
        self.train_dict = {}
        self.test_dict = {}
        self.num_samples = 0

        self.train_tensor_list = []
        self.test_tensor_list = []

        self.tensor_train_path = os.path.join(
            'data' + '/' + folder, 'processed_train.hdf5')
        self.tensor_test_path = os.path.join(
            'data' + '/' + folder, 'processed_test.hdf5')

        self.device = 'cpu'

    def extract_data(self):
        guitar_monophone = os.path.join('data' + '/' + self.folder, 'Samples')
        directory_original = os.path.join(guitar_monophone, 'NoFX')
        directory_distortion = os.path.join(guitar_monophone, 'Distortion')

        # This works. The note and its corresponding distortions
        # are according to the indexing via glob.
        wavelist_original = glob.glob(
            os.path.join(directory_original, '*.wav'))
        wavelist_distortion = glob.glob(
            os.path.join(directory_distortion, '*.wav'))

        for index in range(len(wavelist_original)):
            self.wavelist_dict[index] = {
                'x': wavelist_original[index],
                'y': wavelist_distortion[index]
            }

        self.num_samples = len(self.wavelist_dict)

    def train_test_split(self):
        split_index = int(0.7 * self.num_samples)
        kv_pairs = [v for k, v in self.wavelist_dict.items()]
        random.shuffle(kv_pairs)

        self.train_list = kv_pairs[:split_index]
        self.test_list = kv_pairs[split_index:]

    def write_to_hdf5(self, data, subset_type):
        data_path = self.tensor_train_path if subset_type == "train" else self.tensor_test_path

        sr = 16000
        num_samples = len(data)

        if not os.path.exists(data_path):
            h5_file = h5py.File(data_path, 'w')
            h5_data = h5_file.create_dataset(subset_type, shape=(
                2*num_samples, 1 + 2*sr, 1), dtype=np.float32)

            for index, audio_pair in enumerate(data):
                h5_data[2*index] = torch.tensor(librosa.load(
                    audio_pair['x'], sr=16000)[0]).reshape(1 + 2*sr, 1)
                h5_data[1 + 2*index] = torch.tensor(librosa.load(
                    audio_pair['y'], sr=16000)[0]).reshape(1 + 2*sr, 1)

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
    x = DataFetcher()
    x.extract_data()
    x.train_test_split()
    x.write_hdf5_files()
    x.get_train_test_data()
