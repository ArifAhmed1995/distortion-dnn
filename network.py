import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.modules.activation import ReLU, Softplus

import layers

from utils import tensor_size

class DistortionNetwork(nn.Module):
    '''DNN to learn the distortion effect
    '''
    def __init__(self, window_length, filters, kernel_size, learning_rate):
        super(DistortionNetwork, self).__init__()
        self.window_length = window_length
        self.filters = filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate

        self.conv = nn.Conv1d(1, self.filters, self.kernel_size, stride=1, padding='same', padding_mode='zeros')
        self.convolution1d_locally_connected = layers.Convolution1D_Locally_Connected(self.filters, 2 * self.kernel_size)
        self.dense_layer = layers.LatentSpace_DNN_LocallyConnected_Dense(self.window_length//64, activation=nn.ReLU)
        self.deconvolution = layers.Deconvolution(1, self.kernel_size, self.conv, padding='same')

    def permute_dims(self, x):
        # pytorch initializes layer tensors in column major format.
        return x.permute(0, 2, 1)

    def Dense(self, x, in_features, out_features, activation=None):
        if activation is not None:
            return activation()(nn.Linear(in_features, out_features)(x))
        return nn.Linear(in_features, out_features)(x)

    def forward(self, x):
        x = self.permute_dims(x)
        x_conv = self.conv(x)
        x_abs = torch.abs(x_conv)

        M = self.convolution1d_locally_connected(x_abs)
        M = nn.Softplus()(M)
        M = M.permute(1, 0, 2, 3)

        P = x_conv
        Z = layers.BatchMaxPooling1d(self.window_length//64)(M)

        Z = layers.LatentSpace_DNN_LocallyConnected_Dense(self.window_length//64)(Z)
        Z = layers.TimeDistributed(layer=self.Dense, batch_first=True,
                            in_features=tensor_size(Z)[-1],
                            out_features=self.window_length//64,
                            activation=nn.Softplus, layer_name='Dense')(Z)

        M_ = layers.UpSampling1D(self.window_length)(Z)
        
        Y_ = layers.Multiply()([P, M_])

        Y_ = self.permute_dims(Y_)
        Y_ = self.Dense(Y_, tensor_size(Y_)[-1], self.filters, nn.ReLU)
        Y_ = self.Dense(Y_, tensor_size(Y_)[-1], self.filters//2, nn.ReLU)
        Y_ = self.Dense(Y_, tensor_size(Y_)[-1], self.filters//2, nn.ReLU)
        Y_ = self.Dense(Y_, tensor_size(Y_)[-1], self.filters//2, nn.ReLU)
        Y_ = self.Dense(Y_, tensor_size(Y_)[-1], self.filters)

        Y_ = layers.SAAF(break_points=25, break_range=0.2, magnitude=100)(Y_)
        output = layers.Deconvolution(self.filters, self.kernel_size, self.conv, padding='same')(Y_)

        return output
