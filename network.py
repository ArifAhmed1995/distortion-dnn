import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.modules.activation import ReLU, Softplus

import layers

class DistortionNetwork(nn.Module):
    '''DNN to learn the distortion effect
    '''
    def __init__(self, window_length, filters, kernel_size, learning_rate):
        super(DistortionNetwork, self).__init__()
        self.window_length = window_length
        self.filters = filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate

        self.conv = nn.Conv1d(1, self.filters, self.kernel_size, stride=1, padding='same', padding_mode='zeros').to('cuda')
        self.conv_1 = nn.Conv1d(self.filters, self.filters, 2*self.kernel_size, stride=1, padding='same', padding_mode='zeros').to('cuda')
        
        self.dense_layer = layers.LatentSpace_DNN_LocallyConnected_Dense(self.window_length//64, activation=nn.ReLU).to('cuda')
        self.deconvolution = layers.Deconvolution(128, 1, 32).to('cuda')
        self.maxpooling = layers.BatchMaxPooling1d(16).to('cuda')
        self.dnnlayer = layers.LatentSpace_DNN_LocallyConnected_Dense(128).to('cuda')
        self.timelayer = nn.Linear(4081, 64).to('cuda')
        self.upsampling = layers.UpSampling1D(4096).to('cuda')

        self.linear_1 = nn.Linear(128, 128).to('cuda')
        self.linear_2 = nn.Linear(128, 64).to('cuda')
        self.linear_3 = nn.Linear(64, 64).to('cuda')
        self.linear_4 = nn.Linear(64, 64).to('cuda')
        self.linear_5 = nn.Linear(64, 128).to('cuda')

        self.final_linear = nn.Linear(4097, 4096).to('cuda')

    def permute_dims(self, x):
        # pytorch initializes layer tensors in column major format.
        return x.permute(0, 2, 1)

    def Dense(self, x, in_features, out_features, activation=None):
        if activation is not None:
            return activation()(nn.Linear(in_features, out_features)(x))
        return nn.Linear(in_features, out_features)(x)

    def forward(self, x):
        x = self.permute_dims(x)
        x_abs = self.conv(x)

        M = self.conv_1(x_abs)
        M = nn.Softplus()(M)

        P = x_abs
        Z = self.maxpooling(M)

        Z = self.dnnlayer(Z)
        Z = self.timelayer(Z)
        Z = nn.Softplus()(Z)

        M_ = self.upsampling(Z)
        
        Y_ = torch.mul(P, M_)

        Y_ = self.permute_dims(Y_)

        Y_ = self.linear_1(Y_)
        Y_ = nn.ReLU()(Y_)
        Y_ = self.linear_2(Y_)
        Y_ = nn.ReLU()(Y_)
        Y_ = self.linear_3(Y_)
        Y_ = nn.ReLU()(Y_)
        Y_ = self.linear_4(Y_)
        Y_ = nn.ReLU()(Y_)
        Y_ = self.linear_5(Y_)

        Y_ = Y_.permute(0, 2, 1)
        output = self.deconvolution(Y_)
        output = self.final_linear(output)
        output = output.permute(0, 2, 1)

        return output
