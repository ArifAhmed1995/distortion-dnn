from network import np, nn, functional, torch
from utils import tensor_size

import gc

from torch.nn.modules import activation, padding
from torch.autograd import Variable


class SAAF(nn.Module):
    def __init__(self, break_points, break_range, magnitude):
        super(SAAF, self).__init__()
        self.break_range = break_range
        self.magnitude = magnitude
        self.break_points = list(
            torch.linspace(-self.break_range, self.break_range, break_points, dtype=torch.float32))
        self.num_segs = int(len(self.break_points) / 2)

    def basisf(self, x, s, e):
        cp_start = torch.less_equal(s, x).float()
        cp_end = torch.greater(e, x).float()

        output = self.magnitude * (0.5 * (x - s)**2 * cp_start
                                   * cp_end + ((e - s) * (x - e) + 0.5 * (e - s)**2) * (1 - cp_end))

        return output.type(torch.FloatTensor)

    def forward(self, x):
        input_shape = list(x.size())
        self.kernel_dim = (self.num_segs + 1, input_shape[2])

        self.kernel = nn.Parameter(data=torch.zeros(
            self.kernel_dim,), requires_grad=True)

        output = torch.multiply(x, self.kernel[-1])

        for segment in range(0, self.num_segs):
            output += torch.multiply(self.basisf(
                x, self.break_points[segment * 2], self.break_points[segment * 2 + 1]), self.kernel[segment])

        return output


class Deconvolution(nn.Module):
    def __init__(self, filters, kernel_size, conv_layer,
                 strides=1,
                 padding='valid'):
        super(Deconvolution, self).__init__()

        self.device = torch.device("cpu")

        self.filters = filters
        self.strides = (strides,)
        self.padding = padding
        self.input_dim = None
        self.input_length = None

        self.kernel_size = kernel_size
        self.conv_layer = conv_layer

    def forward(self, x):
        gc.collect()
        input_dim = x.shape[-1]
        self.kernel_shape = (self.kernel_size, input_dim, self.filters)

        x = torch.unsqueeze(x, -1)
        x = x.permute(0, 2, 1, 3)

        W = torch.unsqueeze(self.conv_layer.weight, -1)

        W = W.permute(1, 0, 2, 3)

        conv2 = nn.Conv2d(self.filters, 1, self.kernel_size,
                          padding=self.padding, bias=True, padding_mode='zeros', dtype=None)
        conv2.weight = nn.Parameter(data=W, requires_grad=True)

        output = conv2(x)
        output = torch.squeeze(output, 3)

        return output.permute(0, 2, 1)


class Convolution1D_Locally_Connected(nn.Module):
    def __init__(self, filters, kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1):
        super(Convolution1D_Locally_Connected, self).__init__()

        self.device = torch.device("cpu")
        self.filters = filters
        self.strides = (strides,)
        self.padding = padding
        self.data_format = 'channels_last'
        self.dilation_rate = (dilation_rate,)
        self.activation = 'linear'
        self.input_dim = None
        self.input_length = None

        self.kernel_size = kernel_size
        self.kernel_shape = (filters, 1, kernel_size)

        self.kernel = nn.Parameter(data=torch.zeros(
            self.kernel_shape), requires_grad=True)
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, x):
        x = torch.split(x, 1, dim=1)
        W = torch.split(self.kernel, 1, dim=0)

        out_shape = [self.filters]
        out_shape.extend(list(x[0].shape))
        outputs = torch.zeros(out_shape)

        for i in range(self.filters):
            x_conv1 = nn.Conv1d(1, 1, self.kernel_size, stride=1, padding='same',
                                dilation=1, groups=1, bias=True, padding_mode='zeros', dtype=None)
            x_conv1.weight = nn.Parameter(data=W[i], requires_grad=True)
            x_conv1.bias = nn.Parameter(
                data=torch.zeros(1), requires_grad=True)
            outputs[i] = x_conv1(x[i])

        return outputs


class BatchMaxPooling1d(nn.Module):
    def __init__(self, kernel_size):
        super(BatchMaxPooling1d, self).__init__()

        self.device = torch.device("cpu")
        self.kernel_size = kernel_size

    def forward(self, x):
        # x is a tensor
        in_shape = list(x.shape)
        out_shape = in_shape[:-1]
        out_shape.extend([in_shape[-1]//self.kernel_size])

        output = torch.zeros(out_shape)
        for i in range(in_shape[0]):
            output[i] = nn.MaxPool1d(self.kernel_size)(torch.squeeze(x[i], 0))

        return output


class LatentSpace_DNN_LocallyConnected_Dense(nn.Module):
    def __init__(self, units,
                 activation=None,
                 use_bias=True):
        super(LatentSpace_DNN_LocallyConnected_Dense, self).__init__()

        self.device = torch.device("cpu")

        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.supports_masking = True

    def forward(self, x):
        input_shape = list(x.size())
        input_dim = input_shape[-1]  # 64
        self.split = input_shape[1]  # 128

        kernels = [nn.Parameter(data=torch.zeros(
            input_dim, self.units), requires_grad=True) for i in range(self.split)]
        self.kernel = torch.cat(kernels, -1)

        biases = [nn.Parameter(data=torch.zeros(
            self.units, ), requires_grad=True) for i in range(self.split)]
        self.bias = torch.cat(biases, -1)

        split_input = torch.split(x, 1, dim=1)
        W = torch.split(self.kernel, self.units, dim=1)
        b = torch.split(self.bias, self.units, dim=0)

        outputs = []
        for i, j in enumerate(split_input):
            output = torch.matmul(torch.squeeze(j), W[i]) + b[i]
            if self.activation is not None:
                output = self.activation(output)
            outputs.append(output)

        return_val = torch.cat(outputs, 1)
        return return_val.view(input_shape[0], self.split, input_dim)


class TimeDistributed(nn.Module):
    def __init__(self, layer, batch_first=False, **kwargs):
        super(TimeDistributed, self).__init__()
        self.layer = layer
        self.batch_first = batch_first
        self.kwargs = kwargs

    def forward(self, x):
        if self.kwargs['layer_name'] == 'Dense':
            in_features = self.kwargs['in_features']
            out_features = self.kwargs['out_features']
            activation = self.kwargs['activation']
            c_out = self.layer(x, in_features, out_features, activation)
            return c_out
        return None


class UpSampling1D(nn.Module):
    def __init__(self, size):
        super(UpSampling1D, self).__init__()
        self.size = size

    def forward(self, x):
        return nn.Upsample(size=self.size, mode='linear')(x)


class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, tensors):
        result = torch.ones(tensors[0].size())
        for t in tensors:
            result *= t
        return t
