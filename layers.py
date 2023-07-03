from network import nn, torch

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

class Deconvolution(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1, padding="same", groups=1, bias=True):
        if padding == "same":
            padding = max(0, (kernel_size - 2) // 2)
        super(Deconvolution, self).__init__(in_channels, out_channels, kernel_size,
                                       stride=1, padding=padding, groups=groups,
                                       bias=bias, dilation=dilation)
    def forward(self, inputs):
        return super(Deconvolution, self).forward(inputs)


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
        self.kernel_size = kernel_size

    def forward(self, x):
        # x is a tensor
        return nn.MaxPool1d(self.kernel_size, stride=1)(x)


class LatentSpace_DNN_LocallyConnected_Dense(nn.Module):
    def __init__(self, units,
                 activation=None,
                 use_bias=True):
        super(LatentSpace_DNN_LocallyConnected_Dense, self).__init__()

        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.supports_masking = True

        self.conv_1 = nn.Conv1d(self.units, self.units, self.units, stride=1, padding='same', padding_mode='zeros')
        self.conv_2 = nn.Conv1d(self.units, self.units, self.units, stride=1, padding='same', padding_mode='zeros')

    def forward(self, x):
        return nn.Softplus()(self.conv_2(nn.Softplus()(self.conv_1(x))))


class TimeDistributed(nn.Module):
    def __init__(self, layer, batch_first=False, **kwargs):
        super(TimeDistributed, self).__init__()
        self.layer = layer.to('cuda')
        self.batch_first = batch_first
        self.kwargs = kwargs

    def forward(self, x):
        if self.kwargs['layer_name'] == 'Dense':
            in_features = self.kwargs['in_features']
            out_features = self.kwargs['out_features']
            activation = self.kwargs['activation']
            c_out = self.layer(x, in_features, out_features, activation).to('cuda')
            return c_out
        return None


class UpSampling1D(nn.Module):
    def __init__(self, size):
        super(UpSampling1D, self).__init__()
        self.size = size
        self.upsample = nn.Upsample(size=self.size, mode='linear')

    def forward(self, x):
        return self.upsample(x)
