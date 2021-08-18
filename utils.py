from network import np, torch, functional

from datafetcher import librosa


def tensor_size(x):
    return list(x.size())


class DataGenerator():
    def __init__(self, x, y, batch_size=16, window_length=4096):
        self.x, self.y = x, y
        self.window_length = window_length  # 4096
        self.hop_length = window_length//2  # 2048
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x)

    def window_time_series(self, x, frame_length, hop_length):
        '''Window time series, overlapping'''
        blocks = []
        n = list(x.size())[0]
        ilo = range(0, n, hop_length)
        ihi = range(frame_length, n+1, hop_length)
        ilohi = zip(ilo, ihi)
        blocks = [x[ilo:ihi] for ilo, ihi in ilohi]
        return torch.stack(blocks, 0)

    def slicing(self, x):
        # Padding time series to ensure frames are centered
        x = functional.pad(
            x, (self.window_length//2, self.window_length//2), mode='constant')

        # Window the time series
        return self.window_time_series(x, self.window_length, self.hop_length)

    def ts(self, x):
        return list(x.size())

    def __getitem__(self, idx):
        batch_x = torch.zeros((self.batch_size, self.window_length, 1))
        batch_y = torch.zeros((self.batch_size, self.window_length, 1))

        x_w = self.x[idx].reshape(len(self.x[idx]))
        y_w = self.y[idx].reshape(len(self.y[idx]))

        x_w = self.slicing(x_w)
        y_w = self.slicing(y_w)

        for i in range(self.batch_size):
            batch_x[i] = x_w[i].reshape(self.window_length, 1)
            batch_y[i] = y_w[i].reshape(self.window_length, 1)

        return batch_x, batch_y
