import os
import librosa

from scipy.io import wavfile
from network import torch, DistortionNetwork
from utils import WavSplitter

epochs = 2000
div = 1
window_length = 4096
filters = 128
kernel_size = 64
learning_rate = 0.00001
batch_size = 32

path = os.getcwd() + '/distortion_network_model_params.pt'

state_dict = torch.load(path)

# Load into neural network
distortion_network = DistortionNetwork(
    window_length, filters, kernel_size, learning_rate).to('cuda')

distortion_network.load_state_dict(state_dict['model_state_dict'])
distortion_network.eval()

file_code = "G61-42102-1111-20595.wav"
input_audio = os.path.join(os.getcwd(), "data/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre_monophon/Samples/NoFX/" + file_code)

input_tensor = torch.FloatTensor(librosa.load(input_audio, sr=44100)[0])

wavsplitter = WavSplitter(input_tensor, 1, 4096)
input_splits = wavsplitter.get_splits()

output_tensor = distortion_network.forward(input_splits)
output_tensor = output_tensor.detach().cpu().numpy()
shape = output_tensor.shape
output_tensor = output_tensor.reshape((shape[0] * shape[1], 1))

wavfile.write('file_trim_2s.wav', 88200, output_tensor)
