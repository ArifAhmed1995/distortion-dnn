'''Martínez Ramírez M. A. and Reiss J. D.,

“Modeling nonlinear audio effects with end-to-end deep neural networks”

IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
Brighton, UK, May 2019.
'''

import os

from datafetcher import DataFetcher
from network import torch, nn, DistortionNetwork

from utils import DataGenerator

if __name__ == '__main__':
    # Parameters
    epochs = 2000
    div = 1
    window_length = 4096
    filters = 128
    kernel_size = 64
    learning_rate = 0.00005
    batch_size = 1

    # Get train, test data
    base_path = os.path.join(os.getcwd(), 'data', 'IDMT-SMT-AUDIO-EFFECTS', 'IDMT-SMT-AUDIO-EFFECTS', 'Gitarre_monophon', 'Samples')
    df = DataFetcher(nofx_path=os.path.join(base_path, 'NoFX'),
                fx_path=os.path.join(base_path, 'Distortion'))

    X_train, X_test, y_train, y_test = df.get_train_test_data()

    # Load into neural network
    distortion_network = DistortionNetwork(
        window_length, filters, kernel_size, learning_rate).to('cuda')

    # Loss function and Optimizer
    loss_function = nn.L1Loss()  # Mean Absolute Error
    optimizer = torch.optim.Adam([p for p in distortion_network.parameters() if p.requires_grad], lr=learning_rate, betas=(0.9, 0.99))

    # Data generator to yield batches to train and test data
    train_generator = DataGenerator(
        X_train, y_train, batch_size=batch_size, window_length=window_length)
    test_generator = DataGenerator(
        X_test, y_test, batch_size=batch_size, window_length=window_length)

    model_path = os.getcwd() + '/distortion_network_model_params.pt'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        distortion_network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_generator):
            input_audio, output_audio = data

            optimizer.zero_grad()

            network_output_audio = distortion_network(input_audio)
            mae_loss = loss_function(network_output_audio, output_audio)
            mae_loss.backward()
            optimizer.step()

            div = 5
            running_loss += mae_loss.item()
            if i != 0 and i % div == 0:
                print('[%d, %5d] loss on train data: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
        path = os.getcwd() + '/distortion_network_model_params.pt'

        torch.save({
            'model_state_dict': distortion_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)


    print("Finished training....")

    for i, data in enumerate(test_generator):
        input_audio, output_audio = data
        network_output_audio = distortion_network(input_audio)

        mae_loss = loss_function(network_output_audio, output_audio)

        running_loss += 1000 * mae_loss.item()

        if i != 0 and i % div == 0:
            print('[%d, %5d] loss on test data: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
