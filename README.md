## Modeling non-linear audio effects with end-to-end deep neural networks  
  
Implements a Deep Neural Network detailed in the paper with the above title by Ramı́rez and Reiss.  

The details of the paper are in the docstring of `main.py`.

The network is expensive to train without a relatively powerful GPU. Colab Free GPU mode does train faster
than most common consumer GPUs, but Colab Pro might train much faster. The notebook has modified code to specifically run on a GPU.

Personally I have trained upto 100 epochs on Colab Free GPU mode, saving the model and optimizer states
regularly. Have witnessed a gradual decrease of `running_loss` so the model does seem to train and learn
the distortion audio.

The `running_loss` is indicative of the sum of  `mean absolute error` per `10` batches
after which it resets to zero.

Format: `[epoch, current_batch_index] loss: running_loss`

```
[1,   111] loss: 10.856
[1,   121] loss: 11.123
[1,   131] loss: 11.276
[1,   141] loss: 11.735
[1,   151] loss: 11.537
[1,   161] loss: 11.192
```
100 epochs later
```
[100,   331] loss: 9.823
[100,   321] loss: 10.130
[100,   351] loss: 10.900
[100,   361] loss: 10.199
[100,   371] loss: 9.910
[100,   381] loss: 10.255
```

The train and test data are meant to be in `HDF5` format. You can generate them by running `datafetcher.py`
after downloading the wav files from [this site](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/audio_effects.html). For some reason, it seems to be blocked in India, so please
use a VPN.
