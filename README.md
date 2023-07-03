## Modeling non-linear audio effects with end-to-end deep neural networks  
  
Implements a Deep Neural Network detailed in the paper with the above title by Ramı́rez and Reiss.  

The details of the paper are in the docstring of `main.py`.

The `running_loss` is indicative of the sum of `mean absolute error` per `10` batches
after which it resets to zero. The default learning rate is for the monotone case, where a mapping from nofx to distortion effect is learned.

The `master` branch has the architecture as implemented in the paper, but the training time per step is quite long. If some simplifications are done in the
network then train time is much faster. Personally I do not think the complicated DNN-SAAF layer is required and we get good results for the monotone case without it.

The train and test data are meant to be in `HDF5` format. You can generate them by running `datafetcher.py`
after downloading the wav files from [this site](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/audio_effects.html).
Put the files into a folder called `data` in the main directory.

The training code is in `main.py` and will use this HDF5 data.

The inference code is in `inference.py`. Checkpoint for the guitar monotones: [Monotone Checkpoint](https://drive.google.com/file/d/1EkhbNNBtTkQV0UoTvqYa2M5n9F87HtlP/view?usp=sharing)
