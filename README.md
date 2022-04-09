# VF-VC: Many-to-Many Voice Conversion based on CVAE Augmented with Flow

## development proceeding
  4.8：parallel VC：cvae limit to 128


### Dependencies
- Python 3
- Numpy
- PyTorch >= v0.4.1
- TensorFlow >= v1.3 (only for tensorboard)
- librosa
- tqdm
- wavenet_vocoder ```pip install wavenet_vocoder```
  for more information, please refer to https://github.com/r9y9/wavenet_vocoder

### 0.Convert Mel-Spectrograms

Download pre-trained AUTOVC model, and run the ```conversion.ipynb``` in the same directory.

The fast and high-quality hifi-gan v1 (https://github.com/jik876/hifi-gan) pre-trained model is now available [here.](https://drive.google.com/file/d/1n76jHs8k1sDQ3Eh5ajXwdxuY_EZw4N9N/view?usp=sharing)


### 1.Mel-Spectrograms to waveform

Download pre-trained WaveNet Vocoder model, and run the ```vocoder.ipynb``` in the same the directory.

Please note the training metadata and testing metadata have different formats.


### 2.Train model

We have included a small set of training audio files in the wav folder. However, the data is very small and is for code verification purpose only. Please prepare your own dataset for training.

1.Generate spectrogram data from the wav files: ```python make_spect.py```

2.Generate training metadata, including the GE2E speaker embedding (please use one-hot embeddings if you are not doing zero-shot conversion): ```python make_metadata.py```

3.Run the main training script: ```python main.py```

Converges when the reconstruction loss is around 0.0001.



