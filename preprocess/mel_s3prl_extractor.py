import os
import numpy as np
import soundfile as sf
from scipy.signal import lfilter
import librosa
from tqdm import tqdm


def log_mel_spectrogram(x, preemph=0.97, sample_rate=16000, n_mels=80,
                        n_fft=400, hop_length=320, win_length=400, f_min=0, center=False):
    """
    Create a log Mel spectrogram from a raw audio signal.
    """
    x = lfilter([1, -preemph], [1], x)
    magnitude = np.abs(
        librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False)
    )
    mel_fb = librosa.filters.mel(
        sample_rate, n_fft, n_mels=n_mels, fmin=f_min
    )
    mel_spec = np.dot(mel_fb, magnitude)
    log_mel_spec = np.log(mel_spec + 1e-9)

    return log_mel_spec.T


# audio file directory
rootDir = '/home/hongcz/alab/data/LJSpeech-1.1/wavs16'
# spectrogram directory
targetDir = '/home/hongcz/alab/feature/mel_s3prl'


filelist = os.listdir(rootDir)
for wavname in tqdm(sorted(filelist)):
    # Read audio file
    x, fs = sf.read(os.path.join(rootDir, wavname))
    # extract mel-spectrogram
    mel = log_mel_spectrogram(x)
    # save spect
    # if mel.shape[0] > 128:
    np.save(os.path.join(targetDir, wavname[:-4]), mel.astype(np.float32), allow_pickle=False)

