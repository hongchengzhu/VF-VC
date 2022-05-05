import os
import torch
import librosa.display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import os
from vc_validation import validation
import soundfile as sf
from model_vc import Generator
import sys
sys.path.append('/home/hongcz/alab/code')
print(sys.path)
from hifi_gan_master.meldataset import mel_spectrogram
import pickle


mel1 = np.load('/home/hongcz/alab/feature/mel_VCTK1/p227/p227_003.npy')
mel2 = np.load('/home/hongcz/alab/feature/mel_VCTK1/p228/p228_004.npy')

wav1 = sf.read('/home/hongcz/alab/data/VCTK-Corpus/wav16/p227/p227_003.wav')[0]
wav2 = sf.read('/home/hongcz/alab/data/VCTK-Corpus/wav16/p228/p228_004.wav')[0]

x1 = torch.FloatTensor(wav1).unsqueeze(0)
mel1_hat = mel_spectrogram(x1, 400, 80, 16000, 320, 400, 0, 8000, center=False)

x2 = torch.FloatTensor(wav2).unsqueeze(0)
mel2_hat = mel_spectrogram(x2, 400, 80, 16000, 320, 400, 0, 8000, center=False)

content1 = pickle.load(open('/home/hongcz/alab/feature/wav2vec2_VCTK1/p227/p227_003.pkl', 'rb'))
content2 = pickle.load(open('/home/hongcz/alab/feature/wav2vec2_VCTK1/p228/p228_004.pkl', 'rb'))


print(1)
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2)) # mel [80,T]
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.show()
    plt.savefig('x.png')
    plt.close()

    return fig


# from wav to mel
wav = sf.read('cmp/inference59989.wav')[0]
x = torch.FloatTensor(wav).unsqueeze(0)
# extract mel-spectrogram
mel = mel_spectrogram(x, 400, 80, 16000, 320, 400, 0, 8000, center=False)
# save spect
# if mel.shape[0] > 128:
mel = mel.squeeze(0).cpu().numpy().astype(np.float32)
plot_spectrogram(mel)


# x = np.load('./feature/mel_s3prl/LJ001-0001.npy')
# a = plot_spectrogram(x.T)
# print(1)


# re-wav
tgt_mel = torch.tensor(np.load('/home/hongcz/alab/feature/mel_s3prl/LJ050-0269.npy')).to('cuda:0').unsqueeze(0)
hifigan = validation()
tgt_wav = hifigan.hifigan(tgt_mel.transpose(1, 2))
sf.write('0269-1.wav', tgt_wav, samplerate=16000)


# tgt_wav, _ = sf.read('./wavs/LJ001-0001.wav')
plt.specgram(sf.read('inference.wav')[0], NFFT=320, Fs=16000, window=np.hanning(320))
plt.title('inference.wav')
plt.ylabel('Frequency')
plt.xlabel('Time(s)')
plt.show()
plt.savefig('inference.wav.png')

# path = "./wavs/LJ001-0001.wav"
# path = './t.wav'
# task = 'm2m-non-parallel'
# path = '/home/hongcz/桌面/individualAudio.wav'



wavlist = os.listdir(os.path.join('./testwav', task))

# for wav in sorted(wavlist):
wav = '99800.wav'
path = os.path.join(os.path.join('./testwav', task), wav)
# sr=None声音保持原采样频率， mono=False声音保持原通道数
data, fs = librosa.load(path, sr=None, mono=False)
data1, fs1 = librosa.load('tgt.wav', sr=None, mono=False)

plt.subplot(1, 2, 1)
plt.specgram(data, NFFT=256, Fs=fs, window=np.hanning(256))
plt.title(str(wav))
plt.ylabel('Frequency')
plt.xlabel('Time(s)')

plt.subplot(1, 2, 2)
plt.specgram(data1, NFFT=256, Fs=fs, window=np.hanning(256))
plt.title('tgt.wav')
plt.ylabel('Frequency')
plt.xlabel('Time(s)')

plt.show()
plt.savefig('c.png')