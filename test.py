import os

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

path = "./wavs/LJ001-0001.wav"
path = './t.wav'
# path = '/home/hongcz/桌面/individualAudio.wav'


# p = np.load(open('./spmel_autovc/p225/p225_003.npy', 'rb'))
# p = p.T
# plt.imshow(p.transpose(0,1))
# plt.show()

wavlist = os.listdir('./testwav')

# for wav in sorted(wavlist):
wav = '33200.wav'
path = os.path.join('./testwav', wav)
# sr=None声音保持原采样频率， mono=False声音保持原通道数
data, fs = librosa.load(path, sr=None, mono=False)

plt.specgram(data, NFFT=256, Fs=fs, window=np.hanning(256))
plt.title(str(wav))
plt.ylabel('Frequency')
plt.xlabel('Time(s)')

plt.show()