import os

import librosa
from matplotlib import pyplot as plt
import numpy as np
import librosa.display
import threading
import math
import soundfile as sf
import os


path = "/home/hongcz/alab/data/VCTK-Corpus/wav16"
target = '/home/hongcz/alab/data/VCTK-Corpus/wav16_filter'
if not os.path.exists(target):
    os.mkdir(target)

spk_list = sorted(os.listdir(path))


def func_format(x, pos):
    return "%d" % (1000 * x)


def spk_feature_filter(index):
    spk = spk_list[index]

    wavlist = os.listdir(os.path.join(path, spk))
    for wav in wavlist:
        wavpath = os.path.join(os.path.join(path, spk), wav)
        audio, sr = librosa.load(wavpath, sr=16000)

        # random_noise = np.random.uniform(-0.005,0.005,(len(audio)))
        # plt.figure(2)

        # audio_adv = audio + random_noise
        # amp_adv = np.abs(librosa.stft(audio_adv,n_fft=1024))

        # freq = librosa.stft(audio, n_fft=1024)
        freq = librosa.stft(audio, n_fft=400, hop_length=320, win_length=400, window='hann', center=False)
        amp = np.abs(freq)
        ang = np.angle(freq)

        total_fre = 8000
        n_stft = 201
        shengmen_down = math.floor(n_stft*50/total_fre)
        shengmen_up = math.ceil(n_stft*300/total_fre)
        liwo_down = math.floor(n_stft*4000/total_fre)
        liwo_up = math.ceil(n_stft*5500/total_fre)
        fuyin_down = math.floor(n_stft*6500/total_fre)
        fuyin_up = math.ceil(n_stft*7800/total_fre)
        max_dif = np.zeros(amp.shape[0])

        pitches, magnitudes = librosa.piptrack(S=amp, sr=16000, threshold=1, ref=np.mean, fmin=300, fmax=4000)
        ts = np.average(magnitudes[np.nonzero(magnitudes)])

        for j in range(amp.shape[0]):
            if j in range(shengmen_down,shengmen_up) or j in range(liwo_down, liwo_up) or j in range(fuyin_down, fuyin_up):
                # max_dif[j]=np.max(amp[j,:]-amp_adv[j,:])
                amp1 = amp[j, :]
                amp1[amp1 < ts] = 0
                amp[j, :] = amp1
        # print(max_dif)
        # print(ts)

        # data_tran = librosa.istft(amp*np.exp(ang))
        data_tran = librosa.istft(amp*np.exp(1j*ang), hop_length=320, win_length=400, n_fft=400, window='hann', center=False)

        if not os.path.exists(os.path.join(target, spk)):
            os.mkdir(os.path.join(target, spk))
        to_name = os.path.join(os.path.join(target, spk), wav)
        sf.write(to_name, data_tran, 16000)


# log_pow_db = librosa.amplitude_to_db(amp, ref=np.max)
# plt.figure(1)
# img1 = librosa.display.specshow(log_pow_db, sr=16000, x_axis="time", y_axis="linear")
# plt.title(11)
# plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(func_format))
# plt.colorbar(img1, format="%+2.f dB")
# plt.show()


if __name__ == '__main__':
    for i in range(0, 109, 8):
        t1 = threading.Thread(target=spk_feature_filter, args=(i,))
        t2 = threading.Thread(target=spk_feature_filter, args=(i + 1,))
        t3 = threading.Thread(target=spk_feature_filter, args=(i + 2,))
        t4 = threading.Thread(target=spk_feature_filter, args=(i + 3,))
        t5 = threading.Thread(target=spk_feature_filter, args=(i + 4,))
        t6 = threading.Thread(target=spk_feature_filter, args=(i + 5,))
        t7 = threading.Thread(target=spk_feature_filter, args=(i + 6,))
        t8 = threading.Thread(target=spk_feature_filter, args=(i + 7,))

        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()
        t7.start()
        t8.start()

        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
        t6.join()
        t7.join()
        t8.join()











