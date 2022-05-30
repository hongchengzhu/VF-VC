import os
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import librosa
import math
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import threading
import umap
import pickle


# source11 = '../id_test/spk_emb/ge2e_p225_002.txt'
# source12 = '../id_test/spk_emb/ge2e_p226_002.txt'
# source13 = '../id_test/spk_emb/ge2e_p227_002.txt'
# source21 = '../id_test/spk_emb/p225_filter.txt'
# source22 = '../id_test/spk_emb/p226_filter.txt'
# source23 = '../id_test/spk_emb/p227_filter.txt'
# wav_src = '../id_test/wav/p227/p227_002.wav'
#
# # ge2e
# spk_emb_ge2e11 = np.loadtxt(source11, delimiter=',')
# spk_emb_ge2e12 = np.loadtxt(source12, delimiter=',')
# spk_emb_ge2e13 = np.loadtxt(source13, delimiter=',')

# spk = []
# spk.append(spk_emb_ge2e11)
# spk.append(spk_emb_ge2e12)
# spk.append(spk_emb_ge2e13)

########################################## t-SNE ##########################################################
# spk_emb = []
# spk_emb = np.zeros([1000, 201])
# color = np.zeros([200])
# spk_emb = np.loadtxt('/home/hongcz/alab/code/VF-VC-5-1/id_test/spk_emb/ge2e_infer_postflow.txt', delimiter=',')
# t = 0.1
# for i in range(0, 200, 1):
#     # spk_emb[i, :] = spk_emb_filter[201*i:201*(i+1)]
#     color[i] = t
#     if i != 0 and i % 100 == 0:
#         t = t + 0.5
# #     spk_emb.append(spk_emb_filter[201*i:201*(i+1), :])
#
# tsne = TSNE(n_components=2)
# result = tsne.fit_transform(spk_emb)
# x_min, x_max = result.min(0), result.max(0)
# x_norm = (result - x_min) / (x_max - x_min)
# fig = plt.figure()
# # colors = cm.rainbow(np.linspace(0, 1, 10))
# colors = cm.rainbow(color)
# # ax = fig.add_subplot(111, projection='3d')
# for i in range(len(result)):
#     plt.text(x_norm[i, 0], x_norm[i, 1], '.', color=colors[i], fontdict={'weight': 'bold', 'size': '29'})
# # ax.scatter(x_norm[:, 0], x_norm[:, 1], x_norm[:, 2], c=colors)
# # ax.view_init(4, -72)
# # plt.axis('off')
# # plt.show()
# #
# plt.xticks([])
# plt.yticks([])
# plt.title('test')
# plt.show()

# spk_emb_filter21 = np.loadtxt(source21, delimiter=',')
# spk_emb_filter22 = np.loadtxt(source22, delimiter=',')
# spk_emb_filter23 = np.loadtxt(source23, delimiter=',')


########################################################## U-map #######################################################
spk_emb = np.zeros([1000, 768])
source = '/home/hongcz/alab/feature/wav2vec2_VCTK_filter'
spklist = sorted(os.listdir(source))[:10]
index = 0
for spk in spklist:
    wavlist = os.listdir(os.path.join(source, spk))[:100]
    for wav in wavlist:
        spk_emb[index, :] = torch.mean(pickle.load(open(os.path.join(source, spk, wav), 'rb')), axis=1).squeeze(0)
        index += 1
# wavlist = sorted(os.listdir(source))
# for i in range(len(wavlist)):
#     spk_emb[i, :] = np.loadtxt(os.path.join(source, wavlist[i]), delimiter=',')
# spk_emb = np.loadtxt('/home/hongcz/alab/code/VF-VC-5-1/id_test/spk_emb/filter.txt', delimiter=',')
# spk_emb_1 = np.zeros([100, 201])
# target = np.zeros([100])
# for i in range(100):
#     spk_emb_1[i, :] = spk_emb[201*i: 201*(i+1)]
#     target[i] = (i//10)

target = np.zeros([1000])
for i in range(1000):
    target[i] = i // 100
    # target[i] = int(wavlist[i][-7:-4]) - 225

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(spk_emb)
print(embedding.shape)

# embedding = (embedding - embedding.min(0)) / (embedding.max(0) - embedding.min(0))

plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('wac2vec2_VCTK_emb')
plt.show()

print(1)

# dif121 = spk_emb_ge2e12 - spk_emb_ge2e11
# dif132 = spk_emb_ge2e13 - spk_emb_ge2e12
# dif113 = spk_emb_ge2e11 - spk_emb_ge2e13
#
# dif = []
# dif.append(dif121)
# dif.append(dif132)
# dif.append(dif113)
# dif = np.array(dif)
# np.savetxt('../id_test/spk_emb/ge2e_3p.txt', dif, fmt='%f', delimiter=',')
# print(1)
# dif221 = spk_emb_filter22 - spk_emb_filter21
# dif232 = spk_emb_filter23 - spk_emb_filter22
# dif213 = spk_emb_filter21 - spk_emb_filter23

# dif311 = np.mean(dif221) - np.mean(dif121)
# dif322 = np.mean(dif232) - np.mean(dif132)
# dif333 = np.mean(dif213) - np.mean(dif113)


# print(1)

# # infer filter
# wavlist = sorted(os.listdir('/home/hongcz/alab/code/VF-VC-5-1/id_test/id_infer/ge2e'))
#
# target = '/home/hongcz/alab/code/VF-VC-5-1/id_test/id_infer/ge2e_filter_emb'
# if not os.path.exists(target):
#     os.mkdir(target)
#
# for wav in wavlist:
#     wav_src = os.path.join('/home/hongcz/alab/code/VF-VC-5-1/id_test/id_infer/ge2e', wav)
#     audio, sr = librosa.load(wav_src, sr=16000)
#     freq = librosa.stft(audio, n_fft=400, hop_length=192, win_length=400, window='hann', center=False)
#     amp = np.abs(freq)
#     ang = np.angle(freq)
#
#     total_fre = 8000
#     n_stft = 201
#     shengmen_down = math.floor(n_stft * 50 / total_fre)
#     shengmen_up = math.ceil(n_stft * 300 / total_fre)
#     liwo_down = math.floor(n_stft * 4000 / total_fre)
#     liwo_up = math.ceil(n_stft * 5500 / total_fre)
#     fuyin_down = math.floor(n_stft * 6500 / total_fre)
#     fuyin_up = math.ceil(n_stft * 7800 / total_fre)
#     max_dif = np.zeros(amp.shape[0])
#
#     pitches, magnitudes = librosa.piptrack(S=amp, sr=16000, threshold=1, ref=np.mean, fmin=300, fmax=4000)
#     ts = np.average(magnitudes[np.nonzero(magnitudes)])
#
#     tmp = np.zeros([amp.shape[0], amp.shape[1]])
#
#     for k in range(amp.shape[0]):
#         if k in range(shengmen_down, shengmen_up) or k in range(liwo_down, liwo_up) or k in range(fuyin_down, fuyin_up):
#             amp1 = amp[k, :]
#             # amp1[amp1 < ts] = 0
#             # amp[j, :] = amp1
#             for m in range(len(amp1)):
#                 if amp1[m] < ts:
#                     tmp[k, m] = amp1[m]
#
#     with open(os.path.join(os.path.join(target, wav[:-4])+'.txt'), 'ab') as f:
#         np.savetxt(f, np.mean(tmp, axis=1), fmt='%f', delimiter=',')
#
# print(1)





# extract spk_id_filter
# spk_id_filter
spklist = sorted(os.listdir('/home/hongcz/alab/data/VCTK-Corpus/wav16'))

target = '/home/hongcz/alab/feature/spk_emb_filter'
if not os.path.exists(target):
    os.mkdir(target)


def spk_emb_filter(index):
    result = np.zeros([100, 201])
    spk = spklist[index]
    wavlist = sorted(os.listdir(os.path.join('/home/hongcz/alab/data/VCTK-Corpus/wav16', spk)))
    for j in tqdm(range(100)):
        wav_src = os.path.join('/home/hongcz/alab/data/VCTK-Corpus/wav16', spk, wavlist[j])
        audio, sr = librosa.load(wav_src, sr=16000)
        freq = librosa.stft(audio, n_fft=400, hop_length=192, win_length=400, window='hann', center=False)
        amp = np.abs(freq)
        ang = np.angle(freq)

        total_fre = 8000
        n_stft = 201
        shengmen_down = math.floor(n_stft * 50 / total_fre)
        shengmen_up = math.ceil(n_stft * 300 / total_fre)
        liwo_down = math.floor(n_stft * 4000 / total_fre)
        liwo_up = math.ceil(n_stft * 5500 / total_fre)
        fuyin_down = math.floor(n_stft * 6500 / total_fre)
        fuyin_up = math.ceil(n_stft * 7800 / total_fre)
        max_dif = np.zeros(amp.shape[0])

        pitches, magnitudes = librosa.piptrack(S=amp, sr=16000, threshold=1, ref=np.mean, fmin=300, fmax=4000)
        ts = np.average(magnitudes[np.nonzero(magnitudes)])

        tmp = np.zeros([amp.shape[0], amp.shape[1]])

        for k in range(amp.shape[0]):
            if k in range(shengmen_down, shengmen_up) or k in range(liwo_down, liwo_up) or k in range(fuyin_down, fuyin_up):
                amp1 = amp[k, :]
                # amp1[amp1 < ts] = 0
                # amp[j, :] = amp1
                for m in range(len(amp1)):
                    if amp1[m] < ts:
                        tmp[k, m] = amp1[m]
        result[j, :] = np.mean(tmp, axis=1)
    result = (result - result.min()) / (result.max() - result.min())

    result = np.mean(result, axis=0)

    tgt = os.path.join(target, spk)
    if not os.path.exists(tgt):
        os.mkdir(tgt)

    with open(os.path.join(tgt, spk+'.txt'), 'ab') as f:
        np.savetxt(f, result, fmt='%f', delimiter=',')
        # result.append(torch.FloatTensor(tmp).T)
# result = pad_sequence(result).transpose(0, 1).transpose(1, 2)


# emb = np.zeros([1000, 201])
# for i in range(10):
#     emb[100*i:100*(i+1)] = spk_emb_filter(i)
# print(1)

for i in range(0, 109, 4):
    t1 = threading.Thread(target=spk_emb_filter, args=(i, ))
    t2 = threading.Thread(target=spk_emb_filter, args=(i+1, ))
    t3 = threading.Thread(target=spk_emb_filter, args=(i+2, ))
    t4 = threading.Thread(target=spk_emb_filter, args=(i+3, ))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
