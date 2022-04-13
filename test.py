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


# # inference
# val_mel = torch.tensor(np.load('./feature/mel128/LJ001-0001.npy')).to('cuda:0').unsqueeze(0)
# val_content = torch.tensor(np.load('./feature/mel128/p225_009_16k.npy')).to('cuda:0').unsqueeze(0)
# validate = validation()
# model = Generator()
# model.load_state_dict(torch.load('./log/model/m2m-non-parallel/VF-VC_VAE_99981.ckpt'))
# model.to('cuda:0')
# model.eval()
#
# val_loss = {}
# val_output = {}
# val_nonpadding = torch.ones([1, 1, 128]).to('cuda:0')
#
# val_loss, val_output = model(None, cond=val_mel[:, 128:256, :],
#                                         loss=val_loss, output=val_output,
#                                         nonpadding=val_nonpadding, infer=True)
# # self.val_output = output
# val_input = val_output['x_recon'].transpose(1, 2)
# vc_wav = validate.hifigan(val_input)
#
# name = '99981' + '.wav'
# sf.write(name, vc_wav, samplerate=16000)



#
# tgt_mel = torch.tensor(np.load('./feature/mel/LJ001-0001.npy')).to('cuda:0').unsqueeze(0)
# tgt_mel = tgt_mel[:, 128:256:, ]
#
# hifigan = validation()
# tgt_wav = hifigan.hifigan(tgt_mel.transpose(1, 2))
# sf.write('tgt.wav', tgt_wav, samplerate=16000)

path = "./wavs/LJ001-0001.wav"
path = './t.wav'
task = 'm2m-non-parallel'
# path = '/home/hongcz/桌面/individualAudio.wav'


# p = np.load(open('./spmel_autovc/p225/p225_003.npy', 'rb'))
# p = p.T
# plt.imshow(p.transpose(0,1))
# plt.show()

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