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


# re-wav
tgt_mel = torch.tensor(np.load('/home/hongcz/alab/feature/mel_s3prl/LJ001-0001.npy')).to('cuda:0').unsqueeze(0)
hifigan = validation()
tgt_wav = hifigan.hifigan(tgt_mel.transpose(1, 2))
sf.write('tgt_s3prl_320.wav', tgt_wav, samplerate=16000)
# tgt_wav, _ = sf.read('./wavs/LJ001-0001.wav')
plt.specgram(tgt_wav, NFFT=320, Fs=16000, window=np.hanning(320))
plt.title('tgt_s3prl_320.wav')
plt.ylabel('Frequency')
plt.xlabel('Time(s)')
plt.show()
plt.savefig('tgt_s3prl_320.wav.png')

# path = "./wavs/LJ001-0001.wav"
# path = './t.wav'
# task = 'm2m-non-parallel'
# path = '/home/hongcz/桌面/individualAudio.wav'

# show mel
p = np.load('/home/hongcz/alab/feature/mel128/LJ001-0001.npy')
p = p.T
plt.imshow(p.transpose(0, 1))
plt.show()
plt.savefig('tgt_mel128.wav.png')



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