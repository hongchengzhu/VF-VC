import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
import sys
sys.path.append('/home/hongcz/alab/code')
print(sys.path)
from hifi_gan_master.meldataset import mel_spectrogram

# audio file directory
rootDir = '/home/hongcz/alab/data/LJSpeech-1.1/wavs16'
# spectrogram directory
targetDir = '/home/hongcz/alab/feature/mel_hifigan_padding'


filelist = os.listdir(rootDir)
for wavname in tqdm(sorted(filelist)):
    # Read audio file
    x, fs = sf.read(os.path.join(rootDir, wavname))
    x = torch.FloatTensor(x).unsqueeze(0)
    # extract mel-spectrogram
    mel = mel_spectrogram(x, 400, 80, 16000, 320, 400, 0, 8000, center=False)
    # save spect
    # if mel.shape[0] > 128:
    mel = mel.transpose(1, 2).squeeze(0).cpu().numpy().astype(np.float32)
    tag = mel.shape[0] % 4
    if tag != 0:
        mel_padding = np.zeros([mel.shape[0] + (4 - tag), 80])
        mel_padding[:mel.shape[0], :] = mel
    else:
        mel_padding = mel
    np.save(os.path.join(targetDir, wavname[:-4]), mel_padding.astype('float32'), allow_pickle=False)

