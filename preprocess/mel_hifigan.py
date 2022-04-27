import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
import threading
import sys
sys.path.append('/home/hongcz/alab/code')
print(sys.path)
from hifi_gan_master.meldataset import mel_spectrogram

# audio file directory
rootDir = '/home/hongcz/alab/data/VCTK-Corpus/wav16'
# spectrogram directory
targetDir = '/home/hongcz/alab/feature/mel_hifigan_padding_alignment_VCTK/'
if not os.path.exists(targetDir):
    os.mkdir(targetDir)

global spklist
spklist = sorted(os.listdir(rootDir))


def mel_hifigan_extractor(index):
    spk = spklist[index]
    src = os.path.join(rootDir, spk)
    wav_list = os.listdir(src)
    # Read audio file
    for wavname in wav_list:
        x, fs = sf.read(os.path.join(src, wavname))
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
        tgt = os.path.join(targetDir, spk)
        if not os.path.exists(tgt):
            os.mkdir(tgt)
        np.save(os.path.join(tgt, wavname[:-4]), mel_padding.astype('float32'), allow_pickle=False)


for i in range(0, 109, 4):
    t1 = threading.Thread(target=mel_hifigan_extractor, args=(i, ))
    t2 = threading.Thread(target=mel_hifigan_extractor, args=(i+1, ))
    t3 = threading.Thread(target=mel_hifigan_extractor, args=(i+2, ))
    t4 = threading.Thread(target=mel_hifigan_extractor, args=(i+3, ))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
