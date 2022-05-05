# import upstreaming task model
import os
import torch
import pickle
import numpy as np
import s3prl.hub as hub
import soundfile as sf
from tqdm import tqdm
import threading
import sys
sys.path.append('/home/hongcz/alab/code')
print(sys.path)
from hifi_gan_master.meldataset import mel_spectrogram


device = 'cuda:0'

wav2vec_model = getattr(hub, 'wav2vec2')()
wav2vec_model = wav2vec_model.to(device)


root = '/home/hongcz/alab/data/VCTK-Corpus/wav16'
content_target = '/home/hongcz/alab/feature/wav2vec2_VCTK1/'
mel_target = '/home/hongcz/alab/feature/mel_VCTK1/'

if not os.path.exists(content_target):
    os.mkdir(content_target)

if not os.path.exists(mel_target):
    os.mkdir(mel_target)

global spklist
spklist = sorted(os.listdir(root))


def extract_feature(index):
        with torch.no_grad():
            spk = spklist[index]
            src = os.path.join(root, spk)
            wav_list = sorted(os.listdir(src))
            for wavname in tqdm(wav_list):
                wav, _ = sf.read(os.path.join(src, wavname))
                # 1. content feature extract
                wav = torch.tensor(wav, dtype=torch.float32).to(device).unsqueeze(0)
                ret = wav2vec_model(wav)['last_hidden_state']
                ret = ret.cpu()
                content_tag = ret.shape[1] % 4

                if content_tag != 0:
                    ret_padding = torch.zeros([1, ret.shape[1] + (4 - content_tag), 768])
                    ret_padding[:, :ret.shape[1], :] = ret
                else:
                    ret_padding = ret

                # 2. mel feature extract
                mel = mel_spectrogram(wav, 400, 80, 16000, 320, 400, 0, 8000, center=False)
                # save spect
                mel = mel.transpose(1, 2).squeeze(0).cpu().numpy().astype(np.float32)
                mel_tag = mel.shape[0] % 4
                if mel_tag != 0:
                    mel_padding = np.zeros([mel.shape[0] + (4 - mel_tag), 80])
                    mel_padding[:mel.shape[0], :] = mel
                else:
                    mel_padding = mel

                # 3. forced alignment
                if ret_padding.shape[1] > mel_padding.shape[0]:
                    ret_padding = ret_padding[:, :mel_padding.shape[0], :]
                elif ret_padding.shape[1] < mel_padding.shape[0]:
                    mel_padding = mel_padding[:ret_padding.shape[1], :]

                content_tgt = os.path.join(content_target, spk)
                mel_tgt = os.path.join(mel_target, spk)
                if not os.path.exists(content_tgt):
                    os.mkdir(content_tgt)
                if not os.path.exists(mel_tgt):
                    os.mkdir(mel_tgt)

                content_to_name = os.path.join(content_tgt, wavname[:-4] + '.pkl')
                with open(content_to_name, 'wb') as f:
                    pickle.dump(ret_padding, f)
                mel_to_name = os.path.join(mel_tgt, wavname[:-4])
                np.save(mel_to_name, mel_padding.astype('float32'), allow_pickle=False)
        print(1)


for i in range(0, 109, 8):
    t1 = threading.Thread(target=extract_feature, args=(i, ))
    t2 = threading.Thread(target=extract_feature, args=(i+1, ))
    t3 = threading.Thread(target=extract_feature, args=(i+2, ))
    t4 = threading.Thread(target=extract_feature, args=(i+3, ))
    t5 = threading.Thread(target=extract_feature, args=(i+4, ))
    t6 = threading.Thread(target=extract_feature, args=(i+5, ))
    t7 = threading.Thread(target=extract_feature, args=(i+6, ))
    t8 = threading.Thread(target=extract_feature, args=(i+7, ))

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

