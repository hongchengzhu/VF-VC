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

# wav2vec_model = getattr(hub, 'wav2vec2')()
# wav2vec_model = wav2vec_model.to(device)


root = '/home/hongcz/alab/data/VCTK-Corpus/wav16'
bnf_source = '/home/hongcz/alab/feature/bnf_VCTK_filter_50'
content_target = '/home/hongcz/alab/feature/bnf_VCTK_filter_50'
mel_target = '/home/hongcz/alab/feature/mel_VCTK_bnf_50/'

if not os.path.exists(content_target):
    os.mkdir(content_target)

if not os.path.exists(mel_target):
    os.mkdir(mel_target)

global spklist
spklist = sorted(os.listdir(root))[:50]


def extract_feature(index):
        with torch.no_grad():
            spk = spklist[index]
            src = os.path.join(root, spk)
            bnf_src = os.path.join(bnf_source, spk)
            wav_list = sorted(os.listdir(src))
            for wavname in tqdm(wav_list):
                bnf = torch.FloatTensor(np.load(os.path.join(bnf_src, wavname[:-4]+'.npy')))
                # 1. content feature extract
                # wav = torch.tensor(wav, dtype=torch.float32).to(device).unsqueeze(0)
                # ret = wav2vec_model(wav)['last_hidden_state']
                # ret = ret.cpu()
                # content_tag = ret.shape[1] % 4

                # process bnf feature
                content_tag = bnf.shape[1] % 4

                if content_tag != 0:
                    ret_padding = torch.zeros([1, bnf.shape[1] + (4 - content_tag), 256])
                    ret_padding[:, :bnf.shape[1], :] = bnf
                else:
                    ret_padding = bnf

                # 2. mel feature extract
                wav = torch.FloatTensor(sf.read(os.path.join(src, wavname))[0]).unsqueeze(0)
                mel = mel_spectrogram(wav, 400, 80, 16000, 192, 400, 0, 8000, center=False)
                # save spect
                mel = mel.transpose(1, 2).squeeze(0).cpu().numpy().astype(np.float32)

                # mel = np.load(os.path.join(mel_target, spk, wavname[:-4]+'.npy'))
                mel = mel[:bnf.shape[1], :]

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

                content_to_name = os.path.join(content_tgt, wavname[:-4] + '.npy')
                # with open(content_to_name, 'wb') as f:
                #     pickle.dump(ret_padding, f)
                np.save(content_to_name, ret_padding.numpy().astype('float32'), allow_pickle=False)
                mel_to_name = os.path.join(mel_tgt, wavname[:-4])
                np.save(mel_to_name, mel_padding.astype('float32'), allow_pickle=False)
        print(1)


for i in range(0, 50, 4):
    t1 = threading.Thread(target=extract_feature, args=(i, ))
    t2 = threading.Thread(target=extract_feature, args=(i+1, ))
    t3 = threading.Thread(target=extract_feature, args=(i+2, ))
    t4 = threading.Thread(target=extract_feature, args=(i+3, ))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()


