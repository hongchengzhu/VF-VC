# import upstreaming task model
import os
import torch
import pickle
import s3prl.hub as hub
import soundfile as sf
from tqdm import tqdm
import threading


device = 'cuda:0'

wav2vec_model = getattr(hub, 'wav2vec2')()
wav2vec_model = wav2vec_model.to(device)


root = '/home/hongcz/alab/data/VCTK-Corpus/wav16'
target = '/home/hongcz/alab/feature/wav2vec2_padding_VCTK/'
if not os.path.exists(target):
    os.mkdir(target)

global spklist
spklist = sorted(os.listdir(root))


def extract_wav2vec2(index):
    with torch.no_grad():
        spk = spklist[index]
        src = os.path.join(root, spk)
        wav_list = os.listdir(src)
        for wavname in tqdm(wav_list):
            wav, _ = sf.read(os.path.join(src, wavname))
            wav = torch.tensor(wav, dtype=torch.float32).to(device).unsqueeze(0)
            ret = wav2vec_model(wav)['last_hidden_state']
            ret = ret.cpu()
            tag = ret.shape[1] % 4
            if tag != 0:
                ret_padding = torch.zeros([1, ret.shape[1] + (4 - tag), 768])
                ret_padding[:, :ret.shape[1], :] = ret
            else:
                ret_padding = ret
            tgt = os.path.join(target, spk)
            if not os.path.exists(tgt):
                os.mkdir(tgt)
            to_name = os.path.join(tgt, wavname[:-4] + '.pkl')
            with open(to_name, 'wb') as f:
                pickle.dump(ret_padding, f)
        print(1)


for i in range(0, 109, 4):
    t1 = threading.Thread(target=extract_wav2vec2, args=(i, ))
    t2 = threading.Thread(target=extract_wav2vec2, args=(i+1, ))
    t3 = threading.Thread(target=extract_wav2vec2, args=(i+2, ))
    t4 = threading.Thread(target=extract_wav2vec2, args=(i+3, ))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
