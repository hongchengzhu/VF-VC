"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
import threading

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('../checkpoint/speaker_encoder/3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDir = '/home/hongcz/alab/feature/mel_hifigan_padding_alignment_VCTK'
targetDir = '/home/hongcz/alab/feature/speaker_embedding_VCTK'
if not os.path.exists(targetDir):
    os.mkdir(targetDir)
dirName, subdirList, _ = next(os.walk(rootDir))
sorted(dirName)
sorted(subdirList)
print('Found directory: %s' % dirName)


def spk_embedding(index):
    speaker = subdirList[index]
    print('Processing speaker: %s' % speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))
    
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        while tmp.shape[0] <= len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        left = np.random.randint(0, tmp.shape[0]-len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())     

    tgt = os.path.join(targetDir, speaker)
    if not os.path.exists(tgt):
        os.mkdir(tgt)
    np.save(os.path.join(tgt, speaker), np.mean(embs, axis=0).astype('float32'), allow_pickle=False)


for i in range(0, 109, 4):
    t1 = threading.Thread(target=spk_embedding, args=(i, ))
    t2 = threading.Thread(target=spk_embedding, args=(i+1, ))
    t3 = threading.Thread(target=spk_embedding, args=(i+2, ))
    t4 = threading.Thread(target=spk_embedding, args=(i+3, ))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()


