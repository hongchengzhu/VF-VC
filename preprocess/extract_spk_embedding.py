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
num_uttrs = 3
len_crop = 128

# Directory containing mel-spectrograms
rootDir = '/home/hongcz/alab/code/VF-VC-5-1/id_test/mel'
targetDir = '/home/hongcz/alab/code/VF-VC-5-1/id_test'
if not os.path.exists(targetDir):
    os.mkdir(targetDir)
subdirList = sorted(os.listdir(rootDir))


def spk_embedding(index):
    cnt = 0
    speaker = subdirList[index]
    print('Processing speaker: %s' % speaker)
    fileList = sorted(os.listdir(os.path.join(rootDir, speaker)))
    
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(rootDir, speaker, fileList[idx_uttrs[i]]))
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        while tmp.shape[0] <= len_crop:
            cnt += 1
            print(tmp.shape)
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(rootDir, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        left = np.random.randint(0, tmp.shape[0]-len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())     

    # tgt = os.path.join(targetDir, speaker+'_ge2e.txt')
    # if not os.path.exists(tgt):
    #     os.mkdir(tgt)
    # np.save(os.path.join(tgt, speaker), np.mean(embs, axis=0).astype('float32'), allow_pickle=False)

    # np.savetxt(tgt, np.mean(embs, axis=0), fmt='%f',
    #            delimiter=',')
    return embs


tmp = spk_embedding(0)
tmp = np.array(tmp)
np.savetxt('/home/hongcz/alab/code/VF-VC-5-1/id_test/spk_emb/ge2e_p225_3.txt', tmp, fmt='%f',
           delimiter=',')

spk_emb = np.zeros([200, 256])
for i in range(0, 2, 1):
    tmp = spk_embedding(i)
    for j in range(0, 1, 1):
        spk_emb[100*i+j, :] = tmp[j]
np.savetxt('/home/hongcz/alab/code/VF-VC-5-1/id_test/spk_emb/ge2e_infer_postflow.txt', spk_emb, fmt='%f', delimiter=',')


for i in range(0, 2, 2):
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


