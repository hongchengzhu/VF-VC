import pickle
import numpy as np
import os
from tqdm import tqdm

with open('../feature/wavlist13100.txt', 'r') as f:
    wavlist = sorted(f.readlines())

    mel_root = '/mnt/hongcz/feature/mel_hifigan_padding'
    content_root = '/mnt/hongcz/feature/wav2vec2_padding'

    mel_to_root = '/mnt/hongcz/feature/mel_hifigan_padding_alignment'
    content_to_root = '/mnt/hongcz/feature/wav2vec2_padding_alignment'

    # mel = np.load(os.path.join(mel_root, 'LJ001-0008.npy'))
    # content = pickle.load(open(os.path.join(content_root, 'LJ001-0008.pkl'), 'rb'))
    #
    # mel_alignment = np.load(os.path.join(mel_to_root, 'LJ001-0008.npy'))
    # content_alignment = pickle.load(open(os.path.join(content_to_root, 'LJ001-0008.pkl'), 'rb'))

    # print(1)
    #
    for i in tqdm(wavlist):
        mel = np.load(os.path.join(mel_root, i[:-1]+'.npy'))
        content = pickle.load(open(os.path.join(content_root, i[:-1]+'.pkl'), 'rb'))

        # aligned
        mel_tag = mel.shape[0]
        content_tag = content.shape[1]
        if mel_tag > content_tag:
            mel = mel[:content_tag, :]
        elif content_tag > mel_tag:
            content = content[:, :mel_tag, :]
            print(i[:-1] + ' 第2:', mel_tag, content_tag)

        np.save(os.path.join(mel_to_root, i[:-1]), mel.astype('float32'), allow_pickle=False)
        with open(os.path.join(content_to_root, i[:-1]+'.pkl'), 'wb') as f1:
            pickle.dump(content, f1)


