from torch.utils import data
import torch
import numpy as np
import pickle
import os
import random
import copy
from torch.nn.utils.rnn import pad_sequence

from multiprocessing import Process, Manager


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self):
        """Initialize and preprocess the Utterances dataset."""
        self.conten_dir = './feature/wav2vec2/'
        self.tgt_dir = '/home/hongcz/alab/feature/mel128'
        # self.step = 10
        self.len_crop = 128

        # load utterances and decide which to load
        with open('./feature/wavlist13100.txt', 'r') as f:
            dataset = f.readlines()

        # mel_dataset = copy.deepcopy(dataset)
        # random.shuffle(dataset)

        mel_dataset = []
        content_dataset = []
        for i in sorted(dataset):
            mel_dataset.append('/home/hongcz/alab/feature/mel_s3prl/' + i[:-1] + '.npy')
            content_dataset.append('/home/hongcz/alab/feature/wav2vec2/' + i[:-1] + '.pkl')
        self.train_dataset = [mel_dataset, content_dataset]
        self.lens = len(self.train_dataset[0])

        # """Load data using multiprocessing"""
        # manager = Manager()
        # # meta = manager.list(meta)
        # dataset = manager.list([])
        # processes = []
        # for i in range(0, len(meta), self.step):
        #     p = Process(target=self.load_data,
        #                 args=(meta[i:i + self.step], mel_list[i:i+self.step], dataset, i))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()

        print('Finished loading the dataset...')

    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset
        mel_index = dataset[0][index]
        content_index = dataset[1][index]

        # load tgt mel feature / content feature
        # tgt_mel = np.load(os.path.join(self.tgt_dir, mel_index+'.npy'))
        tgt_mel = np.load(mel_index)
        tgt_mel = torch.tensor(tgt_mel).to('cuda:0')

        # content = np.load(os.path.join(self.tgt_dir, content_index+'.npy'))
        # content = torch.tensor(content).to('cuda:0')
        content = pickle.load(open(content_index, 'rb')).to('cuda:0')

        # # limit to 128
        # start_index = np.random.randint(0, tgt_mel.shape[0]-128)
        # tgt_mel = tgt_mel[start_index: start_index+128, :]
        #
        # content = content[start_index: start_index+128, :]

        # load content feature as condition
        # content = pickle.load(open(os.path.join(self.conten_dir, wav_index+'.pkl'), "rb"))
        # content = tgt_mel
        return [content.squeeze(0), tgt_mel]


    def __len__(self):
        """
        Return the length of training data.
        """
        return self.lens


def collate_fn_vf_vc(dataset):
    batch_size = len(dataset)

    content = [dataset[i][0] for i in range(batch_size)]
    mel = [dataset[i][1] for i in range(batch_size)]

    content_padding = pad_sequence(content, padding_value=0).permute(1, 0, 2)
    mel_padding = pad_sequence(mel, padding_value=0).permute(1, 2, 0)

    nonpadding = (mel_padding != 0).float()[:, :]
    nonpadding_mean = torch.mean(nonpadding, 1, keepdim=True)
    nonpadding_mean[nonpadding_mean > 0] = 1

    # pad because the ConvTranspose
    tag = mel_padding.shape[-1] % 4
    if tag != 0:
        mel_p_padding = torch.zeros([batch_size, 80, mel_padding.shape[-1] + (4 - tag)]).to('cuda:0')
        content_p_padding = torch.zeros([batch_size, content_padding.shape[1] + (4 - tag), 768]).to('cuda:0')

        mel_p_padding[:, :, :mel_padding.shape[-1]] = mel_padding
        content_p_padding[:, :content_padding.shape[1], :] = content_padding

        nonpadding = (mel_p_padding != 0).float()[:, :]
        nonpadding_mean = torch.mean(nonpadding, 1, keepdim=True)
        nonpadding_mean[nonpadding_mean > 0] = 1

        return content_p_padding, mel_p_padding, nonpadding_mean

    # mel_nonpadding = torch.ones([batch_size, 1, mel_padding.shape[-1]]).to('cuda:0')

    return content_padding, mel_padding, nonpadding_mean


def get_loader( batch_size=16, num_workers=0):
    """
    Build and return a data loader.
    """

    dataset = Utterances()

    # worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    data_loader = data.DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        drop_last=False,
                                        worker_init_fn=None,
                                        collate_fn=collate_fn_vf_vc)

    return data_loader

