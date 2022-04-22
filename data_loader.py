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

    def __init__(self, config):
        """Initialize and preprocess the Utterances dataset."""
        self.conten_dir = config['content_folder']
        self.mel_dir = config['mel_folder']
        # self.step = 10


        # load utterances and decide which to load
        with open(config['training_data_list'], 'r') as f:
            dataset = f.readlines()

        # mel_dataset = copy.deepcopy(dataset)
        # random.shuffle(dataset)

        mel_dataset = []
        content_dataset = []
        for i in sorted(dataset):
            mel_dataset.append(os.path.join(self.mel_dir, i[:-1] + '.npy'))
            content_dataset.append(os.path.join(self.conten_dir, i[:-1] + '.pkl'))
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

        print('Finished loading the training dataset...')

    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset
        mel_index = dataset[0][index]
        content_index = dataset[1][index]

        # load tgt mel feature / content feature
        # tgt_mel = np.load(os.path.join(self.tgt_dir, mel_index+'.npy'))
        tgt_mel = np.load(mel_index)
        tgt_mel = torch.tensor(tgt_mel)

        # content = np.load(os.path.join(self.tgt_dir, content_index+'.npy'))
        # content = torch.tensor(content).to('cuda:0')
        content = pickle.load(open(content_index, 'rb'))

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

    # pad because the ConvTranspose

    nonpadding = (mel_padding != 0).float()[:, :]
    nonpadding_mean = torch.mean(nonpadding, 1, keepdim=True)
    nonpadding_mean[nonpadding_mean > 0] = 1

    # print(content_p_padding.shape, mel_p_padding.shape)

    return content_padding, mel_padding, nonpadding_mean

    # mel_nonpadding = torch.ones([batch_size, 1, mel_padding.shape[-1]]).to('cuda:0')

    # return content_padding, mel_padding, nonpadding_mean


def get_loader(config, num_workers=0):
    """
    Build and return a data loader.
    """

    dataset = Utterances(config)

    # worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['training_batch_size'],
                                  shuffle=True,
                                  num_workers=16,
                                  drop_last=False,
                                  worker_init_fn=None,
                                  collate_fn=collate_fn_vf_vc)

    return data_loader


################################################################################################################
# validation dataloader

class validation_Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, config):
        """Initialize and preprocess the Utterances dataset."""
        self.conten_dir = config['content_folder']
        self.mel_dir = config['mel_folder']
        # self.step = 10

        # load utterances and decide which to load
        with open(config['validation_data_list'], 'r') as f:
            dataset = f.readlines()

        # mel_dataset = copy.deepcopy(dataset)
        # random.shuffle(dataset)

        mel_dataset = []
        content_dataset = []
        for i in sorted(dataset):
            mel_dataset.append(os.path.join(self.mel_dir, i[:-1] + '.npy'))
            content_dataset.append(os.path.join(self.conten_dir, i[:-1] + '.pkl'))
        self.train_dataset = [mel_dataset, content_dataset]
        self.lens = len(self.train_dataset[0])

        print('Finished loading the validation dataset...')

    def __getitem__(self, index):
        dataset = self.train_dataset
        mel_index = dataset[0][index]
        content_index = dataset[1][index]

        tgt_mel = np.load(mel_index)
        tgt_mel = torch.tensor(tgt_mel)

        content = pickle.load(open(content_index, 'rb'))

        return [content.squeeze(0), tgt_mel]

    def __len__(self):
        """
        Return the length of training data.
        """
        return self.lens


def validation_collate_fn_vf_vc(dataset):
    batch_size = len(dataset)

    content = [dataset[i][0] for i in range(batch_size)]
    mel = [dataset[i][1] for i in range(batch_size)]

    content_padding = pad_sequence(content, padding_value=0).permute(1, 0, 2)
    mel_padding = pad_sequence(mel, padding_value=0).permute(1, 2, 0)

    # pad because the ConvTranspose
    nonpadding = (mel_padding != 0).float()[:, :]
    nonpadding_mean = torch.mean(nonpadding, 1, keepdim=True)
    nonpadding_mean[nonpadding_mean > 0] = 1

    return content_padding, mel_padding, nonpadding_mean


def validation_get_loader(config, num_workers=0):
    """
    Build and return a data loader.
    """

    dataset = validation_Utterances(config)

    # worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['validation_batch_size'],
                                  shuffle=False,
                                  num_workers=16,
                                  drop_last=False,
                                  worker_init_fn=None,
                                  collate_fn=validation_collate_fn_vf_vc)

    return data_loader


