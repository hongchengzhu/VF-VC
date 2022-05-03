from torch.utils import data
import torch
import numpy as np
import pickle
import os
from torch.nn.utils.rnn import pad_sequence
from math import ceil


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[-1])/base))
    len_pad = len_out - x.shape[-1]
    assert len_pad >= 0
    return np.pad(x, ((0, 0), (0, 0), (0, len_pad)), 'constant')


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, config):
        """Initialize and preprocess the Utterances dataset."""
        # self.conten_dir = config['content_folder']
        self.mel_dir = config['mel_folder']
        self.spk_emb = config['spk_emb']

        # load utterances and decide which to load
        with open(config['training_data_list'], 'r') as f:
            dataset = f.readlines()

        mel_dataset = []
        # content_dataset = []
        speaker_embedding = []
        for i in sorted(dataset):
            spk = i[:4]
            mel_dataset.append(os.path.join(os.path.join(self.mel_dir, spk), i[:-1] + '.npy'))
            # content_dataset.append(os.path.join(os.path.join(self.conten_dir, spk), i[:-1] + '.pkl'))
            speaker_embedding.append((os.path.join(os.path.join(self.spk_emb, spk), spk + '.npy')))
        # self.train_dataset = [mel_dataset, content_dataset, speaker_embedding]
        self.train_dataset = [mel_dataset, speaker_embedding]
        self.lens = len(self.train_dataset[0])

        print('Finished loading the training dataset...')

    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset
        mel_index = dataset[0][index]
        # content_index = dataset[1][index]
        spk_index = dataset[1][index]

        tgt_mel = np.load(mel_index)
        tgt_mel = torch.tensor(tgt_mel)

        # content = pickle.load(open(content_index, 'rb'))

        spk_embed = torch.FloatTensor(np.load(spk_index))
        # return [content.squeeze(0), tgt_mel, spk_embed]
        return [tgt_mel, spk_embed]

    def __len__(self):
        """
        Return the length of training data.
        """
        return self.lens


def collate_fn_vf_vc(dataset):
    batch_size = len(dataset)

    # content = [dataset[i][0] for i in range(batch_size)]
    mel = [dataset[i][0] for i in range(batch_size)]
    spk_emb = [dataset[i][1] for i in range(batch_size)]

    # content_padding = pad_sequence(content, padding_value=0).permute(1, 0, 2)
    mel_padding = pad_sequence(mel, padding_value=0).permute(1, 2, 0)
    spk_emb = torch.stack(spk_emb, 0)

    mel_padding = torch.FloatTensor(pad_seq(mel_padding))

    nonpadding = (mel_padding != 0).float()[:, :]
    nonpadding_mean = torch.mean(nonpadding, 1, keepdim=True)
    nonpadding_mean[nonpadding_mean > 0] = 1

    return mel_padding, nonpadding_mean, spk_emb


def get_loader(config, num_workers=0):
    """
    Build and return a data loader.
    """

    dataset = Utterances(config)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['training_batch_size'],
                                  shuffle=True,
                                  num_workers=0,
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
        # self.conten_dir = config['content_folder']
        self.mel_dir = config['mel_folder']
        self.spk_emb = config['spk_emb']

        # load utterances and decide which to load
        with open(config['validation_data_list'], 'r') as f:
            dataset = f.readlines()

        mel_dataset = []
        # content_dataset = []
        speaker_embedding = []
        for i in sorted(dataset):
            spk = i[:4]
            mel_dataset.append(os.path.join(os.path.join(self.mel_dir, spk), i[:-1] + '.npy'))
            # content_dataset.append(os.path.join(os.path.join(self.conten_dir, spk), i[:-1] + '.pkl'))
            speaker_embedding.append((os.path.join(os.path.join(self.spk_emb, spk), spk + '.npy')))
        self.val_dataset = [mel_dataset, speaker_embedding]
        self.lens = len(self.val_dataset[0])

        print('Finished loading the validation dataset...')

    def __getitem__(self, index):
        dataset = self.val_dataset
        mel_index = dataset[0][index]
        # content_index = dataset[1][index]
        spk_index = dataset[1][index]

        tgt_mel = np.load(mel_index)
        tgt_mel = torch.tensor(tgt_mel)

        # content = pickle.load(open(content_index, 'rb'))

        spk_embed = torch.FloatTensor(np.load(spk_index))

        return [tgt_mel, spk_embed]

    def __len__(self):
        """
        Return the length of training data.
        """
        return self.lens


def validation_collate_fn_vf_vc(dataset):
    batch_size = len(dataset)

    # content = [dataset[i][0] for i in range(batch_size)]
    mel = [dataset[i][0] for i in range(batch_size)]
    spk_emb = [dataset[i][1] for i in range(batch_size)]

    # content_padding = pad_sequence(content, padding_value=0).permute(1, 0, 2)
    mel_padding = pad_sequence(mel, padding_value=0).permute(1, 2, 0)
    spk_emb = torch.stack(spk_emb, 0)

    mel_padding = torch.FloatTensor(pad_seq(mel_padding))

    nonpadding = (mel_padding != 0).float()[:, :]
    nonpadding_mean = torch.mean(nonpadding, 1, keepdim=True)
    nonpadding_mean[nonpadding_mean > 0] = 1

    return mel_padding, nonpadding_mean, spk_emb


def validation_get_loader(config, num_workers=0):
    """
    Build and return a data loader.
    """

    dataset = validation_Utterances(config)

    # worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['validation_batch_size'],
                                  shuffle=False,
                                  num_workers=0,
                                  drop_last=False,
                                  worker_init_fn=None,
                                  collate_fn=validation_collate_fn_vf_vc)

    return data_loader


